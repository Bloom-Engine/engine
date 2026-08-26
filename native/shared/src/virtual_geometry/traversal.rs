use super::hiz::{
    GpuVirtualHiZ, VirtualGeometryHiZFrame, VIRTUAL_HIZ_MIP_COUNT,
    VIRTUAL_HIZ_SELECTION_PARAMS_BYTES,
};
use super::{
    GpuVirtualGeometryPool, VirtualGeometryGpuError, VirtualMeshId,
    GPU_VIRTUAL_MESH_MATERIALS_BOUND,
};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
use super::gpu_pool::MAX_GPU_HIERARCHY_LEVELS;

const WORKGROUP_SIZE: u32 = 64;
const INSTANCE_CONE_CULL_SAFE: u32 = 1 << 0;
const INSTANCE_NEGATIVE_DETERMINANT: u32 = 1 << 1;
const INSTANCE_PREVIOUS_HIZ_ELIGIBLE: u32 = 1 << 2;
const SELECTED_VERTEX_ENCODING_SHIFT: u32 = 28;
const SELECTED_VERTEX_ENCODING_MASK: u32 = 3;
const ID_SLOT_MASK: u32 = (1 << 20) - 1;
const ALL_SOURCE_MESHES: u32 = u32::MAX;
static NEXT_SELECTOR_ID: AtomicU64 = AtomicU64::new(1);

/// Fixed GPU input record for one virtual-geometry instance (208 bytes).
///
/// The first 128 bytes are the traversal-hot current transform, normal rows,
/// and identity. Previous transform and tint are an appended render prefix so
/// later visibility/PBR work can reproduce the established velocity and color
/// contract without making hierarchy selection fetch a separate record.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVirtualInstance {
    model: [[f32; 4]; 4],
    normal_rows: [[f32; 4]; 3],
    /// mesh ID, caller-stable instance ID, flags, source glTF mesh filter.
    /// `u32::MAX` admits every source mesh for single-mesh/procedural assets.
    instance_info: [u32; 4],
    previous_model: [[f32; 4]; 4],
    model_tint: [f32; 4],
}

impl GpuVirtualInstance {
    pub fn new(
        mesh: VirtualMeshId,
        instance_id: u32,
        model: [[f32; 4]; 4],
    ) -> Result<Self, VirtualGeometryTraversalError> {
        Self::with_render_state(mesh, instance_id, model, model, [1.0; 4])
    }

    pub fn with_render_state(
        mesh: VirtualMeshId,
        instance_id: u32,
        model: [[f32; 4]; 4],
        previous_model: [[f32; 4]; 4],
        model_tint: [f32; 4],
    ) -> Result<Self, VirtualGeometryTraversalError> {
        Self::with_source_mesh_render_state(
            mesh,
            ALL_SOURCE_MESHES,
            instance_id,
            model,
            previous_model,
            model_tint,
        )
    }

    /// Create one placement of a specific source glTF mesh within a shared
    /// multi-mesh `.bgeo` archive. Traversal admits only clusters whose cooked
    /// `mesh_index` matches `source_mesh_index`; compatibility primitives from
    /// that source mesh remain owned by the ordinary renderer.
    pub fn for_source_mesh(
        mesh: VirtualMeshId,
        source_mesh_index: u32,
        instance_id: u32,
        model: [[f32; 4]; 4],
    ) -> Result<Self, VirtualGeometryTraversalError> {
        Self::with_source_mesh_render_state(
            mesh,
            source_mesh_index,
            instance_id,
            model,
            model,
            [1.0; 4],
        )
    }

    /// Source-mesh-filtered form with the complete temporal/material state.
    pub fn with_source_mesh_render_state(
        mesh: VirtualMeshId,
        source_mesh_index: u32,
        instance_id: u32,
        model: [[f32; 4]; 4],
        previous_model: [[f32; 4]; 4],
        model_tint: [f32; 4],
    ) -> Result<Self, VirtualGeometryTraversalError> {
        let (normal_rows, cone_safe, negative_determinant) = normal_rows_and_cone_safety(model)
            .ok_or(VirtualGeometryTraversalError::InvalidInstanceTransform {
                instance: instance_id,
            })?;
        if !finite_affine(previous_model) || !model_tint.iter().all(|value| value.is_finite()) {
            return Err(VirtualGeometryTraversalError::InvalidInstanceTransform {
                instance: instance_id,
            });
        }
        Ok(Self {
            model,
            normal_rows,
            instance_info: [
                mesh.raw(),
                instance_id,
                (u32::from(cone_safe) * INSTANCE_CONE_CULL_SAFE)
                    | (u32::from(negative_determinant) * INSTANCE_NEGATIVE_DETERMINANT),
                source_mesh_index,
            ],
            previous_model,
            model_tint,
        })
    }

    pub fn identity(mesh: VirtualMeshId, instance_id: u32) -> Self {
        Self::new(
            mesh,
            instance_id,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        .expect("identity is an invertible cone-safe transform")
    }

    pub const fn mesh_id(self) -> VirtualMeshId {
        VirtualMeshId::from_raw(self.instance_info[0])
    }

    pub const fn instance_id(self) -> u32 {
        self.instance_info[1]
    }

    /// The selected source glTF mesh, or `None` when this instance admits the
    /// complete archive (the established single-mesh/procedural behavior).
    pub const fn source_mesh_index(self) -> Option<u32> {
        if self.instance_info[3] == ALL_SOURCE_MESHES {
            None
        } else {
            Some(self.instance_info[3])
        }
    }

    pub const fn cone_cull_safe(self) -> bool {
        self.instance_info[2] & INSTANCE_CONE_CULL_SAFE != 0
    }

    pub const fn negative_determinant(self) -> bool {
        self.instance_info[2] & INSTANCE_NEGATIVE_DETERMINANT != 0
    }

    pub const fn model(self) -> [[f32; 4]; 4] {
        self.model
    }

    pub const fn normal_rows(self) -> [[f32; 4]; 3] {
        self.normal_rows
    }

    pub const fn previous_model(self) -> [[f32; 4]; 4] {
        self.previous_model
    }

    pub const fn model_tint(self) -> [f32; 4] {
        self.model_tint
    }

    pub(crate) const fn history_identity(self) -> [u32; 3] {
        [
            self.instance_info[0],
            self.instance_info[1],
            self.instance_info[3],
        ]
    }

    pub(crate) fn set_previous_hiz_eligible(&mut self, eligible: bool) {
        self.instance_info[2] = (self.instance_info[2] & !INSTANCE_PREVIOUS_HIZ_ELIGIBLE)
            | u32::from(eligible) * INSTANCE_PREVIOUS_HIZ_ELIGIBLE;
    }
}

const _: () = assert!(std::mem::size_of::<GpuVirtualInstance>() == 208);

/// Camera data for projected-error hierarchy selection.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VirtualGeometryView {
    /// World-space inward-facing planes. Plane normals need not be normalized.
    pub frustum_planes: [[f32; 4]; 6],
    /// Column-major world-to-clip transform.
    pub view_projection: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    /// Pixels per unit at clip `w == 1` (normally half the render height times
    /// the absolute vertical projection scale).
    pub projection_scale: f32,
    pub target_error_pixels: f32,
}

/// One fixed-size selected cluster record consumed by later indirect emission.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd, bytemuck::Pod, bytemuck::Zeroable,
)]
pub struct GpuSelectedVirtualCluster {
    pub mesh_id: u32,
    /// Dense index into the exact instance buffer used for this dispatch.
    pub instance_index: u32,
    /// Absolute index into the pool's GPU cluster table.
    pub cluster_table_index: u32,
    /// Absolute byte base of the selected resident physical page.
    pub physical_page_base: u32,
    pub lod_level: u32,
    pub triangle_count: u32,
    /// Generation-safe renderer material ID; admitted selections never use zero.
    pub material_id: u32,
    /// Low bits retain cooked cluster flags; high bits carry vertex encoding.
    pub flags: u32,
}

impl GpuSelectedVirtualCluster {
    /// Cooked vertex encoding packed into the render-ready selection record.
    pub const fn vertex_encoding(self) -> u32 {
        self.flags >> SELECTED_VERTEX_ENCODING_SHIFT & SELECTED_VERTEX_ENCODING_MASK
    }
}

/// A bounded request for a logical page that prevented hierarchy refinement.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd, bytemuck::Pod, bytemuck::Zeroable,
)]
pub struct GpuVirtualPageRequest {
    pub mesh_id: u32,
    pub page_index: u32,
    pub instance_id: u32,
    pub source_cluster: u32,
}

/// GPU-written traversal telemetry. Attempted counts can exceed output
/// capacities; consumers must use the corresponding overflow fields.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVirtualTraversalCounters {
    pub selected_count: u32,
    pub page_request_count: u32,
    pub visible_groups: u32,
    pub frustum_culled_groups: u32,
    pub cone_culled_clusters: u32,
    pub refined_groups: u32,
    pub fallback_groups: u32,
    pub missing_current_pages: u32,
    pub selected_overflow: u32,
    pub request_overflow: u32,
    pub invalid_records: u32,
    pub depth_limit_fallbacks: u32,
    pub occlusion_culled_groups: u32,
    pub occlusion_uncertain_groups: u32,
}

const _: () = assert!(std::mem::size_of::<GpuSelectedVirtualCluster>() == 32);
const _: () = assert!(std::mem::size_of::<GpuVirtualPageRequest>() == 16);
const _: () = assert!(std::mem::size_of::<GpuVirtualTraversalCounters>() == 56);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GpuVirtualTraversalConfig {
    pub max_instances: u32,
    pub max_selected_clusters: u32,
    pub max_page_requests: u32,
}

impl Default for GpuVirtualTraversalConfig {
    fn default() -> Self {
        Self {
            max_instances: 65_535,
            max_selected_clusters: 1_048_576,
            max_page_requests: 65_536,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct VirtualGeometryTraversalDispatch {
    pub instance_count: u32,
    pub maximum_root_clusters: u32,
    pub workgroups_x: u32,
    pub workgroups_y: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TraversalParams {
    planes: [[f32; 4]; 6],
    view_projection: [[f32; 4]; 4],
    camera_projection: [f32; 4],
    thresholds: [f32; 4],
    dispatch: [u32; 4],
    limits: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<TraversalParams>() == 224);

/// Explicit GPU hierarchy selector with fixed-capacity transient buffers.
/// `Renderer` owns it only after virtual geometry is explicitly enabled, so an
/// ordinary renderer retains no selector cost or pixel effect.
pub struct GpuVirtualHierarchySelector {
    id: u64,
    config: GpuVirtualTraversalConfig,
    pool_id: u64,
    instance_buffer: wgpu::Buffer,
    selected_buffer: wgpu::Buffer,
    request_buffer: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    hiz: GpuVirtualHiZ,
    pipeline: wgpu::ComputePipeline,
    max_workgroups_per_dimension: u32,
}

impl GpuVirtualHierarchySelector {
    pub fn new(
        device: &wgpu::Device,
        pool: &GpuVirtualGeometryPool,
        config: GpuVirtualTraversalConfig,
    ) -> Result<Self, VirtualGeometryTraversalError> {
        validate_selector_config(device, config)?;
        let instance_buffer = create_buffer(
            device,
            "virtual_geometry_instances",
            buffer_bytes::<GpuVirtualInstance>(config.max_instances),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let selected_buffer = create_buffer(
            device,
            "virtual_geometry_selected_clusters",
            buffer_bytes::<GpuSelectedVirtualCluster>(config.max_selected_clusters),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let request_buffer = create_buffer(
            device,
            "virtual_geometry_page_requests",
            buffer_bytes::<GpuVirtualPageRequest>(config.max_page_requests),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let counter_buffer = create_buffer(
            device,
            "virtual_geometry_traversal_counters",
            std::mem::size_of::<GpuVirtualTraversalCounters>() as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let params_buffer = create_buffer(
            device,
            "virtual_geometry_traversal_params",
            std::mem::size_of::<TraversalParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let hiz = GpuVirtualHiZ::new(device);
        let layout = create_layout(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("virtual_geometry_traversal_bind_group"),
            layout: &layout,
            entries: &[
                binding(0, pool.mesh_table_buffer()),
                binding(1, pool.page_table_buffer()),
                binding(2, pool.cluster_table_buffer()),
                binding(3, &instance_buffer),
                binding(4, &selected_buffer),
                binding(5, &request_buffer),
                binding(6, &counter_buffer),
                binding(7, &params_buffer),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("virtual_geometry_traversal_shader"),
            source: wgpu::ShaderSource::Wgsl(TRAVERSAL_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("virtual_geometry_traversal_pipeline_layout"),
            bind_group_layouts: &[Some(&layout), Some(hiz.sample_layout())],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("virtual_geometry_traversal_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("select_virtual_clusters"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            id: NEXT_SELECTOR_ID.fetch_add(1, Ordering::Relaxed),
            config,
            pool_id: pool.id(),
            instance_buffer,
            selected_buffer,
            request_buffer,
            counter_buffer,
            params_buffer,
            bind_group,
            hiz,
            pipeline,
            max_workgroups_per_dimension: device.limits().max_compute_workgroups_per_dimension,
        })
    }

    pub fn record(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pool: &GpuVirtualGeometryPool,
        instances: &[GpuVirtualInstance],
        view: VirtualGeometryView,
    ) -> Result<VirtualGeometryTraversalDispatch, VirtualGeometryTraversalError> {
        self.record_internal(queue, encoder, pool, instances, view, None)
    }

    pub(crate) fn record_with_previous_hiz(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pool: &GpuVirtualGeometryPool,
        instances: &[GpuVirtualInstance],
        view: VirtualGeometryView,
        hiz_frame: VirtualGeometryHiZFrame,
    ) -> Result<VirtualGeometryTraversalDispatch, VirtualGeometryTraversalError> {
        self.record_internal(queue, encoder, pool, instances, view, Some(hiz_frame))
    }

    fn record_internal(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pool: &GpuVirtualGeometryPool,
        instances: &[GpuVirtualInstance],
        view: VirtualGeometryView,
        hiz_frame: Option<VirtualGeometryHiZFrame>,
    ) -> Result<VirtualGeometryTraversalDispatch, VirtualGeometryTraversalError> {
        if pool.id() != self.pool_id {
            return Err(VirtualGeometryTraversalError::PoolMismatch);
        }
        validate_view(view)?;
        if instances.len() > self.config.max_instances as usize {
            return Err(VirtualGeometryTraversalError::TooManyInstances {
                requested: instances.len(),
                capacity: self.config.max_instances,
            });
        }

        let mut maximum_root_clusters = 0;
        for instance in instances {
            validate_instance(*instance)?;
            validate_source_mesh_filter(pool, *instance)?;
            let mesh = pool.mesh_entry(instance.mesh_id())?;
            if mesh.flags & GPU_VIRTUAL_MESH_MATERIALS_BOUND == 0 {
                return Err(VirtualGeometryTraversalError::UnboundMaterials {
                    mesh: instance.mesh_id(),
                });
            }
            maximum_root_clusters = maximum_root_clusters.max(mesh.root_cluster_count);
        }
        let workgroups_x = maximum_root_clusters.div_ceil(WORKGROUP_SIZE);
        if workgroups_x > self.max_workgroups_per_dimension {
            return Err(VirtualGeometryTraversalError::DispatchLimitExceeded {
                requested: workgroups_x,
                maximum: self.max_workgroups_per_dimension,
            });
        }

        queue.write_buffer(
            &self.counter_buffer,
            0,
            bytemuck::bytes_of(&GpuVirtualTraversalCounters::default()),
        );
        let params = TraversalParams {
            planes: view.frustum_planes,
            view_projection: view.view_projection,
            camera_projection: [
                view.camera_position[0],
                view.camera_position[1],
                view.camera_position[2],
                view.projection_scale,
            ],
            thresholds: [view.target_error_pixels, 1.0e-5, 0.0, 0.0],
            dispatch: [
                instances.len() as u32,
                maximum_root_clusters,
                self.config.max_selected_clusters,
                self.config.max_page_requests,
            ],
            limits: [
                pool.config().max_hierarchy_levels,
                pool.config().max_clusters_per_group,
                0,
                0,
            ],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        let fallback_hiz_frame = VirtualGeometryHiZFrame {
            frame_index: 0,
            view_projection: view.view_projection,
            view: identity_matrix(),
            render_extent: (1, 1),
            camera_cut: true,
        };
        let hiz_frame = hiz_frame.unwrap_or(fallback_hiz_frame);
        self.hiz.prepare_selection(
            queue,
            hiz_frame,
            hiz_frame.frame_index != 0 && self.hiz.history_valid_for(hiz_frame),
        );

        if !instances.is_empty() {
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
        }
        if workgroups_x != 0 && !instances.is_empty() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("virtual_geometry_hierarchy_selection"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, self.hiz.sample_bind_group(), &[]);
            pass.dispatch_workgroups(workgroups_x, instances.len() as u32, 1);
        }

        Ok(VirtualGeometryTraversalDispatch {
            instance_count: instances.len() as u32,
            maximum_root_clusters,
            workgroups_x,
            workgroups_y: instances.len() as u32,
        })
    }

    pub(crate) fn previous_hiz_history_valid(&self, frame: VirtualGeometryHiZFrame) -> bool {
        self.hiz.history_valid_for(frame)
    }

    pub(crate) fn previous_hiz_contains(&self, instance: GpuVirtualInstance) -> bool {
        self.hiz.instance_was_captured(instance)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_previous_hiz_capture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        source_size: (u32, u32),
        frame: VirtualGeometryHiZFrame,
        instances: &[GpuVirtualInstance],
    ) {
        self.hiz.record_capture(
            device,
            queue,
            encoder,
            source,
            source_size,
            frame,
            instances,
        );
    }

    pub(crate) fn after_submit_previous_hiz(&mut self) {
        self.hiz.after_submit();
    }

    pub(crate) fn invalidate_previous_hiz(&mut self, source_recreated: bool) {
        self.hiz.invalidate(source_recreated);
    }

    pub fn previous_hiz_telemetry(&self) -> super::GpuVirtualHiZTelemetry {
        self.hiz.telemetry()
    }

    pub const fn config(&self) -> GpuVirtualTraversalConfig {
        self.config
    }

    pub fn selected_buffer(&self) -> &wgpu::Buffer {
        &self.selected_buffer
    }

    pub fn instance_buffer(&self) -> &wgpu::Buffer {
        &self.instance_buffer
    }

    pub fn page_request_buffer(&self) -> &wgpu::Buffer {
        &self.request_buffer
    }

    pub fn counter_buffer(&self) -> &wgpu::Buffer {
        &self.counter_buffer
    }

    pub(super) const fn id(&self) -> u64 {
        self.id
    }

    pub(super) const fn pool_id(&self) -> u64 {
        self.pool_id
    }
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let mut entries = (0..7)
        .map(|binding| storage(binding, binding < 4))
        .collect::<Vec<_>>();
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 7,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("virtual_geometry_traversal_layout"),
        entries: &entries,
    })
}

fn create_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

const fn buffer_bytes<T>(records: u32) -> u64 {
    records as u64 * std::mem::size_of::<T>() as u64
}

fn validate_selector_config(
    device: &wgpu::Device,
    config: GpuVirtualTraversalConfig,
) -> Result<(), VirtualGeometryTraversalError> {
    if config.max_instances == 0
        || config.max_selected_clusters == 0
        || config.max_page_requests == 0
    {
        return Err(VirtualGeometryTraversalError::InvalidConfig);
    }
    let limits = device.limits();
    if limits.max_storage_buffers_per_shader_stage < 7
        || limits.max_bind_groups < 2
        || limits.max_sampled_textures_per_shader_stage < VIRTUAL_HIZ_MIP_COUNT
        || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE
        || limits.max_compute_workgroup_size_x < WORKGROUP_SIZE
        || limits.max_uniform_buffer_binding_size
            < (std::mem::size_of::<TraversalParams>() as u64)
                .max(VIRTUAL_HIZ_SELECTION_PARAMS_BYTES)
        || config.max_instances > limits.max_compute_workgroups_per_dimension
    {
        return Err(VirtualGeometryTraversalError::DeviceUnsupported);
    }
    for (resource, bytes) in [
        (
            "instance table",
            buffer_bytes::<GpuVirtualInstance>(config.max_instances),
        ),
        (
            "selected cluster table",
            buffer_bytes::<GpuSelectedVirtualCluster>(config.max_selected_clusters),
        ),
        (
            "page request table",
            buffer_bytes::<GpuVirtualPageRequest>(config.max_page_requests),
        ),
    ] {
        if bytes > limits.max_buffer_size || bytes > limits.max_storage_buffer_binding_size {
            return Err(VirtualGeometryTraversalError::DeviceLimitExceeded {
                resource,
                requested_bytes: bytes,
                maximum_bytes: limits
                    .max_buffer_size
                    .min(limits.max_storage_buffer_binding_size),
            });
        }
    }
    Ok(())
}

fn validate_view(view: VirtualGeometryView) -> Result<(), VirtualGeometryTraversalError> {
    let finite = view
        .frustum_planes
        .iter()
        .flatten()
        .chain(view.view_projection.iter().flatten())
        .chain(view.camera_position.iter())
        .chain([view.projection_scale, view.target_error_pixels].iter())
        .all(|value| value.is_finite());
    if !finite || view.projection_scale <= 0.0 || view.target_error_pixels < 0.0 {
        return Err(VirtualGeometryTraversalError::InvalidView);
    }
    Ok(())
}

fn validate_instance(instance: GpuVirtualInstance) -> Result<(), VirtualGeometryTraversalError> {
    let finite = instance
        .model
        .iter()
        .flatten()
        .chain(instance.normal_rows.iter().flatten())
        .chain(instance.previous_model.iter().flatten())
        .chain(instance.model_tint.iter())
        .all(|value| value.is_finite());
    let Some((expected_normal_rows, expected_cone_safe, expected_negative_determinant)) =
        normal_rows_and_cone_safety(instance.model)
    else {
        return Err(VirtualGeometryTraversalError::InvalidInstanceTransform {
            instance: instance.instance_id(),
        });
    };
    let expected_flags = (u32::from(expected_cone_safe) * INSTANCE_CONE_CULL_SAFE)
        | (u32::from(expected_negative_determinant) * INSTANCE_NEGATIVE_DETERMINANT);
    if !finite
        || instance.instance_info[0] & ID_SLOT_MASK == 0
        || instance.normal_rows != expected_normal_rows
        || instance.instance_info[2] & !INSTANCE_PREVIOUS_HIZ_ELIGIBLE != expected_flags
        || !finite_affine(instance.previous_model)
    {
        return Err(VirtualGeometryTraversalError::InvalidInstanceTransform {
            instance: instance.instance_id(),
        });
    }
    Ok(())
}

const fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn validate_source_mesh_filter(
    pool: &GpuVirtualGeometryPool,
    instance: GpuVirtualInstance,
) -> Result<(), VirtualGeometryTraversalError> {
    let mesh = instance.mesh_id();
    let archive = pool.asset(mesh)?.archive();
    match instance.source_mesh_index() {
        Some(source_mesh_index) => {
            if archive
                .clusters
                .iter()
                .any(|cluster| cluster.mesh_index == source_mesh_index)
            {
                Ok(())
            } else {
                Err(VirtualGeometryTraversalError::SourceMeshNotVirtual {
                    mesh,
                    source_mesh_index,
                })
            }
        }
        None => {
            let first_source_mesh = archive
                .clusters
                .first()
                .map(|cluster| cluster.mesh_index)
                .unwrap_or(0);
            if archive
                .clusters
                .iter()
                .any(|cluster| cluster.mesh_index != first_source_mesh)
            {
                Err(VirtualGeometryTraversalError::SourceMeshFilterRequired { mesh })
            } else {
                Ok(())
            }
        }
    }
}

fn finite_affine(model: [[f32; 4]; 4]) -> bool {
    model.iter().flatten().all(|value| value.is_finite())
        && model[0][3].abs() <= 1.0e-6
        && model[1][3].abs() <= 1.0e-6
        && model[2][3].abs() <= 1.0e-6
        && (model[3][3] - 1.0).abs() <= 1.0e-6
}

fn normal_rows_and_cone_safety(model: [[f32; 4]; 4]) -> Option<([[f32; 4]; 3], bool, bool)> {
    if !finite_affine(model) {
        return None;
    }
    let a00 = model[0][0];
    let a01 = model[1][0];
    let a02 = model[2][0];
    let a10 = model[0][1];
    let a11 = model[1][1];
    let a12 = model[2][1];
    let a20 = model[0][2];
    let a21 = model[1][2];
    let a22 = model[2][2];
    let cofactors = [
        [
            a11 * a22 - a12 * a21,
            a12 * a20 - a10 * a22,
            a10 * a21 - a11 * a20,
        ],
        [
            a02 * a21 - a01 * a22,
            a00 * a22 - a02 * a20,
            a01 * a20 - a00 * a21,
        ],
        [
            a01 * a12 - a02 * a11,
            a02 * a10 - a00 * a12,
            a00 * a11 - a01 * a10,
        ],
    ];
    let determinant = a00 * cofactors[0][0] + a01 * cofactors[0][1] + a02 * cofactors[0][2];
    if !determinant.is_finite() || determinant.abs() <= 1.0e-12 {
        return None;
    }
    let inverse_determinant = determinant.recip();
    let normal_rows = std::array::from_fn(|row| {
        [
            cofactors[row][0] * inverse_determinant,
            cofactors[row][1] * inverse_determinant,
            cofactors[row][2] * inverse_determinant,
            0.0,
        ]
    });

    let columns = [[a00, a10, a20], [a01, a11, a21], [a02, a12, a22]];
    let squared = columns.map(|column| dot3(column, column));
    let scale2 = squared.into_iter().fold(0.0, f32::max);
    let tolerance = scale2.max(1.0) * 1.0e-4;
    let cone_safe = squared
        .iter()
        .all(|length2| (*length2 - scale2).abs() <= tolerance)
        && dot3(columns[0], columns[1]).abs() <= tolerance
        && dot3(columns[0], columns[2]).abs() <= tolerance
        && dot3(columns[1], columns[2]).abs() <= tolerance;
    Some((normal_rows, cone_safe, determinant < 0.0))
}

const fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VirtualGeometryTraversalError {
    InvalidConfig,
    DeviceUnsupported,
    DeviceLimitExceeded {
        resource: &'static str,
        requested_bytes: u64,
        maximum_bytes: u64,
    },
    PoolMismatch,
    TooManyInstances {
        requested: usize,
        capacity: u32,
    },
    InvalidView,
    InvalidInstanceTransform {
        instance: u32,
    },
    UnboundMaterials {
        mesh: VirtualMeshId,
    },
    SourceMeshFilterRequired {
        mesh: VirtualMeshId,
    },
    SourceMeshNotVirtual {
        mesh: VirtualMeshId,
        source_mesh_index: u32,
    },
    DispatchLimitExceeded {
        requested: u32,
        maximum: u32,
    },
    Pool(VirtualGeometryGpuError),
}

impl From<VirtualGeometryGpuError> for VirtualGeometryTraversalError {
    fn from(value: VirtualGeometryGpuError) -> Self {
        Self::Pool(value)
    }
}

impl fmt::Display for VirtualGeometryTraversalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(formatter, "invalid virtual-geometry traversal config"),
            Self::DeviceUnsupported => write!(
                formatter,
                "device lacks the storage-buffer or compute limits required by virtual-geometry traversal"
            ),
            Self::DeviceLimitExceeded {
                resource,
                requested_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "virtual-geometry traversal {resource} requires {requested_bytes} bytes but the device limit is {maximum_bytes}"
            ),
            Self::PoolMismatch => write!(
                formatter,
                "virtual-geometry selector was recorded with a different page pool"
            ),
            Self::TooManyInstances {
                requested,
                capacity,
            } => write!(
                formatter,
                "virtual-geometry traversal received {requested} instances but has capacity for {capacity}"
            ),
            Self::InvalidView => write!(formatter, "invalid virtual-geometry camera data"),
            Self::InvalidInstanceTransform { instance } => write!(
                formatter,
                "virtual-geometry instance {instance} has a non-finite or singular transform"
            ),
            Self::UnboundMaterials { mesh } => write!(
                formatter,
                "virtual mesh {} has no complete GPU material binding",
                mesh.raw()
            ),
            Self::SourceMeshFilterRequired { mesh } => write!(
                formatter,
                "multi-source virtual mesh {} requires an explicit source glTF mesh filter",
                mesh.raw()
            ),
            Self::SourceMeshNotVirtual {
                mesh,
                source_mesh_index,
            } => write!(
                formatter,
                "virtual mesh {} has no eligible clusters for source glTF mesh {}",
                mesh.raw(), source_mesh_index
            ),
            Self::DispatchLimitExceeded { requested, maximum } => write!(
                formatter,
                "virtual-geometry traversal needs {requested} workgroups in one dimension but the device limit is {maximum}"
            ),
            Self::Pool(error) => error.fmt(formatter),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct CpuTraversalResult {
    pub selected: Vec<GpuSelectedVirtualCluster>,
    pub requests: Vec<GpuVirtualPageRequest>,
    pub counters: GpuVirtualTraversalCounters,
}

#[cfg(test)]
pub(super) fn select_cpu_reference(
    pool: &GpuVirtualGeometryPool,
    config: GpuVirtualTraversalConfig,
    instances: &[GpuVirtualInstance],
    view: VirtualGeometryView,
) -> Result<CpuTraversalResult, VirtualGeometryTraversalError> {
    validate_view(view)?;
    if instances.len() > config.max_instances as usize {
        return Err(VirtualGeometryTraversalError::TooManyInstances {
            requested: instances.len(),
            capacity: config.max_instances,
        });
    }
    let mut result = CpuTraversalResult::default();
    for (instance_index, instance) in instances.iter().enumerate() {
        validate_instance(*instance)?;
        validate_source_mesh_filter(pool, *instance)?;
        let mesh_id = instance.mesh_id();
        let mesh_entry = pool.mesh_entry(mesh_id)?;
        if mesh_entry.flags & GPU_VIRTUAL_MESH_MATERIALS_BOUND == 0 {
            return Err(VirtualGeometryTraversalError::UnboundMaterials { mesh: mesh_id });
        }
        let archive = pool.asset(mesh_id)?.archive();
        for root_index in 0..mesh_entry.root_cluster_count {
            let root = &archive.clusters[root_index as usize];
            if instance
                .source_mesh_index()
                .is_some_and(|source_mesh| root.mesh_index != source_mesh)
            {
                continue;
            }
            let (mut group_first, mut group_count) = (root_index, 1u32);
            if root.child_count != 0 {
                let first_child = &archive.clusters[root.first_child as usize];
                if first_child.parent != bloom_geometry_format::NO_RELATION
                    && first_child.parent_count != 0
                {
                    group_first = first_child.parent;
                    group_count = first_child.parent_count;
                }
            }
            if root_index != group_first {
                continue;
            }

            let scale = cpu_scale_bound(instance.model);
            let mut exhausted_depth = true;
            for _depth in 0..pool.config().max_hierarchy_levels {
                let range = group_first as usize..(group_first + group_count) as usize;
                let mut visible = false;
                let mut maximum_error = 0.0f32;
                for cluster in &archive.clusters[range.clone()] {
                    let sphere = cpu_world_sphere(cluster, *instance, scale);
                    if !cpu_sphere_outside_frustum(sphere, view.frustum_planes) {
                        visible = true;
                        maximum_error =
                            maximum_error.max(cpu_projected_error(cluster, sphere, scale, view));
                    }
                }
                if !visible {
                    result.counters.frustum_culled_groups += 1;
                    exhausted_depth = false;
                    break;
                }
                result.counters.visible_groups += 1;

                let first = &archive.clusters[group_first as usize];
                let wants_refinement =
                    maximum_error > view.target_error_pixels && first.child_count != 0;
                if !wants_refinement {
                    cpu_select_group(
                        pool,
                        config,
                        mesh_id,
                        instance_index as u32,
                        *instance,
                        group_first,
                        group_count,
                        scale,
                        view,
                        &mut result,
                    )?;
                    exhausted_depth = false;
                    break;
                }
                if !cpu_group_is_resident(pool, mesh_id, first.first_child, first.child_count)? {
                    result.counters.fallback_groups += 1;
                    cpu_emit_missing_requests(
                        pool,
                        config,
                        mesh_id,
                        instance.instance_id(),
                        first.first_child,
                        first.child_count,
                        &mut result,
                    )?;
                    cpu_select_group(
                        pool,
                        config,
                        mesh_id,
                        instance_index as u32,
                        *instance,
                        group_first,
                        group_count,
                        scale,
                        view,
                        &mut result,
                    )?;
                    exhausted_depth = false;
                    break;
                }
                result.counters.refined_groups += 1;
                group_first = first.first_child;
                group_count = first.child_count;
            }
            if exhausted_depth {
                result.counters.depth_limit_fallbacks += 1;
                cpu_select_group(
                    pool,
                    config,
                    mesh_id,
                    instance_index as u32,
                    *instance,
                    group_first,
                    group_count,
                    scale,
                    view,
                    &mut result,
                )?;
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
fn cpu_group_is_resident(
    pool: &GpuVirtualGeometryPool,
    mesh_id: VirtualMeshId,
    first: u32,
    count: u32,
) -> Result<bool, VirtualGeometryTraversalError> {
    let archive = pool.asset(mesh_id)?.archive();
    for cluster in &archive.clusters[first as usize..(first + count) as usize] {
        let page = pool.page_entry(super::VirtualPageId {
            mesh: mesh_id,
            page_index: cluster.page_index,
        })?;
        if page.slot_plus_one == 0
            || page.mesh_id != mesh_id.raw()
            || page.flags & super::GPU_VIRTUAL_PAGE_RESIDENT == 0
        {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
fn cpu_emit_missing_requests(
    pool: &GpuVirtualGeometryPool,
    config: GpuVirtualTraversalConfig,
    mesh_id: VirtualMeshId,
    instance_id: u32,
    first: u32,
    count: u32,
    result: &mut CpuTraversalResult,
) -> Result<(), VirtualGeometryTraversalError> {
    let archive = pool.asset(mesh_id)?.archive();
    let mut seen = Vec::new();
    for cluster in &archive.clusters[first as usize..(first + count) as usize] {
        if seen.contains(&cluster.page_index) {
            continue;
        }
        seen.push(cluster.page_index);
        let page = pool.page_entry(super::VirtualPageId {
            mesh: mesh_id,
            page_index: cluster.page_index,
        })?;
        if page.slot_plus_one != 0
            && page.mesh_id == mesh_id.raw()
            && page.flags & super::GPU_VIRTUAL_PAGE_RESIDENT != 0
        {
            continue;
        }
        let output_index = result.counters.page_request_count;
        result.counters.page_request_count += 1;
        if output_index < config.max_page_requests {
            result.requests.push(GpuVirtualPageRequest {
                mesh_id: mesh_id.raw(),
                page_index: cluster.page_index,
                instance_id,
                source_cluster: first,
            });
        } else {
            result.counters.request_overflow += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn cpu_select_group(
    pool: &GpuVirtualGeometryPool,
    config: GpuVirtualTraversalConfig,
    mesh_id: VirtualMeshId,
    instance_index: u32,
    instance: GpuVirtualInstance,
    first: u32,
    count: u32,
    scale: f32,
    view: VirtualGeometryView,
    result: &mut CpuTraversalResult,
) -> Result<(), VirtualGeometryTraversalError> {
    let archive = pool.asset(mesh_id)?.archive();
    for (offset, cluster) in archive.clusters[first as usize..(first + count) as usize]
        .iter()
        .enumerate()
    {
        let sphere = cpu_world_sphere(cluster, instance, scale);
        if cpu_sphere_outside_frustum(sphere, view.frustum_planes) {
            continue;
        }
        if cpu_cone_culled(cluster, instance, sphere, view.camera_position) {
            result.counters.cone_culled_clusters += 1;
            continue;
        }
        let page = pool.page_entry(super::VirtualPageId {
            mesh: mesh_id,
            page_index: cluster.page_index,
        })?;
        if page.slot_plus_one == 0
            || page.mesh_id != mesh_id.raw()
            || page.flags & super::GPU_VIRTUAL_PAGE_RESIDENT == 0
        {
            result.counters.missing_current_pages += 1;
            cpu_emit_missing_requests(
                pool,
                config,
                mesh_id,
                instance.instance_id(),
                first + offset as u32,
                1,
                result,
            )?;
            continue;
        }
        let output_index = result.counters.selected_count;
        result.counters.selected_count += 1;
        if output_index < config.max_selected_clusters {
            let material_id = pool.bound_material_id(mesh_id, cluster.material_index)?;
            let mesh = pool.mesh_entry(mesh_id)?;
            result.selected.push(GpuSelectedVirtualCluster {
                mesh_id: mesh_id.raw(),
                instance_index,
                cluster_table_index: mesh.cluster_table_base + first + offset as u32,
                physical_page_base: (page.slot_plus_one - 1) * mesh.page_stride_bytes,
                lod_level: cluster.lod_level,
                triangle_count: cluster.triangle_count,
                material_id,
                flags: cluster.flags | mesh.vertex_encoding << SELECTED_VERTEX_ENCODING_SHIFT,
            });
        } else {
            result.counters.selected_overflow += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
#[derive(Copy, Clone)]
struct CpuWorldSphere {
    center: [f32; 3],
    radius: f32,
}

#[cfg(test)]
fn cpu_world_sphere(
    cluster: &bloom_geometry_format::ClusterRecord,
    instance: GpuVirtualInstance,
    scale: f32,
) -> CpuWorldSphere {
    let p = cluster.sphere_center;
    let model = instance.model;
    CpuWorldSphere {
        center: [
            model[0][0] * p[0] + model[1][0] * p[1] + model[2][0] * p[2] + model[3][0],
            model[0][1] * p[0] + model[1][1] * p[1] + model[2][1] * p[2] + model[3][1],
            model[0][2] * p[0] + model[1][2] * p[1] + model[2][2] * p[2] + model[3][2],
        ],
        radius: cluster.sphere_radius * scale,
    }
}

#[cfg(test)]
fn cpu_scale_bound(model: [[f32; 4]; 4]) -> f32 {
    let columns = [
        [model[0][0], model[0][1], model[0][2]],
        [model[1][0], model[1][1], model[1][2]],
        [model[2][0], model[2][1], model[2][2]],
    ];
    let gram = [
        [
            dot3(columns[0], columns[0]),
            dot3(columns[0], columns[1]),
            dot3(columns[0], columns[2]),
        ],
        [
            dot3(columns[1], columns[0]),
            dot3(columns[1], columns[1]),
            dot3(columns[1], columns[2]),
        ],
        [
            dot3(columns[2], columns[0]),
            dot3(columns[2], columns[1]),
            dot3(columns[2], columns[2]),
        ],
    ];
    gram.map(|row| row.into_iter().map(f32::abs).sum::<f32>())
        .into_iter()
        .fold(0.0, f32::max)
        .max(0.0)
        .sqrt()
}

#[cfg(test)]
fn cpu_sphere_outside_frustum(sphere: CpuWorldSphere, planes: [[f32; 4]; 6]) -> bool {
    planes.into_iter().any(|plane| {
        let normal = [plane[0], plane[1], plane[2]];
        dot3(normal, sphere.center) + plane[3] < -sphere.radius * dot3(normal, normal).sqrt()
    })
}

#[cfg(test)]
fn cpu_projected_error(
    cluster: &bloom_geometry_format::ClusterRecord,
    sphere: CpuWorldSphere,
    scale: f32,
    view: VirtualGeometryView,
) -> f32 {
    let world_error = cluster.geometric_error * scale;
    if world_error <= 0.0 {
        return 0.0;
    }
    let p = sphere.center;
    let m = view.view_projection;
    let clip_w = m[0][3] * p[0] + m[1][3] * p[1] + m[2][3] * p[2] + m[3][3];
    let clip_w_gradient = [m[0][3], m[1][3], m[2][3]];
    let nearest_w = clip_w - sphere.radius * dot3(clip_w_gradient, clip_w_gradient).sqrt();
    if nearest_w <= 1.0e-5 {
        1.0e30
    } else {
        world_error * view.projection_scale / nearest_w
    }
}

#[cfg(test)]
fn cpu_cone_culled(
    cluster: &bloom_geometry_format::ClusterRecord,
    instance: GpuVirtualInstance,
    sphere: CpuWorldSphere,
    camera: [f32; 3],
) -> bool {
    let cutoff = cluster.normal_cone_cutoff;
    if cutoff <= 0.0 || !instance.cone_cull_safe() {
        return false;
    }
    let axis = [
        dot3(
            [
                instance.normal_rows[0][0],
                instance.normal_rows[0][1],
                instance.normal_rows[0][2],
            ],
            cluster.normal_cone_axis,
        ),
        dot3(
            [
                instance.normal_rows[1][0],
                instance.normal_rows[1][1],
                instance.normal_rows[1][2],
            ],
            cluster.normal_cone_axis,
        ),
        dot3(
            [
                instance.normal_rows[2][0],
                instance.normal_rows[2][1],
                instance.normal_rows[2][2],
            ],
            cluster.normal_cone_axis,
        ),
    ];
    let axis_length = dot3(axis, axis).sqrt();
    let to_camera = [
        camera[0] - sphere.center[0],
        camera[1] - sphere.center[1],
        camera[2] - sphere.center[2],
    ];
    let distance = dot3(to_camera, to_camera).sqrt();
    if axis_length <= 1.0e-8 || distance <= sphere.radius || distance <= 1.0e-8 {
        return false;
    }
    let axis = axis.map(|component| component / axis_length);
    let view_direction = to_camera.map(|component| component / distance);
    let sin_theta = (1.0 - cutoff * cutoff).max(0.0).sqrt();
    let sin_phi = (sphere.radius / distance).clamp(0.0, 1.0);
    let cos_phi = (1.0 - sin_phi * sin_phi).max(0.0).sqrt();
    let threshold = -(sin_theta * cos_phi + cutoff * sin_phi);
    dot3(axis, view_direction) <= threshold
}

const TRAVERSAL_SHADER: &str = r#"
const NO_RELATION: u32 = 0xffffffffu;
const ALL_SOURCE_MESHES: u32 = 0xffffffffu;
const INSTANCE_CONE_CULL_SAFE: u32 = 1u;
const INSTANCE_PREVIOUS_HIZ_ELIGIBLE: u32 = 4u;

struct GpuVirtualMeshEntry {
    mesh_id: u32,
    page_table_base: u32,
    page_count: u32,
    cluster_table_base: u32,
    cluster_count: u32,
    root_cluster_count: u32,
    page_stride_bytes: u32,
    vertex_encoding: u32,
    format_version: u32,
    flags: u32,
    reserved: vec2<u32>,
};
struct GpuVirtualPageEntry {
    slot_plus_one: u32,
    payload_bytes: u32,
    mesh_id: u32,
    flags: u32,
};
struct GpuVirtualClusterEntry {
    aabb_min_error: vec4<f32>,
    aabb_max_radius: vec4<f32>,
    sphere: vec4<f32>,
    normal_cone: vec4<f32>,
    identity: vec4<u32>,
    page_lod_counts: vec4<u32>,
    payload: vec4<u32>,
    relations: vec4<u32>,
};
struct GpuVirtualInstance {
    model: mat4x4<f32>,
    normal_rows: array<vec4<f32>, 3>,
    instance_info: vec4<u32>,
    previous_model: mat4x4<f32>,
    model_tint: vec4<f32>,
};
struct GpuSelectedVirtualCluster {
    mesh_id: u32,
    instance_index: u32,
    cluster_table_index: u32,
    physical_page_base: u32,
    lod_level: u32,
    triangle_count: u32,
    material_id: u32,
    flags: u32,
};
struct GpuVirtualPageRequest {
    mesh_id: u32,
    page_index: u32,
    instance_id: u32,
    source_cluster: u32,
};
struct MeshTable { records: array<GpuVirtualMeshEntry>, };
struct PageTable { records: array<GpuVirtualPageEntry>, };
struct ClusterTable { records: array<GpuVirtualClusterEntry>, };
struct InstanceTable { records: array<GpuVirtualInstance>, };
struct SelectedTable { records: array<GpuSelectedVirtualCluster>, };
struct RequestTable { records: array<GpuVirtualPageRequest>, };
struct TraversalCounters {
    selected_count: atomic<u32>,
    page_request_count: atomic<u32>,
    visible_groups: atomic<u32>,
    frustum_culled_groups: atomic<u32>,
    cone_culled_clusters: atomic<u32>,
    refined_groups: atomic<u32>,
    fallback_groups: atomic<u32>,
    missing_current_pages: atomic<u32>,
    selected_overflow: atomic<u32>,
    request_overflow: atomic<u32>,
    invalid_records: atomic<u32>,
    depth_limit_fallbacks: atomic<u32>,
    occlusion_culled_groups: atomic<u32>,
    occlusion_uncertain_groups: atomic<u32>,
};
struct TraversalParams {
    planes: array<vec4<f32>, 6>,
    view_projection: mat4x4<f32>,
    camera_projection: vec4<f32>,
    thresholds: vec4<f32>,
    dispatch: vec4<u32>,
    limits: vec4<u32>,
};
struct WorldSphere {
    center: vec3<f32>,
    radius: f32,
};
struct HiZParams {
    previous_view_projection: mat4x4<f32>,
    previous_view: mat4x4<f32>,
    current_view_projection: mat4x4<f32>,
    current_view: mat4x4<f32>,
    extent: vec4<u32>,
    thresholds: vec4<f32>,
};
struct ProjectedSphere {
    uv_min: vec2<f32>,
    uv_max: vec2<f32>,
    nearest_depth: f32,
    valid: u32,
};

@group(0) @binding(0) var<storage, read> meshes: MeshTable;
@group(0) @binding(1) var<storage, read> pages: PageTable;
@group(0) @binding(2) var<storage, read> clusters: ClusterTable;
@group(0) @binding(3) var<storage, read> instances: InstanceTable;
@group(0) @binding(4) var<storage, read_write> selected: SelectedTable;
@group(0) @binding(5) var<storage, read_write> requests: RequestTable;
@group(0) @binding(6) var<storage, read_write> counters: TraversalCounters;
@group(0) @binding(7) var<uniform> params: TraversalParams;
@group(1) @binding(0) var<uniform> hiz_params: HiZParams;
@group(1) @binding(1) var hiz_0: texture_2d<f32>;
@group(1) @binding(2) var hiz_1: texture_2d<f32>;
@group(1) @binding(3) var hiz_2: texture_2d<f32>;
@group(1) @binding(4) var hiz_3: texture_2d<f32>;
@group(1) @binding(5) var hiz_4: texture_2d<f32>;
@group(1) @binding(6) var hiz_5: texture_2d<f32>;
@group(1) @binding(7) var hiz_6: texture_2d<f32>;
@group(1) @binding(8) var hiz_7: texture_2d<f32>;
@group(1) @binding(9) var hiz_8: texture_2d<f32>;

fn valid_cluster(mesh: GpuVirtualMeshEntry, local_index: u32) -> bool {
    return local_index < mesh.cluster_count
        && mesh.cluster_table_base + local_index < arrayLength(&clusters.records);
}

fn valid_page(mesh: GpuVirtualMeshEntry, local_index: u32) -> bool {
    return local_index < mesh.page_count
        && mesh.page_table_base + local_index < arrayLength(&pages.records);
}

fn scale_bound(model: mat4x4<f32>) -> f32 {
    let c0 = model[0].xyz;
    let c1 = model[1].xyz;
    let c2 = model[2].xyz;
    let g0 = vec3<f32>(dot(c0, c0), dot(c0, c1), dot(c0, c2));
    let g1 = vec3<f32>(g0.y, dot(c1, c1), dot(c1, c2));
    let g2 = vec3<f32>(g0.z, g1.z, dot(c2, c2));
    let eigen_upper = max(
        dot(abs(g0), vec3<f32>(1.0)),
        max(dot(abs(g1), vec3<f32>(1.0)), dot(abs(g2), vec3<f32>(1.0)))
    );
    return sqrt(max(eigen_upper, 0.0));
}

fn world_sphere(
    cluster: GpuVirtualClusterEntry,
    instance: GpuVirtualInstance,
    scale: f32,
) -> WorldSphere {
    return WorldSphere(
        (instance.model * vec4<f32>(cluster.sphere.xyz, 1.0)).xyz,
        cluster.aabb_max_radius.w * scale
    );
}

fn transformed_sphere(
    cluster: GpuVirtualClusterEntry,
    model: mat4x4<f32>,
    scale: f32,
) -> WorldSphere {
    return WorldSphere(
        (model * vec4<f32>(cluster.sphere.xyz, 1.0)).xyz,
        cluster.aabb_max_radius.w * scale
    );
}

fn project_hiz_sphere(
    sphere: WorldSphere,
    view_projection: mat4x4<f32>,
    view: mat4x4<f32>,
) -> ProjectedSphere {
    var uv_min = vec2<f32>(1.0e30);
    var uv_max = vec2<f32>(-1.0e30);
    for (var corner = 0u; corner < 8u; corner++) {
        let signs = vec3<f32>(
            select(-1.0, 1.0, (corner & 1u) != 0u),
            select(-1.0, 1.0, (corner & 2u) != 0u),
            select(-1.0, 1.0, (corner & 4u) != 0u)
        );
        let clip = view_projection
            * vec4<f32>(sphere.center + signs * sphere.radius, 1.0);
        if (clip.w <= 0.05 || clip.w != clip.w || any(abs(clip.xyz) > vec3<f32>(1.0e30))) {
            return ProjectedSphere(uv_min, uv_max, 0.0, 0u);
        }
        let ndc = clip.xy / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        uv_min = min(uv_min, uv);
        uv_max = max(uv_max, uv);
    }
    let view_center = view * vec4<f32>(sphere.center, 1.0);
    let nearest_depth = -view_center.z - sphere.radius;
    if (nearest_depth != nearest_depth || nearest_depth <= 0.0) {
        return ProjectedSphere(uv_min, uv_max, nearest_depth, 0u);
    }
    return ProjectedSphere(uv_min, uv_max, nearest_depth, 1u);
}

fn hiz_depth(mip: u32, coordinate: vec2<i32>) -> f32 {
    switch mip {
        case 0u: { return textureLoad(hiz_0, coordinate, 0).r; }
        case 1u: { return textureLoad(hiz_1, coordinate, 0).r; }
        case 2u: { return textureLoad(hiz_2, coordinate, 0).r; }
        case 3u: { return textureLoad(hiz_3, coordinate, 0).r; }
        case 4u: { return textureLoad(hiz_4, coordinate, 0).r; }
        case 5u: { return textureLoad(hiz_5, coordinate, 0).r; }
        case 6u: { return textureLoad(hiz_6, coordinate, 0).r; }
        case 7u: { return textureLoad(hiz_7, coordinate, 0).r; }
        default: { return textureLoad(hiz_8, coordinate, 0).r; }
    }
}

// 0 = proven occluded, 1 = sampled and visible, 2 = uncertain/visible.
fn previous_hiz_result(
    cluster: GpuVirtualClusterEntry,
    instance: GpuVirtualInstance,
    current_sphere: WorldSphere,
) -> u32 {
    if (hiz_params.extent.w == 0u
        || (instance.instance_info.z & INSTANCE_PREVIOUS_HIZ_ELIGIBLE) == 0u) {
        return 2u;
    }
    let previous_scale = scale_bound(instance.previous_model);
    let previous_sphere = transformed_sphere(cluster, instance.previous_model, previous_scale);
    let previous = project_hiz_sphere(
        previous_sphere,
        hiz_params.previous_view_projection,
        hiz_params.previous_view
    );
    let current = project_hiz_sphere(
        current_sphere,
        hiz_params.current_view_projection,
        hiz_params.current_view
    );
    if (previous.valid == 0u || current.valid == 0u) { return 2u; }
    if (previous.uv_max.x <= 0.0 || previous.uv_min.x >= 1.0
        || previous.uv_max.y <= 0.0 || previous.uv_min.y >= 1.0
        || current.uv_max.x <= 0.0 || current.uv_min.x >= 1.0
        || current.uv_max.y <= 0.0 || current.uv_min.y >= 1.0) {
        return 2u;
    }
    let minimum_delta = abs(previous.uv_min - current.uv_min);
    let maximum_delta = abs(previous.uv_max - current.uv_max);
    let screen_delta = max(
        max(minimum_delta.x, minimum_delta.y),
        max(maximum_delta.x, maximum_delta.y)
    );
    if (screen_delta > max(hiz_params.thresholds.x, hiz_params.thresholds.y)) {
        return 2u;
    }

    let expansion = hiz_params.thresholds.xy * 2.0;
    let uv_min = clamp(min(previous.uv_min, current.uv_min) - expansion, vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_max = clamp(max(previous.uv_max, current.uv_max) + expansion, vec2<f32>(0.0), vec2<f32>(1.0));
    let base_span = max(
        (uv_max.x - uv_min.x) * f32(hiz_params.extent.x),
        (uv_max.y - uv_min.y) * f32(hiz_params.extent.y)
    );
    var mip = 0u;
    var span = base_span;
    while (span > 2.0 && mip + 1u < hiz_params.extent.z) {
        span *= 0.5;
        mip++;
    }
    let divisor = 1u << mip;
    let dimensions = max(
        vec2<u32>(1u),
        (hiz_params.extent.xy + vec2<u32>(divisor - 1u)) / divisor
    );
    let maximum_coordinate = vec2<i32>(dimensions - vec2<u32>(1u));
    let first = clamp(vec2<i32>(floor(uv_min * vec2<f32>(dimensions))), vec2<i32>(0), maximum_coordinate);
    let last = clamp(vec2<i32>(floor(uv_max * vec2<f32>(dimensions))), vec2<i32>(0), maximum_coordinate);
    var maximum_depth = 0.0;
    for (var y = first.y; y <= last.y; y++) {
        for (var x = first.x; x <= last.x; x++) {
            maximum_depth = max(maximum_depth, hiz_depth(mip, vec2<i32>(x, y)));
        }
    }
    let nearest_depth = min(previous.nearest_depth, current.nearest_depth);
    let occluded = nearest_depth
        > maximum_depth * (1.0 + hiz_params.thresholds.z) + hiz_params.thresholds.w;
    return select(1u, 0u, occluded);
}

fn sphere_outside_frustum(sphere: WorldSphere) -> bool {
    for (var plane_index = 0u; plane_index < 6u; plane_index++) {
        let plane = params.planes[plane_index];
        let normal_length = length(plane.xyz);
        if (dot(plane.xyz, sphere.center) + plane.w < -sphere.radius * normal_length) {
            return true;
        }
    }
    return false;
}

fn projected_error(
    cluster: GpuVirtualClusterEntry,
    sphere: WorldSphere,
    scale: f32,
) -> f32 {
    let world_error = cluster.aabb_min_error.w * scale;
    if (world_error <= 0.0) {
        return 0.0;
    }
    let clip = params.view_projection * vec4<f32>(sphere.center, 1.0);
    let clip_w_gradient = vec3<f32>(
        params.view_projection[0].w,
        params.view_projection[1].w,
        params.view_projection[2].w
    );
    let nearest_w = clip.w - sphere.radius * length(clip_w_gradient);
    if (nearest_w <= params.thresholds.y) {
        return 1.0e30;
    }
    return world_error * params.camera_projection.w / nearest_w;
}

fn cone_culled(
    cluster: GpuVirtualClusterEntry,
    instance: GpuVirtualInstance,
    sphere: WorldSphere,
) -> bool {
    let cutoff = cluster.normal_cone.w;
    if (cutoff <= 0.0 || (instance.instance_info.z & INSTANCE_CONE_CULL_SAFE) == 0u) {
        return false;
    }
    var axis = vec3<f32>(
        dot(instance.normal_rows[0].xyz, cluster.normal_cone.xyz),
        dot(instance.normal_rows[1].xyz, cluster.normal_cone.xyz),
        dot(instance.normal_rows[2].xyz, cluster.normal_cone.xyz)
    );
    let axis_length = length(axis);
    let to_camera = params.camera_projection.xyz - sphere.center;
    let distance = length(to_camera);
    if (axis_length <= 1.0e-8 || distance <= sphere.radius || distance <= 1.0e-8) {
        return false;
    }
    axis /= axis_length;
    let view_direction = to_camera / distance;
    let sin_theta = sqrt(max(1.0 - cutoff * cutoff, 0.0));
    let sin_phi = clamp(sphere.radius / distance, 0.0, 1.0);
    let cos_phi = sqrt(max(1.0 - sin_phi * sin_phi, 0.0));
    let conservative_threshold = -(sin_theta * cos_phi + cutoff * sin_phi);
    return dot(axis, view_direction) <= conservative_threshold;
}

fn group_is_resident(mesh: GpuVirtualMeshEntry, first: u32, count: u32) -> bool {
    for (var offset = 0u; offset < count; offset++) {
        let local_cluster = first + offset;
        if (!valid_cluster(mesh, local_cluster)) {
            return false;
        }
        let cluster = clusters.records[mesh.cluster_table_base + local_cluster];
        let page_index = cluster.page_lod_counts.x;
        if (!valid_page(mesh, page_index)) {
            return false;
        }
        let page = pages.records[mesh.page_table_base + page_index];
        if (page.slot_plus_one == 0u || page.mesh_id != mesh.mesh_id || (page.flags & 1u) == 0u) {
            return false;
        }
    }
    return true;
}

fn emit_missing_requests(
    mesh: GpuVirtualMeshEntry,
    instance: GpuVirtualInstance,
    first: u32,
    count: u32,
) {
    for (var offset = 0u; offset < count; offset++) {
        let local_cluster = first + offset;
        if (!valid_cluster(mesh, local_cluster)) {
            atomicAdd(&counters.invalid_records, 1u);
            continue;
        }
        let cluster = clusters.records[mesh.cluster_table_base + local_cluster];
        let page_index = cluster.page_lod_counts.x;
        if (!valid_page(mesh, page_index)) {
            atomicAdd(&counters.invalid_records, 1u);
            continue;
        }
        let page = pages.records[mesh.page_table_base + page_index];
        if (page.slot_plus_one != 0u && page.mesh_id == mesh.mesh_id && (page.flags & 1u) != 0u) {
            continue;
        }
        var duplicate = false;
        for (var previous = 0u; previous < offset; previous++) {
            let previous_cluster = clusters.records[mesh.cluster_table_base + first + previous];
            if (previous_cluster.page_lod_counts.x == page_index) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            let output_index = atomicAdd(&counters.page_request_count, 1u);
            if (output_index < params.dispatch.w) {
                requests.records[output_index] = GpuVirtualPageRequest(
                    mesh.mesh_id,
                    page_index,
                    instance.instance_info.y,
                    first
                );
            } else {
                atomicAdd(&counters.request_overflow, 1u);
            }
        }
    }
}

fn select_group(
    mesh: GpuVirtualMeshEntry,
    instance_index: u32,
    instance: GpuVirtualInstance,
    first: u32,
    count: u32,
    scale: f32,
) {
    for (var offset = 0u; offset < count; offset++) {
        let local_cluster = first + offset;
        if (!valid_cluster(mesh, local_cluster)) {
            atomicAdd(&counters.invalid_records, 1u);
            continue;
        }
        let cluster = clusters.records[mesh.cluster_table_base + local_cluster];
        let sphere = world_sphere(cluster, instance, scale);
        if (sphere_outside_frustum(sphere)) {
            continue;
        }
        if (cone_culled(cluster, instance, sphere)) {
            atomicAdd(&counters.cone_culled_clusters, 1u);
            continue;
        }
        let page_index = cluster.page_lod_counts.x;
        if (!valid_page(mesh, page_index)) {
            atomicAdd(&counters.invalid_records, 1u);
            continue;
        }
        let page = pages.records[mesh.page_table_base + page_index];
        if (page.slot_plus_one == 0u || page.mesh_id != mesh.mesh_id || (page.flags & 1u) == 0u) {
            atomicAdd(&counters.missing_current_pages, 1u);
            emit_missing_requests(mesh, instance, local_cluster, 1u);
            continue;
        }
        let output_index = atomicAdd(&counters.selected_count, 1u);
        if (output_index < params.dispatch.z) {
            selected.records[output_index] = GpuSelectedVirtualCluster(
                mesh.mesh_id,
                instance_index,
                mesh.cluster_table_base + local_cluster,
                (page.slot_plus_one - 1u) * mesh.page_stride_bytes,
                cluster.page_lod_counts.y,
                cluster.page_lod_counts.w,
                cluster.identity.z,
                cluster.identity.w | (mesh.vertex_encoding << 28u)
            );
        } else {
            atomicAdd(&counters.selected_overflow, 1u);
        }
    }
}

@compute @workgroup_size(64)
fn select_virtual_clusters(@builtin(global_invocation_id) gid: vec3<u32>) {
    let instance_index = gid.y;
    let root_index = gid.x;
    if (instance_index >= params.dispatch.x || root_index >= params.dispatch.y) {
        return;
    }
    if (instance_index >= arrayLength(&instances.records)) {
        atomicAdd(&counters.invalid_records, 1u);
        return;
    }
    let instance = instances.records[instance_index];
    let descriptor_index = instance.instance_info.x & 0xfffffu;
    if (descriptor_index == 0u || descriptor_index - 1u >= arrayLength(&meshes.records)) {
        atomicAdd(&counters.invalid_records, 1u);
        return;
    }
    let mesh = meshes.records[descriptor_index - 1u];
    if (mesh.mesh_id != instance.instance_info.x || root_index >= mesh.root_cluster_count) {
        return;
    }
    if (!valid_cluster(mesh, root_index)) {
        atomicAdd(&counters.invalid_records, 1u);
        return;
    }

    let root = clusters.records[mesh.cluster_table_base + root_index];
    let source_mesh_filter = instance.instance_info.w;
    if (source_mesh_filter != ALL_SOURCE_MESHES && root.identity.x != source_mesh_filter) {
        return;
    }
    var group_first = root_index;
    var group_count = 1u;
    if (root.relations.w != 0u && valid_cluster(mesh, root.relations.z)) {
        let first_child = clusters.records[mesh.cluster_table_base + root.relations.z];
        if (first_child.relations.x != NO_RELATION && first_child.relations.y != 0u) {
            group_first = first_child.relations.x;
            group_count = first_child.relations.y;
        }
    }
    if (root_index != group_first) {
        return;
    }

    let scale = scale_bound(instance.model);
    for (var depth = 0u; depth < 32u; depth++) {
        if (depth >= params.limits.x) {
            break;
        }
        if (group_count == 0u || group_count > params.limits.y
            || group_first + group_count > mesh.cluster_count) {
            atomicAdd(&counters.invalid_records, 1u);
            return;
        }

        var frustum_visible = false;
        var occlusion_visible = hiz_params.extent.w == 0u;
        var occlusion_uncertain = false;
        var maximum_error = 0.0;
        for (var offset = 0u; offset < group_count; offset++) {
            let local_cluster = group_first + offset;
            if (!valid_cluster(mesh, local_cluster)) {
                atomicAdd(&counters.invalid_records, 1u);
                return;
            }
            let cluster = clusters.records[mesh.cluster_table_base + local_cluster];
            let sphere = world_sphere(cluster, instance, scale);
            if (!sphere_outside_frustum(sphere)) {
                frustum_visible = true;
                maximum_error = max(maximum_error, projected_error(cluster, sphere, scale));
                if (hiz_params.extent.w != 0u) {
                    let hiz_result = previous_hiz_result(cluster, instance, sphere);
                    occlusion_visible = occlusion_visible || hiz_result != 0u;
                    occlusion_uncertain = occlusion_uncertain || hiz_result == 2u;
                }
            }
        }
        if (!frustum_visible) {
            atomicAdd(&counters.frustum_culled_groups, 1u);
            return;
        }
        if (!occlusion_visible) {
            atomicAdd(&counters.occlusion_culled_groups, 1u);
            return;
        }
        if (occlusion_uncertain) {
            atomicAdd(&counters.occlusion_uncertain_groups, 1u);
        }
        atomicAdd(&counters.visible_groups, 1u);

        let first_cluster = clusters.records[mesh.cluster_table_base + group_first];
        let child_first = first_cluster.relations.z;
        let child_count = first_cluster.relations.w;
        let wants_refinement = maximum_error > params.thresholds.x && child_count != 0u;
        if (!wants_refinement) {
            select_group(mesh, instance_index, instance, group_first, group_count, scale);
            return;
        }
        if (child_count > params.limits.y || child_first + child_count > mesh.cluster_count) {
            atomicAdd(&counters.invalid_records, 1u);
            select_group(mesh, instance_index, instance, group_first, group_count, scale);
            return;
        }
        if (!group_is_resident(mesh, child_first, child_count)) {
            atomicAdd(&counters.fallback_groups, 1u);
            emit_missing_requests(mesh, instance, child_first, child_count);
            select_group(mesh, instance_index, instance, group_first, group_count, scale);
            return;
        }
        atomicAdd(&counters.refined_groups, 1u);
        group_first = child_first;
        group_count = child_count;
    }

    atomicAdd(&counters.depth_limit_fallbacks, 1u);
    select_group(mesh, instance_index, instance, group_first, group_count, scale);
}
"#;

#[cfg(test)]
#[path = "traversal_shader_tests.rs"]
mod shader_tests;
