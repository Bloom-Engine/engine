use super::hiz::{
    GpuVirtualHiZ, VirtualGeometryHiZFrame, VIRTUAL_HIZ_MIP_COUNT,
    VIRTUAL_HIZ_SELECTION_PARAMS_BYTES,
};
use super::{
    GpuVirtualGeometryPool, VirtualGeometryGpuError, VirtualMeshId,
    GPU_VIRTUAL_MESH_MATERIALS_BOUND,
};
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "traversal_error.rs"]
mod error_impl;
pub use error_impl::VirtualGeometryTraversalError;
#[path = "traversal_math.rs"]
mod math_impl;
use math_impl::{dot3, finite_affine, normal_rows_and_cone_safety};

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
#[cfg(test)]
const TRAVERSAL_GROUP_STACK_CAPACITY: usize = 32;
static NEXT_SELECTOR_ID: AtomicU64 = AtomicU64::new(1);

/// Fixed GPU input record for one virtual-geometry instance (224 bytes).
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
    /// Runtime-populated root-table start/count, followed by reserved words.
    root_span: [u32; 4],
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
            root_span: [0; 4],
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

    fn set_root_span(&mut self, start: u32, count: u32) {
        self.root_span = [start, count, 0, 0];
    }
}

const _: () = assert!(std::mem::size_of::<GpuVirtualInstance>() == 224);

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
    /// Non-negative projected-error bits. IEEE-754 bit order matches numeric order.
    pub priority_bits: u32,
    pub source_cluster: u32,
}

/// One selected atomic group whose resident pages were consumed by the current frame.
/// Streaming uses this bounded feedback to protect visible pages from eviction churn.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd, bytemuck::Pod, bytemuck::Zeroable,
)]
pub struct GpuVirtualPageUse {
    pub mesh_id: u32,
    pub source_cluster: u32,
    /// Non-negative projected-error bits. IEEE-754 bit order matches numeric order.
    pub priority_bits: u32,
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
    pub page_use_count: u32,
    pub page_use_overflow: u32,
}

const _: () = assert!(std::mem::size_of::<GpuSelectedVirtualCluster>() == 32);
const _: () = assert!(std::mem::size_of::<GpuVirtualPageRequest>() == 16);
const _: () = assert!(std::mem::size_of::<GpuVirtualPageUse>() == 12);
const _: () = assert!(std::mem::size_of::<GpuVirtualTraversalCounters>() == 64);

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
    page_use_buffer: wgpu::Buffer,
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
        let page_use_buffer = create_buffer(
            device,
            "virtual_geometry_page_uses",
            buffer_bytes::<GpuVirtualPageUse>(config.max_page_requests),
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
                binding(7, &page_use_buffer),
                binding(8, &params_buffer),
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
            page_use_buffer,
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
        self.record_internal(queue, encoder, pool, instances, view, None, None)
    }

    #[cfg(test)]
    pub(crate) fn record_with_previous_hiz(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pool: &GpuVirtualGeometryPool,
        instances: &[GpuVirtualInstance],
        view: VirtualGeometryView,
        hiz_frame: VirtualGeometryHiZFrame,
    ) -> Result<VirtualGeometryTraversalDispatch, VirtualGeometryTraversalError> {
        self.record_internal(queue, encoder, pool, instances, view, Some(hiz_frame), None)
    }

    pub(crate) fn record_with_previous_hiz_profiled(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pool: &GpuVirtualGeometryPool,
        instances: &[GpuVirtualInstance],
        view: VirtualGeometryView,
        hiz_frame: VirtualGeometryHiZFrame,
        profiler: &mut crate::profiler::Profiler,
    ) -> Result<VirtualGeometryTraversalDispatch, VirtualGeometryTraversalError> {
        const LABEL: &str = "virtual_geometry_hierarchy_selection";
        profiler.begin(LABEL);
        let timestamp_writes = profiler.compute_pass_timestamp_writes(LABEL);
        let result = self.record_internal(
            queue,
            encoder,
            pool,
            instances,
            view,
            Some(hiz_frame),
            timestamp_writes,
        );
        profiler.end(LABEL);
        result
    }

    fn record_internal(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pool: &GpuVirtualGeometryPool,
        instances: &[GpuVirtualInstance],
        view: VirtualGeometryView,
        hiz_frame: Option<VirtualGeometryHiZFrame>,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
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
        let mut prepared_instances = Vec::with_capacity(instances.len());
        for instance in instances {
            validate_instance(*instance)?;
            let (root_start, root_count) = instance_root_span(pool, *instance)?;
            let mesh = pool.mesh_entry(instance.mesh_id())?;
            if mesh.flags & GPU_VIRTUAL_MESH_MATERIALS_BOUND == 0 {
                return Err(VirtualGeometryTraversalError::UnboundMaterials {
                    mesh: instance.mesh_id(),
                });
            }
            maximum_root_clusters = maximum_root_clusters.max(root_count);
            let mut prepared = *instance;
            prepared.set_root_span(root_start, root_count);
            prepared_instances.push(prepared);
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
            thresholds: [
                view.target_error_pixels,
                1.0e-5,
                lateral_frustum_guard_w(view),
                0.0,
            ],
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

        if !prepared_instances.is_empty() {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&prepared_instances),
            );
        }
        if workgroups_x != 0 && !instances.is_empty() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("virtual_geometry_hierarchy_selection"),
                timestamp_writes,
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

    pub fn page_use_buffer(&self) -> &wgpu::Buffer {
        &self.page_use_buffer
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
    let mut entries = (0..8)
        .map(|binding| storage(binding, binding < 4))
        .collect::<Vec<_>>();
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 8,
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
    if limits.max_storage_buffers_per_shader_stage < 8
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
        (
            "page use table",
            buffer_bytes::<GpuVirtualPageUse>(config.max_page_requests),
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

fn lateral_frustum_guard_w(view: VirtualGeometryView) -> f32 {
    const MINIMUM_GUARD_W: f32 = 1.0e-5;
    let near = view.frustum_planes[4];
    let near_normal = [near[0], near[1], near[2]];
    let near_normal_length = dot3(near_normal, near_normal).sqrt();
    let w_gradient = [
        view.view_projection[0][3],
        view.view_projection[1][3],
        view.view_projection[2][3],
    ];
    let w_gradient_length = dot3(w_gradient, w_gradient).sqrt();
    if near_normal_length <= MINIMUM_GUARD_W || w_gradient_length <= MINIMUM_GUARD_W {
        return MINIMUM_GUARD_W;
    }
    let camera_near_signed_distance =
        (dot3(near_normal, view.camera_position) + near[3]) / near_normal_length;
    (-camera_near_signed_distance * w_gradient_length).max(MINIMUM_GUARD_W)
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

fn instance_root_span(
    pool: &GpuVirtualGeometryPool,
    instance: GpuVirtualInstance,
) -> Result<(u32, u32), VirtualGeometryTraversalError> {
    validate_source_mesh_filter(pool, instance)?;
    let mesh = pool.mesh_entry(instance.mesh_id())?;
    match instance.source_mesh_index() {
        Some(source_mesh_index) => {
            let span = pool
                .asset(instance.mesh_id())?
                .source_root_span(source_mesh_index)
                .ok_or(VirtualGeometryTraversalError::SourceMeshNotVirtual {
                    mesh: instance.mesh_id(),
                    source_mesh_index,
                })?;
            Ok((span.start, span.end - span.start))
        }
        None => Ok((0, mesh.root_cluster_count)),
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
        let (root_start, root_count) = instance_root_span(pool, *instance)?;
        let mesh_id = instance.mesh_id();
        let mesh_entry = pool.mesh_entry(mesh_id)?;
        if mesh_entry.flags & GPU_VIRTUAL_MESH_MATERIALS_BOUND == 0 {
            return Err(VirtualGeometryTraversalError::UnboundMaterials { mesh: mesh_id });
        }
        let archive = pool.asset(mesh_id)?.archive();
        for root_index in root_start..root_start + root_count {
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
            // A cooked group can replace several independently refinable child
            // ranges. Keep those branches on a bounded local stack instead of
            // following the first sibling's child range and silently dropping
            // every other branch.
            let mut stack = vec![(group_first, group_count, 1.0e30f32.to_bits(), 0u32)];
            while let Some((group_first, group_count, group_priority_bits, depth)) = stack.pop() {
                if depth >= pool.config().max_hierarchy_levels {
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
                        group_priority_bits,
                        &mut result,
                    )?;
                    continue;
                }
                let range = group_first as usize..(group_first + group_count) as usize;
                let mut common_outside_mask = (1u32 << view.frustum_planes.len()) - 1;
                let mut intersecting_error = 0.0f32;
                let mut group_error = 0.0f32;
                let mut has_intersecting_cluster = false;
                for cluster in &archive.clusters[range.clone()] {
                    let sphere = cpu_world_sphere(cluster, *instance, scale);
                    let outside_mask =
                        cpu_cluster_frustum_outside_mask(cluster, *instance, sphere, view);
                    let error = cpu_projected_error(cluster, sphere, scale, view);
                    common_outside_mask &= outside_mask;
                    group_error = group_error.max(error);
                    if outside_mask == 0 {
                        has_intersecting_cluster = true;
                        intersecting_error = intersecting_error.max(error);
                    }
                }
                if common_outside_mask != 0 {
                    result.counters.frustum_culled_groups += 1;
                    continue;
                }
                let maximum_error = if has_intersecting_cluster {
                    intersecting_error
                } else {
                    group_error
                };
                result.counters.visible_groups += 1;
                let mut child_groups = Vec::<(u32, u32)>::new();
                let mut terminal_clusters = false;
                for cluster in &archive.clusters[range] {
                    if cluster.first_child == bloom_geometry_format::NO_RELATION
                        || cluster.child_count == 0
                    {
                        terminal_clusters = true;
                        continue;
                    }
                    let relation = (cluster.first_child, cluster.child_count);
                    if child_groups.last().copied() != Some(relation) {
                        child_groups.push(relation);
                    }
                }
                let mixed_replacement = terminal_clusters && !child_groups.is_empty();
                let wants_refinement = maximum_error > view.target_error_pixels
                    && !child_groups.is_empty()
                    && !mixed_replacement;
                if mixed_replacement {
                    result.counters.invalid_records += 1;
                }
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
                        group_priority_bits,
                        &mut result,
                    )?;
                    continue;
                }
                let children_resident = child_groups.iter().try_fold(
                    true,
                    |resident, &(child_first, child_count)| {
                        Ok::<_, VirtualGeometryTraversalError>(
                            resident
                                && cpu_group_is_resident(pool, mesh_id, child_first, child_count)?,
                        )
                    },
                )?;
                if !children_resident {
                    result.counters.fallback_groups += 1;
                    for &(child_first, child_count) in &child_groups {
                        cpu_emit_missing_requests(
                            pool,
                            config,
                            mesh_id,
                            child_first,
                            child_count,
                            maximum_error.max(0.0).to_bits(),
                            &mut result,
                        )?;
                    }
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
                        group_priority_bits,
                        &mut result,
                    )?;
                    continue;
                }
                if stack.len() + child_groups.len() > TRAVERSAL_GROUP_STACK_CAPACITY {
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
                        group_priority_bits,
                        &mut result,
                    )?;
                    continue;
                }
                result.counters.refined_groups += 1;
                let child_priority_bits = maximum_error.max(0.0).to_bits();
                for &(child_first, child_count) in child_groups.iter().rev() {
                    stack.push((child_first, child_count, child_priority_bits, depth + 1));
                }
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
    first: u32,
    count: u32,
    priority_bits: u32,
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
                priority_bits,
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
    priority_bits: u32,
    result: &mut CpuTraversalResult,
) -> Result<(), VirtualGeometryTraversalError> {
    let archive = pool.asset(mesh_id)?.archive();
    let page_use_index = result.counters.page_use_count;
    result.counters.page_use_count += 1;
    if page_use_index >= config.max_page_requests {
        result.counters.page_use_overflow += 1;
    }
    for (offset, cluster) in archive.clusters[first as usize..(first + count) as usize]
        .iter()
        .enumerate()
    {
        let sphere = cpu_world_sphere(cluster, instance, scale);
        if cpu_cluster_frustum_outside_mask(cluster, instance, sphere, view) != 0 {
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
                first + offset as u32,
                1,
                priority_bits,
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
fn cpu_cluster_frustum_outside_mask(
    cluster: &bloom_geometry_format::ClusterRecord,
    instance: GpuVirtualInstance,
    sphere: CpuWorldSphere,
    view: VirtualGeometryView,
) -> u32 {
    let local_center: [f32; 3] =
        std::array::from_fn(|axis| (cluster.aabb_min[axis] + cluster.aabb_max[axis]) * 0.5);
    let local_extent: [f32; 3] =
        std::array::from_fn(|axis| (cluster.aabb_max[axis] - cluster.aabb_min[axis]) * 0.5);
    let model = instance.model();
    let world_center = [
        model[0][0] * local_center[0]
            + model[1][0] * local_center[1]
            + model[2][0] * local_center[2]
            + model[3][0],
        model[0][1] * local_center[0]
            + model[1][1] * local_center[1]
            + model[2][1] * local_center[2]
            + model[3][1],
        model[0][2] * local_center[0]
            + model[1][2] * local_center[1]
            + model[2][2] * local_center[2]
            + model[3][2],
    ];
    let outside_mask =
        view.frustum_planes
            .into_iter()
            .enumerate()
            .fold(0u32, |mask, (plane_index, plane)| {
                let normal = [plane[0], plane[1], plane[2]];
                let projected_radius =
                    (dot3(normal, [model[0][0], model[0][1], model[0][2]]).abs() * local_extent[0])
                        + (dot3(normal, [model[1][0], model[1][1], model[1][2]]).abs()
                            * local_extent[1])
                        + (dot3(normal, [model[2][0], model[2][1], model[2][2]]).abs()
                            * local_extent[2]);
                mask | (u32::from(dot3(normal, world_center) + plane[3] < -projected_radius)
                    << plane_index)
            });

    // Hardware clips a primitive that crosses the near plane before rasterization. Its
    // un-clipped cluster bound can simultaneously classify outside a lateral homogeneous plane,
    // so rejecting it here can punch camera-adjacent holes. Fail open only for the four lateral
    // planes when the conservative sphere reaches the view's actual near-clip distance; near and
    // far rejection remain active.
    let clip_w = view.view_projection[0][3] * sphere.center[0]
        + view.view_projection[1][3] * sphere.center[1]
        + view.view_projection[2][3] * sphere.center[2]
        + view.view_projection[3][3];
    let w_gradient = [
        view.view_projection[0][3],
        view.view_projection[1][3],
        view.view_projection[2][3],
    ];
    let nearest_w = clip_w - sphere.radius * dot3(w_gradient, w_gradient).sqrt();
    if nearest_w <= lateral_frustum_guard_w(view) {
        outside_mask & !0x0f
    } else {
        outside_mask
    }
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
    // Near-intersecting groups all need refinement, but assigning every one
    // the same sentinel priority makes pool admission depend on source order.
    // Clamp at the actual near plane so projected error stays conservative,
    // finite, and proportional to the group's authored geometric error.
    world_error * view.projection_scale / nearest_w.max(lateral_frustum_guard_w(view))
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

const TRAVERSAL_SHADER: &str = include_str!("../../shaders/virtual_geometry/traversal.wgsl");

#[cfg(test)]
#[path = "traversal_shader_tests.rs"]
mod shader_tests;
