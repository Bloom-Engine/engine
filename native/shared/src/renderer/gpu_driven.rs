//! GPU-driven static opaque submission (#28).
//!
//! The compatibility renderer remains the oracle. This path is selected only
//! when the device exposes indirect first-instance and Tier-A global material
//! tables. Static cached meshes live in one geometry arena; a compute pass
//! writes one ordered indirect command per submitted draw and zeros
//! `instance_count` for culled draws. Keeping command slots ordered avoids the
//! alpha/cutout ordering changes caused by atomic append compaction.

use super::{
    material_indirection::GpuCompletionTracker, MeshDrawRef, Renderer, Uniforms3D, Vertex3D,
    DEPTH_FORMAT, HDR_FORMAT, MATERIAL_FORMAT, VELOCITY_FORMAT,
};
use std::sync::OnceLock;

pub const GPU_DRIVEN_FEATURES: wgpu::Features = wgpu::Features::INDIRECT_FIRST_INSTANCE;
pub(crate) const DRAW_FLAG_DOUBLE_SIDED: u32 = 1 << 0;
pub(crate) const DRAW_FLAG_VISIBILITY_ELIGIBLE: u32 = 1 << 1;

pub(crate) const fn draw_flags(double_sided: bool, visibility_eligible: bool) -> u32 {
    (if double_sided {
        DRAW_FLAG_DOUBLE_SIDED
    } else {
        0
    }) | if visibility_eligible {
        DRAW_FLAG_VISIBILITY_ELIGIBLE
    } else {
        0
    }
}
/// Below this count, compute/indirect setup costs more than the CPU loop on
/// current Metal hardware. Keep small scenes on the lower-overhead oracle.
pub const GPU_DRIVEN_MIN_DRAWS: usize = 32;

/// Request only portable GPU-driven features the adapter actually exposes.
///
/// `MULTI_DRAW_INDIRECT_COUNT` is optional and is currently absent on Metal;
/// fixed-count multi-draw remains a one-call GPU submission there.
pub fn request_features_if_supported(supported: wgpu::Features, required: &mut wgpu::Features) {
    if supported.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
        *required |= wgpu::Features::INDIRECT_FIRST_INSTANCE;
    }
    if supported.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
        *required |= wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
    }
    super::visibility_buffer::request_feature_if_supported(supported, required);
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct GeometrySlice {
    pub vertex_offset: u64,
    pub vertex_size: u64,
    pub index_offset: u64,
    pub index_size: u64,
    pub first_index: u32,
    pub base_vertex: i32,
}

pub enum MeshGeometry {
    Shared(GeometrySlice),
    Dedicated {
        vertex: wgpu::Buffer,
        index: wgpu::Buffer,
    },
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct FreeRange {
    offset: u64,
    size: u64,
}

struct RetiredGeometry {
    epoch: u64,
    slice: GeometrySlice,
}

struct GeometryArena {
    vertex: wgpu::Buffer,
    index: wgpu::Buffer,
    vertex_capacity: u64,
    index_capacity: u64,
    vertex_end: u64,
    index_end: u64,
    free_vertices: Vec<FreeRange>,
    free_indices: Vec<FreeRange>,
    retired: Vec<RetiredGeometry>,
    completion: GpuCompletionTracker,
    generation: u64,
}

impl GeometryArena {
    const INITIAL_CAPACITY: u64 = 64 * 1024;

    fn new(device: &wgpu::Device) -> Self {
        Self {
            vertex: create_vertex_arena(device, Self::INITIAL_CAPACITY),
            index: create_index_arena(device, Self::INITIAL_CAPACITY),
            vertex_capacity: Self::INITIAL_CAPACITY,
            index_capacity: Self::INITIAL_CAPACITY,
            vertex_end: 0,
            index_end: 0,
            free_vertices: Vec::new(),
            free_indices: Vec::new(),
            retired: Vec::new(),
            completion: GpuCompletionTracker::default(),
            generation: 0,
        }
    }

    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[Vertex3D],
        indices: &[u32],
    ) -> GeometrySlice {
        self.collect();
        let vertex_size = std::mem::size_of_val(vertices) as u64;
        let index_size = std::mem::size_of_val(indices) as u64;
        let previous_vertex_end = self.vertex_end;
        let previous_index_end = self.index_end;
        let vertex_offset = allocate_range(
            &mut self.free_vertices,
            &mut self.vertex_end,
            vertex_size,
            std::mem::align_of::<Vertex3D>() as u64,
        );
        let index_offset =
            allocate_range(&mut self.free_indices, &mut self.index_end, index_size, 4);
        self.grow_if_needed(
            device,
            queue,
            vertex_offset + vertex_size,
            index_offset + index_size,
            previous_vertex_end,
            previous_index_end,
        );
        if vertex_size > 0 {
            queue.write_buffer(&self.vertex, vertex_offset, bytemuck::cast_slice(vertices));
        }
        if index_size > 0 {
            queue.write_buffer(&self.index, index_offset, bytemuck::cast_slice(indices));
        }
        GeometrySlice {
            vertex_offset,
            vertex_size,
            index_offset,
            index_size,
            first_index: (index_offset / 4) as u32,
            base_vertex: (vertex_offset / std::mem::size_of::<Vertex3D>() as u64) as i32,
        }
    }

    fn retire_many(
        &mut self,
        queue: &wgpu::Queue,
        slices: impl IntoIterator<Item = GeometrySlice>,
    ) {
        let slices: Vec<_> = slices.into_iter().collect();
        if slices.is_empty() {
            return;
        }
        let epoch = self.completion.track_submitted_work(queue);
        self.retired.extend(
            slices
                .into_iter()
                .map(|slice| RetiredGeometry { epoch, slice }),
        );
    }

    fn collect(&mut self) {
        let completed = self.completion.completed_epoch();
        let mut keep = Vec::with_capacity(self.retired.len());
        for retired in self.retired.drain(..) {
            if retired.epoch > completed {
                keep.push(retired);
                continue;
            }
            if retired.slice.vertex_size > 0 {
                self.free_vertices.push(FreeRange {
                    offset: retired.slice.vertex_offset,
                    size: retired.slice.vertex_size,
                });
            }
            if retired.slice.index_size > 0 {
                self.free_indices.push(FreeRange {
                    offset: retired.slice.index_offset,
                    size: retired.slice.index_size,
                });
            }
        }
        self.retired = keep;
        merge_ranges(&mut self.free_vertices);
        merge_ranges(&mut self.free_indices);
    }

    fn grow_if_needed(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        required_vertex: u64,
        required_index: u64,
        copy_vertex_end: u64,
        copy_index_end: u64,
    ) {
        let new_vertex_capacity = grown_capacity(self.vertex_capacity, required_vertex);
        let new_index_capacity = grown_capacity(self.index_capacity, required_index);
        if new_vertex_capacity == self.vertex_capacity && new_index_capacity == self.index_capacity
        {
            return;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_geometry_arena_grow"),
        });
        if new_vertex_capacity != self.vertex_capacity {
            let next = create_vertex_arena(device, new_vertex_capacity);
            if copy_vertex_end > 0 {
                encoder.copy_buffer_to_buffer(&self.vertex, 0, &next, 0, copy_vertex_end);
            }
            self.vertex = next;
            self.vertex_capacity = new_vertex_capacity;
        }
        if new_index_capacity != self.index_capacity {
            let next = create_index_arena(device, new_index_capacity);
            if copy_index_end > 0 {
                encoder.copy_buffer_to_buffer(&self.index, 0, &next, 0, copy_index_end);
            }
            self.index = next;
            self.index_capacity = new_index_capacity;
        }
        self.generation = self.generation.wrapping_add(1);
        queue.submit(std::iter::once(encoder.finish()));
    }
}

fn create_vertex_arena(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_driven_shared_vertices"),
        size,
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_index_arena(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_driven_shared_indices"),
        size,
        usage: wgpu::BufferUsages::INDEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn grown_capacity(current: u64, required: u64) -> u64 {
    if required <= current {
        current
    } else {
        required
            .next_power_of_two()
            .max(GeometryArena::INITIAL_CAPACITY)
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

fn allocate_range(free: &mut Vec<FreeRange>, end: &mut u64, size: u64, alignment: u64) -> u64 {
    if size == 0 {
        return 0;
    }
    for index in 0..free.len() {
        let aligned = align_up(free[index].offset, alignment);
        let padding = aligned - free[index].offset;
        if padding + size > free[index].size {
            continue;
        }
        let original = free.swap_remove(index);
        if padding > 0 {
            free.push(FreeRange {
                offset: original.offset,
                size: padding,
            });
        }
        let tail_offset = aligned + size;
        let tail_size = original.size - padding - size;
        if tail_size > 0 {
            free.push(FreeRange {
                offset: tail_offset,
                size: tail_size,
            });
        }
        return aligned;
    }
    let offset = align_up(*end, alignment);
    *end = offset + size;
    offset
}

fn merge_ranges(ranges: &mut Vec<FreeRange>) {
    ranges.sort_by_key(|range| range.offset);
    let mut write = 0usize;
    for read in 0..ranges.len() {
        let current = ranges[read];
        if write > 0 {
            let previous = &mut ranges[write - 1];
            if previous.offset + previous.size == current.offset {
                previous.size += current.size;
                continue;
            }
        }
        ranges[write] = current;
        write += 1;
    }
    ranges.truncate(write);
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuDrawRecord {
    pub(crate) uniforms: Uniforms3D,
    pub(crate) bounds_min: [f32; 4],
    pub(crate) bounds_max: [f32; 4],
    /// x=index count, y=first index, z=bitcast base vertex, w=material ID.
    pub(crate) draw: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CullParams {
    planes: [[f32; 4]; 6],
    meta: [u32; 4],
}

#[derive(Copy, Clone, Debug, Default)]
pub struct SubmissionStats {
    pub submitted: u32,
    pub compatibility: u32,
    pub indirect_calls: u32,
    pub frustum_visible_oracle: u32,
    pub frustum_culled_oracle: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum IndirectRoute {
    All,
    Compatibility,
}

pub struct GpuDrivenRenderer {
    arena: GeometryArena,
    enabled: bool,
    count_supported: bool,
    draw_capacity: usize,
    draw_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    compatibility_indirect_buffer: Option<wgpu::Buffer>,
    cull_params: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,
    cull_layout: wgpu::BindGroupLayout,
    cull_bind_group: wgpu::BindGroup,
    draw_layout: wgpu::BindGroupLayout,
    draw_bind_group: wgpu::BindGroup,
    cull_pipeline: Option<wgpu::ComputePipeline>,
    depth_pipeline: Option<wgpu::RenderPipeline>,
    main_pipeline: Option<wgpu::RenderPipeline>,
    main_prepassed_pipeline: Option<wgpu::RenderPipeline>,
    main_visibility_compat_pipeline: Option<wgpu::RenderPipeline>,
    main_visibility_compat_prepassed_pipeline: Option<wgpu::RenderPipeline>,
    visibility: super::visibility_buffer::VisibilityBufferRuntime,
    pub(super) draw_scratch: Vec<GpuDrawRecord>,
    pub stats: SubmissionStats,
}

impl GpuDrivenRenderer {
    pub fn new(
        device: &wgpu::Device,
        lighting_layout: &wgpu::BindGroupLayout,
        joint_layout: &wgpu::BindGroupLayout,
        global_material_layout: Option<&wgpu::BindGroupLayout>,
        scene_source: &str,
    ) -> Self {
        let forced_off = gpu_driven_forced_off();
        let feature_ready = device
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
        let tier_ready = global_material_layout.is_some();
        let enabled =
            cfg!(not(target_arch = "wasm32")) && feature_ready && tier_ready && !forced_off;
        let count_supported = device
            .features()
            .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        let routed_visibility = super::visibility_buffer::requested_mode().shades() && enabled;
        let draw_capacity = 64;
        let draw_buffer = create_draw_buffer(device, draw_capacity);
        let indirect_buffer =
            create_indirect_buffer(device, draw_capacity, "gpu_driven_indirect_commands");
        let compatibility_indirect_buffer = routed_visibility.then(|| {
            create_indirect_buffer(
                device,
                draw_capacity,
                "gpu_driven_compatibility_indirect_commands",
            )
        });
        let cull_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_driven_cull_params"),
            size: std::mem::size_of::<CullParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_driven_counters"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cull_layout = create_cull_layout(device, routed_visibility);
        let cull_bind_group = create_cull_bind_group(
            device,
            &cull_layout,
            &draw_buffer,
            &indirect_buffer,
            compatibility_indirect_buffer.as_ref(),
            &cull_params,
            &counter_buffer,
        );
        let draw_layout = create_draw_layout(device);
        let draw_bind_group = create_draw_bind_group(device, &draw_layout, &draw_buffer);
        let (
            cull_pipeline,
            depth_pipeline,
            main_pipeline,
            main_prepassed_pipeline,
            main_visibility_compat_pipeline,
            main_visibility_compat_prepassed_pipeline,
        ) = if let Some(global_layout) = global_material_layout.filter(|_| enabled) {
            let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpu_driven_cull_shader"),
                source: wgpu::ShaderSource::Wgsl(cull_shader_source(routed_visibility)),
            });
            let cull_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("gpu_driven_cull_pipeline_layout"),
                    bind_group_layouts: &[Some(&cull_layout)],
                    immediate_size: 0,
                });
            let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpu_driven_cull_pipeline"),
                layout: Some(&cull_pipeline_layout),
                module: &cull_shader,
                entry_point: Some("cs_cull"),
                compilation_options: Default::default(),
                cache: None,
            });

            let shader_source = make_gpu_scene_shader(scene_source);
            let prepassed_source = strip_prepass_discard(&shader_source);
            let visibility_compat_source = super::visibility_buffer::requested_mode()
                .shades()
                .then(|| {
                    super::visibility_shading::make_forward_compatibility_shader(&shader_source)
                });
            let visibility_compat_prepassed_source = visibility_compat_source
                .as_deref()
                .map(strip_prepass_discard);
            let visibility_depth_shader = routed_visibility.then(|| {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("gpu_driven_visibility_depth_shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        super::visibility_shading::make_visibility_depth_shader(&shader_source)
                            .into(),
                    ),
                })
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpu_driven_scene_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
            let prepassed_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpu_driven_scene_prepassed_shader"),
                source: wgpu::ShaderSource::Wgsl(prepassed_source.into()),
            });
            let visibility_compat_shader = visibility_compat_source.as_deref().map(|source| {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("gpu_driven_visibility_compat_shader"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                })
            });
            let visibility_compat_prepassed_shader =
                visibility_compat_prepassed_source.as_deref().map(|source| {
                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("gpu_driven_visibility_compat_prepassed_shader"),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    })
                });
            let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_driven_scene_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&draw_layout),
                    Some(lighting_layout),
                    Some(global_layout),
                    Some(joint_layout),
                ],
                immediate_size: 0,
            });
            let visibility_compat_pipeline = visibility_compat_shader.as_ref().map(|fragment| {
                create_main_pipeline(device, &render_layout, &shader, fragment, false)
            });
            let visibility_compat_prepassed_pipeline =
                visibility_compat_prepassed_shader.as_ref().map(|fragment| {
                    create_main_pipeline(device, &render_layout, &shader, fragment, true)
                });
            (
                Some(cull_pipeline),
                Some(create_depth_pipeline(
                    device,
                    &render_layout,
                    visibility_depth_shader.as_ref().unwrap_or(&shader),
                    routed_visibility,
                )),
                Some(create_main_pipeline(
                    device,
                    &render_layout,
                    &shader,
                    &shader,
                    false,
                )),
                Some(create_main_pipeline(
                    device,
                    &render_layout,
                    &shader,
                    &prepassed_shader,
                    true,
                )),
                visibility_compat_pipeline,
                visibility_compat_prepassed_pipeline,
            )
        } else {
            (None, None, None, None, None, None)
        };

        let visibility_source = (super::visibility_buffer::requested_mode().requested() && enabled)
            .then(|| make_gpu_scene_shader(scene_source));
        let visibility = super::visibility_buffer::VisibilityBufferRuntime::new(
            device,
            enabled,
            &draw_layout,
            lighting_layout,
            global_material_layout,
            joint_layout,
            visibility_source.as_deref(),
        );

        let renderer = Self {
            arena: GeometryArena::new(device),
            enabled,
            count_supported,
            draw_capacity,
            draw_buffer,
            indirect_buffer,
            compatibility_indirect_buffer,
            cull_params,
            counter_buffer,
            cull_layout,
            cull_bind_group,
            draw_layout,
            draw_bind_group,
            cull_pipeline,
            depth_pipeline,
            main_pipeline,
            main_prepassed_pipeline,
            main_visibility_compat_pipeline,
            main_visibility_compat_prepassed_pipeline,
            visibility,
            draw_scratch: Vec::with_capacity(draw_capacity),
            stats: SubmissionStats::default(),
        };
        log::info!(
            "bloom: gpu-driven submission enabled={} indirect_count={} tier_a={} first_instance={}",
            renderer.enabled,
            renderer.count_supported,
            tier_ready,
            feature_ready,
        );
        renderer
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn submitting(&self) -> bool {
        self.enabled && !self.draw_scratch.is_empty()
    }

    pub(crate) fn visibility_routing_enabled(&self) -> bool {
        self.compatibility_indirect_buffer.is_some()
    }

    #[cfg(feature = "models3d")]
    pub(super) fn draw_layout(&self) -> &wgpu::BindGroupLayout {
        &self.draw_layout
    }

    #[cfg(feature = "models3d")]
    pub(super) fn draw_bind_group(&self) -> &wgpu::BindGroup {
        &self.draw_bind_group
    }

    pub(super) fn shared_geometry(&self) -> (&wgpu::Buffer, &wgpu::Buffer) {
        (&self.arena.vertex, &self.arena.index)
    }

    pub fn upload_static(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[Vertex3D],
        indices: &[u32],
    ) -> GeometrySlice {
        self.arena.upload(device, queue, vertices, indices)
    }

    pub fn retire_shared(
        &mut self,
        queue: &wgpu::Queue,
        slices: impl IntoIterator<Item = GeometrySlice>,
    ) {
        self.arena.retire_many(queue, slices);
    }

    pub(super) fn mesh_draw<'a>(
        &'a self,
        geometry: &'a MeshGeometry,
        index_count: u32,
    ) -> MeshDrawRef<'a> {
        match geometry {
            MeshGeometry::Shared(slice) => MeshDrawRef {
                vertex: &self.arena.vertex,
                index: &self.arena.index,
                first_index: slice.first_index,
                index_count,
                base_vertex: slice.base_vertex,
            },
            MeshGeometry::Dedicated { vertex, index } => MeshDrawRef {
                vertex,
                index,
                first_index: 0,
                index_count,
                base_vertex: 0,
            },
        }
    }

    /// Mesh-local binding window for passes that pair the shared primary
    /// arena with a per-mesh sidecar stream. The index values remain
    /// primitive-local, so slicing both arena buffers lets every vertex stream
    /// use base vertex zero.
    pub(super) fn mesh_draw_localized<'a>(
        &'a self,
        geometry: &'a MeshGeometry,
        index_count: u32,
    ) -> (MeshDrawRef<'a>, u64, u64) {
        match geometry {
            MeshGeometry::Shared(slice) => (
                MeshDrawRef {
                    vertex: &self.arena.vertex,
                    index: &self.arena.index,
                    first_index: 0,
                    index_count,
                    base_vertex: 0,
                },
                slice.vertex_offset,
                slice.index_offset,
            ),
            MeshGeometry::Dedicated { vertex, index } => (
                MeshDrawRef {
                    vertex,
                    index,
                    first_index: 0,
                    index_count,
                    base_vertex: 0,
                },
                0,
                0,
            ),
        }
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        planes: [[f32; 4]; 6],
        compatibility_count: u32,
        frustum_visible_oracle: u32,
        frustum_culled_oracle: u32,
    ) {
        self.visibility.begin_frame();
        self.stats = SubmissionStats {
            submitted: self.draw_scratch.len() as u32,
            compatibility: compatibility_count,
            indirect_calls: u32::from(!self.draw_scratch.is_empty()),
            frustum_visible_oracle,
            frustum_culled_oracle,
        };
        if !self.enabled || self.draw_scratch.is_empty() {
            return;
        }
        self.ensure_draw_capacity(device, self.draw_scratch.len());
        queue.write_buffer(
            &self.draw_buffer,
            0,
            bytemuck::cast_slice(&self.draw_scratch),
        );
        let params = CullParams {
            planes,
            meta: [self.draw_scratch.len() as u32, 0, 0, 0],
        };
        queue.write_buffer(&self.cull_params, 0, bytemuck::bytes_of(&params));
        // x is the ordered command count consumed by the count API; y/z are
        // visible/culled telemetry atomics written by compute.
        queue.write_buffer(
            &self.counter_buffer,
            0,
            bytemuck::cast_slice(&[self.draw_scratch.len() as u32, 0u32, 0, 0]),
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_driven_frustum_cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(
            self.cull_pipeline
                .as_ref()
                .expect("enabled gpu-driven path owns cull pipeline"),
        );
        pass.set_bind_group(0, &self.cull_bind_group, &[]);
        pass.dispatch_workgroups((self.draw_scratch.len() as u32).div_ceil(64), 1, 1);
    }

    pub fn report_json(&self) -> String {
        let classified = self.stats.frustum_visible_oracle + self.stats.frustum_culled_oracle;
        let culled_ratio = if classified == 0 {
            0.0
        } else {
            self.stats.frustum_culled_oracle as f64 / classified as f64
        };
        let routed_streams = self.compatibility_indirect_buffer.is_some();
        let routed_indirect_bytes = if routed_streams {
            (self.draw_capacity * std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>()) as u64
        } else {
            0
        };
        format!(
            concat!(
                "{{\"enabled\":{},\"indirect_count_supported\":{},",
                "\"submitted\":{},\"compatibility\":{},\"indirect_calls\":{},",
                "\"frustum_visible_oracle\":{},\"frustum_culled_oracle\":{},",
                "\"frustum_culled_ratio\":{:.6},",
                "\"visibility_routed_indirect_streams\":{},",
                "\"visibility_routed_indirect_bytes\":{},",
                "\"classification_source\":\"retained-scene conservative CPU oracle\",",
                "\"visibility_buffer_contract\":{},",
                "\"visibility_buffer_runtime\":{}}}"
            ),
            self.enabled,
            self.count_supported,
            self.stats.submitted,
            self.stats.compatibility,
            self.stats.indirect_calls,
            self.stats.frustum_visible_oracle,
            self.stats.frustum_culled_oracle,
            culled_ratio,
            routed_streams,
            routed_indirect_bytes,
            super::visibility_buffer::contract_json(),
            self.visibility.report_json(),
        )
    }

    pub fn draw_depth<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        lighting: &'a wgpu::BindGroup,
        global_materials: &'a wgpu::BindGroup,
        joints: &'a wgpu::BindGroup,
    ) -> bool {
        let Some(pipeline) = self.depth_pipeline.as_ref() else {
            return false;
        };
        if self.draw_scratch.is_empty() {
            return false;
        }
        pass.set_pipeline(pipeline);
        self.bind_and_draw(pass, lighting, global_materials, joints, IndirectRoute::All);
        self.visibility.shading_requested()
    }

    pub fn draw_main<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        lighting: &'a wgpu::BindGroup,
        global_materials: &'a wgpu::BindGroup,
        joints: &'a wgpu::BindGroup,
        prepassed: bool,
    ) {
        let pipeline = match (prepassed, self.visibility.shading_active()) {
            (true, true) => self.main_visibility_compat_prepassed_pipeline.as_ref(),
            (false, true) => self.main_visibility_compat_pipeline.as_ref(),
            (true, false) => self.main_prepassed_pipeline.as_ref(),
            (false, false) => self.main_pipeline.as_ref(),
        };
        let Some(pipeline) = pipeline else {
            return;
        };
        if self.draw_scratch.is_empty() {
            return;
        }
        pass.set_pipeline(pipeline);
        self.bind_and_draw(
            pass,
            lighting,
            global_materials,
            joints,
            if self.visibility.shading_active() {
                IndirectRoute::Compatibility
            } else {
                IndirectRoute::All
            },
        );
    }

    pub(crate) fn visibility_diagnostic_enabled(&self) -> bool {
        self.visibility.enabled()
    }

    #[cfg(feature = "models3d")]
    pub(crate) const fn visibility_shading_requested(&self) -> bool {
        self.visibility.shading_requested()
    }

    fn update_visibility_draw_counts(&mut self) -> u32 {
        let eligible = self
            .draw_scratch
            .iter()
            .filter(|draw| {
                bitcast_draw_flags(draw.bounds_min[3]) & DRAW_FLAG_VISIBILITY_ELIGIBLE != 0
            })
            .count() as u32;
        let compatibility = self
            .stats
            .compatibility
            .saturating_add(self.draw_scratch.len() as u32 - eligible);
        self.visibility.set_draw_counts(eligible, compatibility);
        eligible
    }

    #[cfg(feature = "models3d")]
    pub(crate) fn prepare_visibility_shading(
        &mut self,
        device: &wgpu::Device,
        extent: (u32, u32),
        force_target: bool,
    ) -> super::visibility_buffer::ResourceCreations {
        if !self.visibility.shading_requested() {
            return super::visibility_buffer::ResourceCreations::default();
        }
        let eligible = self.update_visibility_draw_counts();
        if eligible == 0 && !force_target {
            return super::visibility_buffer::ResourceCreations::default();
        }
        self.visibility.ensure_resources(
            device,
            extent,
            &self.arena.vertex,
            &self.arena.index,
            &self.draw_buffer,
            self.draw_capacity,
            self.arena.generation,
        )
    }

    #[cfg(not(feature = "models3d"))]
    pub(crate) fn prepare_visibility_shading(
        &mut self,
        device: &wgpu::Device,
        extent: (u32, u32),
    ) -> super::visibility_buffer::ResourceCreations {
        if !self.visibility.shading_requested() || self.draw_scratch.is_empty() {
            return super::visibility_buffer::ResourceCreations::default();
        }
        if self.update_visibility_draw_counts() == 0 {
            return super::visibility_buffer::ResourceCreations::default();
        }
        self.visibility.ensure_resources(
            device,
            extent,
            &self.arena.vertex,
            &self.arena.index,
            &self.draw_buffer,
            self.draw_capacity,
            self.arena.generation,
        )
    }

    pub(crate) fn visibility_raster_attachment(
        &self,
    ) -> Option<wgpu::RenderPassColorAttachment<'_>> {
        self.visibility
            .shading_requested()
            .then(|| self.visibility.raster_attachment())
            .flatten()
    }

    #[cfg(feature = "models3d")]
    pub(crate) fn visibility_texture(&self) -> Option<&wgpu::Texture> {
        self.visibility.visibility_texture()
    }

    pub(crate) fn finish_visibility_raster_inline(&mut self, recorded: bool) {
        if recorded {
            self.visibility.mark_raster_recorded();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_visibility_diagnostic(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        depth_view: &wgpu::TextureView,
        lighting: &wgpu::BindGroup,
        global_materials: &wgpu::BindGroup,
        joints: &wgpu::BindGroup,
        extent: (u32, u32),
    ) -> super::visibility_buffer::ResourceCreations {
        if !self.visibility.enabled() {
            return super::visibility_buffer::ResourceCreations::default();
        }
        if self.visibility.shading_requested() {
            return super::visibility_buffer::ResourceCreations::default();
        }
        if self.draw_scratch.is_empty() {
            self.visibility.set_draw_counts(0, self.stats.compatibility);
            return super::visibility_buffer::ResourceCreations::default();
        }
        let eligible = self.update_visibility_draw_counts();
        if eligible == 0 {
            return super::visibility_buffer::ResourceCreations::default();
        }
        let creations = self.visibility.ensure_resources(
            device,
            extent,
            &self.arena.vertex,
            &self.arena.index,
            &self.draw_buffer,
            self.draw_capacity,
            self.arena.generation,
        );
        profiler.begin("visibility_raster_pass");
        {
            let timestamps = profiler.pass_timestamp_writes("visibility_raster_pass");
            self.visibility.record_raster(
                encoder,
                depth_view,
                &self.draw_bind_group,
                lighting,
                global_materials,
                joints,
                &self.arena.vertex,
                &self.arena.index,
                &self.indirect_buffer,
                &self.counter_buffer,
                self.draw_scratch.len() as u32,
                self.count_supported,
                timestamps,
            );
        }
        profiler.end("visibility_raster_pass");
        if self.visibility.reconstruction_enabled() {
            profiler.begin("visibility_reconstruct_pass");
            {
                let timestamps =
                    profiler.compute_pass_timestamp_writes("visibility_reconstruct_pass");
                self.visibility.record_reconstruct(encoder, timestamps);
            }
            profiler.end("visibility_reconstruct_pass");
        }
        creations
    }

    pub(crate) fn draw_visibility_shading<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        lighting: &'a wgpu::BindGroup,
        global_materials: &'a wgpu::BindGroup,
        joints: &'a wgpu::BindGroup,
    ) {
        self.visibility.draw_shading(
            pass,
            &self.draw_bind_group,
            lighting,
            global_materials,
            joints,
        );
    }

    pub(crate) fn record_visibility_debug_overlay(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        hdr_view: &wgpu::TextureView,
    ) {
        if !self.visibility.debug_overlay_enabled() {
            return;
        }
        profiler.begin("visibility_debug_overlay");
        {
            let timestamps = profiler.pass_timestamp_writes("visibility_debug_overlay");
            self.visibility
                .record_debug_overlay(encoder, hdr_view, timestamps);
        }
        profiler.end("visibility_debug_overlay");
    }

    fn bind_and_draw<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        lighting: &'a wgpu::BindGroup,
        global_materials: &'a wgpu::BindGroup,
        joints: &'a wgpu::BindGroup,
        route: IndirectRoute,
    ) {
        pass.set_bind_group(0, &self.draw_bind_group, &[]);
        pass.set_bind_group(1, lighting, &[]);
        pass.set_bind_group(2, global_materials, &[]);
        pass.set_bind_group(3, joints, &[]);
        pass.set_vertex_buffer(0, self.arena.vertex.slice(..));
        pass.set_index_buffer(self.arena.index.slice(..), wgpu::IndexFormat::Uint32);
        let indirect = match route {
            IndirectRoute::All => &self.indirect_buffer,
            IndirectRoute::Compatibility => self
                .compatibility_indirect_buffer
                .as_ref()
                .expect("active visibility shading owns a compatibility command stream"),
        };
        let count = self.draw_scratch.len() as u32;
        if self.count_supported {
            pass.multi_draw_indexed_indirect_count(indirect, 0, &self.counter_buffer, 0, count);
        } else {
            pass.multi_draw_indexed_indirect(indirect, 0, count);
        }
    }

    fn ensure_draw_capacity(&mut self, device: &wgpu::Device, required: usize) {
        if required <= self.draw_capacity {
            return;
        }
        self.draw_capacity = required.next_power_of_two();
        self.draw_buffer = create_draw_buffer(device, self.draw_capacity);
        self.indirect_buffer =
            create_indirect_buffer(device, self.draw_capacity, "gpu_driven_indirect_commands");
        if self.compatibility_indirect_buffer.is_some() {
            self.compatibility_indirect_buffer = Some(create_indirect_buffer(
                device,
                self.draw_capacity,
                "gpu_driven_compatibility_indirect_commands",
            ));
        }
        self.cull_bind_group = create_cull_bind_group(
            device,
            &self.cull_layout,
            &self.draw_buffer,
            &self.indirect_buffer,
            self.compatibility_indirect_buffer.as_ref(),
            &self.cull_params,
            &self.counter_buffer,
        );
        self.draw_bind_group = create_draw_bind_group(device, &self.draw_layout, &self.draw_buffer);
        self.draw_scratch.reserve(
            self.draw_capacity
                .saturating_sub(self.draw_scratch.capacity()),
        );
    }
}

fn bitcast_draw_flags(value: f32) -> u32 {
    value.to_bits()
}

impl Renderer {
    /// Prepare retained scene resources through both the compatibility and
    /// GPU-driven paths. Keeping this borrow split inside Renderer lets the
    /// scene use the shared geometry arena without exposing device internals.
    pub fn prepare_scene_graph(
        &mut self,
        scene: &mut crate::scene::SceneGraph,
        use_occlusion: bool,
    ) {
        if self.temporal_camera_cut_active {
            scene.reset_motion_history();
        }
        let vp = self.current_vp_matrix;
        let prev_vp = self.velocity_ref_vp;
        let occlusion = use_occlusion.then_some(&self.occlusion);
        scene.prepare_with_refraction(
            &self.device,
            &self.queue,
            &vp,
            &prev_vp,
            &self.uniform_3d_layout,
            occlusion,
            &mut self.gpu_driven,
            self.imported_refraction_enabled,
        );
        scene.prepare_materials(self);
    }

    pub(crate) fn gpu_driven_enabled(&self) -> bool {
        self.gpu_driven.enabled()
    }

    pub(crate) fn allocate_scene_gpu_material(
        &mut self,
        material: &crate::scene::PbrMaterial,
    ) -> super::material_indirection::MaterialId {
        let texture_id = |index: u32| {
            self.global_texture_ids
                .get(index as usize)
                .copied()
                .unwrap_or(super::material_indirection::TextureId::FALLBACK)
        };
        let mut record = super::material_indirection::GpuMaterialRecord::default();
        // Per-node colour/opacity stay in Uniforms3D.model_tint, matching the
        // compatibility scene shader. The global record carries only factors
        // that were previously held by SceneMaterialUniforms.
        record.metal_rough = [
            material.metalness,
            material.roughness,
            if material.specular_glossiness_factor.is_some() {
                2.0
            } else {
                (material.metallic_roughness_texture_idx != 0) as u8 as f32
            },
            material
                .alpha_mode
                .shader_alpha_value(material.alpha_cutoff),
        ];
        record.emissive = [
            material.emissive[0],
            material.emissive[1],
            material.emissive[2],
            if material.alpha_coverage_mips {
                1.0
            } else {
                0.0
            },
        ];
        record.spec_gloss = material.specular_glossiness_factor.unwrap_or([1.0; 4]);
        record.texture_ids_0 = [
            texture_id(material.texture_idx).raw(),
            texture_id(material.normal_texture_idx).raw(),
            texture_id(material.metallic_roughness_texture_idx).raw(),
            texture_id(material.emissive_texture_idx).raw(),
        ];
        record.texture_ids_1[0] = texture_id(material.occlusion_texture_idx).raw();
        record.sampler_ids_0 = [self.global_linear_sampler_id.raw(); 4];
        record.sampler_ids_1[0] = self.global_linear_sampler_id.raw();
        self.material_system
            .indirection
            .allocate_material(&self.device, record)
    }

    pub(crate) fn retire_scene_gpu_materials(
        &mut self,
        ids: impl IntoIterator<Item = super::material_indirection::MaterialId>,
    ) {
        self.material_system
            .indirection
            .retire_materials(&self.queue, ids);
    }
}

fn gpu_driven_forced_off() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| {
        let explicitly_disabled = std::env::var("BLOOM_GPU_DRIVEN")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "off" | "false" | "disabled"
                )
            })
            .unwrap_or(false);
        explicitly_disabled
            || !super::capabilities::RendererCapabilities::forced_path_allowed(
                super::capabilities::RendererCapabilityTier::HighEnd,
            )
    })
}

fn create_draw_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_driven_draw_records"),
        size: (capacity * std::mem::size_of::<GpuDrawRecord>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_indirect_buffer(
    device: &wgpu::Device,
    capacity: usize,
    label: &'static str,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (capacity * std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        mapped_at_creation: false,
    })
}

fn create_cull_layout(device: &wgpu::Device, routed_visibility: bool) -> wgpu::BindGroupLayout {
    let storage = |binding, read_only, visibility| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let mut entries = vec![
        storage(
            0,
            true,
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
        ),
        storage(1, false, wgpu::ShaderStages::COMPUTE),
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        storage(3, false, wgpu::ShaderStages::COMPUTE),
    ];
    if routed_visibility {
        entries.push(storage(4, false, wgpu::ShaderStages::COMPUTE));
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gpu_driven_cull_layout"),
        entries: &entries,
    })
}

fn create_draw_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gpu_driven_draw_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn create_draw_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    draws: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu_driven_draw_bind_group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: draws.as_entire_binding(),
        }],
    })
}

fn create_cull_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    draws: &wgpu::Buffer,
    indirect: &wgpu::Buffer,
    compatibility_indirect: Option<&wgpu::Buffer>,
    params: &wgpu::Buffer,
    counters: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: draws.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: indirect.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: params.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: counters.as_entire_binding(),
        },
    ];
    if let Some(compatibility) = compatibility_indirect {
        entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: compatibility.as_entire_binding(),
        });
    }
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu_driven_cull_bind_group"),
        layout,
        entries: &entries,
    })
}

fn create_depth_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    visibility_target: bool,
) -> wgpu::RenderPipeline {
    let targets = [visibility_target.then_some(wgpu::ColorTargetState {
        format: super::visibility_buffer::VISIBILITY_FORMAT,
        blend: None,
        // Shade mode's specialized depth fragment writes packed IDs while
        // priming depth; the ordinary path has no color target at all.
        write_mask: wgpu::ColorWrites::ALL,
    })];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gpu_driven_depth_pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main_scene"),
            buffers: &[Vertex3D::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_depth_prepass"),
            // Slot zero is normally unattached. Shade mode declares the
            // packed-ID target so one traversal owns both depth and IDs.
            targets: &targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_main_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    vertex_shader: &wgpu::ShaderModule,
    fragment_shader: &wgpu::ShaderModule,
    prepassed: bool,
) -> wgpu::RenderPipeline {
    #[cfg(lean_mrt)]
    let targets = &[
        Some(wgpu::ColorTargetState {
            format: HDR_FORMAT,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        }),
        None,
        Some(wgpu::ColorTargetState {
            format: VELOCITY_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        None,
    ];
    #[cfg(not(lean_mrt))]
    let targets = &[
        Some(wgpu::ColorTargetState {
            format: HDR_FORMAT,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: MATERIAL_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: VELOCITY_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
    ];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(if prepassed {
            "gpu_driven_main_prepassed_pipeline"
        } else {
            "gpu_driven_main_pipeline"
        }),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: vertex_shader,
            entry_point: Some("vs_main_scene"),
            buffers: &[Vertex3D::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: fragment_shader,
            entry_point: Some("fs_main_scene"),
            targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            // Match the canonical prepassed scene pipeline: the depth pass is
            // two-sided, so the Equal-test main pass must shade the same face
            // if overlapping geometry made a back face win depth.
            cull_mode: if prepassed {
                None
            } else {
                Some(wgpu::Face::Back)
            },
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(!prepassed),
            depth_compare: Some(if prepassed {
                wgpu::CompareFunction::Equal
            } else {
                wgpu::CompareFunction::Less
            }),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

pub(super) fn make_gpu_scene_shader(source: &str) -> String {
    const LEGACY_MATERIALS: &str = r#"@group(2) @binding(0) var base_color_tex: texture_2d<f32>;
@group(2) @binding(1) var base_color_samp: sampler;
@group(2) @binding(2) var normal_tex: texture_2d<f32>;
@group(2) @binding(3) var normal_samp: sampler;
@group(2) @binding(4) var mr_tex: texture_2d<f32>;
@group(2) @binding(5) var mr_samp: sampler;
@group(2) @binding(6) var em_tex: texture_2d<f32>;
@group(2) @binding(7) var em_samp: sampler;
@group(2) @binding(8) var<uniform> material: MaterialFactors;
@group(2) @binding(9) var occ_tex: texture_2d<f32>;
@group(2) @binding(10) var occ_samp: sampler;"#;
    const GPU_DRAWS: &str = r#"struct GpuDrawRecord {
    uniforms: Uniforms3D,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    draw: vec4<u32>,
};
struct GpuDrawTable { records: array<GpuDrawRecord>, };
@group(0) @binding(0) var<storage, read> gpu_draws: GpuDrawTable;"#;
    let mut out = source
        .replace("@group(0) @binding(0) var<uniform> u: Uniforms3D;", GPU_DRAWS)
        .replace(
            LEGACY_MATERIALS,
            include_str!("../../shaders/material_indirection.wgsl"),
        )
        .replace(
            "    @location(6) prev_clip: vec4<f32>,\n};",
            "    @location(6) prev_clip: vec4<f32>,\n    @location(7) @interpolate(flat) material_id: u32,\n    @location(8) @interpolate(flat) draw_flags: u32,\n};",
        )
        .replace(
            "fn vs_main_scene(in: VertexInputScene) -> VertexOutputScene {",
            "fn vs_main_scene(in: VertexInputScene, @builtin(instance_index) draw_index: u32) -> VertexOutputScene {\n    let gpu_draw = gpu_draws.records[draw_index];\n    let u = gpu_draw.uniforms;\n    let material = bloom_material_record(gpu_draw.draw.w);",
        )
        .replace(
            "        return o;",
            "        o.material_id = gpu_draw.draw.w;\n        o.draw_flags = bitcast<u32>(gpu_draw.bounds_min.w);\n        return o;",
        )
        .replace(
            "    return out;\n}",
            "    out.material_id = gpu_draw.draw.w;\n    out.draw_flags = bitcast<u32>(gpu_draw.bounds_min.w);\n    return out;\n}",
        )
        .replace(
            "fn fs_depth_prepass(in: VertexOutputScene) {",
            "fn fs_depth_prepass(in: VertexOutputScene, @builtin(front_facing) front_facing: bool) {\n    let material = bloom_material_record(in.material_id);\n    if ((in.draw_flags & 1u) == 0u && !front_facing) { discard; }",
        )
        .replace(
            "fn shade_main_scene(in: VertexOutputScene, front_facing: bool) -> SceneOut {",
            "fn shade_main_scene(in: VertexOutputScene, front_facing: bool) -> SceneOut {\n    let material = bloom_material_record(in.material_id);",
        );
    out = out
        .replace(
            "textureSample(base_color_tex, base_color_samp, in.uv).a",
            "bloom_sample_raw(material.texture_ids_0.x, material.sampler_ids_0.x, in.uv).a",
        )
        .replace(
            "textureSampleBias(normal_tex, normal_samp, in.uv, 0.25 + lod_bias)",
            "bloom_sample_normal_raw_bias(material, in.uv, 0.25 + lod_bias)",
        )
        .replace(
            "textureSampleBias(base_color_tex, base_color_samp, in.uv, lod_bias)",
            "bloom_sample_raw_bias(material.texture_ids_0.x, material.sampler_ids_0.x, in.uv, lod_bias)",
        )
        .replace(
            "textureSampleLevel(base_color_tex, base_color_samp, in.uv, 0.0)",
            "bloom_sample_raw_level(material.texture_ids_0.x, material.sampler_ids_0.x, in.uv, 0.0)",
        )
        .replace(
            "textureSampleLevel(base_color_tex, base_color_samp, in.uv, mask_lod)",
            "bloom_sample_raw_level(material.texture_ids_0.x, material.sampler_ids_0.x, in.uv, mask_lod)",
        )
        .replace(
            "textureSampleLevel(base_color_tex, base_color_samp, in.uv, 1.0)",
            "bloom_sample_raw_level(material.texture_ids_0.x, material.sampler_ids_0.x, in.uv, 1.0)",
        )
        .replace(
            "textureDimensions(base_color_tex)",
            "bloom_base_color_dimensions(material)",
        )
        .replace(
            "textureSample(mr_tex, mr_samp, in.uv)",
            "bloom_sample_raw(material.texture_ids_0.z, material.sampler_ids_0.z, in.uv)",
        )
        .replace(
            "textureSample(em_tex, em_samp, in.uv)",
            "bloom_sample_raw_bias(material.texture_ids_0.w, material.sampler_ids_0.w, in.uv, 0.0)",
        )
        .replace(
            "textureSample(occ_tex, occ_samp, in.uv)",
            "bloom_sample_raw(material.texture_ids_1.x, material.sampler_ids_1.x, in.uv)",
        );
    assert!(
        out.contains("var<storage, read> gpu_draws"),
        "scene shader group-0 ABI changed"
    );
    assert!(
        !out.contains("base_color_tex"),
        "legacy material declarations or samples remain in GPU scene shader"
    );
    out
}

fn strip_prepass_discard(source: &str) -> String {
    match (
        source.find("//PREPASS_STRIP_BEGIN"),
        source.find("//PREPASS_STRIP_END"),
    ) {
        (Some(begin), Some(end)) if end > begin => {
            let suffix = end + "//PREPASS_STRIP_END".len();
            format!("{}{}", &source[..begin], &source[suffix..])
        }
        _ => source.to_string(),
    }
}

const CULL_SHADER: &str = r#"
struct Uniforms3D {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    prev_mvp: mat4x4<f32>,
    model_tint: vec4<f32>,
    misc: vec4<f32>,
};
struct GpuDrawRecord {
    uniforms: Uniforms3D,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    draw: vec4<u32>,
};
struct GpuDrawTable { records: array<GpuDrawRecord>, };
struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};
struct IndirectTable { commands: array<DrawIndexedIndirect>, };
struct CullParams {
    planes: array<vec4<f32>, 6>,
    draw_info: vec4<u32>,
};
struct Counters {
    draw_count: u32,
    visible: atomic<u32>,
    culled: atomic<u32>,
    padding: u32,
};
@group(0) @binding(0) var<storage, read> draws: GpuDrawTable;
@group(0) @binding(1) var<storage, read_write> indirect: IndirectTable;
@group(0) @binding(2) var<uniform> params: CullParams;
@group(0) @binding(3) var<storage, read_write> counters: Counters;

fn outside_frustum(bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
    if (bmin.x > bmax.x) {
        return false;
    }
    for (var i = 0u; i < 6u; i++) {
        let plane = params.planes[i];
        let p = vec3<f32>(
            select(bmin.x, bmax.x, plane.x >= 0.0),
            select(bmin.y, bmax.y, plane.y >= 0.0),
            select(bmin.z, bmax.z, plane.z >= 0.0)
        );
        if (dot(plane.xyz, p) + plane.w < 0.0) {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let draw_index = gid.x;
    if (draw_index >= params.draw_info.x) {
        return;
    }
    let draw = draws.records[draw_index];
    let visible = !outside_frustum(draw.bounds_min.xyz, draw.bounds_max.xyz);
    indirect.commands[draw_index] = DrawIndexedIndirect(
        draw.draw.x,
        select(0u, 1u, visible),
        draw.draw.y,
        bitcast<i32>(draw.draw.z),
        draw_index
    );
    if (visible) {
        atomicAdd(&counters.visible, 1u);
    } else {
        atomicAdd(&counters.culled, 1u);
    }
}
"#;

fn cull_shader_source(routed_visibility: bool) -> std::borrow::Cow<'static, str> {
    if !routed_visibility {
        return CULL_SHADER.into();
    }
    const BINDING_ANCHOR: &str =
        "@group(0) @binding(3) var<storage, read_write> counters: Counters;";
    const ROUTED_BINDINGS: &str = concat!(
        "@group(0) @binding(3) var<storage, read_write> counters: Counters;\n",
        "@group(0) @binding(4) var<storage, read_write> compatibility_indirect: IndirectTable;"
    );
    const COMMAND_ANCHOR: &str = r#"    indirect.commands[draw_index] = DrawIndexedIndirect(
        draw.draw.x,
        select(0u, 1u, visible),
        draw.draw.y,
        bitcast<i32>(draw.draw.z),
        draw_index
    );"#;
    const ROUTED_COMMANDS: &str = r#"    indirect.commands[draw_index] = DrawIndexedIndirect(
        draw.draw.x,
        select(0u, 1u, visible),
        draw.draw.y,
        bitcast<i32>(draw.draw.z),
        draw_index
    );
    let visibility_eligible = (bitcast<u32>(draw.bounds_min.w) & 2u) != 0u;
    compatibility_indirect.commands[draw_index] = DrawIndexedIndirect(
        draw.draw.x,
        select(0u, 1u, visible && !visibility_eligible),
        draw.draw.y,
        bitcast<i32>(draw.draw.z),
        draw_index
    );"#;
    assert_eq!(CULL_SHADER.matches(BINDING_ANCHOR).count(), 1);
    assert_eq!(CULL_SHADER.matches(COMMAND_ANCHOR).count(), 1);
    CULL_SHADER
        .replacen(BINDING_ANCHOR, ROUTED_BINDINGS, 1)
        .replacen(COMMAND_ANCHOR, ROUTED_COMMANDS, 1)
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_reuses_and_merges_ranges() {
        let mut free = vec![FreeRange {
            offset: 16,
            size: 16,
        }];
        let mut end = 64;
        assert_eq!(allocate_range(&mut free, &mut end, 8, 4), 16);
        free.push(FreeRange {
            offset: 24,
            size: 8,
        });
        free.push(FreeRange {
            offset: 16,
            size: 8,
        });
        merge_ranges(&mut free);
        assert!(free.iter().any(|range| *range
            == FreeRange {
                offset: 16,
                size: 16
            }));
    }

    #[test]
    fn gpu_record_matches_wgsl_alignment() {
        assert_eq!(std::mem::size_of::<Uniforms3D>(), 224);
        assert_eq!(std::mem::size_of::<GpuDrawRecord>(), 272);
        assert_eq!(
            std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>(),
            20
        );
        assert_eq!(draw_flags(false, false), 0);
        assert_eq!(draw_flags(true, false), DRAW_FLAG_DOUBLE_SIDED);
        assert_eq!(
            draw_flags(true, true),
            DRAW_FLAG_DOUBLE_SIDED | DRAW_FLAG_VISIBILITY_ELIGIBLE
        );
    }

    #[test]
    fn generated_scene_shader_preserves_legacy_color_decode() {
        let generated = make_gpu_scene_shader(super::super::SCENE_SHADER);
        wgpu::naga::front::wgsl::parse_str(&generated)
            .unwrap_or_else(|error| panic!("GPU-driven scene WGSL failed to parse: {error:?}"));
        assert!(generated.contains("bloom_sample_raw_bias(material.texture_ids_0.x"));
        assert!(generated.contains("bloom_sample_raw_bias(material.texture_ids_0.w"));
        assert!(generated.contains("128.0 / 255.0, 128.0 / 255.0, 1.0, 0.0"));
        assert!(!generated.contains("bloom_sample_registered_color_bias(material.texture_ids_0"));
        assert!(generated.contains("@builtin(front_facing) front_facing: bool"));
        assert!(generated.contains("(in.draw_flags & 1u) == 0u"));
    }

    #[test]
    fn routed_cull_shader_is_opt_in_and_partitions_instance_counts() {
        let ordinary = cull_shader_source(false);
        let routed = cull_shader_source(true);
        wgpu::naga::front::wgsl::parse_str(&ordinary)
            .unwrap_or_else(|error| panic!("ordinary cull WGSL failed: {error:?}"));
        wgpu::naga::front::wgsl::parse_str(&routed)
            .unwrap_or_else(|error| panic!("routed cull WGSL failed: {error:?}"));
        assert!(!ordinary.contains("visibility_indirect"));
        assert!(!ordinary.contains("compatibility_indirect"));
        assert!(!routed.contains("visibility_indirect"));
        assert!(routed.contains("visible && !visibility_eligible"));
        assert_eq!(routed.matches("first_instance: u32").count(), 1);
        assert_eq!(routed.matches("draw_index\n    );").count(), 2);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn routed_cull_gpu_preserves_slots_and_excludes_cross_route_instances() {
        use bytemuck::Zeroable as _;
        use wgpu::util::DeviceExt as _;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("no GPU adapter — skipping routed-cull oracle");
            return;
        };
        let mut required_limits = wgpu::Limits::downlevel_defaults();
        required_limits.max_storage_buffers_per_shader_stage = 5;
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("gpu_driven_routed_cull_oracle_device"),
                required_limits,
                ..Default::default()
            }))
        else {
            eprintln!("adapter rejected routed-cull oracle device");
            return;
        };

        let make_draw = |index: u32, eligible: bool, outside: bool| {
            let mut draw = GpuDrawRecord::zeroed();
            draw.bounds_min = if outside {
                [
                    -2.0,
                    -0.1,
                    -0.1,
                    f32::from_bits(draw_flags(false, eligible)),
                ]
            } else {
                // Sentinel bounds (min.x > max.x) are conservatively visible.
                [1.0, 0.0, 0.0, f32::from_bits(draw_flags(false, eligible))]
            };
            draw.bounds_max = if outside {
                [-1.0, 0.1, 0.1, 0.0]
            } else {
                [0.0, 0.0, 0.0, 0.0]
            };
            draw.draw = [3, index * 3, 0, 0];
            draw
        };
        let draws = [
            make_draw(0, true, false),
            make_draw(1, false, false),
            make_draw(2, true, true),
        ];
        let mut planes = [[0.0, 0.0, 0.0, 1.0]; 6];
        planes[0] = [1.0, 0.0, 0.0, 0.0];
        let params = CullParams {
            planes,
            meta: [draws.len() as u32, 0, 0, 0],
        };
        let draws_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("routed_cull_oracle_draws"),
            contents: bytemuck::cast_slice(&draws),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("routed_cull_oracle_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let counters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("routed_cull_oracle_counters"),
            contents: bytemuck::cast_slice(&[draws.len() as u32, 0, 0, 0]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let command_bytes =
            (draws.len() * std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>()) as u64;
        let make_commands = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: command_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let all = make_commands("routed_cull_oracle_all");
        let compatibility = make_commands("routed_cull_oracle_compatibility");
        let layout = create_cull_layout(&device, true);
        let bind_group = create_cull_bind_group(
            &device,
            &layout,
            &draws_buffer,
            &all,
            Some(&compatibility),
            &params_buffer,
            &counters,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("routed_cull_oracle_shader"),
            source: wgpu::ShaderSource::Wgsl(cull_shader_source(true)),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("routed_cull_oracle_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("routed_cull_oracle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_cull"),
            compilation_options: Default::default(),
            cache: None,
        });
        let readbacks = std::array::from_fn::<_, 2, _>(|index| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match index {
                    0 => "routed_cull_oracle_all_readback",
                    _ => "routed_cull_oracle_compatibility_readback",
                }),
                size: command_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("routed_cull_oracle_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("routed_cull_oracle_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        for (source, target) in [(&all, &readbacks[0]), (&compatibility, &readbacks[1])] {
            encoder.copy_buffer_to_buffer(source, 0, target, 0, command_bytes);
        }
        queue.submit(std::iter::once(encoder.finish()));

        let read = |buffer: &wgpu::Buffer| {
            let slice = buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            receiver
                .recv()
                .expect("routed-cull readback callback dropped")
                .expect("routed-cull readback mapping failed");
            let mapped = slice.get_mapped_range();
            let words = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
            drop(mapped);
            buffer.unmap();
            words
        };
        let all = read(&readbacks[0]);
        let compatibility = read(&readbacks[1]);
        let instance_counts = |words: &[u32]| [words[1], words[6], words[11]];
        let first_instances = |words: &[u32]| [words[4], words[9], words[14]];
        assert_eq!(instance_counts(&all), [1, 1, 0]);
        assert_eq!(instance_counts(&compatibility), [0, 1, 0]);
        assert_eq!(first_instances(&all), [0, 1, 2]);
        assert_eq!(first_instances(&compatibility), [0, 1, 2]);
    }
}
