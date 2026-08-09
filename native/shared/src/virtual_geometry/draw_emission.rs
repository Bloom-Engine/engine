use super::GpuVirtualHierarchySelector;
use std::fmt;

const WORKGROUP_SIZE: u32 = 64;

/// Non-indexed indirect command for raw-page vertex pulling. `first_instance`
/// addresses the matching selected-cluster record.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVirtualDrawIndirect {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// GPU-written draw-emission state. `draw_count` is at byte offset zero for
/// direct use by `multi_draw_indirect_count`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVirtualDrawEmissionState {
    pub draw_count: u32,
    pub batch_fallback: u32,
    pub selector_selected_count: u32,
    pub selector_selected_overflow: u32,
    pub selector_invalid_or_missing: u32,
    pub emitted_triangles: u32,
    pub emitted_draws: u32,
    pub reserved: [u32; 5],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVirtualDispatchIndirect {
    pub workgroups_x: u32,
    pub workgroups_y: u32,
    pub workgroups_z: u32,
}

const _: () = assert!(std::mem::size_of::<GpuVirtualDrawIndirect>() == 16);
const _: () = assert!(std::mem::size_of::<GpuVirtualDrawEmissionState>() == 48);
const _: () = assert!(std::mem::size_of::<GpuVirtualDispatchIndirect>() == 12);

/// Converts bounded hierarchy output into a compact non-indexed indirect
/// stream. If selection overflowed or observed an invalid/missing current
/// record, the GPU publishes a zero draw count for the entire virtual batch.
pub struct GpuVirtualDrawEmitter {
    selector_id: u64,
    draw_capacity: u32,
    command_buffer: wgpu::Buffer,
    state_buffer: wgpu::Buffer,
    dispatch_buffer: wgpu::Buffer,
    prepare_bind_group: wgpu::BindGroup,
    emit_bind_group: wgpu::BindGroup,
    prepare_pipeline: wgpu::ComputePipeline,
    emit_pipeline: wgpu::ComputePipeline,
}

impl GpuVirtualDrawEmitter {
    pub fn new(
        device: &wgpu::Device,
        selector: &GpuVirtualHierarchySelector,
    ) -> Result<Self, VirtualGeometryDrawEmissionError> {
        let draw_capacity = selector.config().max_selected_clusters;
        let command_bytes =
            u64::from(draw_capacity) * std::mem::size_of::<GpuVirtualDrawIndirect>() as u64;
        let limits = device.limits();
        if limits.max_storage_buffers_per_shader_stage < 5
            || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE
            || limits.max_compute_workgroup_size_x < WORKGROUP_SIZE
        {
            return Err(VirtualGeometryDrawEmissionError::DeviceUnsupported);
        }
        if command_bytes > limits.max_buffer_size
            || command_bytes > limits.max_storage_buffer_binding_size
        {
            return Err(VirtualGeometryDrawEmissionError::DeviceLimitExceeded {
                requested_bytes: command_bytes,
                maximum_bytes: limits
                    .max_buffer_size
                    .min(limits.max_storage_buffer_binding_size),
            });
        }
        let workgroups = draw_capacity.div_ceil(WORKGROUP_SIZE);
        if workgroups > limits.max_compute_workgroups_per_dimension {
            return Err(VirtualGeometryDrawEmissionError::DispatchLimitExceeded {
                requested: workgroups,
                maximum: limits.max_compute_workgroups_per_dimension,
            });
        }
        let command_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("virtual_geometry_indirect_commands"),
            size: command_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("virtual_geometry_draw_emission_state"),
            size: std::mem::size_of::<GpuVirtualDrawEmissionState>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dispatch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("virtual_geometry_draw_dispatch_args"),
            size: std::mem::size_of::<GpuVirtualDispatchIndirect>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let prepare_layout = create_prepare_layout(device);
        let prepare_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("virtual_geometry_draw_emission_bind_group"),
            layout: &prepare_layout,
            entries: &[
                binding(0, selector.selected_buffer()),
                binding(1, selector.counter_buffer()),
                binding(2, &command_buffer),
                binding(3, &state_buffer),
                binding(4, &dispatch_buffer),
            ],
        });
        let emit_layout = create_emit_layout(device);
        let emit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("virtual_geometry_draw_emit_bind_group"),
            layout: &emit_layout,
            entries: &[
                binding(0, selector.selected_buffer()),
                binding(1, &command_buffer),
                binding(2, &state_buffer),
            ],
        });
        let prepare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("virtual_geometry_draw_prepare_shader"),
            source: wgpu::ShaderSource::Wgsl(DRAW_PREPARE_SHADER.into()),
        });
        let prepare_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("virtual_geometry_draw_prepare_pipeline_layout"),
                bind_group_layouts: &[Some(&prepare_layout)],
                immediate_size: 0,
            });
        let prepare_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("virtual_geometry_draw_prepare_pipeline"),
            layout: Some(&prepare_pipeline_layout),
            module: &prepare_shader,
            entry_point: Some("prepare_virtual_draws"),
            compilation_options: Default::default(),
            cache: None,
        });
        let emit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("virtual_geometry_draw_emit_shader"),
            source: wgpu::ShaderSource::Wgsl(DRAW_EMIT_SHADER.into()),
        });
        let emit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("virtual_geometry_draw_emit_pipeline_layout"),
            bind_group_layouts: &[Some(&emit_layout)],
            immediate_size: 0,
        });
        let emit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("virtual_geometry_draw_emit_pipeline"),
            layout: Some(&emit_pipeline_layout),
            module: &emit_shader,
            entry_point: Some("emit_virtual_draws"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            selector_id: selector.id(),
            draw_capacity,
            command_buffer,
            state_buffer,
            dispatch_buffer,
            prepare_bind_group,
            emit_bind_group,
            prepare_pipeline,
            emit_pipeline,
        })
    }

    pub fn record(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        selector: &GpuVirtualHierarchySelector,
    ) -> Result<(), VirtualGeometryDrawEmissionError> {
        if selector.id() != self.selector_id {
            return Err(VirtualGeometryDrawEmissionError::SelectorMismatch);
        }
        queue.write_buffer(
            &self.state_buffer,
            0,
            bytemuck::bytes_of(&GpuVirtualDrawEmissionState::default()),
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("virtual_geometry_draw_prepare"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prepare_pipeline);
            pass.set_bind_group(0, &self.prepare_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("virtual_geometry_draw_emit"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.emit_pipeline);
            pass.set_bind_group(0, &self.emit_bind_group, &[]);
            pass.dispatch_workgroups_indirect(&self.dispatch_buffer, 0);
        }
        Ok(())
    }

    pub const fn draw_capacity(&self) -> u32 {
        self.draw_capacity
    }

    pub fn command_buffer(&self) -> &wgpu::Buffer {
        &self.command_buffer
    }

    /// Offset zero is a GPU-written draw count suitable for
    /// `multi_draw_indirect_count` after the emitter passes complete.
    pub fn state_buffer(&self) -> &wgpu::Buffer {
        &self.state_buffer
    }

    pub fn dispatch_buffer(&self) -> &wgpu::Buffer {
        &self.dispatch_buffer
    }
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_prepare_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("virtual_geometry_draw_prepare_layout"),
        entries: &[
            storage_layout_entry(0, true),
            storage_layout_entry(1, true),
            storage_layout_entry(2, true),
            storage_layout_entry(3, false),
            storage_layout_entry(4, false),
        ],
    })
}

fn create_emit_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("virtual_geometry_draw_emit_layout"),
        entries: &[
            storage_layout_entry(0, true),
            storage_layout_entry(1, false),
            storage_layout_entry(2, false),
        ],
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VirtualGeometryDrawEmissionError {
    DeviceUnsupported,
    DeviceLimitExceeded {
        requested_bytes: u64,
        maximum_bytes: u64,
    },
    DispatchLimitExceeded {
        requested: u32,
        maximum: u32,
    },
    SelectorMismatch,
}

impl fmt::Display for VirtualGeometryDrawEmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceUnsupported => write!(
                formatter,
                "device lacks the compute limits required by virtual draw emission"
            ),
            Self::DeviceLimitExceeded {
                requested_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "virtual indirect commands require {requested_bytes} bytes but the device limit is {maximum_bytes}"
            ),
            Self::DispatchLimitExceeded { requested, maximum } => write!(
                formatter,
                "virtual draw emission needs {requested} workgroups but one dimension is limited to {maximum}"
            ),
            Self::SelectorMismatch => write!(
                formatter,
                "virtual draw emitter was recorded with a different hierarchy selector"
            ),
        }
    }
}

impl std::error::Error for VirtualGeometryDrawEmissionError {}

const DRAW_PREPARE_SHADER: &str = r#"
struct GpuSelectedVirtualCluster {
    mesh_id: u32,
    instance_id: u32,
    cluster_index: u32,
    physical_slot: u32,
    lod_level: u32,
    triangle_count: u32,
    material_index: u32,
    flags: u32,
};
struct SelectedTable { records: array<GpuSelectedVirtualCluster>, };
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
};
struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};
struct DrawTable { records: array<DrawIndirect>, };
struct DrawEmissionState {
    draw_count: u32,
    batch_fallback: u32,
    selector_selected_count: u32,
    selector_selected_overflow: u32,
    selector_invalid_or_missing: u32,
    emitted_triangles: atomic<u32>,
    emitted_draws: atomic<u32>,
    reserved_0: u32,
    reserved_1: u32,
    reserved_2: u32,
    reserved_3: u32,
    reserved_4: u32,
};
struct DispatchIndirect {
    workgroups_x: u32,
    workgroups_y: u32,
    workgroups_z: u32,
};

@group(0) @binding(0) var<storage, read> selected: SelectedTable;
@group(0) @binding(1) var<storage, read> traversal: TraversalCounters;
@group(0) @binding(2) var<storage, read> commands: DrawTable;
@group(0) @binding(3) var<storage, read_write> state: DrawEmissionState;
@group(0) @binding(4) var<storage, read_write> dispatch_args: DispatchIndirect;

@compute @workgroup_size(1)
fn prepare_virtual_draws() {
    let selected_count = atomicLoad(&traversal.selected_count);
    let selected_overflow = atomicLoad(&traversal.selected_overflow);
    let invalid_records = atomicLoad(&traversal.invalid_records);
    let missing_current_pages = atomicLoad(&traversal.missing_current_pages);
    let invalid_or_missing = invalid_records
        + min(missing_current_pages, 0xffffffffu - invalid_records);
    let capacity = min(arrayLength(&selected.records), arrayLength(&commands.records));
    let bounded_count = min(selected_count, capacity);
    let fallback = selected_overflow != 0u
        || invalid_or_missing != 0u
        || selected_count > capacity;
    let draw_count = select(bounded_count, 0u, fallback);
    dispatch_args.workgroups_x = draw_count / 64u + select(0u, 1u, draw_count % 64u != 0u);
    dispatch_args.workgroups_y = 1u;
    dispatch_args.workgroups_z = 1u;
    state.draw_count = draw_count;
    state.batch_fallback = select(0u, 1u, fallback);
    state.selector_selected_count = selected_count;
    state.selector_selected_overflow = selected_overflow;
    state.selector_invalid_or_missing = invalid_or_missing;
}

"#;

const DRAW_EMIT_SHADER: &str = r#"
struct GpuSelectedVirtualCluster {
    mesh_id: u32,
    instance_id: u32,
    cluster_index: u32,
    physical_slot: u32,
    lod_level: u32,
    triangle_count: u32,
    material_index: u32,
    flags: u32,
};
struct SelectedTable { records: array<GpuSelectedVirtualCluster>, };
struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};
struct DrawTable { records: array<DrawIndirect>, };
struct DrawEmissionState {
    draw_count: u32,
    batch_fallback: u32,
    selector_selected_count: u32,
    selector_selected_overflow: u32,
    selector_invalid_or_missing: u32,
    emitted_triangles: atomic<u32>,
    emitted_draws: atomic<u32>,
    reserved_0: u32,
    reserved_1: u32,
    reserved_2: u32,
    reserved_3: u32,
    reserved_4: u32,
};

@group(0) @binding(0) var<storage, read> selected: SelectedTable;
@group(0) @binding(1) var<storage, read_write> commands: DrawTable;
@group(0) @binding(2) var<storage, read_write> state: DrawEmissionState;

@compute @workgroup_size(64)
fn emit_virtual_draws(@builtin(global_invocation_id) gid: vec3<u32>) {
    let draw_index = gid.x;
    if (draw_index >= state.draw_count) {
        return;
    }
    let record = selected.records[draw_index];
    commands.records[draw_index] = DrawIndirect(
        record.triangle_count * 3u,
        1u,
        0u,
        draw_index
    );
    atomicAdd(&state.emitted_triangles, record.triangle_count);
    atomicAdd(&state.emitted_draws, 1u);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_emission_shader_parses_and_keeps_whole_batch_fallback() {
        wgpu::naga::front::wgsl::parse_str(DRAW_PREPARE_SHADER)
            .unwrap_or_else(|error| panic!("virtual draw prepare WGSL failed: {error:?}"));
        wgpu::naga::front::wgsl::parse_str(DRAW_EMIT_SHADER)
            .unwrap_or_else(|error| panic!("virtual draw emit WGSL failed: {error:?}"));
        assert!(DRAW_PREPARE_SHADER.contains("state.draw_count = draw_count"));
        assert!(DRAW_PREPARE_SHADER.contains("select(bounded_count, 0u, fallback)"));
    }
}
