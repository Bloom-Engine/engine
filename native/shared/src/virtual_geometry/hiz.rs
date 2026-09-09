use super::GpuVirtualInstance;

pub(crate) const VIRTUAL_HIZ_BASE_SIZE: u32 = 256;
pub(crate) const VIRTUAL_HIZ_MIP_COUNT: u32 = 9;
const VIRTUAL_HIZ_RELATIVE_DEPTH_BIAS: f32 = 0.02;
const VIRTUAL_HIZ_ABSOLUTE_DEPTH_BIAS: f32 = 0.1;

#[derive(Clone, Copy, Debug)]
pub(crate) struct VirtualGeometryHiZFrame {
    pub frame_index: u64,
    pub view_projection: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub render_extent: (u32, u32),
    pub camera_cut: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuVirtualHiZTelemetry {
    pub texture_bytes: u64,
    pub captures_recorded: u64,
    pub captures_submitted: u64,
    pub history_valid: bool,
    pub history_frame: u64,
    pub history_instances: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SelectionParams {
    previous_view_projection: [[f32; 4]; 4],
    previous_view: [[f32; 4]; 4],
    current_view_projection: [[f32; 4]; 4],
    current_view: [[f32; 4]; 4],
    extent: [u32; 4],
    thresholds: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<SelectionParams>() == 288);
pub(crate) const VIRTUAL_HIZ_SELECTION_PARAMS_BYTES: u64 =
    std::mem::size_of::<SelectionParams>() as u64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    source_size: [u32; 2],
    output_size: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DownsampleParams {
    source_size: [u32; 2],
    output_size: [u32; 2],
}

#[derive(Clone)]
struct History {
    frame_index: u64,
    view_projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    render_extent: (u32, u32),
    instances: Vec<[u32; 3]>,
}

pub(crate) struct GpuVirtualHiZ {
    _textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
    sample_layout: wgpu::BindGroupLayout,
    sample_bind_group: wgpu::BindGroup,
    selection_params: wgpu::Buffer,
    reduce_layout: wgpu::BindGroupLayout,
    reduce_pipeline: wgpu::ComputePipeline,
    reduce_params: wgpu::Buffer,
    reduce_bind_group: Option<wgpu::BindGroup>,
    downsample_pipeline: wgpu::ComputePipeline,
    downsample_params: Vec<wgpu::Buffer>,
    downsample_bind_groups: Vec<wgpu::BindGroup>,
    history: Option<History>,
    pending: Option<History>,
    captures_recorded: u64,
    captures_submitted: u64,
}

impl GpuVirtualHiZ {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let mut textures = Vec::with_capacity(VIRTUAL_HIZ_MIP_COUNT as usize);
        let mut views = Vec::with_capacity(VIRTUAL_HIZ_MIP_COUNT as usize);
        for mip in 0..VIRTUAL_HIZ_MIP_COUNT {
            let size = VIRTUAL_HIZ_BASE_SIZE >> mip;
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("virtual_geometry_previous_hiz_mip"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            views.push(texture.create_view(&wgpu::TextureViewDescriptor::default()));
            textures.push(texture);
        }

        let selection_params = uniform_buffer(
            device,
            "virtual_geometry_hiz_selection_params",
            std::mem::size_of::<SelectionParams>() as u64,
        );
        let sample_layout = sample_layout(device);
        let mut sample_entries = Vec::with_capacity(VIRTUAL_HIZ_MIP_COUNT as usize + 1);
        sample_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: selection_params.as_entire_binding(),
        });
        sample_entries.extend(
            views
                .iter()
                .enumerate()
                .map(|(index, view)| wgpu::BindGroupEntry {
                    binding: index as u32 + 1,
                    resource: wgpu::BindingResource::TextureView(view),
                }),
        );
        let sample_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("virtual_geometry_hiz_sample_bind_group"),
            layout: &sample_layout,
            entries: &sample_entries,
        });

        let reduce_layout = reduce_layout(device);
        let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("virtual_geometry_hiz_reduce_shader"),
            source: wgpu::ShaderSource::Wgsl(REDUCE_SHADER.into()),
        });
        let reduce_pipeline = compute_pipeline(
            device,
            "virtual_geometry_hiz_reduce_pipeline",
            &reduce_layout,
            &reduce_shader,
        );
        let reduce_params = uniform_buffer(
            device,
            "virtual_geometry_hiz_reduce_params",
            std::mem::size_of::<ReduceParams>() as u64,
        );

        let downsample_layout = downsample_layout(device);
        let downsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("virtual_geometry_hiz_downsample_shader"),
            source: wgpu::ShaderSource::Wgsl(DOWNSAMPLE_SHADER.into()),
        });
        let downsample_pipeline = compute_pipeline(
            device,
            "virtual_geometry_hiz_downsample_pipeline",
            &downsample_layout,
            &downsample_shader,
        );
        let mut downsample_params = Vec::with_capacity(VIRTUAL_HIZ_MIP_COUNT as usize - 1);
        let mut downsample_bind_groups = Vec::with_capacity(VIRTUAL_HIZ_MIP_COUNT as usize - 1);
        for mip in 0..VIRTUAL_HIZ_MIP_COUNT - 1 {
            let params = uniform_buffer(
                device,
                "virtual_geometry_hiz_downsample_params",
                std::mem::size_of::<DownsampleParams>() as u64,
            );
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("virtual_geometry_hiz_downsample_bind_group"),
                layout: &downsample_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&views[mip as usize]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&views[mip as usize + 1]),
                    },
                ],
            });
            downsample_params.push(params);
            downsample_bind_groups.push(bind_group);
        }

        Self {
            _textures: textures,
            views,
            sample_layout,
            sample_bind_group,
            selection_params,
            reduce_layout,
            reduce_pipeline,
            reduce_params,
            reduce_bind_group: None,
            downsample_pipeline,
            downsample_params,
            downsample_bind_groups,
            history: None,
            pending: None,
            captures_recorded: 0,
            captures_submitted: 0,
        }
    }

    pub(crate) fn sample_layout(&self) -> &wgpu::BindGroupLayout {
        &self.sample_layout
    }

    pub(crate) fn sample_bind_group(&self) -> &wgpu::BindGroup {
        &self.sample_bind_group
    }

    pub(crate) fn history_valid_for(&self, frame: VirtualGeometryHiZFrame) -> bool {
        !frame.camera_cut
            && self.history.as_ref().is_some_and(|history| {
                history.frame_index.wrapping_add(1) == frame.frame_index
                    && history.render_extent == frame.render_extent
            })
    }

    pub(crate) fn instance_was_captured(&self, instance: GpuVirtualInstance) -> bool {
        self.history.as_ref().is_some_and(|history| {
            history
                .instances
                .binary_search(&instance.history_identity())
                .is_ok()
        })
    }

    pub(crate) fn prepare_selection(
        &self,
        queue: &wgpu::Queue,
        frame: VirtualGeometryHiZFrame,
        enabled: bool,
    ) {
        let history = enabled.then(|| self.history.as_ref()).flatten();
        let previous_view_projection = history
            .map(|history| history.view_projection)
            .unwrap_or(frame.view_projection);
        let previous_view = history.map(|history| history.view).unwrap_or(frame.view);
        let width = (frame.render_extent.0 / 2)
            .max(1)
            .min(VIRTUAL_HIZ_BASE_SIZE);
        let height = (frame.render_extent.1 / 2)
            .max(1)
            .min(VIRTUAL_HIZ_BASE_SIZE);
        queue.write_buffer(
            &self.selection_params,
            0,
            bytemuck::bytes_of(&SelectionParams {
                previous_view_projection,
                previous_view,
                current_view_projection: frame.view_projection,
                current_view: frame.view,
                extent: [width, height, VIRTUAL_HIZ_MIP_COUNT, u32::from(enabled)],
                // One base-grid texel of accepted screen motion, then a
                // two-texel query expansion plus relative/absolute depth bias.
                thresholds: [
                    1.0 / width as f32,
                    1.0 / height as f32,
                    VIRTUAL_HIZ_RELATIVE_DEPTH_BIAS,
                    VIRTUAL_HIZ_ABSOLUTE_DEPTH_BIAS,
                ],
            }),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_capture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        source_size: (u32, u32),
        frame: VirtualGeometryHiZFrame,
        instances: &[GpuVirtualInstance],
    ) {
        let output_size = (
            source_size.0.min(VIRTUAL_HIZ_BASE_SIZE).max(1),
            source_size.1.min(VIRTUAL_HIZ_BASE_SIZE).max(1),
        );
        queue.write_buffer(
            &self.reduce_params,
            0,
            bytemuck::bytes_of(&ReduceParams {
                source_size: [source_size.0.max(1), source_size.1.max(1)],
                output_size: [output_size.0, output_size.1],
            }),
        );
        if self.reduce_bind_group.is_none() {
            self.reduce_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("virtual_geometry_hiz_reduce_bind_group"),
                layout: &self.reduce_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.reduce_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.views[0]),
                    },
                ],
            }));
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("virtual_geometry_hiz_reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reduce_pipeline);
            pass.set_bind_group(0, self.reduce_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(output_size.0.div_ceil(8), output_size.1.div_ceil(8), 1);
        }

        let mut source_extent = output_size;
        for mip in 0..VIRTUAL_HIZ_MIP_COUNT - 1 {
            let output_extent = (source_extent.0.div_ceil(2), source_extent.1.div_ceil(2));
            queue.write_buffer(
                &self.downsample_params[mip as usize],
                0,
                bytemuck::bytes_of(&DownsampleParams {
                    source_size: [source_extent.0, source_extent.1],
                    output_size: [output_extent.0, output_extent.1],
                }),
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("virtual_geometry_hiz_downsample"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.downsample_pipeline);
            pass.set_bind_group(0, &self.downsample_bind_groups[mip as usize], &[]);
            pass.dispatch_workgroups(output_extent.0.div_ceil(8), output_extent.1.div_ceil(8), 1);
            source_extent = output_extent;
        }

        let mut captured_instances = instances
            .iter()
            .map(|instance| instance.history_identity())
            .collect::<Vec<_>>();
        captured_instances.sort_unstable();
        captured_instances.dedup();
        self.pending = Some(History {
            frame_index: frame.frame_index,
            view_projection: frame.view_projection,
            view: frame.view,
            render_extent: frame.render_extent,
            instances: captured_instances,
        });
        self.captures_recorded = self.captures_recorded.saturating_add(1);
    }

    pub(crate) fn after_submit(&mut self) {
        if let Some(pending) = self.pending.take() {
            self.history = Some(pending);
            self.captures_submitted = self.captures_submitted.saturating_add(1);
        }
    }

    pub(crate) fn invalidate(&mut self, source_recreated: bool) {
        self.history = None;
        self.pending = None;
        if source_recreated {
            self.reduce_bind_group = None;
        }
    }

    pub(crate) fn telemetry(&self) -> GpuVirtualHiZTelemetry {
        let history = self.history.as_ref();
        GpuVirtualHiZTelemetry {
            texture_bytes: virtual_hiz_texture_bytes(),
            captures_recorded: self.captures_recorded,
            captures_submitted: self.captures_submitted,
            history_valid: history.is_some(),
            history_frame: history.map_or(0, |history| history.frame_index),
            history_instances: history.map_or(0, |history| history.instances.len() as u32),
        }
    }
}

fn virtual_hiz_texture_bytes() -> u64 {
    (0..VIRTUAL_HIZ_MIP_COUNT)
        .map(|mip| {
            let size = u64::from(VIRTUAL_HIZ_BASE_SIZE >> mip);
            size * size * 4
        })
        .sum()
}

fn uniform_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn sample_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(VIRTUAL_HIZ_MIP_COUNT as usize + 1);
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    entries.extend(
        (0..VIRTUAL_HIZ_MIP_COUNT).map(|mip| wgpu::BindGroupLayoutEntry {
            binding: mip + 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }),
    );
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("virtual_geometry_hiz_sample_layout"),
        entries: &entries,
    })
}

fn reduce_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("virtual_geometry_hiz_reduce_layout"),
        entries: &[
            uniform_layout_entry(0),
            sampled_layout_entry(1),
            storage_layout_entry(2),
        ],
    })
}

fn downsample_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("virtual_geometry_hiz_downsample_layout"),
        entries: &[
            uniform_layout_entry(0),
            sampled_layout_entry(1),
            storage_layout_entry(2),
        ],
    })
}

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn sampled_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::R32Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn compute_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

const REDUCE_SHADER: &str = r#"
struct Params { source_size: vec2<u32>, output_size: vec2<u32>, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= params.output_size)) { return; }
    let begin = gid.xy * params.source_size / params.output_size;
    let end = ((gid.xy + vec2<u32>(1u)) * params.source_size
        + params.output_size - vec2<u32>(1u)) / params.output_size;
    var maximum_depth = 0.0;
    for (var y = begin.y; y < end.y; y++) {
        for (var x = begin.x; x < end.x; x++) {
            maximum_depth = max(maximum_depth, textureLoad(source, vec2<i32>(i32(x), i32(y)), 0).r);
        }
    }
    textureStore(output, vec2<i32>(gid.xy), vec4<f32>(maximum_depth, 0.0, 0.0, 0.0));
}
"#;

const DOWNSAMPLE_SHADER: &str = r#"
struct Params { source_size: vec2<u32>, output_size: vec2<u32>, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= params.output_size)) { return; }
    let begin = gid.xy * 2u;
    let maximum_source = vec2<i32>(params.source_size - vec2<u32>(1u));
    var maximum_depth = 0.0;
    for (var y = 0u; y < 2u; y++) {
        for (var x = 0u; x < 2u; x++) {
            let coordinate = min(vec2<i32>(begin + vec2<u32>(x, y)), maximum_source);
            maximum_depth = max(maximum_depth, textureLoad(source, coordinate, 0).r);
        }
    }
    textureStore(output, vec2<i32>(gid.xy), vec4<f32>(maximum_depth, 0.0, 0.0, 0.0));
}
"#;

#[cfg(test)]
mod shader_tests {
    use super::*;

    #[test]
    fn virtual_hiz_shaders_parse_and_texture_budget_is_fixed() {
        wgpu::naga::front::wgsl::parse_str(REDUCE_SHADER).unwrap();
        wgpu::naga::front::wgsl::parse_str(DOWNSAMPLE_SHADER).unwrap();
        assert_eq!(VIRTUAL_HIZ_BASE_SIZE, 256);
        assert_eq!(VIRTUAL_HIZ_MIP_COUNT, 9);
        assert_eq!(VIRTUAL_HIZ_RELATIVE_DEPTH_BIAS, 0.02);
        assert_eq!(VIRTUAL_HIZ_ABSOLUTE_DEPTH_BIAS, 0.1);
        assert_eq!(virtual_hiz_texture_bytes(), 349_524);
    }
}
