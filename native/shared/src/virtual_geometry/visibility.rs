use super::{
    draw_emission::BINNED_FALLBACK_DRAW_COUNT, GpuVirtualDrawEmitter, GpuVirtualGeometryPool,
    GpuVirtualHierarchySelector, VirtualGeometrySubmissionMode,
};
use std::fmt;

const VIRTUAL_VISIBILITY_RASTER_WGSL: &str =
    include_str!("../../shaders/virtual_geometry/visibility_raster.wgsl");
const VIRTUAL_DECODE_WGSL: &str = include_str!("../../shaders/virtual_geometry/decode.wgsl");
const VIRTUAL_RENDER_ABI_WGSL: &str =
    include_str!("../../shaders/virtual_geometry/render_abi.wgsl");
const VIRTUAL_VISIBILITY_RASTER_BINNED_WGSL: &str =
    include_str!("../../shaders/virtual_geometry/visibility_raster_binned.wgsl");

/// Frame transforms shared by virtual visibility raster and its future exact
/// PBR reconstruction consumer (128 bytes, column-major).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVirtualVisibilityFrame {
    view_projection: [[f32; 4]; 4],
    previous_view_projection: [[f32; 4]; 4],
}

impl GpuVirtualVisibilityFrame {
    pub fn new(
        view_projection: [[f32; 4]; 4],
        previous_view_projection: [[f32; 4]; 4],
    ) -> Result<Self, VirtualGeometryVisibilityError> {
        if !view_projection
            .iter()
            .flatten()
            .chain(previous_view_projection.iter().flatten())
            .all(|value| value.is_finite())
        {
            return Err(VirtualGeometryVisibilityError::InvalidFrame);
        }
        Ok(Self {
            view_projection,
            previous_view_projection,
        })
    }

    pub const fn view_projection(self) -> [[f32; 4]; 4] {
        self.view_projection
    }

    pub const fn previous_view_projection(self) -> [[f32; 4]; 4] {
        self.previous_view_projection
    }
}

const _: () = assert!(std::mem::size_of::<GpuVirtualVisibilityFrame>() == 128);

/// Raw-page raster stage for the opt-in shared `Rg32Uint` visibility target.
/// It owns no geometry and can only draw commands emitted by its selector.
pub struct GpuVirtualVisibilityRaster {
    selector_id: u64,
    draw_capacity: u32,
    count_supported: bool,
    frame_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl GpuVirtualVisibilityRaster {
    pub fn new(
        device: &wgpu::Device,
        pool: &GpuVirtualGeometryPool,
        selector: &GpuVirtualHierarchySelector,
        emitter: &GpuVirtualDrawEmitter,
    ) -> Result<Self, VirtualGeometryVisibilityError> {
        if selector.pool_id() != pool.id() {
            return Err(VirtualGeometryVisibilityError::PoolMismatch);
        }
        if emitter.selector_id() != selector.id() {
            return Err(VirtualGeometryVisibilityError::SelectorMismatch);
        }
        let features = device.features();
        let limits = device.limits();
        let binned_fallback =
            emitter.submission_mode() == VirtualGeometrySubmissionMode::BinnedFallback;
        let required_storage_buffers = if binned_fallback { 5 } else { 4 };
        if !features
            .contains(wgpu::Features::PRIMITIVE_INDEX | wgpu::Features::INDIRECT_FIRST_INSTANCE)
            || limits.max_storage_buffers_per_shader_stage < required_storage_buffers
        {
            return Err(VirtualGeometryVisibilityError::DeviceUnsupported);
        }
        if emitter.draw_capacity() >= crate::renderer::visibility_buffer::DRAW_INDEX_MASK {
            return Err(VirtualGeometryVisibilityError::NamespaceCapacityExceeded {
                capacity: emitter.draw_capacity(),
            });
        }

        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("virtual_geometry_visibility_frame"),
            size: std::mem::size_of::<GpuVirtualVisibilityFrame>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = create_layout(device, binned_fallback);
        let mut entries = vec![
            binding(0, pool.physical_buffer()),
            binding(1, pool.cluster_table_buffer()),
            binding(2, selector.selected_buffer()),
            binding(3, selector.instance_buffer()),
            binding(4, &frame_buffer),
        ];
        if binned_fallback {
            let (indices, _) = emitter
                .binned_buffers()
                .expect("binned emitter owns its selection indirection");
            entries.push(binding(5, indices));
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("virtual_geometry_visibility_bind_group"),
            layout: &layout,
            entries: &entries,
        });
        let mut source = format!(
            "enable primitive_index;\n{}\n{}\n{}\n{}",
            crate::renderer::visibility_buffer::RECONSTRUCTION_WGSL,
            VIRTUAL_RENDER_ABI_WGSL,
            VIRTUAL_DECODE_WGSL,
            VIRTUAL_VISIBILITY_RASTER_WGSL,
        );
        if binned_fallback {
            source.push('\n');
            source.push_str(VIRTUAL_VISIBILITY_RASTER_BINNED_WGSL);
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("virtual_geometry_visibility_shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("virtual_geometry_visibility_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("virtual_geometry_visibility_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(if binned_fallback {
                    "vs_virtual_visibility_binned"
                } else {
                    "vs_virtual_visibility"
                }),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_virtual_visibility"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::renderer::visibility_buffer::VISIBILITY_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::renderer::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Ok(Self {
            selector_id: selector.id(),
            draw_capacity: emitter.draw_capacity(),
            count_supported: features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT),
            frame_buffer,
            bind_group,
            pipeline,
        })
    }

    pub fn prepare_frame(
        &self,
        queue: &wgpu::Queue,
        frame: GpuVirtualVisibilityFrame,
    ) -> Result<(), VirtualGeometryVisibilityError> {
        let validated =
            GpuVirtualVisibilityFrame::new(frame.view_projection, frame.previous_view_projection)?;
        queue.write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&validated));
        Ok(())
    }

    pub const fn counted_submission_supported(&self) -> bool {
        self.count_supported
    }

    pub fn draw<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        emitter: &'a GpuVirtualDrawEmitter,
    ) -> Result<(), VirtualGeometryVisibilityError> {
        self.validate_emitter(emitter)?;
        self.bind(pass);
        match emitter.submission_mode() {
            VirtualGeometrySubmissionMode::Counted => {
                if !self.count_supported {
                    return Err(VirtualGeometryVisibilityError::IndirectCountUnsupported);
                }
                pass.multi_draw_indirect_count(
                    emitter.command_buffer(),
                    0,
                    emitter.state_buffer(),
                    0,
                    self.draw_capacity,
                );
            }
            VirtualGeometrySubmissionMode::BinnedFallback => {
                let (_, commands) = emitter
                    .binned_buffers()
                    .expect("binned emitter owns fixed indirect commands");
                pass.multi_draw_indirect(commands, 0, BINNED_FALLBACK_DRAW_COUNT);
            }
        }
        Ok(())
    }

    fn validate_emitter(
        &self,
        emitter: &GpuVirtualDrawEmitter,
    ) -> Result<(), VirtualGeometryVisibilityError> {
        if emitter.selector_id() != self.selector_id
            || emitter.draw_capacity() != self.draw_capacity
        {
            return Err(VirtualGeometryVisibilityError::SelectorMismatch);
        }
        Ok(())
    }

    fn bind<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
    }

    pub(super) const fn selector_id(&self) -> u64 {
        self.selector_id
    }

    pub(super) fn frame_buffer(&self) -> &wgpu::Buffer {
        &self.frame_buffer
    }
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_layout(device: &wgpu::Device, binned_fallback: bool) -> wgpu::BindGroupLayout {
    let storage = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let mut entries = vec![
        storage(0),
        storage(1),
        storage(2),
        storage(3),
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ];
    if binned_fallback {
        entries.push(storage(5));
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("virtual_geometry_visibility_layout"),
        entries: &entries,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VirtualGeometryVisibilityError {
    InvalidFrame,
    DeviceUnsupported,
    PoolMismatch,
    SelectorMismatch,
    NamespaceCapacityExceeded { capacity: u32 },
    IndirectCountUnsupported,
    PbrDeviceUnsupported,
    InvalidVisibilityTarget,
}

impl fmt::Display for VirtualGeometryVisibilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFrame => write!(formatter, "invalid virtual visibility frame transforms"),
            Self::DeviceUnsupported => write!(
                formatter,
                "device lacks virtual visibility vertex-pulling or primitive-index support"
            ),
            Self::PoolMismatch => {
                write!(formatter, "virtual visibility pool does not match selector")
            }
            Self::SelectorMismatch => {
                write!(formatter, "virtual visibility selector/emitter mismatch")
            }
            Self::NamespaceCapacityExceeded { capacity } => write!(
                formatter,
                "virtual draw capacity {capacity} exceeds the packed visibility namespace"
            ),
            Self::IndirectCountUnsupported => write!(
                formatter,
                "counted virtual submission requires indirect-count device support"
            ),
            Self::PbrDeviceUnsupported => write!(
                formatter,
                "device limits cannot run virtual visibility PBR composition"
            ),
            Self::InvalidVisibilityTarget => write!(
                formatter,
                "virtual PBR requires an Rg32Uint texture-binding visibility target"
            ),
        }
    }
}

impl std::error::Error for VirtualGeometryVisibilityError {}

#[cfg(test)]
mod shader_tests {
    use super::*;

    #[test]
    fn raw_virtual_visibility_shader_parses_and_uses_the_shared_namespace() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(std::mem::size_of::<GpuVirtualVisibilityFrame>(), 128);
        let mut invalid = identity;
        invalid[2][1] = f32::NAN;
        assert_eq!(
            GpuVirtualVisibilityFrame::new(identity, invalid),
            Err(VirtualGeometryVisibilityError::InvalidFrame)
        );
        let source = format!(
            "enable primitive_index;\n{}\n{}\n{}\n{}",
            crate::renderer::visibility_buffer::RECONSTRUCTION_WGSL,
            VIRTUAL_RENDER_ABI_WGSL,
            VIRTUAL_DECODE_WGSL,
            VIRTUAL_VISIBILITY_RASTER_WGSL,
        );
        wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("virtual visibility WGSL failed: {error:?}"));
        assert!(source.contains("bloom_encode_virtual_visibility"));
        assert!(source.contains("previous_model: mat4x4<f32>"));
        assert!(source.contains("BLOOM_VIRTUAL_FLAG_ALPHA_MASKED) != 0u"));
        assert!(!source.contains("BLOOM_VIRTUAL_FLAG_DOUBLE_SIDED) == 0u && !front_facing"));
    }
}
