use super::{
    GpuVirtualGeometryPool, GpuVirtualHierarchySelector, GpuVirtualVisibilityRaster,
    VirtualGeometryVisibilityError,
};

#[derive(Copy, Clone)]
pub(crate) struct VirtualVisibilityPbrLayouts<'a> {
    pub draw: &'a wgpu::BindGroupLayout,
    pub lighting: &'a wgpu::BindGroupLayout,
    pub global_materials: &'a wgpu::BindGroupLayout,
    pub joints: &'a wgpu::BindGroupLayout,
}

/// Explicit, unattached full-PBR consumer for namespaced virtual visibility
/// IDs. It owns only its pipeline and group-4 bindings; the caller retains all
/// scene-global bind groups and MRT/depth attachments.
pub struct GpuVirtualVisibilityShading {
    selector_id: u64,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

impl GpuVirtualVisibilityShading {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &wgpu::Device,
        pool: &GpuVirtualGeometryPool,
        selector: &GpuVirtualHierarchySelector,
        raster: &GpuVirtualVisibilityRaster,
        visibility: &wgpu::Texture,
        layouts: VirtualVisibilityPbrLayouts<'_>,
        gpu_scene_source: &str,
    ) -> Result<Self, VirtualGeometryVisibilityError> {
        if selector.pool_id() != pool.id() {
            return Err(VirtualGeometryVisibilityError::PoolMismatch);
        }
        if raster.selector_id() != selector.id() {
            return Err(VirtualGeometryVisibilityError::SelectorMismatch);
        }
        let limits = device.limits();
        let required_color_attachments = if cfg!(lean_mrt) { 2 } else { 4 };
        if limits.max_bind_groups < 5
            || limits.max_storage_buffers_per_shader_stage < 8
            || limits.max_color_attachments < required_color_attachments
        {
            return Err(VirtualGeometryVisibilityError::PbrDeviceUnsupported);
        }
        if visibility.format() != crate::renderer::visibility_buffer::VISIBILITY_FORMAT
            || !visibility
                .usage()
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        {
            return Err(VirtualGeometryVisibilityError::InvalidVisibilityTarget);
        }

        let layout = create_layout(device);
        let visibility_view = visibility.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("virtual_geometry_visibility_pbr_bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&visibility_view),
                },
                binding(1, pool.physical_buffer()),
                binding(2, pool.cluster_table_buffer()),
                binding(3, selector.selected_buffer()),
                binding(4, selector.instance_buffer()),
                binding(5, raster.frame_buffer()),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("virtual_geometry_visibility_pbr_pipeline_layout"),
            bind_group_layouts: &[
                Some(layouts.draw),
                Some(layouts.lighting),
                Some(layouts.global_materials),
                Some(layouts.joints),
                Some(&layout),
            ],
            immediate_size: 0,
        });
        let source = crate::renderer::visibility_shading::make_virtual_shader(gpu_scene_source);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("virtual_geometry_visibility_pbr_shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = crate::renderer::visibility_shading::create_virtual_pipeline(
            device,
            &pipeline_layout,
            &shader,
        );
        Ok(Self {
            selector_id: selector.id(),
            pipeline,
            bind_group,
        })
    }

    pub(crate) fn draw<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        selector: &GpuVirtualHierarchySelector,
        draw: &'a wgpu::BindGroup,
        lighting: &'a wgpu::BindGroup,
        global_materials: &'a wgpu::BindGroup,
        joints: &'a wgpu::BindGroup,
    ) -> Result<(), VirtualGeometryVisibilityError> {
        if selector.id() != self.selector_id {
            return Err(VirtualGeometryVisibilityError::SelectorMismatch);
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, draw, &[]);
        pass.set_bind_group(1, lighting, &[]);
        pass.set_bind_group(2, global_materials, &[]);
        pass.set_bind_group(3, joints, &[]);
        pass.set_bind_group(4, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("virtual_geometry_visibility_pbr_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            storage(1),
            storage(2),
            storage(3),
            storage(4),
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}
