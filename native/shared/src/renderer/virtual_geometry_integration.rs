use super::{gpu_driven, specialized_scene_shader_source, Renderer};
use crate::virtual_geometry::{
    GpuVirtualGeometryPool, GpuVirtualHierarchySelector, GpuVirtualVisibilityRaster,
    GpuVirtualVisibilityShading, VirtualGeometryVisibilityError,
};

impl Renderer {
    /// Build the explicit, unattached virtual-geometry PBR consumer against
    /// this renderer's exact lighting/material ABI. Ordinary frames do not
    /// call this and therefore allocate no virtual shading resources.
    pub fn create_virtual_visibility_shading(
        &self,
        pool: &GpuVirtualGeometryPool,
        selector: &GpuVirtualHierarchySelector,
        raster: &GpuVirtualVisibilityRaster,
        visibility: &wgpu::Texture,
    ) -> Result<GpuVirtualVisibilityShading, VirtualGeometryVisibilityError> {
        let Some(global_materials) = self.material_system.indirection.global_layout.as_ref() else {
            return Err(VirtualGeometryVisibilityError::PbrDeviceUnsupported);
        };
        let specialized = specialized_scene_shader_source(
            self.froxel.is_some(),
            self.shadow_map.virtual_map.requested(),
        );
        let gpu_source = gpu_driven::make_gpu_scene_shader(&specialized);
        GpuVirtualVisibilityShading::new(
            &self.device,
            pool,
            selector,
            raster,
            visibility,
            crate::virtual_geometry::shading::VirtualVisibilityPbrLayouts {
                draw: self.gpu_driven.draw_layout(),
                lighting: &self.lighting_layout,
                global_materials,
                joints: &self.joint_layout,
            },
            &gpu_source,
        )
    }

    /// Bind the renderer-owned scene globals and record one disjoint virtual
    /// fullscreen PBR pass into caller-provided four-MRT attachments.
    pub fn draw_virtual_visibility_shading<'a>(
        &'a self,
        shading: &'a GpuVirtualVisibilityShading,
        pass: &mut wgpu::RenderPass<'a>,
        selector: &GpuVirtualHierarchySelector,
    ) -> Result<(), VirtualGeometryVisibilityError> {
        let Some(global_materials) = self.material_system.indirection.global_bind_group.as_ref()
        else {
            return Err(VirtualGeometryVisibilityError::PbrDeviceUnsupported);
        };
        shading.draw(
            pass,
            selector,
            self.gpu_driven.draw_bind_group(),
            &self.lighting_bind_group,
            global_materials,
            &self.joint_bind_group,
        )
    }
}
