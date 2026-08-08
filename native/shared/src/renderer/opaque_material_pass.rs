//! Opaque user-material recording inside the compiled `hdr_scene` node.

use super::*;

impl Renderer {
    pub(super) fn record_opaque_material_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
    ) {
        if self.material_system.commands.is_empty() || self.dbg_skip("material_pass") {
            return;
        }
        let camera_planes = crate::scene::extract_frustum_planes(&mat4_multiply(
            self.current_proj_matrix_unjittered,
            self.current_view_matrix,
        ));
        profiler.begin("material_pass");
        {
            let timestamp_writes = profiler.pass_timestamp_writes("material_pass");
            #[cfg(lean_mrt)]
            let attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>] = &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                None,
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.velocity_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                None,
            ];
            #[cfg(not(lean_mrt))]
            let attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>] = &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.material_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.velocity_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.albedo_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_material_pass"),
                color_attachments: attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.material_system
                .dispatch(&mut pass, Some(&camera_planes), |handle, mesh_index| {
                    let mesh = self
                        .model_gpu_cache
                        .get(&handle)?
                        .as_ref()?
                        .get(mesh_index)?;
                    Some((
                        self.gpu_driven.mesh_draw(&mesh.geometry, mesh.index_count),
                        mesh.local_min,
                        mesh.local_max,
                    ))
                });
        }
        profiler.end("material_pass");
    }
}
