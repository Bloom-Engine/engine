//! The render pass and atlas copies of one frame's mesh-card captures.

use super::{Renderer, CARD_SLOTS_PER_ROW, CARD_SLOT_SIZE};

/// One axis of one mesh: what to draw, the scratch-grid cell it draws into
/// and the atlas slot that cell is copied to.
pub(super) struct CardCaptureDraw {
    pub uniform_offset: u32,
    pub texture_bg: wgpu::BindGroup,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub cell_x: u32,
    pub cell_y: u32,
    pub slot: u32,
}

impl Renderer {
    pub(super) fn record_card_capture_draws(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        uniform_bg: &wgpu::BindGroup,
        draws: &[CardCaptureDraw],
    ) {
        if draws.is_empty() {
            return;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("card_capture_pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.card_capture_albedo_scratch_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.card_capture_emissive_scratch_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.card_capture_depth_scratch_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.card_capture_pipeline);
            for draw in draws {
                // The viewport maps the capture's clip space onto its cell;
                // the scissor keeps geometry outside the card bounds from
                // spilling into a neighbouring cell.
                pass.set_viewport(
                    draw.cell_x as f32,
                    draw.cell_y as f32,
                    CARD_SLOT_SIZE as f32,
                    CARD_SLOT_SIZE as f32,
                    0.0,
                    1.0,
                );
                pass.set_scissor_rect(draw.cell_x, draw.cell_y, CARD_SLOT_SIZE, CARD_SLOT_SIZE);
                pass.set_bind_group(0, uniform_bg, &[draw.uniform_offset]);
                pass.set_bind_group(1, &draw.texture_bg, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                pass.set_index_buffer(draw.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..draw.index_count, 0, 0..1);
            }
        }

        let copy_extent = wgpu::Extent3d {
            width: CARD_SLOT_SIZE,
            height: CARD_SLOT_SIZE,
            depth_or_array_layers: 1,
        };
        for draw in draws {
            let cell_origin = wgpu::Origin3d {
                x: draw.cell_x,
                y: draw.cell_y,
                z: 0,
            };
            let atlas_origin = wgpu::Origin3d {
                x: (draw.slot % CARD_SLOTS_PER_ROW) * CARD_SLOT_SIZE,
                y: (draw.slot / CARD_SLOTS_PER_ROW) * CARD_SLOT_SIZE,
                z: 0,
            };
            for (scratch, atlas) in [
                (&self.card_capture_albedo_scratch, &self.mesh_card_atlas),
                (
                    &self.card_capture_emissive_scratch,
                    &self.mesh_card_emissive_tex,
                ),
            ] {
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: scratch,
                        mip_level: 0,
                        origin: cell_origin,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: atlas,
                        mip_level: 0,
                        origin: atlas_origin,
                        aspect: wgpu::TextureAspect::All,
                    },
                    copy_extent,
                );
            }
        }
    }
}
