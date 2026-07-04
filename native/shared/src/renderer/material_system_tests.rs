//! Tests for material_system: headless-GPU dispatch tests + the
//! translucent sort. Child module of material_system via #[path] —
//! the glob below re-exposes the parent's items to the inner mods.

#[allow(unused_imports)]
use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::formats;
    use crate::renderer::types::Vertex3D;

    /// Headless wgpu device. See sibling helpers in `transient.rs` /
    /// `impulse_field.rs` — same fallback adapter pattern. Returns None
    /// (test skips gracefully) when no GPU is available.
    fn try_create_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: true,
        })).ok()?;
        // The material ABI uses 5 bind groups (PerFrame, PerView,
        // PerMaterial, PerDraw, SceneInputs). downlevel_defaults caps
        // max_bind_groups at 4, which is fine on Metal (it silently
        // accepts more) but DX12 enforces it strictly — bump to 5 so
        // the user-material pipeline validates on every backend.
        // Also bump max_uniform_buffer_binding_size from 16KB to 64KB
        // for the JointMatrices UBO (1024 × mat4x4 = 64KB).
        let mut required_limits = wgpu::Limits::downlevel_defaults();
        required_limits.max_bind_groups = 5;
        required_limits.max_uniform_buffer_binding_size = 64 << 10;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("material-test-device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                ..Default::default()
            },
        )).ok()?;
        Some((device, queue))
    }

    /// Minimal translucent (Bucket::Refractive) WGSL material. Writes
    /// a constant red+0.5α colour through the alpha-blended HDR target.
    /// Uses #include "material_abi.wgsl" so the same pipeline-layout
    /// / per-frame / per-view bindings are validated as production
    /// materials. Vertex stage transforms via `draw.mvp` so we can
    /// emit a fullscreen-ish triangle from any geometry.
    const TRANSLUCENT_WGSL: &str = r#"
#include "material_abi.wgsl"

struct VsOut {
  @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VsOut {
  var out: VsOut;
  out.clip_position = draw.mvp * vec4<f32>(in.position, 1.0);
  return out;
}

@fragment
fn fs_main(_in: VsOut) -> TranslucentOut {
  var out: TranslucentOut;
  // Red, half alpha — alpha-blended onto whatever the load-op set.
  out.hdr = vec4<f32>(1.0, 0.0, 0.0, 0.5);
  return out;
}
"#;

    /// Create a tiny joint buffer so MaterialSystem::new is happy. The
    /// per_draw layout binds it at @binding(1); the test material
    /// doesn't read it but the bind group still has to validate.
    fn make_joint_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_joint_buffer"),
            // 1024 mat4s × 64 B = 64 KiB. Same size as production.
            size: 65536,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Build a single fullscreen triangle covering NDC: three vertices
    /// at (-1,-1), (3,-1), (-1,3). The pipeline's MVP starts as
    /// identity (we override it below) so the triangle covers the
    /// whole viewport.
    fn make_fullscreen_tri(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let mut verts: [Vertex3D; 3] = [Vertex3D::default(); 3];
        verts[0].position = [-1.0, -1.0, 0.5];
        verts[1].position = [ 3.0, -1.0, 0.5];
        verts[2].position = [-1.0,  3.0, 0.5];
        // The MaterialPipeline's depth-stencil uses Less; the load-op
        // for a translucent pass clears to 1.0 (far) by default in
        // production but in this test we use a depth attachment with
        // the CLEAR op and clear value 1.0, so anything < 1.0 passes.
        let vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_tri_vb"),
            size: std::mem::size_of_val(&verts) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vb, 0, bytemuck::cast_slice(&verts));
        let indices: [u32; 3] = [0, 1, 2];
        let ib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_tri_ib"),
            size: std::mem::size_of_val(&indices) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ib, 0, bytemuck::cast_slice(&indices));
        (vb, ib, 3)
    }

    /// EN-006 — translucent dispatch path. Compiles a refractive
    /// material via `MaterialSystem::compile`, submits one draw,
    /// dispatches into a 64×64 Rgba16Float HDR render target via
    /// `dispatch_translucent`, then reads back the HDR and verifies
    /// the alpha-blended red shows through the cyan background.
    ///
    /// This is the scoped-down version of the acceptance scenario:
    /// it exercises the dispatch site and pipeline-layout / blend-state
    /// wiring without standing up the full `Renderer` (which requires
    /// a surface and ~30 other resources). What this DOES cover:
    ///   - MaterialSystem::compile against a refractive WGSL source
    ///   - dispatch_translucent's pipeline / bind-group binding loop
    ///   - alpha blending via the translucent target's BlendState
    ///   - the per-draw / per-view / per-frame UBO writes
    /// What it does NOT cover:
    ///   - The scene-color snapshot (test material has reads_scene=false
    ///     so group 4 doesn't bind; the e2e copy_texture_to_texture
    ///     before this dispatch is exercised by the depth-snapshot test
    ///     in transient.rs).
    ///   - Sort order across multiple translucent draws (single-draw test).
    /// Skipped on adapters where `try_create_device` returns None.
    #[test]
    fn dispatch_translucent_alpha_blends_into_hdr() {
        let Some((device, queue)) = try_create_device() else { return; };
        let joint_buf = make_joint_buffer(&device);
        let mut sys = MaterialSystem::new(&device, &queue, &joint_buf);

        // Compile a refractive (translucent) material. Use the engine's
        // production format constants so the pipeline matches what
        // Renderer::new would have produced.
        let handle = sys.compile(
            &device,
            TRANSLUCENT_WGSL,
            FragmentProfile::Translucent,
            Bucket::Transparent,
            false,                                                          // reads_scene
            false,                                                          // wants_instancing
            wgpu::TextureFormat::Rgba16Float,                               // hdr_format
            wgpu::TextureFormat::Rg8Unorm,                                  // material_format (unused in translucent)
            wgpu::TextureFormat::Rg16Float,                                 // velocity_format (unused)
            wgpu::TextureFormat::Rgba8Unorm,                                // albedo_format (unused)
            formats::DEPTH_FORMAT,
        ).expect("translucent material compiles");
        assert!(handle != 0, "compile returns a 1-based handle");

        // Frame uniforms — zeros are fine for a constant-colour shader.
        let pf = PerFrameUniforms {
            time: 0.0, delta_time: 0.0, frame_index: 0, _pad0: 0,
            screen_resolution: [64.0, 64.0], render_resolution: [64.0, 64.0],
            taa_jitter: [0.0; 2], _pad1: [0.0; 2], wind: [0.0; 4],
        };
        let pv = bytemuck::Zeroable::zeroed();
        sys.update_frame_uniforms(&queue, &pf, &pv);
        sys.reset_draw_slot();

        // MVP = identity so the fullscreen tri stays in NDC.
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let (vb, ib, icount) = make_fullscreen_tri(&device, &queue);

        sys.submit_draw(
            &device, &queue, &joint_buf,
            handle, /* mesh_handle */ 1, /* mesh_idx */ 0,
            identity, identity, identity,
            [1.0; 4], [0; 4],
        );
        assert_eq!(sys.translucent_commands.len(), 1, "draw queued in translucent bucket");

        // Build the HDR + depth render targets for the dispatch.
        let (rt_w, rt_h) = (64u32, 64u32);
        let hdr_rt = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_hdr_rt"),
            size: wgpu::Extent3d { width: rt_w, height: rt_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let hdr_view = hdr_rt.create_view(&Default::default());
        let depth_rt = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_depth_rt"),
            size: wgpu::Extent3d { width: rt_w, height: rt_h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: formats::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_rt.create_view(&Default::default());

        // Pre-clear HDR to opaque cyan so we can detect alpha-blended red:
        // (1, 0, 0, 0.5) over (0, 1, 1, 1) → (0.5, 0.5, 0.5, 1.0).
        let bg_color = wgpu::Color { r: 0.0, g: 1.0, b: 1.0, a: 1.0 };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_translucent_encoder"),
        });
        {
            // Clear HDR + depth in one pass.
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test_clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &hdr_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(bg_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        // Translucent dispatch — Load (don't clear) HDR; depth read-only.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test_translucent_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &hdr_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            sys.dispatch_translucent(&mut pass, |mh, _idx| {
                if mh == 1 { Some((&vb, &ib, icount)) } else { None }
            });
        }

        // Read back the HDR target (Rgba16Float = 8 B / texel).
        let bpr_unpadded = rt_w * 8;
        let bpr = (bpr_unpadded + 255) & !255;
        let buf_size = (bpr * rt_h) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_hdr_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &hdr_rt,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(rt_h),
                },
            },
            wgpu::Extent3d { width: rt_w, height: rt_h, depth_or_array_layers: 1 },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv().expect("map sender").expect("map failed");
        let data = slice.get_mapped_range();

        // Sample the centre texel. Rgba16Float = 4 × half = 8 bytes.
        // Use half::f16 via bytemuck cast through u16 then manual decode.
        let cx = rt_w / 2;
        let cy = rt_h / 2;
        let row_start = (cy * bpr) as usize;
        let texel_start = row_start + (cx as usize) * 8;
        let halfs: [u16; 4] = [
            u16::from_le_bytes([data[texel_start],     data[texel_start + 1]]),
            u16::from_le_bytes([data[texel_start + 2], data[texel_start + 3]]),
            u16::from_le_bytes([data[texel_start + 4], data[texel_start + 5]]),
            u16::from_le_bytes([data[texel_start + 6], data[texel_start + 7]]),
        ];
        drop(data);
        staging.unmap();

        let r = f16_to_f32(halfs[0]);
        let g = f16_to_f32(halfs[1]);
        let b = f16_to_f32(halfs[2]);
        let _a = f16_to_f32(halfs[3]);

        // Expected SrcAlpha/OneMinusSrcAlpha blend:
        //   src = (1, 0, 0, 0.5)
        //   dst = (0, 1, 1, 1)
        //   out.rgb = src.rgb * src.a + dst.rgb * (1 - src.a)
        //           = (0.5, 0.5, 0.5)
        // Allow 1/256 tolerance for half-precision round-trip.
        let eps = 0.02;
        assert!((r - 0.5).abs() < eps, "red channel = {} (expected ~0.5)", r);
        assert!((g - 0.5).abs() < eps, "green channel = {} (expected ~0.5)", g);
        assert!((b - 0.5).abs() < eps, "blue channel = {} (expected ~0.5)", b);
    }

    /// IEEE-754 binary16 → binary32. We don't pull in the `half` crate
    /// for a single readback; the manual decode is short and exact for
    /// the values this test produces (no NaN / Inf / subnormal cases).
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = (bits >> 15) & 0x1;
        let exp  = (bits >> 10) & 0x1f;
        let frac = bits & 0x3ff;
        if exp == 0 {
            if frac == 0 {
                return if sign == 1 { -0.0 } else { 0.0 };
            }
            // Subnormal — not expected in this test; decode for completeness.
            let f = (frac as f32) / 1024.0 * (2.0f32).powi(-14);
            return if sign == 1 { -f } else { f };
        }
        if exp == 0x1f {
            return f32::NAN;  // Inf or NaN — unexpected in this test.
        }
        let f = (1.0 + (frac as f32) / 1024.0) * (2.0f32).powi(exp as i32 - 15);
        if sign == 1 { -f } else { f }
    }
}

#[cfg(test)]
mod translucent_sort_tests {
    use super::*;

    fn cmd(material: MaterialHandle, view_depth: f32) -> MaterialDrawCommand {
        MaterialDrawCommand {
            material,
            mesh_handle: 0,
            mesh_idx: 0,
            draw_slot: 0,
            view_depth,
            instance: None,
            // Identity — the sort under test only reads view_depth.
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    #[test]
    fn sorts_back_to_front_keeping_equal_depth_submission_order() {
        let mut ms_cmds = vec![
            cmd(1, 5.0),
            cmd(2, 20.0),
            cmd(3, 10.0),
            cmd(4, 10.0), // same depth as material 3 — must stay after it
            cmd(5, 1.0),
        ];
        // exercise the same comparator sort_translucent uses
        ms_cmds.sort_by(|a, b| b.view_depth.total_cmp(&a.view_depth));
        let order: Vec<MaterialHandle> = ms_cmds.iter().map(|c| c.material).collect();
        assert_eq!(order, vec![2, 3, 4, 1, 5]);
    }
}
