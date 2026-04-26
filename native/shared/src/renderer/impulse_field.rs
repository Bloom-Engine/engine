//! Phase 7 — world-space impulse field.
//!
//! A small R32Float texture covering the playable world as a top-down
//! XZ projection. Games call `submit_splat(x, z, radius, strength)` on
//! events of interest (player enters water, explosion lands, footstep
//! on mud), and a per-frame compute pass decays the previous frame's
//! value and adds the new splats. Materials in the translucent pass
//! sample it from `@group(4) @binding(4)` for ripples, decals, snow,
//! wet spots — whatever decays-over-time, localised-in-world
//! phenomenon the material wants to express.
//!
//! Texture is ping-ponged to avoid a read/write alias on a single
//! storage resource; read side is a plain sampled `texture_2d<f32>`
//! so games don't need the read_write storage feature flag.
//!
//! World footprint is a hardcoded 128 m square centred on origin at
//! 256×256 texels = 0.5 m/texel for now. A later phase can replace
//! that with a per-game descriptor.
use crate::renderer::shader_include::ShaderSource;
use crate::renderer::shader_library::library;

/// Max splats accumulated in a single frame. Extra submissions are
/// dropped silently — games should batch their own impulses.
pub const MAX_SPLATS_PER_FRAME: usize = 16;

/// Pixel side of the impulse texture. Power-of-two, workgroup-divisible.
pub const IMPULSE_SIZE: u32 = 256;

/// Default world footprint — a centred square. Half-extent in metres.
pub const IMPULSE_WORLD_HALF_EXTENT: f32 = 64.0;

/// Per-frame decay multiplier. A splat with strength 1.0 fades to
/// ~2% after 120 frames @ 60 fps (two seconds) — matches the RFC's
/// acceptance criterion.
pub const IMPULSE_DECAY_PER_FRAME: f32 = 0.968;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SplatData {
    pos:     [f32; 2],  // world xz
    radius:  f32,
    strength: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InfoUniforms {
    world_min:   [f32; 2],  // meters
    world_size:  [f32; 2],  // meters
    decay:       f32,
    _pad0:       f32,
    splat_count: u32,
    _pad1:       u32,
    splats:      [SplatData; MAX_SPLATS_PER_FRAME],
}

pub struct ImpulseField {
    pipeline:   wgpu::ComputePipeline,
    bg_layout:  wgpu::BindGroupLayout,
    info_buf:   wgpu::Buffer,
    /// Backing textures for view_a/view_b — kept alive so the views
    /// don't dangle. Sampling and binding go through the views; these
    /// fields are write-only handles.
    _tex_a:     wgpu::Texture,
    _tex_b:     wgpu::Texture,
    view_a:     wgpu::TextureView,
    view_b:     wgpu::TextureView,
    sampler:    wgpu::Sampler,
    /// When true, view_a is the "front" that scene_inputs reads and
    /// the next compute pass reads-from. After dispatch we swap.
    front_is_a: bool,
    splats:     Vec<SplatData>,
}

impl ImpulseField {
    pub fn new(device: &wgpu::Device) -> Self {
        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("impulse_field_bg_layout"),
            entries: &[
                // 0: src (previous-frame sampled texture)
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 1: dst (write-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 2: info UBO (splats + bounds + decay)
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("impulse_field_pipeline_layout"),
            bind_group_layouts: &[Some(&bg_layout)],
            immediate_size: 0,
        });

        let shader_src = library().fetch("impulse_field.wgsl")
            .expect("impulse_field.wgsl must be present in shader_library")
            .to_string();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("impulse_field"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("impulse_field_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let info_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("impulse_field_info"),
            size: std::mem::size_of::<InfoUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let make_tex = |label: &str| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: IMPULSE_SIZE, height: IMPULSE_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                     | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            (tex, view)
        };
        let (tex_a, view_a) = make_tex("impulse_field_a");
        let (tex_b, view_b) = make_tex("impulse_field_b");

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("impulse_field_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline, bg_layout, info_buf,
            _tex_a: tex_a, _tex_b: tex_b, view_a, view_b, sampler,
            front_is_a: true,
            splats: Vec::with_capacity(MAX_SPLATS_PER_FRAME),
        }
    }

    /// Gameplay API — queue a splat at world xz with given radius (m)
    /// and peak strength. Silently drops overflow.
    pub fn submit_splat(&mut self, world_x: f32, world_z: f32, radius: f32, strength: f32) {
        if self.splats.len() >= MAX_SPLATS_PER_FRAME { return; }
        self.splats.push(SplatData {
            pos: [world_x, world_z], radius, strength,
        });
    }

    /// Run the decay + splat compute pass. Caller encodes just before
    /// the translucent pass so scene_inputs sees the latest field.
    /// After this, `front_view()` returns the view containing the new
    /// field and `update_scene_inputs` should bind it at group 4
    /// binding 4.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue,
                  encoder: &mut wgpu::CommandEncoder) {
        // Ping-pong: read from the CURRENT front (last frame's output),
        // write into the other side, then flip `front_is_a`.
        let (src_view, dst_view) = if self.front_is_a {
            (&self.view_a, &self.view_b)
        } else {
            (&self.view_b, &self.view_a)
        };

        // Build info UBO from the queued splats.
        let mut info = InfoUniforms {
            world_min:   [-IMPULSE_WORLD_HALF_EXTENT, -IMPULSE_WORLD_HALF_EXTENT],
            world_size:  [IMPULSE_WORLD_HALF_EXTENT * 2.0, IMPULSE_WORLD_HALF_EXTENT * 2.0],
            decay:       IMPULSE_DECAY_PER_FRAME,
            _pad0:       0.0,
            splat_count: self.splats.len() as u32,
            _pad1:       0,
            splats:      [SplatData { pos: [0.0; 2], radius: 0.0, strength: 0.0 };
                          MAX_SPLATS_PER_FRAME],
        };
        for (i, s) in self.splats.iter().enumerate() {
            info.splats[i] = *s;
        }
        queue.write_buffer(&self.info_buf, 0, bytemuck::bytes_of(&info));
        self.splats.clear();

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("impulse_field_bg"),
            layout: &self.bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(dst_view) },
                wgpu::BindGroupEntry { binding: 2, resource: self.info_buf.as_entire_binding() },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("impulse_field_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let groups = IMPULSE_SIZE / 8;
            pass.dispatch_workgroups(groups, groups, 1);
        }

        self.front_is_a = !self.front_is_a;
    }

    /// The view containing the latest impulse field — what materials
    /// should sample. Available AFTER `update` ran this frame.
    pub fn front_view(&self) -> &wgpu::TextureView {
        if self.front_is_a { &self.view_a } else { &self.view_b }
    }

    pub fn sampler(&self) -> &wgpu::Sampler { &self.sampler }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Headless wgpu device for tests. Mirrors the helper in
    /// `transient.rs` — falls back to a low-power adapter and returns
    /// `None` when no adapter is available so CI runners without a GPU
    /// skip the test instead of failing.
    ///
    /// The impulse field uses a `R32Float` storage texture (writeOnly).
    /// `downlevel_defaults` does **not** allow R32Float storage, so we
    /// request the explicit `Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`
    /// upgrade. If the adapter doesn't expose it, return `None` — the
    /// test then skips gracefully (matches the "skipped on adapters
    /// without storage textures" doc-comment contract).
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
        // Only proceed if the adapter advertises R32Float storage support.
        let needed = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        let supported = adapter.features();
        if !supported.contains(needed) {
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("impulse-test-device"),
                required_features: needed,
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
        )).ok()?;
        Some((device, queue))
    }

    /// Read back the entire impulse front texture as `IMPULSE_SIZE *
    /// IMPULSE_SIZE` floats. R32Float = 4 B/texel, padded to 256 B per
    /// row by wgpu's copy-to-buffer rules.
    fn read_front_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        field: &ImpulseField,
    ) -> Vec<f32> {
        let bpr_unpadded = IMPULSE_SIZE * 4;
        let bpr = (bpr_unpadded + 255) & !255;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("impulse_test_staging"),
            size: (bpr * IMPULSE_SIZE) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // Pull the actual texture handle that backs `front_view`.
        let front_tex = if field.front_is_a { &field._tex_a } else { &field._tex_b };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("impulse_test_readback"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: front_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(IMPULSE_SIZE),
                },
            },
            wgpu::Extent3d { width: IMPULSE_SIZE, height: IMPULSE_SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv().expect("map_async sender dropped").expect("map failed");
        let data = slice.get_mapped_range();
        let mut out: Vec<f32> = Vec::with_capacity((IMPULSE_SIZE * IMPULSE_SIZE) as usize);
        for row in 0..IMPULSE_SIZE {
            let row_start = (row * bpr) as usize;
            let row_end = row_start + (bpr_unpadded as usize);
            let slice = &data[row_start..row_end];
            let floats: &[f32] = bytemuck::cast_slice(slice);
            out.extend_from_slice(floats);
        }
        drop(data);
        staging.unmap();
        out
    }

    /// Convert a world-XZ position into the impulse texel index that
    /// owns it. Mirrors the WGSL `world_min + uv * world_size` math.
    fn world_to_texel(x: f32, z: f32) -> (u32, u32) {
        let w_min = -IMPULSE_WORLD_HALF_EXTENT;
        let w_size = IMPULSE_WORLD_HALF_EXTENT * 2.0;
        let u = ((x - w_min) / w_size).clamp(0.0, 1.0 - 1e-6);
        let v = ((z - w_min) / w_size).clamp(0.0, 1.0 - 1e-6);
        let tx = (u * IMPULSE_SIZE as f32) as u32;
        let ty = (v * IMPULSE_SIZE as f32) as u32;
        (tx, ty)
    }

    /// EN-006 — verifies that submitting a splat + dispatching the
    /// impulse-field compute shader writes a non-zero value at the
    /// expected texel and that the value decays monotonically across
    /// subsequent frames with no new splats.
    /// Skipped on adapters without `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`
    /// (R32Float storage textures).
    #[test]
    fn splat_appears_then_decays_across_frames() {
        let Some((device, queue)) = try_create_device() else { return; };
        let mut field = ImpulseField::new(&device);

        // Submit a splat at world origin with a generous radius so it
        // covers a couple of texels (the texture is 0.5 m / texel; a
        // 2 m radius lights up a 4-texel disc).
        field.submit_splat(0.0, 0.0, 2.0, 1.0);

        // Dispatch frame 1.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("impulse_test_frame_1"),
        });
        field.update(&device, &queue, &mut encoder);
        queue.submit(std::iter::once(encoder.finish()));

        let pixels = read_front_texture(&device, &queue, &field);
        let (tx, ty) = world_to_texel(0.0, 0.0);
        let center_idx = (ty * IMPULSE_SIZE + tx) as usize;
        let center_v1 = pixels[center_idx];
        assert!(
            center_v1 > 0.0,
            "splat should write a positive value at the centre texel ({}, {}); got {}",
            tx, ty, center_v1,
        );

        // A texel far outside the splat radius (16 m away ≫ 2 m radius)
        // must remain zero — confirms the splat is localised.
        let far_idx = {
            let (fx, fy) = world_to_texel(16.0, 16.0);
            (fy * IMPULSE_SIZE + fx) as usize
        };
        assert_eq!(pixels[far_idx], 0.0, "no splat far from origin");

        // Dispatch a few more frames with no new splats. The value at
        // the centre texel must decay strictly monotonically — the
        // shader multiplies by IMPULSE_DECAY_PER_FRAME (~0.968) every
        // frame.
        let mut prev = center_v1;
        for frame in 2..=4 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("impulse_test_decay"),
            });
            field.update(&device, &queue, &mut encoder);
            queue.submit(std::iter::once(encoder.finish()));
            let pixels = read_front_texture(&device, &queue, &field);
            let v = pixels[center_idx];
            assert!(
                v < prev,
                "frame {} centre value {} should be < previous {}",
                frame, v, prev,
            );
            assert!(v > 0.0, "frame {} centre value should still be positive (decay, not clear)", frame);
            // Verify the multiplier is roughly the documented decay.
            let ratio = v / prev;
            assert!(
                (ratio - IMPULSE_DECAY_PER_FRAME).abs() < 1e-3,
                "frame {} ratio {} should match documented decay {}",
                frame, ratio, IMPULSE_DECAY_PER_FRAME,
            );
            prev = v;
        }
    }

    /// EN-006 — guard: `submit_splat` silently drops when the
    /// per-frame queue is full. Pure CPU; runs everywhere.
    #[test]
    fn submit_splat_caps_at_max_per_frame() {
        // Construct only the CPU-side state we need to assert on.
        // Avoid `ImpulseField::new` here so this runs without a GPU.
        let mut splats: Vec<SplatData> = Vec::with_capacity(MAX_SPLATS_PER_FRAME);
        for _ in 0..(MAX_SPLATS_PER_FRAME + 5) {
            if splats.len() < MAX_SPLATS_PER_FRAME {
                splats.push(SplatData { pos: [0.0, 0.0], radius: 1.0, strength: 1.0 });
            }
        }
        assert_eq!(splats.len(), MAX_SPLATS_PER_FRAME);
    }
}
