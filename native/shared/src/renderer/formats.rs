//! Texture formats + render-target creation helpers.
//!
//! All `*_FORMAT` constants, the mip-count constants, and every
//! `create_*` helper that hands the Renderer a `(Texture, View)`
//! pair live here. Pure data/helpers — no Renderer state. Each
//! helper is `pub(super)` so only the surrounding `renderer::`
//! module can call it.

use wgpu;

// ============================================================
// Depth texture helper
// ============================================================

pub(super) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Linear HDR format for the offscreen render target. The scene + sky
/// + immediate-mode 3D passes write here in linear space; a final
/// composite pass tonemaps to the sRGB surface format.
pub(super) const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Number of bloom mip levels. 5 mips gives a long-tail glow that
/// covers ~32× the source pixel size. More mips = more haloing,
/// fewer = less coverage. Each mip is half the previous size.
pub(super) const BLOOM_MIP_COUNT: u32 = 5;

/// SSAO RT layout: R = GTAO occlusion (bilaterally blurred), G =
/// contact-shadow factor (passed through blur unchanged so the fine-
/// detail ray-march result survives). Rgba8Unorm because WebGPU
/// requires `rgba8unorm` for storage-texture writes by default —
/// the compute GTAO pass (SSAO_SHADER_WGSL) uses `textureStore`.
/// Extra two channels left 0; downstream samplers only read .r/.g,
/// so the only cost is 4 B/px vs 2 B/px at half-res
/// (~180 kB extra on a 1600×900 surface).
pub(super) const SSAO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Material G-buffer format. Rg8Unorm: R = metallic, G = roughness.
/// Written as a second color attachment in the HDR pass; SSR (and
/// any future deferred passes) reads it for per-pixel material info.
pub(super) const MATERIAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;

/// Linear-depth Hi-Z pyramid format. R32Float (not R16Float) because
/// WebGPU only mandates r32-family formats for single-channel storage
/// textures. The pyramid stores *positive* view-space distance
/// (|view_z|) so compute GTAO skips per-sample linearization. Sky
/// pixels get `HIZ_SKY_Z` (10 000) and the downsample uses `min` so
/// any near-field geometry in a tile dominates surrounding sky.
pub(super) const HIZ_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Number of mip levels in the linear-depth Hi-Z pyramid. 5 covers
/// a 16-pixel-radius footprint at the coarsest mip — enough for
/// the 0.25 UV clamp SSAO uses (~100 px at half-res 400-wide).
/// One linearize pass plus `HIZ_MIP_COUNT - 1` downsample passes.
pub(super) const HIZ_MIP_COUNT: u32 = 5;

/// Velocity buffer format. Rg16Float: two-channel 16-bit float for
/// sub-pixel precision screen-space velocity. Written as a third
/// color attachment in the HDR pass; motion blur and TAA read it.
pub(super) const VELOCITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

pub(super) fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        // SSAO samples this texture in a separate pass after the
        // depth-write HDR pass — needs TEXTURE_BINDING in addition
        // to RENDER_ATTACHMENT.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub(super) fn create_hdr_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hdr_rt"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the two ping-pong 1×1 exposure textures. Single fragment
/// writes to one, composite samples the other, swap each frame.
pub(super) fn create_exposure_textures(device: &wgpu::Device) -> ([wgpu::Texture; 2], [wgpu::TextureView; 2]) {
    let make = |label: &str| -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    };
    let (a, av) = make("exposure_a");
    let (b, bv) = make("exposure_b");
    ([a, b], [av, bv])
}

/// Create the material G-buffer (Rg8Unorm, surface size).
pub(super) fn create_material_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("material_rt"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: MATERIAL_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the albedo G-buffer (Rgba8Unorm, surface size). Written by
/// the scene pass so post-passes can modulate bounce light by the
/// receiving surface's diffuse albedo (SSGI etc.).
pub(super) fn create_albedo_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("albedo_rt"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the composed HDR render target. Scene HDR + SSR + SSGI *
/// albedo + bloom + fog + sun shafts all merged into one texture by
/// the `scene_compose` pass. Both the TAA-on path (TAA consumes this
/// as its "current frame" input) and the TAA-off path (composite
/// reads it directly) read from the same buffer, so fog / shafts /
/// post-effects stay visible regardless of TAA state.
pub(super) fn create_composed_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("composed_rt"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the velocity render target (Rg16Float, surface size).
/// Per-pixel screen-space velocity for motion blur and TAA.
pub(super) fn create_velocity_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("velocity_rt"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: VELOCITY_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the SSR render target (quarter-res HDR). Stochastic SSR
/// traces one GGX-sampled ray per pixel per frame and relies on the
/// temporal denoiser to converge the cone over 4–8 frames; pushing
/// from half to quarter-res cuts the march + temporal-filter pixel
/// counts 4× and the bilinear upsample in compose is imperceptible
/// because the GGX cone + temporal blend is already wider than one
/// quarter-res texel.
pub(super) fn create_ssr_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let w = (width / 4).max(1);
    let h = (height / 4).max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ssr_rt"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the SSR temporal history textures (ping-pong pair, same
/// format/size as ssr_rt — half-res HDR). The temporal denoiser
/// blends the noisy current-frame stochastic SSR into the
/// reprojected previous-frame history so 4–8 frames of accumulation
/// converge to a smooth reflection.
pub(super) fn create_ssr_history_textures(
    device: &wgpu::Device, width: u32, height: u32,
) -> ([wgpu::Texture; 2], [wgpu::TextureView; 2]) {
    let w = (width / 4).max(1);
    let h = (height / 4).max(1);
    let make = || {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssr_history"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    };
    let (t0, v0) = make();
    let (t1, v1) = make();
    ([t0, t1], [v0, v1])
}

/// Create the SSGI render target (half-res HDR — indirect diffuse bounce light).
/// Same half-res HDR strategy as SSR: keeps the per-pixel ray march cheap
/// while still providing enough color resolution for colored bounce light.
pub(super) fn create_ssgi_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let w = (width / 2).max(1);
    let h = (height / 2).max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ssgi_rt"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Probe grid = ceil(half_w / 16) × ceil(half_h / 16). 50×29 on a
/// 800×450 half-res Sponza RT = 1450 probes, each holding an 8×8
/// octahedral atlas = 64 radiance samples (ticket 007a).
pub(super) const PROBE_TILE_SIZE: u32 = 16;
pub(super) const PROBE_OCT_SIZE: u32 = 8;
pub(super) const PROBE_OCT_TEXELS: u32 = PROBE_OCT_SIZE * PROBE_OCT_SIZE;

pub(super) fn probe_grid_dims(width: u32, height: u32) -> (u32, u32) {
    let half_w = (width / 2).max(1);
    let half_h = (height / 2).max(1);
    let gw = (half_w + PROBE_TILE_SIZE - 1) / PROBE_TILE_SIZE;
    let gh = (half_h + PROBE_TILE_SIZE - 1) / PROBE_TILE_SIZE;
    (gw.max(1), gh.max(1))
}

/// 3D Rgba16Float texture with dimensions `(probe_grid_w, probe_grid_h,
/// 64)` — one voxel per probe × octahedral texel. Shared shape for the
/// trace output and the ping-pong history textures.
fn create_probe_3d_tex(
    device: &wgpu::Device, label: &'static str, gw: u32, gh: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: gw, height: gh, depth_or_array_layers: PROBE_OCT_TEXELS },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Per-frame trace output. The trace pass writes into this, the temporal
/// pass reads it as the "current" input. Not ping-pong because its
/// contents are fully regenerated every frame.
pub(super) fn create_probe_trace_tex(
    device: &wgpu::Device, width: u32, height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let (gw, gh) = probe_grid_dims(width, height);
    create_probe_3d_tex(device, "probe_trace", gw, gh)
}

/// Ticket 013 — Mesh Cards. Shared 2D atlases sampled by the HW probe
/// trace at hit. V2 stores 6 signed-axis slots per mesh (±X ±Y ±Z)
/// at 64×64 each; 4096² atlas ⇒ 64×64 = 4096 slots ⇒ 682 meshes at
/// full 6-axis capture (Sponza's ~405 fits comfortably).
///
/// Two atlases are kept in lockstep:
///   - `mesh_card_albedo_atlas`   — baked once per mesh at load.
///   - `mesh_card_radiance_atlas` — written every frame by the card-
///     lighting compute pass (albedo × sun × NdotL + sky × NdotUp).
/// The HW trace samples radiance directly at hit, amortising shading
/// cost across all rays that land in the same card texel.
pub(super) const CARD_ATLAS_SIZE: u32 = 4096;
pub(super) const CARD_SLOT_SIZE: u32 = 64;
pub(super) const CARD_SLOTS_PER_ROW: u32 = CARD_ATLAS_SIZE / CARD_SLOT_SIZE;
pub(super) const CARD_MAX_SLOTS: u32 = CARD_SLOTS_PER_ROW * CARD_SLOTS_PER_ROW;
/// V2: 6 directed axes per mesh (+X, -X, +Y, -Y, +Z, -Z).
pub(super) const CARD_AXES_PER_MESH: u32 = 6;

/// Create the mesh-card albedo atlas. `RENDER_ATTACHMENT` for capture,
/// `TEXTURE_BINDING` for both the card-lighting compute input and a
/// direct HW-trace fallback.
pub(super) fn create_mesh_card_atlas(
    device: &wgpu::Device,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mesh_card_albedo_atlas"),
        size: wgpu::Extent3d {
            width: CARD_ATLAS_SIZE,
            height: CARD_ATLAS_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the mesh-card radiance atlas. Written every frame by the
/// card-lighting compute pass; sampled at hit by the HW probe trace.
/// Rgba16Float so we can carry multiplicatively-composed sun + sky
/// without banding. `STORAGE_BINDING` for the compute write,
/// `TEXTURE_BINDING` for the trace sample.
pub(super) fn create_mesh_card_radiance_atlas(
    device: &wgpu::Device,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mesh_card_radiance_atlas"),
        size: wgpu::Extent3d {
            width: CARD_ATLAS_SIZE,
            height: CARD_ATLAS_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Ping-pong probe-radiance history. Each frame the temporal pass
/// reads `[prev_idx]` (last frame's blended history), blends it with
/// the fresh trace, and writes to `[write_idx]`. The resolve pass then
/// samples `[write_idx]`. Separate textures — the temporal pass cannot
/// bind the same view as both sampled input and storage-write output.
pub(super) fn create_probe_history_textures(
    device: &wgpu::Device, width: u32, height: u32,
) -> ([wgpu::Texture; 2], [wgpu::TextureView; 2]) {
    let (gw, gh) = probe_grid_dims(width, height);
    let (t0, v0) = create_probe_3d_tex(device, "probe_history_0", gw, gh);
    let (t1, v1) = create_probe_3d_tex(device, "probe_history_1", gw, gh);
    ([t0, t1], [v0, v1])
}

/// Create the DoF render target (full-res HDR, same format as TAA output).
/// DoF reads the TAA output + depth, writes the blurred result here.
/// Composite then reads this instead of the TAA output.
pub(super) fn create_dof_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dof_rt"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the SSS render target (full-res HDR, same format as DoF/motion-blur).
pub(super) fn create_sss_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sss_rt"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Halton low-discrepancy sequence (base `b`, index `i`, 1-based).
/// Returns a value in [0, 1). Used to generate sub-pixel jitter
/// offsets that are well-distributed across the pixel — the TAA
/// accumulation effectively integrates over those sample points
/// to produce a stably anti-aliased image.
pub(super) fn halton(mut i: u32, b: u32) -> f32 {
    let mut f = 1.0_f32;
    let mut r = 0.0_f32;
    while i > 0 {
        f /= b as f32;
        r += f * (i % b) as f32;
        i /= b;
    }
    r
}

/// Create the two TAA history textures (HDR format, surface size).
pub(super) fn create_taa_textures(device: &wgpu::Device, width: u32, height: u32) -> ([wgpu::Texture; 2], [wgpu::TextureView; 2]) {
    let make = |label: &str| -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    };
    let (a, av) = make("taa_a");
    let (b, bv) = make("taa_b");
    ([a, b], [av, bv])
}

/// Create the SSAO render target. Written by the compute GTAO pass
/// (via `STORAGE_BINDING`) and sampled by the bilateral blur +
/// downstream passes.
pub(super) fn create_ssao_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let w = (width / 2).max(1);
    let h = (height / 2).max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ssao_rt"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SSAO_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Ping-pong half-res SSAO history textures (temporal accumulation).
/// Same size/format as `ssao_rt`. The compute pass reads the previous
/// frame's history via `TEXTURE_BINDING` and writes the blended
/// current-frame result back via `STORAGE_BINDING`. Downstream the
/// bilateral blur samples the current-frame ssao_rt as before —
/// history is only used as an input to the GTAO compute.
pub(super) fn create_ssao_history_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> ([wgpu::Texture; 2], [wgpu::TextureView; 2]) {
    let w = (width / 2).max(1);
    let h = (height / 2).max(1);
    let make = |label: &str| -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SSAO_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    };
    let (a, av) = make("ssao_history_a");
    let (b, bv) = make("ssao_history_b");
    ([a, b], [av, bv])
}

/// Create the SSAO bilateral-blur render target (same format/size as ssao_rt).
pub(super) fn create_ssao_blur_rt(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let w = (width / 2).max(1);
    let h = (height / 2).max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ssao_blur_rt"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SSAO_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Build the linear-depth Hi-Z pyramid as `HIZ_MIP_COUNT` separate
/// single-mip textures. One multi-mip texture is cheaper on paper
/// but Metal's per-subresource state tracking trips when wgpu
/// writes one mip and samples another in the same encoder — the
/// bloom chain has the same issue and uses this same layout.
pub(super) fn create_linear_depth_hiz_chain(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (Vec<wgpu::Texture>, Vec<wgpu::TextureView>) {
    let mut textures = Vec::with_capacity(HIZ_MIP_COUNT as usize);
    let mut views = Vec::with_capacity(HIZ_MIP_COUNT as usize);
    for i in 0..HIZ_MIP_COUNT {
        let w = ((width / 2) >> i).max(1);
        let h = ((height / 2) >> i).max(1);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("linear_depth_hiz_mip"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HIZ_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        textures.push(tex);
        views.push(view);
    }
    (textures, views)
}

/// Create the bloom mip-chain texture + per-mip render views + a
/// full-chain view for sampling. Mip 0 starts at surface/2 size and
/// each subsequent mip halves down to ~surface/2^N. Caller is
/// responsible for deciding N (usually BLOOM_MIP_COUNT). At least
/// 1×1 is enforced per mip.
/// Build the bloom chain as N separate single-mip textures rather
/// than one multi-mip texture. Multi-mip textures with one mip
/// bound as render target while another mip is sampled in the
/// same encoder trips wgpu/Metal's per-subresource state tracking
/// — symptoms include large black bars in the sampled output. N
/// separate textures sidestep the problem entirely (each pass's
/// read/write hits a distinct texture). `bloom_full_view` is a
/// view onto mip 0's texture, kept for backward compatibility.
pub(super) fn create_bloom_chain(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    mip_count: u32,
) -> (Vec<wgpu::Texture>, Vec<wgpu::TextureView>, wgpu::TextureView) {
    let mut textures = Vec::with_capacity(mip_count as usize);
    let mut views = Vec::with_capacity(mip_count as usize);
    for i in 0..mip_count {
        let w = ((width / 2) >> i).max(1);
        let h = ((height / 2) >> i).max(1);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom_mip_tex"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        textures.push(tex);
        views.push(view);
    }
    let full_view = textures[0].create_view(&wgpu::TextureViewDescriptor::default());
    (textures, views, full_view)
}
