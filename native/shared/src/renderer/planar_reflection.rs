//! EN-011 — Planar reflection probes.
//!
//! A planar reflection probe owns an off-screen RT (`Rgba16Float` HDR
//! colour + `Depth32Float` depth) into which the engine renders the
//! scene from a camera mirrored across a flat reflective plane. The
//! resulting texture is bound at `@group(2) @binding(12)` on materials
//! that opt in via `Renderer::set_material_reflection_probe`, so e.g.
//! a water shader can sample the *actual* trees / bridge above the
//! surface instead of just the static `env_tex` skybox.
//!
//! ## V1 design
//!
//! - **One probe per plane.** A river is one plane; lakes are one
//!   plane. Multi-probe blending lives in V2.
//! - **Rebuilt every frame** at the requested resolution. Half of the
//!   swapchain width × height is the typical caller-supplied number;
//!   nothing in this module enforces that — the FFI just defaults to
//!   it when the game passes 0.
//! - **Cull list = "everything in the opaque material bucket minus a
//!   hardcoded exclude list".** Implemented in `Renderer` (this
//!   module just owns the probe RT + plane parameters); the dispatch
//!   loop walks `material_system.commands` and skips the same
//!   material handles for every probe.
//!
//! ## Coordinate convention
//!
//! The reflection plane is defined by `plane_y` (a single y-value
//! offset from origin) and a unit `normal`. We mirror world-space
//! points across plane `n · p = d`, where `d = n · plane_origin` and
//! `plane_origin = (0, plane_y, 0)`. For the typical horizontal water
//! surface, `normal = (0, 1, 0)` and `d = plane_y`. Non-axis-aligned
//! planes work too — the math is general — but the FFI surface
//! exposes the simpler horizontal case explicitly.
//!
//! See `Renderer::dispatch_planar_reflections` for the per-frame
//! render-graph node that drives all registered probes.

use crate::renderer::util::{mat4_invert, mat4_multiply};

/// One planar reflection probe + its dedicated RT pair.
///
/// The texture views (`color_view`, `depth_view`) are stable for the
/// lifetime of the probe, so per-material bind groups built once at
/// `set_material_reflection_probe` time stay valid frame after frame
/// even as the engine repaints the texture each frame.
pub struct PlanarReflectionProbe {
    /// World-space y of the reflective plane (only used to build `d`
    /// when normal == +Y; otherwise carried for diagnostics).
    pub plane_y: f32,
    /// Unit normal of the plane in world space. The mirror matrix
    /// reflects across `n · p = d`, where `d = n · (0, plane_y, 0)`.
    pub normal:  [f32; 3],
    /// Texture extent — width = height for a square probe; both
    /// dimensions get rounded to ≥ 16 px in `new` so a tiny RT can't
    /// crash the renderer.
    pub resolution: u32,

    pub color_rt:    wgpu::Texture,
    pub color_view:  wgpu::TextureView,
    pub depth_rt:    wgpu::Texture,
    pub depth_view:  wgpu::TextureView,
}

impl PlanarReflectionProbe {
    /// Allocate the colour + depth textures sized to `resolution²`.
    /// Format choices match the engine's HDR pipeline so the probe
    /// texture interoperates cleanly with the rest of `material_abi`:
    ///
    ///   - colour: `Rgba16Float` (`HDR_FORMAT`) so emissive geometry
    ///     reflected into the probe doesn't clamp to LDR
    ///   - depth: `Depth32Float` so the mirrored draws can z-test
    ///     against each other without a separate downsample
    pub fn new(
        device: &wgpu::Device,
        plane_y: f32,
        normal:  [f32; 3],
        resolution: u32,
    ) -> Self {
        // Clamp absurd inputs — a 0-px texture is illegal in wgpu
        // and a 16k-px probe would cost more than the rest of the
        // frame combined. 16..=4096 covers every realistic case.
        let res = resolution.clamp(16, 4096);

        let color_rt = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("planar_reflection_color"),
            size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: super::formats::HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_rt.create_view(&Default::default());

        let depth_rt = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("planar_reflection_depth"),
            size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: super::formats::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_rt.create_view(&Default::default());

        // Normalise the supplied normal — caller may pass a non-unit
        // vector; downstream math (specifically the reflection
        // matrix below) assumes |n| == 1.
        let n = normalise(normal);

        Self {
            plane_y, normal: n, resolution: res,
            color_rt, color_view, depth_rt, depth_view,
        }
    }
}

/// Build the world-space reflection matrix for plane (n, plane_y).
///
/// Reflects a world-space point `p` across the plane `n · p = d`
/// where `d = n · (0, plane_y, 0) = n.y * plane_y`. The returned 4×4
/// matrix R has `R · p = p - 2 (n·p - d) n`, with R applied
/// post-multiply on column vectors (Bloom convention).
///
/// Plug this into the view chain via `mirror_view = view * R` —
/// post-multiplying R into the view matrix means: world → mirror
/// (R) → camera (view).
pub fn reflection_matrix(plane_y: f32, normal: [f32; 3]) -> [[f32; 4]; 4] {
    let n = normalise(normal);
    let d = n[1] * plane_y;
    // Standard Householder-style reflection across the plane.
    // Column-major; multiplies p as `R * p` (post-mul).
    let nx = n[0]; let ny = n[1]; let nz = n[2];
    [
        [1.0 - 2.0 * nx * nx, -2.0 * nx * ny,       -2.0 * nx * nz,       0.0],
        [-2.0 * ny * nx,        1.0 - 2.0 * ny * ny, -2.0 * ny * nz,       0.0],
        [-2.0 * nz * nx,       -2.0 * nz * ny,        1.0 - 2.0 * nz * nz, 0.0],
        [2.0 * nx * d,          2.0 * ny * d,         2.0 * nz * d,         1.0],
    ]
}

/// Compose a mirrored view matrix from the camera's current view
/// matrix and the reflection plane.
///
/// In Bloom's column-major / post-mul convention, `view` transforms
/// world points into camera space (`p_cam = view * p_world`). To
/// render the mirror image we want `p_cam = view * R * p_world` —
/// reflect first, then apply the existing view. The returned matrix
/// is `view * R`.
///
/// **Caller MUST flip front-face cull mode** when using this view —
/// reflection inverts triangle winding. The renderer handles that
/// by binding pipelines compiled with `cull_mode = None` for the
/// mirrored pass; we accept the small fragment-shader cost for V1.
pub fn mirrored_view(view: [[f32; 4]; 4], plane_y: f32, normal: [f32; 3]) -> [[f32; 4]; 4] {
    let r = reflection_matrix(plane_y, normal);
    mat4_multiply(view, r)
}

/// Reflect the camera's world position across the plane. Used for
/// `PerView.camera_pos` in the mirrored UBO so view-dependent shading
/// (Fresnel, specular, parallax) sees the mirror camera, not the
/// real one.
pub fn mirrored_camera_pos(pos: [f32; 3], plane_y: f32, normal: [f32; 3]) -> [f32; 3] {
    let n = normalise(normal);
    let d = n[1] * plane_y;
    let dist = n[0] * pos[0] + n[1] * pos[1] + n[2] * pos[2] - d;
    [
        pos[0] - 2.0 * dist * n[0],
        pos[1] - 2.0 * dist * n[1],
        pos[2] - 2.0 * dist * n[2],
    ]
}

/// Recompute `inv_proj` from a possibly-modified projection. The
/// reflection pass uses the same projection as the main camera in V1
/// (no oblique near-plane clip yet — that's a Phase 2 polish item),
/// so this is currently a passthrough wrapper, but isolated here so
/// future oblique-clip work has one place to land.
pub fn inv_proj_for(proj: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    mat4_invert(proj)
}

fn normalise(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq < 1e-10 {
        // Default to +Y — a horizontal mirror like a calm lake.
        return [0.0, 1.0, 0.0];
    }
    let inv = 1.0 / len_sq.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::util::mat4_mul_vec4;

    /// A point above a horizontal y=0 plane reflects to the mirror
    /// position below it — y-coordinate flips sign, x and z preserved.
    #[test]
    fn reflection_matrix_mirrors_above_to_below() {
        let r = reflection_matrix(0.0, [0.0, 1.0, 0.0]);
        let p = [3.0_f32, 5.0, -7.0, 1.0];
        let out = mat4_mul_vec4(&r, &p);
        assert!((out[0] - 3.0).abs() < 1e-5, "x preserved (got {})", out[0]);
        assert!((out[1] + 5.0).abs() < 1e-5, "y flipped (got {})", out[1]);
        assert!((out[2] + 7.0).abs() < 1e-5, "z preserved (got {})", out[2]);
        assert!((out[3] - 1.0).abs() < 1e-5, "w preserved");
    }

    /// A non-zero `plane_y` shifts the mirror — point at y=10 across
    /// plane y=2 lands at y = 2 - (10 - 2) = -6.
    #[test]
    fn reflection_matrix_offset_plane() {
        let r = reflection_matrix(2.0, [0.0, 1.0, 0.0]);
        let p = [0.0_f32, 10.0, 0.0, 1.0];
        let out = mat4_mul_vec4(&r, &p);
        assert!((out[1] - (-6.0)).abs() < 1e-4, "y reflected across y=2 (got {})", out[1]);
    }

    /// Camera position helper agrees with applying the matrix to a
    /// (cam_pos, 1) vec4 — sanity check.
    #[test]
    fn mirrored_camera_pos_matches_matrix() {
        let plane_y = 0.5;
        let n = [0.0, 1.0, 0.0];
        let cam = [1.5_f32, 4.0, -2.0];

        let helper_out = mirrored_camera_pos(cam, plane_y, n);
        let r = reflection_matrix(plane_y, n);
        let mat_out = mat4_mul_vec4(&r, &[cam[0], cam[1], cam[2], 1.0]);

        assert!((helper_out[0] - mat_out[0]).abs() < 1e-5);
        assert!((helper_out[1] - mat_out[1]).abs() < 1e-5);
        assert!((helper_out[2] - mat_out[2]).abs() < 1e-5);
    }

    /// Reflection is its own inverse — applying it twice returns
    /// the original point.
    #[test]
    fn reflection_is_involution() {
        let r = reflection_matrix(0.5, [0.0, 1.0, 0.0]);
        let p = [3.0_f32, 5.0, -7.0, 1.0];
        let once = mat4_mul_vec4(&r, &p);
        let twice = mat4_mul_vec4(&r, &once);
        for i in 0..4 {
            assert!((twice[i] - p[i]).abs() < 1e-4, "comp {} mismatch", i);
        }
    }

    /// Headless wgpu device. Mirrors the EN-006 pattern from
    /// `transient.rs` / `impulse_field.rs`. Returns `None` when no GPU
    /// is available so the test skips gracefully on bare CI.
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
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("planar-reflection-test-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
        )).ok()?;
        Some((device, queue))
    }

    /// EN-011 — verify `PlanarReflectionProbe::new` allocates HDR
    /// colour + Depth32 attachments at the requested resolution.
    /// Pure construction test — render-graph integration lives in
    /// `Renderer::dispatch_planar_reflections`.
    #[test]
    fn probe_creation_allocates_hdr_and_depth_attachments() {
        let Some((device, _queue)) = try_create_device() else { return; };
        let probe = PlanarReflectionProbe::new(&device, 0.5, [0.0, 1.0, 0.0], 256);
        assert_eq!(probe.resolution, 256, "resolution clamped to caller value");
        assert_eq!(probe.normal, [0.0, 1.0, 0.0], "normalised +Y stays +Y");
        let color_size = probe.color_rt.size();
        assert_eq!(color_size.width, 256);
        assert_eq!(color_size.height, 256);
        assert_eq!(probe.color_rt.format(), super::super::formats::HDR_FORMAT);
        assert_eq!(probe.depth_rt.format(), super::super::formats::DEPTH_FORMAT);
    }

    /// Tiny resolutions are clamped up to 16 px to keep wgpu happy.
    #[test]
    fn probe_resolution_clamps_minimum() {
        let Some((device, _queue)) = try_create_device() else { return; };
        let probe = PlanarReflectionProbe::new(&device, 0.0, [0.0, 1.0, 0.0], 4);
        assert_eq!(probe.resolution, 16, "clamps to 16 px floor");
    }
}
