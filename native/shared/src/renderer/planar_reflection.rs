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

use crate::renderer::util::{mat4_invert, mat4_mul_vec4, mat4_multiply};

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

    /// Dummy G-buffer attachments for the user-material probe pass.
    /// Opaque-profile material pipelines target the full 4-attachment
    /// opaque layout (hdr + material + velocity + albedo), so the
    /// probe's material pass must present the same four attachments —
    /// wgpu validates pipeline targets against pass attachments
    /// exactly. Only the hdr result is kept; these three are cleared
    /// each frame and their stores discarded.
    pub aux_material_rt:   wgpu::Texture,
    pub aux_material_view: wgpu::TextureView,
    pub aux_velocity_rt:   wgpu::Texture,
    pub aux_velocity_view: wgpu::TextureView,
    pub aux_albedo_rt:     wgpu::Texture,
    pub aux_albedo_view:   wgpu::TextureView,
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

        // Aux G-buffer dummies — see the struct field comment. Formats
        // must byte-match what `Renderer::compile_material` passes to
        // the pipeline descriptors or the probe pass fails validation.
        let make_aux = |label: &str, format: wgpu::TextureFormat| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            (tex, view)
        };
        let (aux_material_rt, aux_material_view) =
            make_aux("planar_reflection_aux_material", super::formats::MATERIAL_FORMAT);
        let (aux_velocity_rt, aux_velocity_view) =
            make_aux("planar_reflection_aux_velocity", super::formats::VELOCITY_FORMAT);
        let (aux_albedo_rt, aux_albedo_view) =
            make_aux("planar_reflection_aux_albedo", wgpu::TextureFormat::Rgba8Unorm);

        // Normalise the supplied normal — caller may pass a non-unit
        // vector; downstream math (specifically the reflection
        // matrix below) assumes |n| == 1.
        let n = normalise(normal);

        Self {
            plane_y, normal: n, resolution: res,
            color_rt, color_view, depth_rt, depth_view,
            aux_material_rt, aux_material_view,
            aux_velocity_rt, aux_velocity_view,
            aux_albedo_rt, aux_albedo_view,
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

/// Recompute `inv_proj` from a possibly-modified projection. After the
/// EN-011 V2 `oblique_proj` rewrite, the reflection pass uses a near-
/// plane-clipped projection that differs from the main camera's, so
/// the inverse needs to be recomputed for any view-space reconstruction
/// downstream (SSR, deferred unprojection).
pub fn inv_proj_for(proj: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    mat4_invert(proj)
}

/// Transform a world-space plane `(Nx, Ny, Nz, d)` (such that
/// `N · p + d = 0` for points on the plane) into eye / view space
/// using the view matrix. Plane equations transform by the inverse-
/// transpose of the matrix; for a rigid view matrix this is the same
/// as the matrix itself, but we do it the general way to stay safe
/// for skewed views.
///
/// For Bloom's column-major / post-mul convention, applying the
/// inverse-transpose of `view` to a plane `(N, d)` equates to the
/// standard direct-3D / OpenGL formulation.
pub fn world_plane_to_eye_space(
    view: [[f32; 4]; 4],
    plane_world: [f32; 4],
) -> [f32; 4] {
    let inv = mat4_invert(view);
    // Inverse-transpose, column-major. Multiplying the *transpose* of
    // `inv` by the plane vector is the same as multiplying `inv`
    // post-multiplied by the row vector — but we have a column-vec
    // mul helper so we transpose `inv` first by reading rows as cols.
    let inv_t: [[f32; 4]; 4] = [
        [inv[0][0], inv[1][0], inv[2][0], inv[3][0]],
        [inv[0][1], inv[1][1], inv[2][1], inv[3][1]],
        [inv[0][2], inv[1][2], inv[2][2], inv[3][2]],
        [inv[0][3], inv[1][3], inv[2][3], inv[3][3]],
    ];
    mat4_mul_vec4(&inv_t, &plane_world)
}

/// EN-011 V2 — modify a projection matrix so its near plane is clipped
/// at the given eye-space plane. Used for planar reflection to prevent
/// geometry below the water from polluting the reflection along the
/// shoreline edge.
///
/// Reference: Eric Lengyel, "Oblique View Frustum Depth Projection
/// and Clipping" (Journal of Game Development, 2005). The technique
/// shifts the projection's near plane to coincide with the supplied
/// plane (typically the water plane), then any geometry on the wrong
/// side gets clipped at the rasterizer.
///
/// `proj` is the original column-major projection matrix.
/// `plane_eye_space` is `(Nx, Ny, Nz, d)` such that points on the
/// plane satisfy `N · p_eye + d = 0`. Use `world_plane_to_eye_space`
/// to convert from a world-space plane.
pub fn oblique_proj(
    proj: [[f32; 4]; 4],
    plane_eye_space: [f32; 4],
) -> [[f32; 4]; 4] {
    let c = plane_eye_space;

    // Far-plane corner in clip space is in the direction of (sgn(c.x),
    // sgn(c.y), 1, 1) — Lengyel §2. Pulled back into eye space by
    // multiplying with `inv(proj)`.
    let sx = if c[0] >= 0.0 { 1.0 } else { -1.0 };
    let sy = if c[1] >= 0.0 { 1.0 } else { -1.0 };
    let q_clip = [sx, sy, 1.0, 1.0];
    let inv_p = mat4_invert(proj);
    let q = mat4_mul_vec4(&inv_p, &q_clip);

    // Scale `c` so the near-plane crosses through the supplied plane:
    //   M = (2 / dot(c, q)) · c
    // Then the new third row of P (which controls the depth output) is
    //   P_row2 = M - P_row3
    // (P_row3 is the standard perspective w-row, all stays the same.)
    let denom = c[0] * q[0] + c[1] * q[1] + c[2] * q[2] + c[3] * q[3];
    if denom.abs() < 1e-10 {
        // Degenerate plane orientation w.r.t. the frustum — leave
        // projection unchanged rather than divide-by-zero. The
        // reflection still renders, just without near-plane clipping.
        return proj;
    }
    let scale = 2.0 / denom;
    let m = [c[0] * scale, c[1] * scale, c[2] * scale, c[3] * scale];

    // proj is column-major: proj[col][row]. The "third row" we want
    // to replace is at row index 2 across all four columns. The
    // "fourth row" is at row index 3. New row-2 = M - row3 — but
    // since wgpu's clip-space z range is [0, 1] (not [-1, 1] like
    // OpenGL), the depth-rescaling pre-step is `M - P_row3` exactly
    // as in Lengyel's original derivation; the [-1, 1] vs [0, 1]
    // difference is absorbed by the scale.
    let mut out = proj;
    out[0][2] = m[0] - proj[0][3];
    out[1][2] = m[1] - proj[1][3];
    out[2][2] = m[2] - proj[2][3];
    out[3][2] = m[3] - proj[3][3];
    out
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

    /// EN-011 V2 — oblique projection clips a point on the wrong side
    /// of the supplied plane. Point ABOVE the eye-space plane should
    /// project inside the [-w, w] z range (clip-space-z divides to
    /// [0, 1] in wgpu); point BELOW the plane projects with z > w
    /// (i.e. ndc.z > 1 in wgpu) so the rasterizer clips it.
    ///
    /// Setup: identity view (eye-space == world-space), a horizontal
    /// plane y = 0 (so eye-space plane = (0, 1, 0, 0)), and a perspective
    /// projection. We then check a point ABOVE (y = +5) renders, and a
    /// point BELOW (y = -5) gets clipped.
    #[test]
    fn oblique_proj_clips_below_plane() {
        use crate::renderer::util::mat4_perspective;
        // Standard perspective matching mat4_perspective conventions.
        let proj = mat4_perspective(70.0_f32.to_radians(), 16.0/9.0, 0.1, 100.0);
        // Horizontal plane y = 0 in eye-space, normal +Y, points above
        // satisfy y > 0 → N·p + d > 0 (kept). The plane equation
        // (Nx, Ny, Nz, d) for "y >= 0 is above" is (0, 1, 0, 0).
        let plane_eye = [0.0_f32, 1.0, 0.0, 0.0];
        let oblique = oblique_proj(proj, plane_eye);

        // A point in eye space ABOVE the plane, in front of camera:
        // y = +5 (above), z = -10 (in front, right-handed view).
        let p_above = [3.0_f32, 5.0, -10.0, 1.0];
        let clip_above = mat4_mul_vec4(&oblique, &p_above);
        // wgpu / D3D / Metal clip space: visible when 0 <= z <= w.
        // For a point above the plane, z/w should be inside [0, 1].
        assert!(clip_above[3] > 0.0, "point above plane has positive w (got {})", clip_above[3]);
        let ndc_z_above = clip_above[2] / clip_above[3];
        assert!(ndc_z_above >= 0.0 && ndc_z_above <= 1.0,
            "above-plane point ndc.z within [0,1] (got {})", ndc_z_above);

        // A point BELOW the plane: y = -5 (below), z = -10 (in front).
        // Oblique projection moves the near plane to coincide with
        // y = 0, so this point is on the wrong side and should clip.
        // wgpu's clip space requires 0 <= z <= w to be visible — any
        // ndc.z outside [0, 1] (or w <= 0) means the rasterizer drops
        // the fragment.
        let p_below = [3.0_f32, -5.0, -10.0, 1.0];
        let clip_below = mat4_mul_vec4(&oblique, &p_below);
        let visible = clip_below[3] > 0.0
            && clip_below[2] >= 0.0
            && clip_below[2] <= clip_below[3];
        assert!(!visible,
            "below-plane point should be clipped; got clip = ({}, {}, {}, {})",
            clip_below[0], clip_below[1], clip_below[2], clip_below[3]);
    }

    /// `world_plane_to_eye_space` round-trip: a horizontal world plane
    /// y = 0 (normal +Y) under an identity view stays as (0, 1, 0, 0)
    /// in eye space — sanity check the inverse-transpose plumbing.
    #[test]
    fn world_plane_to_eye_space_identity_view() {
        use crate::renderer::util::IDENTITY_MAT4;
        let plane_world = [0.0_f32, 1.0, 0.0, 0.0]; // y = 0 plane
        let plane_eye = world_plane_to_eye_space(IDENTITY_MAT4, plane_world);
        for i in 0..4 {
            assert!((plane_eye[i] - plane_world[i]).abs() < 1e-5,
                "identity view leaves plane unchanged (comp {}: {} vs {})",
                i, plane_eye[i], plane_world[i]);
        }
    }
}
