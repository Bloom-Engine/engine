//! Depth of field.
//!
//! One fullscreen pass: a circle of confusion from scene depth, then a 16-tap
//! disc gather. Three things here are deliberate, because each was a bug:
//!
//! - **Depth is linearised from the projection's two constants**, not by
//!   unprojecting through an inverse matrix. The engine's perspective is the
//!   GL-style one drawn into a 0..1 depth buffer, so the stored value is that
//!   matrix's NDC z as it stands: `dist = B / (d + A)` with `A = proj[2][2]`,
//!   `B = proj[3][2]`.
//! - **There is a band that stays sharp.** A thin-lens circle is zero at one
//!   distance only; with it, whatever the game focuses on is soft a metre to
//!   either side, which is the whole target in a shooter. `range.xy` are the
//!   metres in front of and behind the focus distance that keep a zero circle;
//!   the blur grows from the band's edges.
//! - **The centre pixel and the samples use the same circle function.** They
//!   used two different ones (one divided by depth, one not), so a sample's
//!   weight compared values in different units.

pub(in crate::renderer) const DOF_SHADER_WGSL: &str = "
struct DofParams {
    // x = focus distance, y = aperture (circle scale), z = max radius (fraction of
    // the frame's height), w unused
    params: vec4<f32>,
    // x = sharp metres in front of the focus distance, y = sharp metres behind
    // it, z = height / width (keeps the disc round), w unused
    range: vec4<f32>,
    // x = proj[2][2], y = proj[3][2]
    proj: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: DofParams;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var color_samp: sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var depth_samp: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid & 1u) * 4u) - 1.0;
    let y = f32((vid >> 1u) * 4u) - 1.0;
    var out: VsOut;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Distance from the camera along its view axis, positive. The sky is far.
fn view_distance(uv: vec2<f32>) -> f32 {
    let d = textureSampleLevel(depth_tex, depth_samp, uv, 0u);
    if (d >= 0.9999) {
        return 1000.0;
    }
    return u.proj.y / (d + u.proj.x);
}

// Circle of confusion radius for a surface at `z`, as a fraction of the
// frame's height. Zero inside the sharp band.
fn circle(z: f32) -> f32 {
    let focus = u.params.x;
    var off = 0.0;
    if (z > focus) {
        off = max(z - focus - u.range.y, 0.0);
    } else {
        off = max(focus - z - u.range.x, 0.0);
    }
    return clamp(u.params.y * off / max(z, 0.1), 0.0, u.params.z);
}

// 16-sample Poisson disc.
const POISSON_16: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>( 0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870),
    vec2<f32>( 0.34495938,  0.29387760),
    vec2<f32>(-0.91588581,  0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543,  0.27676845),
    vec2<f32>( 0.97484398,  0.75648379),
    vec2<f32>( 0.44323325, -0.97511554),
    vec2<f32>( 0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>( 0.79197514,  0.19090188),
    vec2<f32>(-0.24188840,  0.99706507),
    vec2<f32>(-0.81409955,  0.91437590),
    vec2<f32>( 0.19984126,  0.78641367),
    vec2<f32>( 0.14383161, -0.14100790),
);

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let view_z = view_distance(in.uv);
    let coc = circle(view_z);
    let center_color = textureSampleLevel(color_tex, color_samp, in.uv, 0.0).rgb;

    // In focus: the source pixel, untouched.
    if (coc < 0.0005) {
        return vec4<f32>(center_color, 1.0);
    }

    // The disc is the same for every pixel. Turning it per pixel (interleaved
    // gradient noise) was tried: with nothing after this pass to average it,
    // the noise's row structure showed as horizontal stripes in the blur.
    let to_uv = vec2<f32>(u.range.z, 1.0);

    var color_sum = center_color;
    var weight_sum = 1.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let sample_uv = in.uv + POISSON_16[i] * coc * to_uv;
        let sample_color = textureSampleLevel(color_tex, color_samp, sample_uv, 0.0).rgb;
        let sample_z = view_distance(sample_uv);

        // A sample in FRONT of this pixel only spreads onto it as far as its
        // own circle reaches: a sharp foreground edge must not smear into the
        // blurred background behind it. A sample behind contributes fully.
        var w = 1.0;
        if (sample_z < view_z) {
            w = saturate(circle(sample_z) / coc);
        }
        color_sum += sample_color * w;
        weight_sum += w;
    }
    return vec4<f32>(color_sum / weight_sum, 1.0);
}
";

#[cfg(test)]
mod tests {
    /// The shader's linearisation, in Rust: `B / (d + A)`.
    fn view_distance(d: f32, near: f32, far: f32) -> f32 {
        let proj = crate::renderer::util::mat4_perspective(1.0, 1.6, near, far);
        proj[3][2] / (d + proj[2][2])
    }

    #[test]
    fn depth_linearises_to_the_distance_it_was_drawn_at() {
        let (near, far) = (0.1f32, 1000.0f32);
        let proj = crate::renderer::util::mat4_perspective(1.0, 1.6, near, far);
        for z in [0.25f32, 1.0, 7.5, 40.0, 300.0] {
            // What the rasteriser stores for a point `z` in front of the camera.
            let clip_z = proj[2][2] * -z + proj[3][2];
            let clip_w = z;
            let d = clip_z / clip_w;
            let back = view_distance(d, near, far);
            assert!(
                (back - z).abs() < z * 0.01,
                "depth {d} came back as {back}, drawn at {z}"
            );
        }
    }
}
