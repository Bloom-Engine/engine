//! Custom WGSL shaders for Pascal Editor effects.
//!
//! These are hand-ported equivalents of the Three.js TSL (Texture Shading Language)
//! node materials used in the Pascal Editor:
//! - Wall cutaway dot pattern
//! - Zone gradient
//! - Item preview shimmer
//!
//! In the browser, these are defined as JavaScript function calls (TSL nodes) that
//! compile to WGSL at runtime. Since we compile natively, we write the WGSL directly.

/// Wall cutaway dot pattern — shows dots that fade with height when a wall
/// is in "cutaway" mode. Replaces the TSL `Fn(() => { ... fract ... step ... })`.
pub const WALL_CUTAWAY_SHADER: &str = "
struct CutawayParams {
    dot_scale: vec4<f32>,      // [scale, dot_size, fade_height, opacity]
    base_color: vec4<f32>,     // rgba
};

@group(0) @binding(0) var<uniform> params: CutawayParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@fragment
fn fs_wall_cutaway(in: VertexOutput) -> @location(0) vec4<f32> {
    let scale = params.dot_scale.x;
    let dot_size = params.dot_scale.y;
    let fade_height = params.dot_scale.z;

    // Create repeating dot grid
    let uv = vec2<f32>(in.local_pos.x, in.local_pos.y) / scale;
    let grid_uv = fract(uv);
    let dist = length(grid_uv - vec2<f32>(0.5));
    let dots = step(dist, dot_size * 0.5);

    // Vertical fade (transparent near bottom, opaque at top)
    let y_fade = 1.0 - smoothstep(0.0, fade_height, in.local_pos.y);

    let alpha = dots * y_fade * params.dot_scale.w;
    if (alpha < 0.01) { discard; }

    return vec4<f32>(params.base_color.rgb * in.color.rgb, alpha);
}
";

/// Zone gradient — vertical gradient for zone visualization walls.
/// Replaces the TSL `uv().y → mul(opacity)` with uniform.
pub const ZONE_GRADIENT_SHADER: &str = "
struct ZoneParams {
    base_color: vec4<f32>,    // rgba
    opacity: vec4<f32>,       // [max_opacity, 0, 0, 0]
};

@group(0) @binding(0) var<uniform> zone_params: ZoneParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@fragment
fn fs_zone_gradient(in: VertexOutput) -> @location(0) vec4<f32> {
    // Gradient: fully opaque at bottom, transparent at top
    let gradient_t = in.uv.y;
    let alpha = zone_params.opacity.x * (1.0 - gradient_t) * 0.6;
    return vec4<f32>(zone_params.base_color.rgb, alpha);
}
";

/// Item preview shimmer — animated shimmer effect for item placement preview.
/// Replaces the TSL `positionLocal.y + time → fract → smoothstep`.
pub const PREVIEW_SHIMMER_SHADER: &str = "
struct ShimmerParams {
    time: vec4<f32>,          // [time, speed, 0, 0]
    base_color: vec4<f32>,    // rgba
};

@group(0) @binding(0) var<uniform> shimmer_params: ShimmerParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@fragment
fn fs_preview_shimmer(in: VertexOutput) -> @location(0) vec4<f32> {
    let time = shimmer_params.time.x;
    let speed = shimmer_params.time.y;

    // Scrolling shimmer band
    let shimmer_y = fract((in.local_pos.y + time * speed) * 10.0);
    let shimmer = smoothstep(0.0, 0.1, shimmer_y) * smoothstep(0.3, 0.2, shimmer_y);

    let base = shimmer_params.base_color;
    let brightness = 1.0 + shimmer * 0.3;
    return vec4<f32>(base.rgb * brightness, base.a * 0.7);
}
";
