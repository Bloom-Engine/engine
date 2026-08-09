pub(in crate::renderer) const SHADER_2D: &str = include_str!("../../../shaders/legacy/2d.wgsl");

pub(in crate::renderer) const SHADER_3D: &str = "
struct Uniforms3D {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    prev_mvp: mat4x4<f32>,
    model_tint: vec4<f32>,
    // x = joint-buffer offset, y = skinned flag (cached skinned draws).
    // Always zero on the immediate path — its verts arrive with joint
    // indices pre-offset CPU-side, so vs_main_3d ignores this field.
    misc: vec4<f32>,
};

struct DirLight {
    direction: vec4<f32>,
    color: vec4<f32>,
};

struct PointLight {
    position: vec4<f32>,
    color: vec4<f32>,
};

struct Lighting {
    ambient: vec4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    dir_light_count: vec4<f32>,
    dir_lights: array<DirLight, 8>,
    point_light_count: vec4<f32>,
    point_lights: array<PointLight, 256>,
};

struct JointMatrices {
    matrices: array<mat4x4<f32>, 1024>,
};

struct VertexInput3D {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) joints: vec4<f32>,
    @location(5) weights: vec4<f32>,
    // Immediate primitives use their otherwise-unused tangent lane for
    // previous world position. w=2 distinguishes it from model tangents.
    @location(6) previous_position: vec4<f32>,
};

struct VertexOutput3D {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_pos: vec3<f32>,
    @location(4) curr_clip: vec4<f32>,
    @location(5) prev_clip: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms3D;
@group(1) @binding(0) var<uniform> lighting: Lighting;
@group(2) @binding(0) var tex3d: texture_2d<f32>;
@group(2) @binding(1) var tex3d_sampler: sampler;
@group(3) @binding(0) var<uniform> joints: JointMatrices; @group(3) @binding(1) var<uniform> joints_prev: JointMatrices;

@vertex
fn vs_main_3d(in: VertexInput3D) -> VertexOutput3D {
    var out: VertexOutput3D;
    let total_weight = in.weights.x + in.weights.y + in.weights.z + in.weights.w;
    var pos = vec4<f32>(in.position, 1.0); var prev_pos = pos;
    var norm = vec4<f32>(in.normal, 0.0);
    if (total_weight > 0.01) {
        let j0 = u32(in.joints.x); let j1 = u32(in.joints.y);
        let j2 = u32(in.joints.z); let j3 = u32(in.joints.w);
        let skinned_pos = joints.matrices[j0] * pos * in.weights.x
                        + joints.matrices[j1] * pos * in.weights.y
                        + joints.matrices[j2] * pos * in.weights.z
                        + joints.matrices[j3] * pos * in.weights.w;
        let skinned_norm = joints.matrices[j0] * norm * in.weights.x
                         + joints.matrices[j1] * norm * in.weights.y
                         + joints.matrices[j2] * norm * in.weights.z
                         + joints.matrices[j3] * norm * in.weights.w;
        prev_pos = joints_prev.matrices[j0] * pos * in.weights.x + joints_prev.matrices[j1] * pos * in.weights.y + joints_prev.matrices[j2] * pos * in.weights.z + joints_prev.matrices[j3] * pos * in.weights.w;
        pos = skinned_pos;
        norm = skinned_norm;
    } else if (in.previous_position.w > 1.5) { prev_pos = vec4<f32>(in.previous_position.xyz, 1.0); } // immediate primitive history
    let curr = u.mvp * pos;
    out.clip_position = curr;
    out.curr_clip = curr;
    out.prev_clip = u.prev_mvp * prev_pos;
    out.normal = normalize((u.model * norm).xyz);
    out.world_pos = (u.model * pos).xyz;
    out.color = in.color * u.model_tint;
    out.uv = in.uv;
    return out;
}

struct Fs3DOut {
    @location(0) color: vec4<f32>,
    @location(1) material: vec2<f32>,
    @location(2) velocity: vec2<f32>,
    @location(3) albedo: vec4<f32>,
};

@fragment
fn fs_main_3d(in: VertexOutput3D) -> Fs3DOut {
    let n = normalize(in.normal);

    // Ambient
    var lit = lighting.ambient.rgb * lighting.ambient.a;

    // Legacy directional light (backward compat)
    let legacy_dir = normalize(lighting.light_dir.xyz);
    let legacy_diffuse = max(dot(n, legacy_dir), 0.0);
    lit += lighting.light_color.rgb * lighting.light_dir.w * legacy_diffuse;

    // Additional directional lights
    let dir_count = u32(lighting.dir_light_count.x);
    for (var i = 0u; i < dir_count; i++) {
        let dl = lighting.dir_lights[i];
        let dir = normalize(dl.direction.xyz);
        let diff = max(dot(n, dir), 0.0);
        lit += dl.color.rgb * dl.direction.w * diff;
    }

    // Point lights
    let pt_count = u32(lighting.point_light_count.x);
    for (var i = 0u; i < pt_count; i++) {
        let pl = lighting.point_lights[i];
        let to_light = pl.position.xyz - in.world_pos;
        let dist = length(to_light);
        let range = pl.position.w;
        if (dist < range) {
            let dir = to_light / dist;
            let diff = max(dot(n, dir), 0.0);
            let atten = 1.0 - (dist / range);
            let atten2 = atten * atten;
            lit += pl.color.rgb * pl.color.w * diff * atten2;
        }
    }

    let tex_color = textureSample(tex3d, tex3d_sampler, in.uv);
    // Per-pixel velocity for motion blur / TAA reprojection.
    let curr_ndc = in.curr_clip.xy / in.curr_clip.w;
    let prev_ndc = in.prev_clip.xy / in.prev_clip.w;
    let vel = (curr_ndc - prev_ndc) * 0.5;
    // Immediate-mode 3D draws (drawCube etc.) aren't PBR — output
    // 0 metallic / 1 roughness so SSR doesn't try to reflect them.
    //
    // Alpha comes from the TINT only. Game textures routinely carry a
    // non-opacity alpha channel (Unvanquished armor packs a gloss mask
    // there), and this batch also renders CPU-skinned characters — the
    // player turned semi-transparent through its gloss mask when texture
    // alpha fed the blend. Deliberate fades still work via tint alpha;
    // untextured effect quads bind the white texture (alpha 1) anyway.
    return Fs3DOut(
        vec4<f32>(tex_color.rgb * in.color.rgb * lit, in.color.a),
        vec2<f32>(0.0, 1.0),
        vel,
        vec4<f32>(0.0),
    );
}
";
