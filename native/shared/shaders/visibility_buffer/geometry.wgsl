// Bloom shared-geometry storage ABI for visibility shading — version 1.
//
// Vertex3D is a tightly packed 96-byte Rust/vertex-buffer record. WGSL
// storage structs give vec3 values 16-byte alignment, so spelling the record
// with native vec3 fields would silently read the wrong offsets. Six vec4<u32>
// lanes preserve the exact 24-word byte layout and are decoded explicitly.

const BLOOM_VERTEX3D_WORDS: u32 = 24u;

struct BloomPackedVertex3D {
    words_0: vec4<u32>,
    words_1: vec4<u32>,
    words_2: vec4<u32>,
    words_3: vec4<u32>,
    words_4: vec4<u32>,
    words_5: vec4<u32>,
};

struct BloomVertex3D {
    position: vec3<f32>,
    normal: vec3<f32>,
    color: vec4<f32>,
    uv: vec2<f32>,
    joints: vec4<f32>,
    weights: vec4<f32>,
    tangent: vec4<f32>,
};

fn bloom_decode_vertex3d(packed: BloomPackedVertex3D) -> BloomVertex3D {
    return BloomVertex3D(
        bitcast<vec3<f32>>(packed.words_0.xyz),
        bitcast<vec3<f32>>(vec3<u32>(
            packed.words_0.w,
            packed.words_1.x,
            packed.words_1.y,
        )),
        bitcast<vec4<f32>>(vec4<u32>(packed.words_1.zw, packed.words_2.xy)),
        bitcast<vec2<f32>>(packed.words_2.zw),
        bitcast<vec4<f32>>(packed.words_3),
        bitcast<vec4<f32>>(packed.words_4),
        bitcast<vec4<f32>>(packed.words_5),
    );
}

fn bloom_interpolate2(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, bary: vec3<f32>) -> vec2<f32> {
    return a * bary.x + b * bary.y + c * bary.z;
}

fn bloom_interpolate3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, bary: vec3<f32>) -> vec3<f32> {
    return a * bary.x + b * bary.y + c * bary.z;
}

fn bloom_interpolate4(a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, bary: vec3<f32>) -> vec4<f32> {
    return a * bary.x + b * bary.y + c * bary.z;
}
