// Bloom packed visibility-buffer ABI — version 1.
//
// Rg32Uint stores one full draw ID plus a 31-bit primitive ID and one face
// orientation bit. Barycentrics are reconstructed from the triangle's clip
// positions, avoiding the incorrect 16-byte Rgba32Uint proposal and avoiding
// a per-corner barycentric vertex stream.

const BLOOM_VISIBILITY_INVALID_DRAW_ID: u32 = 0xffffffffu;
const BLOOM_VISIBILITY_VIRTUAL_DRAW_BIT: u32 = 0x80000000u;
const BLOOM_VISIBILITY_DRAW_INDEX_MASK: u32 = 0x7fffffffu;
const BLOOM_VISIBILITY_FRONT_FACE_BIT: u32 = 0x80000000u;
const BLOOM_VISIBILITY_PRIMITIVE_MASK: u32 = 0x7fffffffu;

struct BloomVisibilityRecord {
    draw_id: u32,
    primitive_id: u32,
    front_facing: bool,
    virtual_geometry: bool,
};

fn bloom_visibility_valid(raw: vec2<u32>) -> bool {
    return raw.x != BLOOM_VISIBILITY_INVALID_DRAW_ID;
}

fn bloom_decode_visibility(raw: vec2<u32>) -> BloomVisibilityRecord {
    return BloomVisibilityRecord(
        raw.x & BLOOM_VISIBILITY_DRAW_INDEX_MASK,
        raw.y & BLOOM_VISIBILITY_PRIMITIVE_MASK,
        (raw.y & BLOOM_VISIBILITY_FRONT_FACE_BIT) != 0u,
        (raw.x & BLOOM_VISIBILITY_VIRTUAL_DRAW_BIT) != 0u,
    );
}

fn bloom_encode_virtual_visibility(
    draw_index: u32,
    primitive_id: u32,
    front_facing: bool,
) -> vec2<u32> {
    let face = select(0u, BLOOM_VISIBILITY_FRONT_FACE_BIT, front_facing);
    return vec2<u32>(
        BLOOM_VISIBILITY_VIRTUAL_DRAW_BIT | draw_index,
        primitive_id | face,
    );
}

fn bloom_encode_visibility(
    draw_id: u32,
    primitive_id: u32,
    front_facing: bool,
) -> vec2<u32> {
    let face = select(0u, BLOOM_VISIBILITY_FRONT_FACE_BIT, front_facing);
    return vec2<u32>(draw_id, primitive_id | face);
}

fn bloom_edge(a: vec2<f32>, b: vec2<f32>, point: vec2<f32>) -> f32 {
    let pa = point - a;
    let ba = b - a;
    return pa.x * ba.y - pa.y * ba.x;
}

fn bloom_perspective_barycentrics(
    point_ndc: vec2<f32>,
    clip0: vec4<f32>,
    clip1: vec4<f32>,
    clip2: vec4<f32>,
) -> vec3<f32> {
    let ndc0 = clip0.xy / clip0.w;
    let ndc1 = clip1.xy / clip1.w;
    let ndc2 = clip2.xy / clip2.w;
    let signed_area = bloom_edge(ndc1, ndc2, ndc0);
    let safe_area = select(
        select(-0.000000000001, 0.000000000001, signed_area >= 0.0),
        signed_area,
        abs(signed_area) > 0.000000000001,
    );
    let linear = vec3<f32>(
        bloom_edge(ndc1, ndc2, point_ndc),
        bloom_edge(ndc2, ndc0, point_ndc),
        bloom_edge(ndc0, ndc1, point_ndc),
    ) / safe_area;
    let weighted = linear / vec3<f32>(clip0.w, clip1.w, clip2.w);
    let weighted_sum = weighted.x + weighted.y + weighted.z;
    let safe_sum = select(
        select(-0.000000000001, 0.000000000001, weighted_sum >= 0.0),
        weighted_sum,
        abs(weighted_sum) > 0.000000000001,
    );
    return weighted / safe_sum;
}
