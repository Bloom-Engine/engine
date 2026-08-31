// Bloom cooked virtual-geometry vertex decoder — version 1.
//
// Consumers provide a storage binding named `virtual_page_words`. Runtime
// validation guarantees every requested byte lies inside its resident page;
// the decode path therefore performs only address arithmetic needed by the
// raster/visibility workload.

struct BloomVirtualRawWords { values: array<u32>, };

struct BloomVirtualVertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    tangent: vec4<f32>,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    color: vec4<f32>,
};

fn bloom_virtual_load_u8(byte_offset: u32) -> u32 {
    let word = virtual_page_words.values[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 0xffu;
}

fn bloom_virtual_load_u16(byte_offset: u32) -> u32 {
    return bloom_virtual_load_u8(byte_offset)
        | (bloom_virtual_load_u8(byte_offset + 1u) << 8u);
}

fn bloom_virtual_load_i16(byte_offset: u32) -> i32 {
    let raw = bloom_virtual_load_u16(byte_offset);
    let extended = select(raw, raw | 0xffff0000u, (raw & 0x8000u) != 0u);
    return bitcast<i32>(extended);
}

fn bloom_virtual_load_f32(byte_offset: u32) -> f32 {
    return bitcast<f32>(virtual_page_words.values[byte_offset >> 2u]);
}

fn bloom_virtual_load_f16(byte_offset: u32) -> f32 {
    return unpack2x16float(bloom_virtual_load_u16(byte_offset)).x;
}

fn bloom_virtual_decode_snorm16(value: i32) -> f32 {
    return max(f32(value) / 32767.0, -1.0);
}

fn bloom_virtual_sign_not_zero(value: f32) -> f32 {
    return select(1.0, -1.0, value < 0.0);
}

fn bloom_virtual_decode_octahedral(encoded: vec2<i32>) -> vec3<f32> {
    let x = bloom_virtual_decode_snorm16(encoded.x);
    let y = bloom_virtual_decode_snorm16(encoded.y);
    var normal = vec3<f32>(x, y, 1.0 - abs(x) - abs(y));
    if (normal.z < 0.0) {
        let old_x = normal.x;
        normal.x = (1.0 - abs(normal.y)) * bloom_virtual_sign_not_zero(old_x);
        normal.y = (1.0 - abs(old_x)) * bloom_virtual_sign_not_zero(normal.y);
    }
    return normalize(normal);
}

fn bloom_virtual_decode_float32_vertex(byte_offset: u32) -> BloomVirtualVertex {
    return BloomVirtualVertex(
        vec3<f32>(
            bloom_virtual_load_f32(byte_offset),
            bloom_virtual_load_f32(byte_offset + 4u),
            bloom_virtual_load_f32(byte_offset + 8u),
        ),
        vec3<f32>(
            bloom_virtual_load_f32(byte_offset + 12u),
            bloom_virtual_load_f32(byte_offset + 16u),
            bloom_virtual_load_f32(byte_offset + 20u),
        ),
        vec4<f32>(
            bloom_virtual_load_f32(byte_offset + 24u),
            bloom_virtual_load_f32(byte_offset + 28u),
            bloom_virtual_load_f32(byte_offset + 32u),
            bloom_virtual_load_f32(byte_offset + 36u),
        ),
        vec2<f32>(
            bloom_virtual_load_f32(byte_offset + 40u),
            bloom_virtual_load_f32(byte_offset + 44u),
        ),
        vec2<f32>(
            bloom_virtual_load_f32(byte_offset + 48u),
            bloom_virtual_load_f32(byte_offset + 52u),
        ),
        vec4<f32>(
            bloom_virtual_load_f32(byte_offset + 56u),
            bloom_virtual_load_f32(byte_offset + 60u),
            bloom_virtual_load_f32(byte_offset + 64u),
            bloom_virtual_load_f32(byte_offset + 68u),
        ),
    );
}

fn bloom_virtual_decode_quantized_vertex(
    byte_offset: u32,
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
) -> BloomVirtualVertex {
    let unorm_scale = 1.0 / 65535.0;
    let encoded_position = vec3<f32>(
        f32(bloom_virtual_load_u16(byte_offset)),
        f32(bloom_virtual_load_u16(byte_offset + 2u)),
        f32(bloom_virtual_load_u16(byte_offset + 4u)),
    ) * unorm_scale;
    let position = bounds_min + (bounds_max - bounds_min) * encoded_position;
    let normal = bloom_virtual_decode_octahedral(vec2<i32>(
        bloom_virtual_load_i16(byte_offset + 6u),
        bloom_virtual_load_i16(byte_offset + 8u),
    ));
    let flags = bloom_virtual_load_u16(byte_offset + 28u);
    var tangent = vec4<f32>(0.0);
    if ((flags & 1u) != 0u) {
        tangent = vec4<f32>(
            bloom_virtual_decode_octahedral(vec2<i32>(
                bloom_virtual_load_i16(byte_offset + 10u),
                bloom_virtual_load_i16(byte_offset + 12u),
            )),
            bloom_virtual_decode_snorm16(bloom_virtual_load_i16(byte_offset + 26u)),
        );
    }
    return BloomVirtualVertex(
        position,
        normal,
        tangent,
        vec2<f32>(
            bloom_virtual_load_f16(byte_offset + 14u),
            bloom_virtual_load_f16(byte_offset + 16u),
        ),
        vec2<f32>(
            bloom_virtual_load_f16(byte_offset + 18u),
            bloom_virtual_load_f16(byte_offset + 20u),
        ),
        vec4<f32>(
            f32(bloom_virtual_load_u8(byte_offset + 22u)),
            f32(bloom_virtual_load_u8(byte_offset + 23u)),
            f32(bloom_virtual_load_u8(byte_offset + 24u)),
            f32(bloom_virtual_load_u8(byte_offset + 25u)),
        ) / 255.0,
    );
}

fn bloom_virtual_decode_vertex(
    byte_offset: u32,
    encoding: u32,
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
) -> BloomVirtualVertex {
    if (encoding == 1u) {
        return bloom_virtual_decode_float32_vertex(byte_offset);
    }
    if (encoding == 2u) {
        return bloom_virtual_decode_quantized_vertex(byte_offset, bounds_min, bounds_max);
    }
    return BloomVirtualVertex(
        vec3<f32>(0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec4<f32>(0.0),
        vec2<f32>(0.0),
        vec2<f32>(0.0),
        vec4<f32>(0.0),
    );
}

fn bloom_virtual_load_local_index(index_byte_offset: u32) -> u32 {
    return bloom_virtual_load_u8(index_byte_offset);
}
