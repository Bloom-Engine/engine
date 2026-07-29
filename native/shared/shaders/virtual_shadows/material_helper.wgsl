fn sample_shadow_cascade(
    cascade_idx: u32,
    world_pos: vec3<f32>,
) -> f32 {
    if (vsm_params.words.x == 0u) {
        return sample_shadow_cascade_csm(cascade_idx, world_pos);
    }
    let light_clip = vsm_params.level_vps[cascade_idx]
                   * vec4<f32>(world_pos, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;
    if (abs(light_ndc.x) > 1.0 || abs(light_ndc.y) > 1.0
        || light_ndc.z < 0.0 || light_ndc.z > 1.0) {
        return sample_shadow_cascade_csm(cascade_idx, world_pos);
    }
    let shadow_uv = vec2<f32>(
        light_ndc.x * 0.5 + 0.5,
        1.0 - (light_ndc.y * 0.5 + 0.5),
    );
    let depth_ref = light_ndc.z - 0.001;
    let axis = vsm_params.words.y;
    let scaled_uv = shadow_uv * f32(axis);
    let page_xy = min(vec2<u32>(scaled_uv), vec2<u32>(axis - 1u));
    let encoded = textureLoad(
        vsm_page_table,
        vec2<i32>(page_xy),
        i32(cascade_idx),
        0,
    ).x;
    if (encoded == 0u) {
        return sample_shadow_cascade_csm(cascade_idx, world_pos);
    }
    let physical_layer = i32((encoded & 0xffffu) - 1u);
    let interior = f32(vsm_params.words.z);
    let border = f32(vsm_params.words.w);
    let physical_size = interior + 2.0 * border;
    let local_uv = clamp(
        scaled_uv - vec2<f32>(page_xy),
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );
    let page_uv = (vec2<f32>(border) + local_uv * interior) / physical_size;
    let texel = vec2<f32>(1.0 / physical_size);
    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>( 0.5,  0.5),
    );
    var virtual_value = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        virtual_value += textureSampleCompareLevel(
            vsm_physical_pages,
            shadow_samp,
            page_uv + offsets[i] * texel,
            physical_layer,
            depth_ref,
        );
    }
    virtual_value *= 0.25;
    let residency_age = f32(encoded >> 16u);
    if (residency_age < 8.0) {
        return mix(
            sample_shadow_cascade_csm(cascade_idx, world_pos),
            virtual_value,
            residency_age / 8.0,
        );
    }
    return virtual_value;
}
