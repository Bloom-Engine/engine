fn sample_virtual_shadow(
    cascade: i32,
    world_pos: vec3<f32>,
    csm_uv: vec2<f32>,
    csm_depth_ref: f32,
) -> f32 {
    if (vsm_params.words.x == 0u) {
        return sample_cascade(cascade, csm_uv, csm_depth_ref);
    }
    let light_clip = vsm_params.level_vps[cascade] * vec4<f32>(world_pos, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2<f32>(
        light_ndc.x * 0.5 + 0.5,
        1.0 - (light_ndc.y * 0.5 + 0.5),
    );
    if (any(shadow_uv < vec2<f32>(0.0)) || any(shadow_uv > vec2<f32>(1.0))
        || light_ndc.z < 0.0 || light_ndc.z > 1.0) {
        return sample_cascade(cascade, csm_uv, csm_depth_ref);
    }
    let depth_ref = light_ndc.z - 0.001;
    let axis = vsm_params.words.y;
    let scaled_uv = shadow_uv * f32(axis);
    let page_xy = min(vec2<u32>(scaled_uv), vec2<u32>(axis - 1u));
    let encoded = textureLoad(
        vsm_page_table,
        vec2<i32>(page_xy),
        cascade,
        0,
    ).x;
    if (encoded == 0u) {
        return sample_cascade(cascade, csm_uv, csm_depth_ref);
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
            sample_cascade(cascade, csm_uv, csm_depth_ref),
            virtual_value,
            residency_age / 8.0,
        );
    }
    return virtual_value;
}

fn local_shadow_face(direction: vec3<f32>) -> u32 {
    let absolute = abs(direction);
    if (absolute.x >= absolute.y && absolute.x >= absolute.z) {
        return select(1u, 0u, direction.x >= 0.0);
    }
    if (absolute.y >= absolute.z) {
        return select(3u, 2u, direction.y >= 0.0);
    }
    return select(5u, 4u, direction.z >= 0.0);
}

fn local_shadow_page(slot: LocalVsmSamplingSlot, face: u32) -> u32 {
    if (face < 4u) {
        return slot.face_pages_0_3[face];
    }
    return slot.face_pages_4_5[face - 4u];
}

// Metadata x states are deliberately fail-closed:
//   0 = ordinary point light (unshadowed),
//   1 = shadow requested but not fully resident (suppress contribution),
//   2..6 = fully resident local slot + 2.
// Metadata y is 0 for a point cube and 1 for a spot projection. For spots,
// metadata z stores the inner/outer projected-radius ratio as f32 bits.
fn sample_local_shadow(light_index: u32, world_pos: vec3<f32>) -> f32 {
    // words.x bit 1 is uniform for the whole draw. Keep the established
    // directional-only VSM path out of the dynamically indexed metadata
    // array when no local-shadow request was admitted this frame.
    if ((vsm_params.words.x & 2u) == 0u) {
        return 1.0;
    }
    let metadata = vsm_params.local_light_meta[light_index];
    let state = metadata.x;
    if (state == 0u) {
        return 1.0;
    }
    if (state == 1u) {
        return 0.0;
    }
    let slot = vsm_params.local_slots[state - 2u];
    let direction = world_pos - lighting.point_lights[light_index].position.xyz;
    let face = select(local_shadow_face(direction), 0u, metadata.y == 1u);
    let encoded = local_shadow_page(slot, face);
    if (encoded == 0u) {
        return 0.0;
    }
    let light_clip = slot.face_vps[face] * vec4<f32>(world_pos, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;
    if (abs(light_ndc.x) > 1.0 || abs(light_ndc.y) > 1.0
        || light_ndc.z < 0.0 || light_ndc.z > 1.0) {
        return 0.0;
    }
    var cone_attenuation = 1.0;
    if (metadata.y == 1u) {
        let projected_radius = length(light_ndc.xy);
        if (projected_radius > 1.0) {
            return 0.0;
        }
        let inner_radius = bitcast<f32>(metadata.z);
        // Equal inner/outer angles author a hard-edged cone. Avoid passing
        // equal edges to smoothstep (undefined by WGSL) while preserving
        // the exact outer boundary.
        if (inner_radius >= 1.0) {
            cone_attenuation = select(0.0, 1.0, projected_radius < 1.0);
        } else {
            cone_attenuation = 1.0 - smoothstep(
                inner_radius,
                1.0,
                projected_radius,
            );
        }
    }
    let shadow_uv = vec2<f32>(
        light_ndc.x * 0.5 + 0.5,
        1.0 - (light_ndc.y * 0.5 + 0.5),
    );
    let interior = f32(vsm_params.words.z);
    let border = f32(vsm_params.words.w);
    let physical_size = interior + 2.0 * border;
    let page_uv = (vec2<f32>(border) + shadow_uv * interior) / physical_size;
    let texel = vec2<f32>(1.0 / physical_size);
    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>( 0.5,  0.5),
    );
    let physical_layer = i32((encoded & 0xffffu) - 1u);
    let depth_bias = max(0.0005, 0.002 * (1.0 - light_ndc.z));
    var shadow_value = 0.0;
    for (var tap = 0; tap < 4; tap = tap + 1) {
        shadow_value += textureSampleCompareLevel(
            vsm_physical_pages,
            shadow_samp,
            page_uv + offsets[tap] * texel,
            physical_layer,
            light_ndc.z - depth_bias,
        );
    }
    shadow_value *= 0.25;
    // New pages fade in from the fail-closed state, never from an
    // unshadowed local light.
    return shadow_value * cone_attenuation
        * min(f32(encoded >> 16u) / 8.0, 1.0);
}
