struct LocalVsmSamplingSlot {
    face_vps: array<mat4x4<f32>, 6>,
    face_pages_0_3: vec4<u32>,
    face_pages_4_5: vec4<u32>,
};
struct DirectionalVsmParams {
    level_vps: array<mat4x4<f32>, 3>,
    words: vec4<u32>,
    local_light_meta: array<vec4<u32>, 256>,
    local_slots: array<LocalVsmSamplingSlot, 5>,
};
@group(1) @binding(13) var vsm_page_table: texture_2d_array<u32>;
@group(1) @binding(14) var vsm_physical_pages: texture_depth_2d_array;
@group(1) @binding(15) var<uniform> vsm_params: DirectionalVsmParams;
