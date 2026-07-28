struct DirectionalVsmParams {
    level_vps: array<mat4x4<f32>, 3>,
    enabled: u32,
    virtual_pages_per_axis: u32,
    page_interior: u32,
    page_border: u32,
};
@group(1) @binding(13) var vsm_page_table: texture_2d_array<u32>;
@group(1) @binding(14) var vsm_physical_pages: texture_depth_2d_array;
@group(1) @binding(15) var<uniform> vsm_params: DirectionalVsmParams;
