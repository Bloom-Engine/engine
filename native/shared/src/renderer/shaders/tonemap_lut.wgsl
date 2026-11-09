// The AgX curve as a 3D table (renderer/tonemap_lut.rs): the axes are the HDR
// colour's channels in log2, over the range below, one texel at each end.
// Below the range AgX is black (it clamps at 2^-12.47); above it, it is white
// (its inset matrix saturates every channel once any exceeds about 2^8.6).
// Trilinear filtering stays within 1/255 of the curve in mid-tones and 2/255
// anywhere, display-encoded (tonemap_lut.rs tests).
const TONEMAP_LUT_SIZE: f32 = 64.0;
const TONEMAP_LUT_LOG2_MIN: f32 = -12.5;
const TONEMAP_LUT_LOG2_MAX: f32 = 9.0;

// Texture coordinate of an HDR colour.
fn tonemap_lut_uvw(hdr: vec3<f32>) -> vec3<f32> {
    let s = clamp(
        (log2(max(hdr, vec3<f32>(1e-30))) - TONEMAP_LUT_LOG2_MIN) /
            (TONEMAP_LUT_LOG2_MAX - TONEMAP_LUT_LOG2_MIN),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    return (s * (TONEMAP_LUT_SIZE - 1.0) + 0.5) / TONEMAP_LUT_SIZE;
}

// The HDR colour a texel holds.
fn tonemap_lut_hdr(texel: vec3<u32>) -> vec3<f32> {
    let s = vec3<f32>(texel) / (TONEMAP_LUT_SIZE - 1.0);
    return exp2(mix(vec3<f32>(TONEMAP_LUT_LOG2_MIN), vec3<f32>(TONEMAP_LUT_LOG2_MAX), s));
}
