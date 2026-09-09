#[cfg(test)]
pub(super) const VIRTUAL_GEOMETRY_DECODE_WGSL: &str =
    include_str!("../../shaders/virtual_geometry/decode.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_page_decoder_parses_for_float_and_quantized_payloads() {
        let source = [
            "@group(0) @binding(0) var<storage, read> ",
            "virtual_page_words: BloomVirtualRawWords;\n",
            VIRTUAL_GEOMETRY_DECODE_WGSL,
            "\n@compute @workgroup_size(1) fn decode_probe() {\n",
            "  _ = bloom_virtual_decode_vertex(0u, 1u, vec3<f32>(0.0), vec3<f32>(1.0));\n",
            "  _ = bloom_virtual_decode_vertex(0u, 2u, vec3<f32>(0.0), vec3<f32>(1.0));\n",
            "}\n",
        ]
        .concat();
        wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("virtual raw-page decoder WGSL failed: {error:?}"));
    }
}
