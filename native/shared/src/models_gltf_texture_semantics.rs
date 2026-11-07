//! glTF image color-space classification used by material texture upload.

use std::collections::HashSet;

/// Images whose RGB channels are defined as sRGB by glTF. All unlisted
/// material-response images (metallic/roughness, occlusion, transmission,
/// thickness, and scalar layered lobes) are linear data. This distinction is
/// required while building mips even though Bloom performs the eventual sRGB
/// decode explicitly in its material shader.
pub(super) fn srgb_material_image_indices(gltf: &gltf::Gltf) -> HashSet<usize> {
    let mut result = HashSet::new();
    for material in gltf.materials() {
        let pbr = material.pbr_metallic_roughness();
        if let Some(info) = pbr.base_color_texture() {
            result.insert(info.texture().source().index());
        }
        if let Some(info) = material.emissive_texture() {
            result.insert(info.texture().source().index());
        }
        if let Some(spec_gloss) = material.pbr_specular_glossiness() {
            if let Some(info) = spec_gloss.diffuse_texture() {
                result.insert(info.texture().source().index());
            }
            // RGB is sRGB specular colour; alpha is linear glossiness. The
            // color mip builder filters RGB in linear light and alpha as raw
            // data, which is exactly this packed texture's contract.
            if let Some(info) = spec_gloss.specular_glossiness_texture() {
                result.insert(info.texture().source().index());
            }
        }
        if let Some(specular) = material.specular() {
            if let Some(info) = specular.specular_color_texture() {
                result.insert(info.texture().source().index());
            }
        }
        if let Some(texture_index) = material
            .extension_value("KHR_materials_sheen")
            .and_then(serde_json::Value::as_object)
            .and_then(|sheen| sheen.get("sheenColorTexture"))
            .and_then(serde_json::Value::as_object)
            .and_then(|texture| texture.get("index"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
        {
            if let Some(texture) = gltf.textures().nth(texture_index) {
                result.insert(texture.source().index());
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::srgb_material_image_indices;

    #[test]
    fn separates_color_images_from_linear_material_data() {
        let document = br#"{
            "asset":{"version":"2.0"},
            "images":[{"uri":"base.png"},{"uri":"mr.png"},{"uri":"emit.png"},{"uri":"normal.png"}],
            "textures":[{"source":0},{"source":1},{"source":2},{"source":3}],
            "materials":[{"pbrMetallicRoughness":{"baseColorTexture":{"index":0},"metallicRoughnessTexture":{"index":1}},"emissiveTexture":{"index":2},"normalTexture":{"index":3}}]
        }"#;
        let gltf = gltf::Gltf::from_slice(document).expect("minimal glTF parses");
        let srgb = srgb_material_image_indices(&gltf);
        assert_eq!(srgb, [0, 2].into_iter().collect());
    }
}
