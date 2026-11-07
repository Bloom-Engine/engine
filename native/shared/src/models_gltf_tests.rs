use super::*;
use crate::models::{
    MaterialAlphaMode, MaterialLayeredPbr, MaterialThicknessSource, MaterialTransmission,
};

fn minimal_triangle_glb(material: &str) -> Vec<u8> {
    minimal_triangle_glb_with_node_scale(material, None)
}

#[test]
fn non_finite_optional_tangents_become_the_missing_tangent_sentinel() {
    assert_eq!(
        sanitize_imported_tangent([1.0, 0.0, 0.0, -1.0]),
        [1.0, 0.0, 0.0, -1.0]
    );
    for invalid in [
        [f32::NAN, f32::NAN, f32::NAN, 0.0],
        [1.0, f32::INFINITY, 0.0, 1.0],
    ] {
        assert_eq!(sanitize_imported_tangent(invalid), [0.0; 4]);
    }
}

fn minimal_triangle_glb_with_node_scale(material: &str, node_scale: Option<[f32; 3]>) -> Vec<u8> {
    let mut binary = Vec::new();
    for value in [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        binary.extend_from_slice(&value.to_le_bytes());
    }
    for tangent in [
        [1.0_f32, 0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, -1.0],
    ] {
        for value in tangent {
            binary.extend_from_slice(&value.to_le_bytes());
        }
    }
    for index in [0_u16, 1, 2] {
        binary.extend_from_slice(&index.to_le_bytes());
    }
    let binary_byte_length = binary.len();
    while binary.len() % 4 != 0 {
        binary.push(0);
    }

    let node = node_scale.map_or_else(
        || r#"{"mesh":0}"#.to_string(),
        |scale| {
            format!(
                r#"{{"mesh":0,"scale":[{},{},{}]}}"#,
                scale[0], scale[1], scale[2]
            )
        },
    );
    let mut json = format!(
        r#"{{
            "asset":{{"version":"2.0"}},
            "scene":0,
            "scenes":[{{"nodes":[0]}}],
            "nodes":[{node}],
            "extensionsUsed":[
                "KHR_materials_transmission",
                "KHR_materials_volume",
                "KHR_materials_ior",
                "KHR_materials_clearcoat",
                "KHR_materials_specular",
                "KHR_materials_sheen",
                "KHR_materials_anisotropy"
            ],
            "buffers":[{{"byteLength":{binary_byte_length}}}],
            "bufferViews":[
                {{"buffer":0,"byteOffset":0,"byteLength":36,"target":34962}},
                {{"buffer":0,"byteOffset":36,"byteLength":48,"target":34962}},
                {{"buffer":0,"byteOffset":84,"byteLength":6,"target":34963}}
            ],
            "accessors":[
                {{
                    "bufferView":0,
                    "componentType":5126,
                    "count":3,
                    "type":"VEC3",
                    "min":[0,0,0],
                    "max":[1,1,0]
                }},
                {{"bufferView":1,"componentType":5126,"count":3,"type":"VEC4"}},
                {{"bufferView":2,"componentType":5123,"count":3,"type":"SCALAR"}}
            ],
            "materials":[{material}],
            "meshes":[{{"primitives":[{{
                "attributes":{{"POSITION":0,"TANGENT":1}},
                "indices":2,
                "material":0
            }}]}}]
        }}"#
    )
    .into_bytes();
    while json.len() % 4 != 0 {
        json.push(b' ');
    }

    let total_length = 12 + 8 + json.len() + 8 + binary.len();
    let mut glb = Vec::with_capacity(total_length);
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2_u32.to_le_bytes());
    glb.extend_from_slice(&(total_length as u32).to_le_bytes());
    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes());
    glb.extend_from_slice(&json);
    glb.extend_from_slice(&(binary.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E_4942_u32.to_le_bytes());
    glb.extend_from_slice(&binary);
    glb
}

fn minimal_instanced_triangle_glb() -> Vec<u8> {
    let source = minimal_triangle_glb(r#"{}"#);
    let json_len = u32::from_le_bytes(source[12..16].try_into().unwrap()) as usize;
    let json = String::from_utf8(source[20..20 + json_len].to_vec()).unwrap();
    let json = json
        .trim_end()
        .replace(
            r#""scenes":[{"nodes":[0]}]"#,
            r#""scenes":[{"nodes":[0,1]}]"#,
        )
        .replace(
            r#""nodes":[{"mesh":0}]"#,
            r#""nodes":[{"mesh":0},{"mesh":0,"translation":[10,0,0]}]"#,
        );
    let binary_chunk = &source[20 + json_len..];
    let mut json = json.into_bytes();
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let total_length = 12 + 8 + json.len() + binary_chunk.len();
    let mut glb = Vec::with_capacity(total_length);
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2_u32.to_le_bytes());
    glb.extend_from_slice(&(total_length as u32).to_le_bytes());
    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes());
    glb.extend_from_slice(&json);
    glb.extend_from_slice(binary_chunk);
    glb
}

fn physical_uv_triangle_glb(texture_tex_coord: u32, include_texcoord_1: bool) -> Vec<u8> {
    let mut binary = Vec::new();
    for value in [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        binary.extend_from_slice(&value.to_le_bytes());
    }
    for uv in [[0.0_f32, 0.0], [1.0, 0.0], [0.0, 1.0]] {
        binary.extend_from_slice(&uv[0].to_le_bytes());
        binary.extend_from_slice(&uv[1].to_le_bytes());
    }
    if include_texcoord_1 {
        for uv in [[0.2_f32, 0.3], [0.8, 0.3], [0.2, 0.9]] {
            binary.extend_from_slice(&uv[0].to_le_bytes());
            binary.extend_from_slice(&uv[1].to_le_bytes());
        }
    }
    let index_offset = binary.len();
    for index in [0_u16, 1, 2] {
        binary.extend_from_slice(&index.to_le_bytes());
    }
    let binary_byte_length = binary.len();
    while binary.len() % 4 != 0 {
        binary.push(0);
    }

    let uv1_view = if include_texcoord_1 {
        r#",{"buffer":0,"byteOffset":60,"byteLength":24,"target":34962}"#
    } else {
        ""
    };
    let uv1_accessor = if include_texcoord_1 {
        r#",{"bufferView":2,"componentType":5126,"count":3,"type":"VEC2"}"#
    } else {
        ""
    };
    let index_view = if include_texcoord_1 { 3 } else { 2 };
    let index_accessor = if include_texcoord_1 { 3 } else { 2 };
    let uv1_attribute = if include_texcoord_1 {
        r#","TEXCOORD_1":2"#
    } else {
        ""
    };
    let mut json = format!(
        r#"{{
            "asset":{{"version":"2.0"}},
            "extensionsUsed":["KHR_materials_transmission","KHR_materials_volume"],
            "buffers":[{{"byteLength":{binary_byte_length}}}],
            "bufferViews":[
                {{"buffer":0,"byteOffset":0,"byteLength":36,"target":34962}},
                {{"buffer":0,"byteOffset":36,"byteLength":24,"target":34962}}
                {uv1_view},
                {{"buffer":0,"byteOffset":{index_offset},"byteLength":6,"target":34963}}
            ],
            "accessors":[
                {{
                    "bufferView":0,
                    "componentType":5126,
                    "count":3,
                    "type":"VEC3",
                    "min":[0,0,0],
                    "max":[1,1,0]
                }},
                {{"bufferView":1,"componentType":5126,"count":3,"type":"VEC2"}}
                {uv1_accessor},
                {{"bufferView":{index_view},"componentType":5123,"count":3,"type":"SCALAR"}}
            ],
            "images":[{{"uri":"glass.png"}}],
            "textures":[{{"source":0}}],
            "materials":[{{
                "extensions":{{
                    "KHR_materials_transmission":{{
                        "transmissionFactor":1.0,
                        "transmissionTexture":{{
                            "index":0,
                            "texCoord":{texture_tex_coord}
                        }}
                    }},
                    "KHR_materials_volume":{{
                        "thicknessFactor":0.25,
                        "thicknessTexture":{{"index":0,"texCoord":0}}
                    }}
                }}
            }}],
            "meshes":[{{"primitives":[{{
                "attributes":{{"POSITION":0,"TEXCOORD_0":1{uv1_attribute}}},
                "indices":{index_accessor},
                "material":0
            }}]}}]
        }}"#
    )
    .into_bytes();
    while json.len() % 4 != 0 {
        json.push(b' ');
    }

    let total_length = 12 + 8 + json.len() + 8 + binary.len();
    let mut glb = Vec::with_capacity(total_length);
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2_u32.to_le_bytes());
    glb.extend_from_slice(&(total_length as u32).to_le_bytes());
    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes());
    glb.extend_from_slice(&json);
    glb.extend_from_slice(&(binary.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E_4942_u32.to_le_bytes());
    glb.extend_from_slice(&binary);
    glb
}

fn textured_mask_triangle_glb() -> Vec<u8> {
    let mut binary = Vec::new();
    for value in [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        binary.extend_from_slice(&value.to_le_bytes());
    }
    for _ in 0..3 {
        for value in [0.2_f32, 0.4, 0.6, 0.5] {
            binary.extend_from_slice(&value.to_le_bytes());
        }
    }
    for index in [0_u16, 1, 2] {
        binary.extend_from_slice(&index.to_le_bytes());
    }
    while binary.len() % 4 != 0 {
        binary.push(0);
    }
    let image_offset = binary.len();
    let image = image::RgbaImage::from_raw(
        2,
        2,
        vec![
            20, 200, 30, 255, 255, 0, 255, 0, 20, 200, 30, 255, 255, 0, 255, 0,
        ],
    )
    .unwrap();
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    let png = png.into_inner();
    let image_length = png.len();
    binary.extend_from_slice(&png);
    let binary_byte_length = binary.len();
    while binary.len() % 4 != 0 {
        binary.push(0);
    }

    let mut json = format!(
        r#"{{
            "asset":{{"version":"2.0"}},
            "buffers":[{{"byteLength":{binary_byte_length}}}],
            "bufferViews":[
                {{"buffer":0,"byteOffset":0,"byteLength":36,"target":34962}},
                {{"buffer":0,"byteOffset":36,"byteLength":48,"target":34962}},
                {{"buffer":0,"byteOffset":84,"byteLength":6,"target":34963}},
                {{"buffer":0,"byteOffset":{image_offset},"byteLength":{image_length}}}
            ],
            "accessors":[
                {{
                    "bufferView":0,
                    "componentType":5126,
                    "count":3,
                    "type":"VEC3",
                    "min":[0,0,0],
                    "max":[1,1,0]
                }},
                {{"bufferView":1,"componentType":5126,"count":3,"type":"VEC4"}},
                {{"bufferView":2,"componentType":5123,"count":3,"type":"SCALAR"}}
            ],
            "images":[{{"bufferView":3,"mimeType":"image/png"}}],
            "textures":[{{"source":0}}],
            "materials":[{{
                "alphaMode":"MASK",
                "alphaCutoff":0.5,
                "pbrMetallicRoughness":{{
                    "baseColorFactor":[0.5,0.25,0.75,0.8],
                    "baseColorTexture":{{"index":0}}
                }}
            }}],
            "meshes":[{{"primitives":[{{
                "attributes":{{"POSITION":0,"COLOR_0":1}},
                "indices":2,
                "material":0
            }}]}}]
        }}"#
    )
    .into_bytes();
    while json.len() % 4 != 0 {
        json.push(b' ');
    }

    let total_length = 12 + 8 + json.len() + 8 + binary.len();
    let mut glb = Vec::with_capacity(total_length);
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2_u32.to_le_bytes());
    glb.extend_from_slice(&(total_length as u32).to_le_bytes());
    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes());
    glb.extend_from_slice(&json);
    glb.extend_from_slice(&(binary.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E_4942_u32.to_le_bytes());
    glb.extend_from_slice(&binary);
    glb
}

#[test]
fn gltf_alpha_modes_remain_distinct() {
    let source = br#"{
      "asset":{"version":"2.0"},
      "materials":[
        {"name":"opaque","alphaMode":"OPAQUE"},
        {"name":"mask","alphaMode":"MASK","alphaCutoff":0.37},
        {"name":"blend","alphaMode":"BLEND","doubleSided":true}
      ]
    }"#;
    let document = gltf::Gltf::from_slice(source).unwrap();
    let materials: Vec<_> = document.materials().collect();
    assert_eq!(
        alpha_mode_from_material(&materials[0]),
        MaterialAlphaMode::Opaque
    );
    assert_eq!(
        alpha_mode_from_material(&materials[1]),
        MaterialAlphaMode::Mask
    );
    assert_eq!(
        alpha_mode_from_material(&materials[2]),
        MaterialAlphaMode::Blend
    );
    assert_eq!(alpha_cutoff_from_material(&materials[0]), 0.0);
    assert!((alpha_cutoff_from_material(&materials[1]) - 0.37).abs() < 1e-6);
    assert_eq!(alpha_cutoff_from_material(&materials[2]), 0.0);
    assert!(materials[2].double_sided());
}

#[test]
fn shader_alpha_tag_preserves_fractional_blend_coverage() {
    assert_eq!(MaterialAlphaMode::Opaque.shader_alpha_value(0.8), 0.0);
    assert_eq!(MaterialAlphaMode::Mask.shader_alpha_value(0.37), 0.37);
    assert_eq!(MaterialAlphaMode::Blend.shader_alpha_value(0.8), -1.0);
}

#[test]
fn mask_textures_select_cutoff_specific_coverage_variants_only() {
    let source = br#"{
      "asset":{"version":"2.0"},
      "images":[{"uri":"shared.png"}],
      "textures":[{"source":0}],
      "materials":[
        {
          "name":"mask-a",
          "alphaMode":"MASK",
          "alphaCutoff":0.5,
          "pbrMetallicRoughness":{
            "baseColorFactor":[1,1,1,0.8],
            "baseColorTexture":{"index":0}
          }
        },
        {
          "name":"mask-b",
          "alphaMode":"MASK",
          "alphaCutoff":0.25,
          "pbrMetallicRoughness":{"baseColorTexture":{"index":0}}
        },
        {
          "name":"blend",
          "alphaMode":"BLEND",
          "pbrMetallicRoughness":{"baseColorTexture":{"index":0}}
        },
        {
          "name":"mask-zero-cutoff",
          "alphaMode":"MASK",
          "alphaCutoff":0,
          "pbrMetallicRoughness":{"baseColorTexture":{"index":0}}
        }
      ]
    }"#;
    let document = gltf::Gltf::from_slice(source).unwrap();
    let materials: Vec<_> = document.materials().collect();
    let references = mask_texture_coverage_references(&document);
    let image_references = references.get(&0).unwrap();
    assert_eq!(image_references.len(), 2);
    assert!(
        mask_only_texture_images(&document, references.keys().copied()).is_empty(),
        "BLEND and zero-cutoff MASK users require the shared ordinary chain"
    );
    assert!(image_references
        .iter()
        .any(|reference| (*reference - 0.625).abs() < 1e-6));
    assert!(image_references
        .iter()
        .any(|reference| (*reference - 0.25).abs() < 1e-6));

    let mut variants = std::collections::HashMap::new();
    variants.insert((0, 0.625f32.to_bits()), 21);
    variants.insert((0, 0.25f32.to_bits()), 22);
    assert_eq!(
        base_color_texture_selection(&materials[0], &[11], &variants),
        (Some(21), true)
    );
    assert_eq!(
        base_color_texture_selection(&materials[1], &[11], &variants),
        (Some(22), true)
    );
    assert_eq!(
        base_color_texture_selection(&materials[2], &[11], &variants),
        (Some(11), false),
        "BLEND must retain its ordinary opacity mip chain"
    );
    assert_eq!(
        base_color_texture_selection(&materials[3], &[11], &variants),
        (Some(11), false),
        "zero-cutoff MASK is fully covered and needs no duplicate chain"
    );
}

#[test]
fn textured_specular_glossiness_preserves_runtime_map_and_factors() {
    let source = br#"{
      "asset":{"version":"2.0"},
      "extensionsUsed":["KHR_materials_pbrSpecularGlossiness"],
      "images":[{"uri":"diffuse.png"},{"uri":"spec-gloss.png"}],
      "textures":[{"source":0},{"source":1}],
      "materials":[{
        "extensions":{"KHR_materials_pbrSpecularGlossiness":{
          "diffuseFactor":[0.8,0.7,0.6,1.0],
          "diffuseTexture":{"index":0},
          "specularFactor":[0.2,0.4,0.8],
          "glossinessFactor":0.65,
          "specularGlossinessTexture":{"index":1}
        }}
      }]
    }"#;
    let document = gltf::Gltf::from_slice(source).expect("valid spec-gloss glTF material");
    let material = document.materials().next().unwrap();
    let spec_gloss = material.pbr_specular_glossiness().unwrap();
    assert_eq!(
        specular_glossiness_texture_selection(&spec_gloss, &[11, 22]),
        Some((22, [0.2, 0.4, 0.8, 0.65]))
    );
    assert_eq!(
        unsupported_material_extension_diagnostics(&document, "bistro.gltf"),
        Vec::<String>::new(),
        "the now-supported workflow must not emit a misleading warning"
    );
}

#[test]
fn vertex_color_multiplies_the_material_factor_per_gltf() {
    assert_eq!(
        multiply_rgba([0.5, 0.25, 0.8, 0.4], [0.2, 0.6, 0.5, 0.75]),
        [0.1, 0.15, 0.4, 0.3]
    );
}

#[test]
fn staged_mask_loader_routes_coverage_variant_and_keeps_plain_loader_image_neutral() {
    let glb = textured_mask_triangle_glb();
    let staged = load_gltf_staged(&glb).expect("textured MASK GLB stages");
    assert_eq!(
        staged.textures.len(),
        1,
        "MASK-only images must not retain an unreachable ordinary mip chain"
    );
    assert_eq!(staged.textures[0].alpha_coverage_reference, Some(0.625));
    assert_eq!(staged.model.meshes[0].texture_idx, Some(1));
    assert!(staged.model.meshes[0].alpha_coverage_mips);
    let color = staged.model.meshes[0].vertices[0].color;
    for (actual, expected) in color.into_iter().zip([0.1, 0.1, 0.45, 0.4]) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "COLOR_0 must multiply baseColorFactor: {color:?}"
        );
    }

    let plain = load_gltf(&glb).expect("plain CPU loader accepts the same GLB");
    assert_eq!(plain.meshes[0].texture_idx, None);
    assert!(
        !plain.meshes[0].alpha_coverage_mips,
        "CPU-only loading must not claim an unregistered coverage texture"
    );
}

#[test]
fn mask_coverage_diagnostic_setting_defaults_on_and_accepts_false_spellings() {
    assert!(mask_coverage_setting_enabled(None));
    assert!(mask_coverage_setting_enabled(Some("1")));
    assert!(mask_coverage_setting_enabled(Some("yes")));
    for value in ["0", "off", "FALSE", " disabled "] {
        assert!(!mask_coverage_setting_enabled(Some(value)), "{value}");
    }
}

#[test]
fn transmission_volume_ior_and_texture_transforms_are_preserved() {
    let json = br#"{
        "asset":{"version":"2.0"},
        "extensionsUsed":[
            "KHR_materials_transmission",
            "KHR_materials_volume",
            "KHR_materials_ior",
            "KHR_texture_transform"
        ],
        "images":[{"uri":"glass.png"}],
        "textures":[{"source":0},{"source":0}],
        "materials":[{
            "name":"Blue glass",
            "extensions":{
                "KHR_materials_transmission":{
                    "transmissionFactor":0.72,
                    "transmissionTexture":{
                        "index":0,
                        "texCoord":1,
                        "extensions":{"KHR_texture_transform":{
                            "offset":[0.1,0.2],
                            "rotation":0.3,
                            "scale":[0.4,0.5],
                            "texCoord":2
                        }}
                    }
                },
                "KHR_materials_ior":{"ior":1.33},
                "KHR_materials_volume":{
                    "thicknessFactor":0.42,
                    "thicknessTexture":{"index":1,"texCoord":3},
                    "attenuationDistance":3.5,
                    "attenuationColor":[0.2,0.4,0.8]
                }
            }
        }]
    }"#;
    let document = gltf::Gltf::from_slice(json).expect("valid physical glTF material");
    let material = document.materials().next().unwrap();
    let physical = transmission_from_material(&material, Some(&[17])).unwrap();

    assert!(physical.authored);
    assert_eq!(physical.factor, 0.72);
    let transmission_texture = physical.texture.unwrap();
    assert_eq!(transmission_texture.source_texture_index, 0);
    assert_eq!(transmission_texture.source_image_index, 0);
    assert_eq!(transmission_texture.runtime_texture_idx, Some(17));
    assert_eq!(transmission_texture.transform.offset, [0.1, 0.2]);
    assert_eq!(transmission_texture.transform.rotation, 0.3);
    assert_eq!(transmission_texture.transform.scale, [0.4, 0.5]);
    assert_eq!(transmission_texture.transform.tex_coord, 2);

    assert!(physical.ior_authored);
    assert_eq!(physical.ior, 1.33);
    assert!(physical.volume_authored);
    assert_eq!(physical.thickness_factor, 0.42);
    assert_eq!(physical.thickness_source, MaterialThicknessSource::Authored);
    let thickness_texture = physical.thickness_texture.unwrap();
    assert_eq!(thickness_texture.source_texture_index, 1);
    assert_eq!(thickness_texture.source_image_index, 0);
    assert_eq!(thickness_texture.runtime_texture_idx, Some(17));
    assert_eq!(thickness_texture.transform.tex_coord, 3);
    assert_eq!(physical.attenuation_distance, 3.5);
    assert_eq!(physical.attenuation_color, [0.2, 0.4, 0.8]);

    let cpu_only = transmission_from_material(&material, None).unwrap();
    assert_eq!(
        cpu_only.texture.unwrap().runtime_texture_idx,
        None,
        "CPU-only loading must preserve the source binding without inventing a GPU handle"
    );
}

#[test]
fn missing_physical_extensions_keep_spec_defaults_and_no_fake_thickness() {
    let document =
        gltf::Gltf::from_slice(br#"{"asset":{"version":"2.0"},"materials":[{"name":"ordinary"}]}"#)
            .unwrap();
    let material = document.materials().next().unwrap();
    assert_eq!(
        transmission_from_material(&material, None),
        Ok(MaterialTransmission::default())
    );
    assert_eq!(
        layered_pbr_from_material(&document, &material, None),
        Ok(MaterialLayeredPbr::default())
    );
}

#[test]
fn clearcoat_specular_ior_and_texture_transforms_are_preserved() {
    let json = br#"{
        "asset":{"version":"2.0"},
        "extensionsUsed":[
            "KHR_materials_clearcoat",
            "KHR_materials_specular",
            "KHR_materials_ior",
            "KHR_texture_transform"
        ],
        "images":[{"uri":"layers.png"}],
        "textures":[{"source":0},{"source":0},{"source":0},{"source":0},{"source":0}],
        "materials":[{
            "name":"Car paint",
            "extensions":{
                "KHR_materials_clearcoat":{
                    "clearcoatFactor":0.8,
                    "clearcoatTexture":{"index":0},
                    "clearcoatRoughnessFactor":0.22,
                    "clearcoatRoughnessTexture":{
                        "index":1,
                        "texCoord":1,
                        "extensions":{"KHR_texture_transform":{
                            "offset":[0.1,0.2],
                            "rotation":0.3,
                            "scale":[0.4,0.5],
                            "texCoord":2
                        }}
                    },
                    "clearcoatNormalTexture":{"index":2,"scale":0.65}
                },
                "KHR_materials_specular":{
                    "specularFactor":0.7,
                    "specularTexture":{"index":3,"texCoord":1},
                    "specularColorFactor":[1.4,0.6,0.2],
                    "specularColorTexture":{"index":4}
                },
                "KHR_materials_ior":{"ior":1.76}
            }
        }]
    }"#;
    let document = gltf::Gltf::from_slice(json).expect("valid layered glTF material");
    let material = document.materials().next().unwrap();
    let layered = layered_pbr_from_material(&document, &material, Some(&[17])).unwrap();

    assert!(layered.clearcoat_authored);
    assert_eq!(layered.clearcoat_factor, 0.8);
    assert_eq!(layered.clearcoat_roughness_factor, 0.22);
    assert_eq!(
        layered.clearcoat_texture.unwrap().runtime_texture_idx,
        Some(17)
    );
    let roughness = layered.clearcoat_roughness_texture.unwrap();
    assert_eq!(roughness.source_texture_index, 1);
    assert_eq!(roughness.transform.offset, [0.1, 0.2]);
    assert_eq!(roughness.transform.rotation, 0.3);
    assert_eq!(roughness.transform.scale, [0.4, 0.5]);
    assert_eq!(roughness.transform.tex_coord, 2);
    assert_eq!(layered.clearcoat_normal_scale, 0.65);

    assert!(layered.specular_authored);
    assert_eq!(layered.specular_factor, 0.7);
    assert_eq!(layered.specular_color_factor, [1.4, 0.6, 0.2]);
    assert_eq!(layered.specular_texture.unwrap().transform.tex_coord, 1);
    assert_eq!(
        layered.specular_color_texture.unwrap().source_texture_index,
        4
    );
    assert!(layered.ior_authored);
    assert_eq!(layered.ior, 1.76);
    assert!(layered.has_clearcoat());
    assert!(layered.has_specular_ior());
    assert!(layered.requests_tex_coord(1));
    assert!(layered.requests_tex_coord(2));

    let cpu_only = layered_pbr_from_material(&document, &material, None).unwrap();
    assert_eq!(
        cpu_only
            .clearcoat_roughness_texture
            .unwrap()
            .runtime_texture_idx,
        None
    );
}

#[test]
fn sheen_anisotropy_and_texture_transforms_are_preserved() {
    let json = br#"{
        "asset":{"version":"2.0"},
        "extensionsUsed":[
            "KHR_materials_sheen",
            "KHR_materials_anisotropy",
            "KHR_texture_transform"
        ],
        "images":[{"uri":"fabric.png"}],
        "textures":[{"source":0},{"source":0},{"source":0}],
        "materials":[{
            "name":"Brushed velvet",
            "extensions":{
                "KHR_materials_sheen":{
                    "sheenColorFactor":[0.8,0.25,0.1],
                    "sheenColorTexture":{"index":0},
                    "sheenRoughnessFactor":0.37,
                    "sheenRoughnessTexture":{
                        "index":1,
                        "texCoord":1,
                        "extensions":{"KHR_texture_transform":{
                            "offset":[0.15,0.2],
                            "rotation":0.4,
                            "scale":[0.5,0.75]
                        }}
                    }
                },
                "KHR_materials_anisotropy":{
                    "anisotropyStrength":0.72,
                    "anisotropyRotation":1.25,
                    "anisotropyTexture":{"index":2,"texCoord":1}
                }
            }
        }]
    }"#;
    let document = gltf::Gltf::from_slice(json).expect("valid sheen/anisotropy material");
    let material = document.materials().next().unwrap();
    let layered = layered_pbr_from_material(&document, &material, Some(&[23])).unwrap();

    assert!(layered.sheen_authored);
    assert_eq!(layered.sheen_color_factor, [0.8, 0.25, 0.1]);
    assert_eq!(layered.sheen_roughness_factor, 0.37);
    assert_eq!(
        layered.sheen_color_texture.unwrap().runtime_texture_idx,
        Some(23)
    );
    let roughness = layered.sheen_roughness_texture.unwrap();
    assert_eq!(roughness.transform.offset, [0.15, 0.2]);
    assert_eq!(roughness.transform.rotation, 0.4);
    assert_eq!(roughness.transform.scale, [0.5, 0.75]);
    assert_eq!(roughness.transform.tex_coord, 1);
    assert!(layered.has_sheen());

    assert!(layered.anisotropy_authored);
    assert_eq!(layered.anisotropy_strength, 0.72);
    assert_eq!(layered.anisotropy_rotation, 1.25);
    assert_eq!(layered.anisotropy_texture.unwrap().transform.tex_coord, 1);
    assert!(layered.has_anisotropy());
    assert!(layered.requests_tex_coord(1));

    let cpu_only = layered_pbr_from_material(&document, &material, None).unwrap();
    assert_eq!(
        cpu_only.anisotropy_texture.unwrap().runtime_texture_idx,
        None
    );
}

#[test]
fn iridescence_parameters_channels_and_texture_transforms_are_preserved() {
    let json = br#"{
        "asset":{"version":"2.0"},
        "extensionsUsed":["KHR_materials_iridescence","KHR_texture_transform"],
        "images":[{"uri":"thin-film.png"}],
        "textures":[{"source":0},{"source":0}],
        "materials":[{
            "name":"Oil film",
            "extensions":{"KHR_materials_iridescence":{
                "iridescenceFactor":0.82,
                "iridescenceTexture":{
                    "index":0,
                    "texCoord":1,
                    "extensions":{"KHR_texture_transform":{
                        "offset":[0.1,0.2],
                        "rotation":0.3,
                        "scale":[0.4,0.5]
                    }}
                },
                "iridescenceIor":1.42,
                "iridescenceThicknessMinimum":620.0,
                "iridescenceThicknessMaximum":180.0,
                "iridescenceThicknessTexture":{"index":1,"texCoord":1}
            }}
        }]
    }"#;
    let document = gltf::Gltf::from_slice(json).expect("valid iridescence material");
    let material = document.materials().next().unwrap();
    let layered = layered_pbr_from_material(&document, &material, Some(&[29])).unwrap();

    assert!(layered.iridescence_authored);
    assert_eq!(layered.iridescence_factor, 0.82);
    assert_eq!(layered.iridescence_ior, 1.42);
    assert_eq!(layered.iridescence_thickness_minimum, 620.0);
    assert_eq!(layered.iridescence_thickness_maximum, 180.0);
    let factor = layered.iridescence_texture.unwrap();
    assert_eq!(factor.runtime_texture_idx, Some(29));
    assert_eq!(factor.transform.offset, [0.1, 0.2]);
    assert_eq!(factor.transform.rotation, 0.3);
    assert_eq!(factor.transform.scale, [0.4, 0.5]);
    assert_eq!(factor.transform.tex_coord, 1);
    assert_eq!(
        layered
            .iridescence_thickness_texture
            .unwrap()
            .transform
            .tex_coord,
        1
    );
    assert!(layered.has_iridescence());
    assert!(layered.is_active());
    assert!(layered.requests_tex_coord(1));

    let cpu_only = layered_pbr_from_material(&document, &material, None).unwrap();
    assert_eq!(
        cpu_only
            .iridescence_thickness_texture
            .unwrap()
            .runtime_texture_idx,
        None
    );
}

#[test]
fn clearcoat_normal_images_use_the_normal_map_upload_path() {
    let document = gltf::Gltf::from_slice(
        br#"{
            "asset":{"version":"2.0"},
            "extensionsUsed":["KHR_materials_clearcoat"],
            "images":[{"uri":"color.png"},{"uri":"coat-normal.png"}],
            "textures":[{"source":0},{"source":1}],
            "materials":[{
                "extensions":{"KHR_materials_clearcoat":{
                    "clearcoatNormalTexture":{"index":1}
                }}
            }]
        }"#,
    )
    .unwrap();
    let mut normal_images = std::collections::HashSet::new();
    retain_layered_normal_image_indices(&document, &mut normal_images);
    assert_eq!(normal_images, std::collections::HashSet::from([1]));
}

#[test]
fn invalid_physical_extension_ranges_are_rejected_by_the_importer() {
    let cases = [
        (
            r#""KHR_materials_transmission":{"transmissionFactor":1.5}"#,
            "transmissionFactor",
        ),
        (r#""KHR_materials_ior":{"ior":0.8}"#, "ior"),
        (
            r#""KHR_materials_volume":{"thicknessFactor":-0.1}"#,
            "thicknessFactor",
        ),
        (
            r#""KHR_materials_volume":{"attenuationDistance":0.0}"#,
            "attenuationDistance",
        ),
        (
            r#""KHR_materials_volume":{"attenuationColor":[1.2,0.5,0.5]}"#,
            "attenuationColor",
        ),
    ];
    for (extension, expected) in cases {
        let invalid = format!(
            r#"{{
                "asset":{{"version":"2.0"}},
                "materials":[{{"name":"Bad glass","extensions":{{{extension}}}}}]
            }}"#
        );
        let document = gltf::Gltf::from_slice(invalid.as_bytes()).unwrap();
        let material = document.materials().next().unwrap();
        let error = transmission_from_material(&material, None).unwrap_err();
        assert!(error.contains("material \"Bad glass\""), "{error}");
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn invalid_layered_extension_ranges_are_rejected_and_ior_zero_is_preserved() {
    let cases = [
        (
            r#""KHR_materials_clearcoat":{"clearcoatFactor":1.2}"#,
            "clearcoatFactor",
        ),
        (
            r#""KHR_materials_clearcoat":{"clearcoatRoughnessFactor":-0.1}"#,
            "clearcoatRoughnessFactor",
        ),
        (
            r#""KHR_materials_specular":{"specularFactor":-0.1}"#,
            "specularFactor",
        ),
        (
            r#""KHR_materials_specular":{"specularColorFactor":[1.0,-0.1,1.0]}"#,
            "specularColorFactor",
        ),
        (
            r#""KHR_materials_sheen":{"sheenColorFactor":[1.0,-0.1,1.0]}"#,
            "sheenColorFactor",
        ),
        (
            r#""KHR_materials_sheen":{"sheenRoughnessFactor":1.1}"#,
            "sheenRoughnessFactor",
        ),
        (
            r#""KHR_materials_anisotropy":{"anisotropyStrength":-0.1}"#,
            "anisotropyStrength",
        ),
        (
            r#""KHR_materials_iridescence":{"iridescenceFactor":1.1}"#,
            "iridescenceFactor",
        ),
        (
            r#""KHR_materials_iridescence":{"iridescenceIor":0.9}"#,
            "iridescenceIor",
        ),
        (
            r#""KHR_materials_iridescence":{"iridescenceThicknessMinimum":-1.0}"#,
            "iridescenceThicknessMinimum",
        ),
        (
            r#""KHR_materials_iridescence":{"iridescenceThicknessMaximum":-1.0}"#,
            "iridescenceThicknessMaximum",
        ),
    ];
    for (extension, expected) in cases {
        let invalid = format!(
            r#"{{
                "asset":{{"version":"2.0"}},
                "materials":[{{"name":"Bad layer","extensions":{{{extension}}}}}]
            }}"#
        );
        let document = gltf::Gltf::from_slice(invalid.as_bytes()).unwrap();
        let material = document.materials().next().unwrap();
        let error = layered_pbr_from_material(&document, &material, None).unwrap_err();
        assert!(error.contains("material \"Bad layer\""), "{error}");
        assert!(error.contains(expected), "{error}");
    }

    let compatibility = gltf::Gltf::from_slice(
        br#"{
            "asset":{"version":"2.0"},
            "materials":[{
                "extensions":{"KHR_materials_ior":{"ior":0.0}}
            }]
        }"#,
    )
    .unwrap();
    let material = compatibility.materials().next().unwrap();
    assert_eq!(
        layered_pbr_from_material(&compatibility, &material, None)
            .unwrap()
            .ior,
        0.0
    );
    assert_eq!(
        transmission_from_material(&material, None).unwrap().ior,
        0.0
    );
}

#[test]
fn unsupported_material_extensions_name_the_asset_and_material() {
    let source = br#"{
        "asset":{"version":"2.0"},
        "extensionsUsed":["VENDOR_materials_velvet"],
        "materials":[{
            "name":"Velvet",
            "extensions":{"VENDOR_materials_velvet":{"roughnessFactor":0.8}}
        }]
    }"#;
    let document = gltf::Gltf::from_slice(source).unwrap();
    let diagnostics = unsupported_material_extension_diagnostics(&document, "vehicles/coupe.glb");
    assert_eq!(diagnostics.len(), 1);
    assert!(diagnostics[0].contains("vehicles/coupe.glb"));
    assert!(diagnostics[0].contains("Velvet"));
    assert!(diagnostics[0].contains("VENDOR_materials_velvet"));
    assert!(diagnostics[0].contains("ignored"));
}

#[test]
fn physical_metadata_round_trips_through_plain_and_staged_glb_loaders() {
    let glb = minimal_triangle_glb(
        r#"{
            "name":"Window",
            "alphaMode":"BLEND",
            "doubleSided":true,
            "extensions":{
                "KHR_materials_transmission":{"transmissionFactor":0.8},
                "KHR_materials_ior":{"ior":1.45},
                "KHR_materials_specular":{
                    "specularFactor":0.65,
                    "specularColorFactor":[1.2,0.8,0.4]
                },
                "KHR_materials_clearcoat":{
                    "clearcoatFactor":0.75,
                    "clearcoatRoughnessFactor":0.18
                },
                "KHR_materials_sheen":{
                    "sheenColorFactor":[0.7,0.2,0.1],
                    "sheenRoughnessFactor":0.4
                },
                "KHR_materials_anisotropy":{
                    "anisotropyStrength":0.6,
                    "anisotropyRotation":0.75
                },
                "KHR_materials_volume":{
                    "thicknessFactor":0.25,
                    "attenuationDistance":2.0,
                    "attenuationColor":[0.8,0.9,1.0]
                }
            }
        }"#,
    );
    let plain = load_gltf(&glb).expect("plain GLB load");
    let staged = load_gltf_staged(&glb).expect("staged GLB load").model;
    for (label, model) in [("plain", plain), ("staged", staged)] {
        assert_eq!(model.meshes.len(), 1, "{label}");
        let mesh = &model.meshes[0];
        assert_eq!(mesh.alpha_mode, MaterialAlphaMode::Blend, "{label}");
        assert!(mesh.double_sided, "{label}");
        assert_eq!(mesh.alpha_cutoff, 0.0, "{label}");
        assert!(mesh.transmission.authored, "{label}");
        assert_eq!(mesh.transmission.factor, 0.8, "{label}");
        assert!(mesh.transmission.ior_authored, "{label}");
        assert_eq!(mesh.transmission.ior, 1.45, "{label}");
        assert!(mesh.layered_pbr.clearcoat_authored, "{label}");
        assert_eq!(mesh.layered_pbr.clearcoat_factor, 0.75, "{label}");
        assert_eq!(mesh.layered_pbr.clearcoat_roughness_factor, 0.18, "{label}");
        assert!(mesh.layered_pbr.specular_authored, "{label}");
        assert_eq!(mesh.layered_pbr.specular_factor, 0.65, "{label}");
        assert_eq!(
            mesh.layered_pbr.specular_color_factor,
            [1.2, 0.8, 0.4],
            "{label}"
        );
        assert!(mesh.layered_pbr.ior_authored, "{label}");
        assert_eq!(mesh.layered_pbr.ior, 1.45, "{label}");
        assert!(mesh.layered_pbr.sheen_authored, "{label}");
        assert_eq!(
            mesh.layered_pbr.sheen_color_factor,
            [0.7, 0.2, 0.1],
            "{label}"
        );
        assert_eq!(mesh.layered_pbr.sheen_roughness_factor, 0.4, "{label}");
        assert!(mesh.layered_pbr.anisotropy_authored, "{label}");
        assert_eq!(mesh.layered_pbr.anisotropy_strength, 0.6, "{label}");
        assert_eq!(mesh.layered_pbr.anisotropy_rotation, 0.75, "{label}");
        assert!(
            mesh.vertices
                .iter()
                .all(|vertex| vertex.tangent == [1.0, 0.0, 0.0, -1.0]),
            "{label}: authored mirrored tangents were not preserved"
        );
        assert!(mesh.transmission.volume_authored, "{label}");
        assert_eq!(mesh.transmission.thickness_factor, 0.25, "{label}");
        assert_eq!(
            mesh.transmission.thickness_source,
            MaterialThicknessSource::Authored,
            "{label}"
        );
        assert_eq!(mesh.transmission.attenuation_distance, 2.0, "{label}");
        assert_eq!(
            mesh.transmission.attenuation_color,
            [0.8, 0.9, 1.0],
            "{label}"
        );
    }
}

#[test]
fn shared_static_node_scale_preserves_world_bounds_and_volume_contract() {
    let glb = minimal_triangle_glb_with_node_scale(
        r#"{
            "extensions":{
                "KHR_materials_transmission":{"transmissionFactor":1.0},
                "KHR_materials_volume":{
                    "thicknessFactor":0.25,
                    "attenuationDistance":1.0,
                    "attenuationColor":[0.2,0.5,0.8]
                }
            }
        }"#,
        Some([2.0, 2.0, 2.0]),
    );
    for (label, model) in [
        ("plain", load_gltf(&glb).expect("plain scaled GLB load")),
        (
            "staged",
            load_gltf_staged(&glb)
                .expect("staged scaled GLB load")
                .model,
        ),
    ] {
        assert_eq!(model.meshes.len(), 1, "{label}");
        let mesh = &model.meshes[0];
        assert_eq!(mesh.vertices[1].position, [1.0, 0.0, 0.0], "{label}");
        assert_eq!(mesh.vertices[2].position, [0.0, 1.0, 0.0], "{label}");
        assert_eq!(model.mesh_transform(0)[0][0], 2.0, "{label}");
        assert_eq!(model.mesh_transform(0)[1][1], 2.0, "{label}");
        assert_eq!(model.mesh_transform(0)[2][2], 2.0, "{label}");
        assert_eq!(model.bbox_min, [0.0, 0.0, 0.0], "{label}");
        assert_eq!(model.bbox_max, [2.0, 2.0, 0.0], "{label}");
        assert_eq!(mesh.transmission.thickness_factor, 0.25, "{label}");
        // Import ownership stays canonical; attach/cache specialize the
        // per-placement material copy with the transform's mean scale.
        assert_eq!(mesh.transmission.baked_thickness_scale, 1.0, "{label}");
        assert_eq!(
            mesh.transmission.effective_thickness_factor(),
            0.25,
            "{label}"
        );
    }
}

#[test]
fn repeated_nodes_share_one_immutable_primitive_payload() {
    let glb = minimal_instanced_triangle_glb();
    let parsed = gltf::Gltf::from_slice(&glb).expect("instanced GLB parses");
    let blob = parsed
        .blob
        .as_deref()
        .expect("instanced GLB has a BIN chunk");
    let expected_source_hash = bloom_geometry_format::geometry_source_sha256(&glb, &[blob]);
    for (label, model) in [
        ("plain", load_gltf(&glb).expect("plain instanced GLB load")),
        (
            "staged",
            load_gltf_staged(&glb)
                .expect("staged instanced GLB load")
                .model,
        ),
    ] {
        assert_eq!(model.meshes.len(), 2, "{label}");
        assert!(
            std::sync::Arc::ptr_eq(&model.meshes[0], &model.meshes[1]),
            "{label}: repeated placement copied primitive ownership"
        );
        assert_eq!(model.mesh_transform(0)[3][0], 0.0, "{label}");
        assert_eq!(model.mesh_transform(1)[3][0], 10.0, "{label}");
        assert_eq!(model.bbox_min, [0.0, 0.0, 0.0], "{label}");
        assert_eq!(model.bbox_max, [11.0, 1.0, 0.0], "{label}");
        assert_eq!(
            model.source_geometry_sha256,
            Some(expected_source_hash),
            "{label}: runtime and cooker source closures diverged"
        );
        assert_eq!(
            model.mesh_source(0),
            Some(ModelPrimitiveSource {
                mesh_index: 0,
                primitive_index: 0,
                placement_index: 0,
            }),
            "{label}"
        );
        assert_eq!(
            model.mesh_source(1),
            Some(ModelPrimitiveSource {
                mesh_index: 0,
                primitive_index: 0,
                placement_index: 1,
            }),
            "{label}"
        );
    }
}

#[test]
fn physical_texcoord_1_is_retained_lazily_and_exactly() {
    let uv1_glb = physical_uv_triangle_glb(1, true);
    for (label, model) in [
        ("plain", load_gltf(&uv1_glb).expect("plain UV1 GLB load")),
        (
            "staged",
            load_gltf_staged(&uv1_glb)
                .expect("staged UV1 GLB load")
                .model,
        ),
    ] {
        assert_eq!(
            model.meshes[0].secondary_tex_coords.as_deref(),
            Some(&[[0.2, 0.3], [0.8, 0.3], [0.2, 0.9]][..]),
            "{label} loader must preserve authored TEXCOORD_1 exactly"
        );
        assert_eq!(
            model.meshes[0].vertices[1].uv,
            [1.0, 0.0],
            "{label} loader must leave the established UV0 vertex ABI intact"
        );
    }

    let uv0_glb = physical_uv_triangle_glb(0, true);
    for model in [
        load_gltf(&uv0_glb).expect("plain UV0 GLB load"),
        load_gltf_staged(&uv0_glb)
            .expect("staged UV0 GLB load")
            .model,
    ] {
        assert!(
            model.meshes[0].secondary_tex_coords.is_none(),
            "an unreferenced TEXCOORD_1 accessor must add no CPU/GPU footprint"
        );
    }
}

#[test]
fn missing_physical_texcoord_1_preserves_material_and_uses_scalar_fallback() {
    let glb = physical_uv_triangle_glb(1, false);
    for model in [
        load_gltf(&glb).expect("plain missing-UV1 GLB load"),
        load_gltf_staged(&glb)
            .expect("staged missing-UV1 GLB load")
            .model,
    ] {
        let mesh = &model.meshes[0];
        assert!(mesh.transmission.is_active());
        assert_eq!(
            mesh.transmission.texture.unwrap().transform.tex_coord,
            1,
            "source metadata must remain lossless for diagnostics/re-export"
        );
        assert!(
            mesh.secondary_tex_coords.is_none(),
            "missing UV1 must never synthesize or silently substitute UV0"
        );
    }
}
