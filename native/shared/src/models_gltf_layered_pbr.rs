//! Import and validation for glTF layered-PBR material extensions.
//!
//! Kept separate from the core mesh loader so adding physical lobes does not
//! grow `models_gltf.rs` beyond the repository's 2,000-line file policy.

use crate::models::{MaterialLayeredPbr, MaterialTextureBinding, MaterialTextureTransform};

pub(super) fn retain_layered_normal_image_indices(
    gltf: &gltf::Gltf,
    out: &mut std::collections::HashSet<usize>,
) {
    for material in gltf.materials() {
        let Some(texture_index) = material
            .extension_value("KHR_materials_clearcoat")
            .and_then(serde_json::Value::as_object)
            .and_then(|clearcoat| clearcoat.get("clearcoatNormalTexture"))
            .and_then(serde_json::Value::as_object)
            .and_then(|texture| texture.get("index"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
        else {
            continue;
        };
        if let Some(texture) = gltf.textures().nth(texture_index) {
            out.insert(texture.source().index());
        }
    }
}

pub(super) fn retain_material_tex_coords_1(
    transmission: crate::models::MaterialTransmission,
    layered_pbr: MaterialLayeredPbr,
    position_count: usize,
    read_values: impl FnOnce() -> Option<Vec<[f32; 2]>>,
) -> Option<Vec<[f32; 2]>> {
    if !transmission.requests_tex_coord(1) && !layered_pbr.requests_tex_coord(1) {
        return None;
    }
    match read_values() {
        Some(values) if values.len() == position_count => Some(values),
        Some(values) => {
            log::warn!(
                "bloom glTF: physical TEXCOORD_1 count {} does not match POSITION count {}; \
                 preserving the material but using its scalar physical factor",
                values.len(),
                position_count,
            );
            None
        }
        None => {
            log::warn!(
                "bloom glTF: physical texture requests TEXCOORD_1 but the primitive has no \
                 TEXCOORD_1 accessor; preserving the material but using its scalar physical factor"
            );
            None
        }
    }
}

pub(super) fn texture_binding_from_info(
    info: gltf::texture::Info<'_>,
    runtime_texture_indices: Option<&[u32]>,
) -> MaterialTextureBinding {
    let texture = info.texture();
    let source_image_index = texture.source().index();
    let mut transform = MaterialTextureTransform {
        tex_coord: info.tex_coord(),
        ..Default::default()
    };
    if let Some(authored) = info.texture_transform() {
        transform.offset = authored.offset();
        transform.rotation = authored.rotation();
        transform.scale = authored.scale();
        transform.tex_coord = authored.tex_coord().unwrap_or(transform.tex_coord);
    }
    MaterialTextureBinding {
        source_texture_index: texture.index() as u32,
        source_image_index: source_image_index as u32,
        runtime_texture_idx: runtime_texture_indices
            .and_then(|indices| indices.get(source_image_index).copied()),
        transform,
    }
}

fn texture_binding_from_extension_value(
    info: &serde_json::Value,
    gltf: &gltf::Gltf,
    runtime_texture_indices: Option<&[u32]>,
    field: &str,
) -> Result<MaterialTextureBinding, String> {
    let object = info
        .as_object()
        .ok_or_else(|| format!("{field} must be an object"))?;
    let texture_index = object
        .get("index")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("{field}.index must be a non-negative integer"))?
        as usize;
    let texture = gltf
        .textures()
        .nth(texture_index)
        .ok_or_else(|| format!("{field}.index {texture_index} is out of range"))?;
    let source_image_index = texture.source().index();
    let mut transform = MaterialTextureTransform {
        tex_coord: object
            .get("texCoord")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u32,
        ..Default::default()
    };
    if let Some(authored) = object
        .get("extensions")
        .and_then(serde_json::Value::as_object)
        .and_then(|extensions| extensions.get("KHR_texture_transform"))
        .and_then(serde_json::Value::as_object)
    {
        let vec2 = |name: &str, fallback: [f32; 2]| -> Result<[f32; 2], String> {
            let Some(value) = authored.get(name) else {
                return Ok(fallback);
            };
            let values = value
                .as_array()
                .filter(|values| values.len() == 2)
                .ok_or_else(|| format!("{field}.KHR_texture_transform.{name} must be vec2"))?;
            let result = [
                values[0]
                    .as_f64()
                    .ok_or_else(|| format!("{field}.{name}[0] must be numeric"))?
                    as f32,
                values[1]
                    .as_f64()
                    .ok_or_else(|| format!("{field}.{name}[1] must be numeric"))?
                    as f32,
            ];
            if result.iter().any(|value| !value.is_finite()) {
                return Err(format!("{field}.{name} must be finite"));
            }
            Ok(result)
        };
        transform.offset = vec2("offset", transform.offset)?;
        transform.scale = vec2("scale", transform.scale)?;
        if let Some(rotation) = authored.get("rotation") {
            transform.rotation = rotation
                .as_f64()
                .ok_or_else(|| format!("{field}.rotation must be numeric"))?
                as f32;
            if !transform.rotation.is_finite() {
                return Err(format!("{field}.rotation must be finite"));
            }
        }
        if let Some(tex_coord) = authored.get("texCoord") {
            transform.tex_coord = tex_coord
                .as_u64()
                .ok_or_else(|| format!("{field}.texCoord must be a non-negative integer"))?
                as u32;
        }
    }
    Ok(MaterialTextureBinding {
        source_texture_index: texture_index as u32,
        source_image_index: source_image_index as u32,
        runtime_texture_idx: runtime_texture_indices
            .and_then(|indices| indices.get(source_image_index).copied()),
        transform,
    })
}

pub(super) fn layered_pbr_from_material(
    gltf: &gltf::Gltf,
    mat: &gltf::Material<'_>,
    runtime_texture_indices: Option<&[u32]>,
) -> Result<MaterialLayeredPbr, String> {
    let mut out = MaterialLayeredPbr::default();
    if let Some(ior) = mat.ior() {
        out.ior_authored = true;
        out.ior = ior;
    }
    if let Some(specular) = mat.specular() {
        out.specular_authored = true;
        out.specular_factor = specular.specular_factor();
        out.specular_color_factor = specular.specular_color_factor();
        out.specular_texture = specular
            .specular_texture()
            .map(|info| texture_binding_from_info(info, runtime_texture_indices));
        out.specular_color_texture = specular
            .specular_color_texture()
            .map(|info| texture_binding_from_info(info, runtime_texture_indices));
    }
    if let Some(clearcoat) = mat
        .extension_value("KHR_materials_clearcoat")
        .and_then(serde_json::Value::as_object)
    {
        out.clearcoat_authored = true;
        if let Some(factor) = clearcoat.get("clearcoatFactor") {
            out.clearcoat_factor = factor.as_f64().ok_or("clearcoatFactor must be numeric")? as f32;
        }
        if let Some(roughness) = clearcoat.get("clearcoatRoughnessFactor") {
            out.clearcoat_roughness_factor = roughness
                .as_f64()
                .ok_or("clearcoatRoughnessFactor must be numeric")?
                as f32;
        }
        if let Some(info) = clearcoat.get("clearcoatTexture") {
            out.clearcoat_texture = Some(texture_binding_from_extension_value(
                info,
                gltf,
                runtime_texture_indices,
                "clearcoatTexture",
            )?);
        }
        if let Some(info) = clearcoat.get("clearcoatRoughnessTexture") {
            out.clearcoat_roughness_texture = Some(texture_binding_from_extension_value(
                info,
                gltf,
                runtime_texture_indices,
                "clearcoatRoughnessTexture",
            )?);
        }
        if let Some(info) = clearcoat.get("clearcoatNormalTexture") {
            out.clearcoat_normal_texture = Some(texture_binding_from_extension_value(
                info,
                gltf,
                runtime_texture_indices,
                "clearcoatNormalTexture",
            )?);
            if let Some(scale) = info
                .as_object()
                .and_then(|object| object.get("scale"))
                .and_then(serde_json::Value::as_f64)
            {
                out.clearcoat_normal_scale = scale as f32;
            }
        }
    }
    if let Some(sheen) = mat
        .extension_value("KHR_materials_sheen")
        .and_then(serde_json::Value::as_object)
    {
        out.sheen_authored = true;
        if let Some(color) = sheen.get("sheenColorFactor") {
            out.sheen_color_factor = extension_vec3(color, "sheenColorFactor")?;
        }
        if let Some(roughness) = sheen.get("sheenRoughnessFactor") {
            out.sheen_roughness_factor = roughness
                .as_f64()
                .ok_or("sheenRoughnessFactor must be numeric")?
                as f32;
        }
        if let Some(info) = sheen.get("sheenColorTexture") {
            out.sheen_color_texture = Some(texture_binding_from_extension_value(
                info,
                gltf,
                runtime_texture_indices,
                "sheenColorTexture",
            )?);
        }
        if let Some(info) = sheen.get("sheenRoughnessTexture") {
            out.sheen_roughness_texture = Some(texture_binding_from_extension_value(
                info,
                gltf,
                runtime_texture_indices,
                "sheenRoughnessTexture",
            )?);
        }
    }
    if let Some(anisotropy) = mat
        .extension_value("KHR_materials_anisotropy")
        .and_then(serde_json::Value::as_object)
    {
        out.anisotropy_authored = true;
        if let Some(strength) = anisotropy.get("anisotropyStrength") {
            out.anisotropy_strength = strength
                .as_f64()
                .ok_or("anisotropyStrength must be numeric")?
                as f32;
        }
        if let Some(rotation) = anisotropy.get("anisotropyRotation") {
            out.anisotropy_rotation = rotation
                .as_f64()
                .ok_or("anisotropyRotation must be numeric")?
                as f32;
        }
        if let Some(info) = anisotropy.get("anisotropyTexture") {
            out.anisotropy_texture = Some(texture_binding_from_extension_value(
                info,
                gltf,
                runtime_texture_indices,
                "anisotropyTexture",
            )?);
        }
    }
    if let Some(iridescence) = mat
        .extension_value("KHR_materials_iridescence")
        .and_then(serde_json::Value::as_object)
    {
        out.iridescence_authored = true;
        if let Some(factor) = iridescence.get("iridescenceFactor") {
            out.iridescence_factor =
                factor.as_f64().ok_or("iridescenceFactor must be numeric")? as f32;
        }
        if let Some(info) = iridescence.get("iridescenceTexture") {
            out.iridescence_texture = Some(texture_binding_from_extension_value(
                info,
                gltf,
                runtime_texture_indices,
                "iridescenceTexture",
            )?);
        }
        if let Some(ior) = iridescence.get("iridescenceIor") {
            out.iridescence_ior = ior.as_f64().ok_or("iridescenceIor must be numeric")? as f32;
        }
        if let Some(thickness) = iridescence.get("iridescenceThicknessMinimum") {
            out.iridescence_thickness_minimum = thickness
                .as_f64()
                .ok_or("iridescenceThicknessMinimum must be numeric")?
                as f32;
        }
        if let Some(thickness) = iridescence.get("iridescenceThicknessMaximum") {
            out.iridescence_thickness_maximum = thickness
                .as_f64()
                .ok_or("iridescenceThicknessMaximum must be numeric")?
                as f32;
        }
        if let Some(info) = iridescence.get("iridescenceThicknessTexture") {
            out.iridescence_thickness_texture = Some(texture_binding_from_extension_value(
                info,
                gltf,
                runtime_texture_indices,
                "iridescenceThicknessTexture",
            )?);
        }
    }
    validate_layered_material(mat, out)
}

fn extension_vec3(value: &serde_json::Value, field: &str) -> Result<[f32; 3], String> {
    let values = value
        .as_array()
        .filter(|values| values.len() == 3)
        .ok_or_else(|| format!("{field} must be a three-component array"))?;
    let mut result = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        result[index] = value
            .as_f64()
            .ok_or_else(|| format!("{field}[{index}] must be numeric"))?
            as f32;
    }
    Ok(result)
}

fn validate_layered_material(
    mat: &gltf::Material<'_>,
    out: MaterialLayeredPbr,
) -> Result<MaterialLayeredPbr, String> {
    let material = mat
        .name()
        .map(|name| format!("\"{name}\""))
        .or_else(|| mat.index().map(|index| format!("#{index}")))
        .unwrap_or_else(|| "<default>".to_owned());
    let invalid = if !out.clearcoat_factor.is_finite()
        || !(0.0..=1.0).contains(&out.clearcoat_factor)
    {
        Some(format!(
            "clearcoatFactor {} outside [0, 1]",
            out.clearcoat_factor
        ))
    } else if !out.clearcoat_roughness_factor.is_finite()
        || !(0.0..=1.0).contains(&out.clearcoat_roughness_factor)
    {
        Some(format!(
            "clearcoatRoughnessFactor {} outside [0, 1]",
            out.clearcoat_roughness_factor
        ))
    } else if !out.clearcoat_normal_scale.is_finite() {
        Some(format!(
            "clearcoatNormalTexture.scale {} must be finite",
            out.clearcoat_normal_scale
        ))
    } else if !out.specular_factor.is_finite() || !(0.0..=1.0).contains(&out.specular_factor) {
        Some(format!(
            "specularFactor {} outside [0, 1]",
            out.specular_factor
        ))
    } else if out
        .specular_color_factor
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        Some(format!(
            "specularColorFactor {:?} has a negative or non-finite component",
            out.specular_color_factor
        ))
    } else if !out.ior.is_finite() || (out.ior != 0.0 && out.ior < 1.0) {
        Some(format!("ior {} must be zero or at least 1.0", out.ior))
    } else if out
        .sheen_color_factor
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        Some(format!(
            "sheenColorFactor {:?} has a component outside [0, 1]",
            out.sheen_color_factor
        ))
    } else if !out.sheen_roughness_factor.is_finite()
        || !(0.0..=1.0).contains(&out.sheen_roughness_factor)
    {
        Some(format!(
            "sheenRoughnessFactor {} outside [0, 1]",
            out.sheen_roughness_factor
        ))
    } else if !out.anisotropy_strength.is_finite()
        || !(0.0..=1.0).contains(&out.anisotropy_strength)
    {
        Some(format!(
            "anisotropyStrength {} outside [0, 1]",
            out.anisotropy_strength
        ))
    } else if !out.anisotropy_rotation.is_finite() {
        Some(format!(
            "anisotropyRotation {} must be finite",
            out.anisotropy_rotation
        ))
    } else if !out.iridescence_factor.is_finite() || !(0.0..=1.0).contains(&out.iridescence_factor)
    {
        Some(format!(
            "iridescenceFactor {} outside [0, 1]",
            out.iridescence_factor
        ))
    } else if !out.iridescence_ior.is_finite() || out.iridescence_ior < 1.0 {
        Some(format!(
            "iridescenceIor {} must be at least 1.0",
            out.iridescence_ior
        ))
    } else if !out.iridescence_thickness_minimum.is_finite()
        || out.iridescence_thickness_minimum < 0.0
    {
        Some(format!(
            "iridescenceThicknessMinimum {} must be non-negative",
            out.iridescence_thickness_minimum
        ))
    } else if !out.iridescence_thickness_maximum.is_finite()
        || out.iridescence_thickness_maximum < 0.0
    {
        Some(format!(
            "iridescenceThicknessMaximum {} must be non-negative",
            out.iridescence_thickness_maximum
        ))
    } else {
        None
    };
    if let Some(reason) = invalid {
        Err(format!(
            "glTF material {material}: invalid layered-PBR extension data: {reason}"
        ))
    } else {
        Ok(out)
    }
}
