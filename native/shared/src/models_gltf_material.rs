//! Physical glTF material validation and compatibility diagnostics.

use super::{texture_binding_from_info, MaterialThicknessSource, MaterialTransmission};

pub(super) fn unsupported_material_extension_diagnostics(
    gltf: &gltf::Gltf,
    source_label: &str,
) -> Vec<String> {
    let mut diagnostics = Vec::new();
    for material in gltf.materials() {
        let name = material
            .name()
            .map(|name| format!("\"{name}\""))
            .or_else(|| material.index().map(|index| format!("#{index}")))
            .unwrap_or_else(|| "<default>".to_owned());
        if let Some(extensions) = material.extensions() {
            for extension in extensions.keys() {
                if matches!(
                    extension.as_str(),
                    "KHR_materials_pbrSpecularGlossiness"
                        | "KHR_materials_clearcoat"
                        | "KHR_materials_sheen"
                        | "KHR_materials_anisotropy"
                ) {
                    continue;
                }
                diagnostics.push(format!(
                    "glTF asset \"{source_label}\", material {name}: unsupported extension \
                     \"{extension}\" is ignored"
                ));
            }
        }
    }
    diagnostics
}

pub(super) fn emit_unsupported_material_extension_diagnostics(
    gltf: &gltf::Gltf,
    source_label: &str,
) {
    for diagnostic in unsupported_material_extension_diagnostics(gltf, source_label) {
        log::warn!("{diagnostic}");
    }
}

pub(super) fn transmission_from_material(
    mat: &gltf::Material<'_>,
    runtime_texture_indices: Option<&[u32]>,
) -> Result<MaterialTransmission, String> {
    let mut out = MaterialTransmission::default();
    if let Some(transmission) = mat.transmission() {
        out.authored = true;
        out.factor = transmission.transmission_factor();
        out.texture = transmission
            .transmission_texture()
            .map(|info| texture_binding_from_info(info, runtime_texture_indices));
    }
    if let Some(ior) = mat.ior() {
        out.ior_authored = true;
        out.ior = ior;
    }
    if let Some(volume) = mat.volume() {
        out.volume_authored = true;
        out.thickness_factor = volume.thickness_factor();
        out.thickness_texture = volume
            .thickness_texture()
            .map(|info| texture_binding_from_info(info, runtime_texture_indices));
        out.attenuation_distance = volume.attenuation_distance();
        out.attenuation_color = volume.attenuation_color();
        out.thickness_source = MaterialThicknessSource::Authored;
    }
    let material = mat
        .name()
        .map(|name| format!("\"{name}\""))
        .or_else(|| mat.index().map(|index| format!("#{index}")))
        .unwrap_or_else(|| "<default>".to_owned());
    let invalid = if !out.factor.is_finite() || !(0.0..=1.0).contains(&out.factor) {
        Some(format!("transmissionFactor {} outside [0, 1]", out.factor))
    } else if !out.ior.is_finite() || (out.ior != 0.0 && out.ior < 1.0) {
        Some(format!("ior {} must be zero or at least 1.0", out.ior))
    } else if !out.thickness_factor.is_finite() || out.thickness_factor < 0.0 {
        Some(format!("thicknessFactor {} below 0", out.thickness_factor))
    } else if out.attenuation_distance.is_nan() || out.attenuation_distance <= 0.0 {
        Some(format!(
            "attenuationDistance {} must be positive",
            out.attenuation_distance
        ))
    } else if out
        .attenuation_color
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        Some(format!(
            "attenuationColor {:?} has a component outside [0, 1]",
            out.attenuation_color
        ))
    } else {
        None
    };
    if let Some(reason) = invalid {
        Err(format!(
            "glTF material {material}: invalid physical extension data: {reason}"
        ))
    } else {
        Ok(out)
    }
}

/// Exact pre-refraction approximation retained only for the diagnostic
/// `BLOOM_GLTF_REFRACTION=0` path.
pub(super) fn apply_transmission_hack(
    transmission: f32,
    base_color: &mut [f32; 4],
    metallic: &mut f32,
    roughness: &mut f32,
) {
    if transmission > 0.5 {
        *metallic = 1.0;
        *roughness = roughness.min(0.05);
        base_color[0] *= 0.85;
        base_color[1] *= 0.85;
        base_color[2] *= 0.85;
        base_color[3] = 1.0;
    }
}
