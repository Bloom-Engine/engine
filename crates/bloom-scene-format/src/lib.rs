//! Versioned source-free scene archive shared by Bloom's cooker and runtime.
//!
//! The payload deliberately mirrors Bloom's runtime model contract rather
//! than retaining glTF JSON. Shipping loads validate and decode this format;
//! only the offline cooker parses source glTF/GLB documents.

use bincode::{Decode, Encode};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub const MAGIC: [u8; 8] = *b"BSCENE\0\0";
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 8 + 4 + 8 + 32;
pub const MAX_ARCHIVE_BYTES: u64 = 16 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Encode, Decode, PartialEq)]
pub struct SceneArchive {
    pub source_geometry_sha256: [u8; 32],
    pub bbox_min: [f32; 3],
    pub bbox_max: [f32; 3],
    pub primitives: Vec<Primitive>,
    pub placements: Vec<Placement>,
    pub textures: Vec<TextureDependency>,
    pub animation: Option<AnimationArchive>,
    pub diagnostics: SceneDiagnostics,
}

#[derive(Copy, Clone, Debug, Default, Encode, Decode, Eq, PartialEq)]
pub struct SceneDiagnostics {
    pub non_finite_position_vertices: u64,
    pub non_finite_attribute_vertices: u64,
    pub dropped_triangles: u64,
    pub dropped_primitives: u64,
    pub dropped_placements: u64,
}

#[derive(Clone, Debug, Encode, Decode, PartialEq)]
pub struct Primitive {
    pub vertices: Vec<Vertex>,
    pub secondary_tex_coords: Option<Vec<[f32; 2]>>,
    pub indices: Vec<u32>,
    pub material: Material,
}

#[derive(Copy, Clone, Debug, Encode, Decode, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub joints: [f32; 4],
    pub weights: [f32; 4],
    pub tangent: [f32; 4],
}

#[derive(Clone, Debug, Encode, Decode, PartialEq)]
pub struct Placement {
    pub primitive: u32,
    pub transform: [[f32; 4]; 4],
    pub cast_shadow: bool,
    pub source: Option<PrimitiveSource>,
}

#[derive(Copy, Clone, Debug, Encode, Decode, Eq, PartialEq)]
pub struct PrimitiveSource {
    pub mesh_index: u32,
    pub primitive_index: u32,
    pub placement_index: u32,
}

#[derive(Clone, Debug, Encode, Decode, PartialEq)]
pub struct TextureDependency {
    pub logical_id: String,
    pub source_sha256: [u8; 32],
    pub width: u32,
    pub height: u32,
    pub is_normal: bool,
    pub is_srgb: bool,
    pub alpha_coverage_reference: Option<f32>,
}

#[derive(Clone, Debug, Encode, Decode, PartialEq)]
pub struct Material {
    /// One-based index into `SceneArchive::textures`; zero is Bloom's white
    /// fallback and `None` means the material did not author the binding.
    pub base_color_texture: Option<u32>,
    pub normal_texture: Option<u32>,
    pub metallic_roughness_texture: Option<u32>,
    pub specular_glossiness_factor: Option<[f32; 4]>,
    pub emissive_texture: Option<u32>,
    pub occlusion_texture: Option<u32>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 3],
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub alpha_coverage_mips: bool,
    pub double_sided: bool,
    pub transmission: Transmission,
    pub layered_pbr: LayeredPbr,
}

#[derive(Copy, Clone, Debug, Encode, Decode, Eq, PartialEq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

#[derive(Copy, Clone, Debug, Encode, Decode, Eq, PartialEq)]
pub enum ThicknessSource {
    Unavailable,
    Authored,
    Approximated,
}

#[derive(Copy, Clone, Debug, Encode, Decode, PartialEq)]
pub struct TextureTransform {
    pub offset: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
    pub tex_coord: u32,
}

#[derive(Copy, Clone, Debug, Encode, Decode, PartialEq)]
pub struct TextureBinding {
    pub source_texture_index: u32,
    pub source_image_index: u32,
    pub texture: Option<u32>,
    pub transform: TextureTransform,
}

#[derive(Copy, Clone, Debug, Encode, Decode, PartialEq)]
pub struct Transmission {
    pub authored: bool,
    pub factor: f32,
    pub texture: Option<TextureBinding>,
    pub ior_authored: bool,
    pub ior: f32,
    pub volume_authored: bool,
    pub thickness_factor: f32,
    pub thickness_texture: Option<TextureBinding>,
    pub attenuation_distance: f32,
    pub attenuation_color: [f32; 3],
    pub thickness_source: ThicknessSource,
    pub baked_thickness_scale: f32,
}

#[derive(Copy, Clone, Debug, Encode, Decode, PartialEq)]
pub struct LayeredPbr {
    pub clearcoat_authored: bool,
    pub clearcoat_factor: f32,
    pub clearcoat_roughness_factor: f32,
    pub clearcoat_texture: Option<TextureBinding>,
    pub clearcoat_roughness_texture: Option<TextureBinding>,
    pub clearcoat_normal_texture: Option<TextureBinding>,
    pub clearcoat_normal_scale: f32,
    pub specular_authored: bool,
    pub specular_factor: f32,
    pub specular_texture: Option<TextureBinding>,
    pub specular_color_factor: [f32; 3],
    pub specular_color_texture: Option<TextureBinding>,
    pub ior_authored: bool,
    pub ior: f32,
    pub sheen_authored: bool,
    pub sheen_color_factor: [f32; 3],
    pub sheen_color_texture: Option<TextureBinding>,
    pub sheen_roughness_factor: f32,
    pub sheen_roughness_texture: Option<TextureBinding>,
    pub anisotropy_authored: bool,
    pub anisotropy_strength: f32,
    pub anisotropy_rotation: f32,
    pub anisotropy_texture: Option<TextureBinding>,
    pub iridescence_authored: bool,
    pub iridescence_factor: f32,
    pub iridescence_texture: Option<TextureBinding>,
    pub iridescence_ior: f32,
    pub iridescence_thickness_minimum: f32,
    pub iridescence_thickness_maximum: f32,
    pub iridescence_thickness_texture: Option<TextureBinding>,
}

#[derive(Clone, Debug, Encode, Decode, PartialEq)]
pub struct AnimationArchive {
    pub skeleton: Option<Skeleton>,
    pub clips: Vec<AnimationClip>,
    pub reference_rest_rotations: Option<Vec<[f32; 4]>>,
}

#[derive(Clone, Debug, Encode, Decode, PartialEq)]
pub struct Skeleton {
    pub joints: Vec<Joint>,
    pub root_joints: Vec<u32>,
}

#[derive(Clone, Debug, Encode, Decode, PartialEq)]
pub struct Joint {
    pub inverse_bind: [[f32; 4]; 4],
    pub children: Vec<u32>,
    pub name: String,
    pub rest_translation: [f32; 3],
    pub rest_rotation: [f32; 4],
    pub rest_scale: [f32; 3],
}

#[derive(Clone, Debug, Encode, Decode, PartialEq)]
pub struct AnimationClip {
    pub channels: Vec<AnimationChannel>,
    pub duration: f32,
    pub name: String,
}

#[derive(Clone, Debug, Encode, Decode, PartialEq)]
pub struct AnimationChannel {
    pub joint_index: u32,
    pub timestamps: Vec<f32>,
    pub translations: Vec<[f32; 3]>,
    pub rotation_timestamps: Vec<f32>,
    pub rotations: Vec<[f32; 4]>,
    pub scale_timestamps: Vec<f32>,
    pub scales: Vec<[f32; 3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedScene {
    pub archive: SceneArchive,
    pub payload_sha256: [u8; 32],
    pub payload_bytes: u64,
}

pub fn encode_scene(archive: &SceneArchive) -> Result<Vec<u8>, String> {
    validate_scene(archive)?;
    let config = bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding();
    let payload = bincode::encode_to_vec(archive, config)
        .map_err(|error| format!("encode cooked scene payload: {error}"))?;
    let payload_bytes = u64::try_from(payload.len())
        .map_err(|_| "cooked scene payload length exceeds u64".to_string())?;
    if payload_bytes > MAX_ARCHIVE_BYTES {
        return Err(format!(
            "cooked scene payload is {payload_bytes} bytes, maximum is {MAX_ARCHIVE_BYTES}"
        ));
    }
    let hash = sha256(&payload);
    let mut output = Vec::with_capacity(HEADER_BYTES + payload.len());
    output.extend_from_slice(&MAGIC);
    output.extend_from_slice(&VERSION.to_le_bytes());
    output.extend_from_slice(&payload_bytes.to_le_bytes());
    output.extend_from_slice(&hash);
    output.extend_from_slice(&payload);
    Ok(output)
}

pub fn decode_scene(bytes: &[u8]) -> Result<DecodedScene, String> {
    if bytes.len() < HEADER_BYTES {
        return Err("cooked scene header is truncated".to_string());
    }
    if bytes[..8] != MAGIC {
        return Err("cooked scene magic is invalid".to_string());
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(format!(
            "cooked scene version {version} is incompatible; recook with format {VERSION}"
        ));
    }
    let payload_bytes = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
    if payload_bytes > MAX_ARCHIVE_BYTES {
        return Err(format!(
            "cooked scene payload declares {payload_bytes} bytes, maximum is {MAX_ARCHIVE_BYTES}"
        ));
    }
    let actual_payload = bytes.len() - HEADER_BYTES;
    if payload_bytes != actual_payload as u64 {
        return Err(format!(
            "cooked scene payload length mismatch: header {payload_bytes}, actual {actual_payload}"
        ));
    }
    let expected_hash: [u8; 32] = bytes[20..52].try_into().unwrap();
    let payload = &bytes[HEADER_BYTES..];
    let actual_hash = sha256(payload);
    if expected_hash != actual_hash {
        return Err("cooked scene payload hash mismatch".to_string());
    }
    let config = bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding();
    let (archive, consumed): (SceneArchive, usize) = bincode::decode_from_slice(payload, config)
        .map_err(|error| format!("decode cooked scene payload: {error}"))?;
    if consumed != payload.len() {
        return Err("cooked scene payload has trailing bytes".to_string());
    }
    validate_scene(&archive)?;
    let canonical = bincode::encode_to_vec(&archive, config)
        .map_err(|error| format!("re-encode cooked scene payload: {error}"))?;
    if canonical != payload {
        return Err("cooked scene payload is not canonical".to_string());
    }
    Ok(DecodedScene {
        archive,
        payload_sha256: actual_hash,
        payload_bytes,
    })
}

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

pub fn hex_hash(hash: [u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn validate_scene(scene: &SceneArchive) -> Result<(), String> {
    if scene.primitives.is_empty() || scene.placements.is_empty() {
        return Err("cooked scene requires primitives and placements".to_string());
    }
    finite_slice(&scene.bbox_min, "scene bbox minimum")?;
    finite_slice(&scene.bbox_max, "scene bbox maximum")?;
    if (0..3).any(|axis| scene.bbox_min[axis] > scene.bbox_max[axis]) {
        return Err("cooked scene bounds are inverted".to_string());
    }
    let mut texture_ids = HashSet::new();
    for (index, texture) in scene.textures.iter().enumerate() {
        validate_logical_id(&texture.logical_id)?;
        if !texture_ids.insert(texture.logical_id.as_str()) {
            return Err(format!(
                "duplicate cooked texture logical ID {:?}",
                texture.logical_id
            ));
        }
        if texture.width == 0 || texture.height == 0 {
            return Err(format!("texture {index} has zero dimensions"));
        }
        if texture.is_normal && texture.is_srgb {
            return Err(format!("texture {index} is both normal and sRGB"));
        }
        if texture
            .alpha_coverage_reference
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(format!(
                "texture {index} has invalid alpha coverage reference"
            ));
        }
    }
    for (index, primitive) in scene.primitives.iter().enumerate() {
        validate_primitive(index, primitive, scene.textures.len())?;
    }
    for (index, placement) in scene.placements.iter().enumerate() {
        if placement.primitive as usize >= scene.primitives.len() {
            return Err(format!(
                "placement {index} references absent primitive {}",
                placement.primitive
            ));
        }
        finite_matrix(
            &placement.transform,
            &format!("placement {index} transform"),
        )?;
    }
    if let Some(animation) = &scene.animation {
        validate_animation(animation)?;
    }
    Ok(())
}

fn validate_primitive(
    index: usize,
    primitive: &Primitive,
    texture_count: usize,
) -> Result<(), String> {
    if primitive.vertices.is_empty() {
        return Err(format!("primitive {index} has no vertices"));
    }
    if primitive.indices.is_empty() || !primitive.indices.len().is_multiple_of(3) {
        return Err(format!(
            "primitive {index} is not a non-empty triangle list"
        ));
    }
    if let Some(bad) = primitive
        .indices
        .iter()
        .find(|value| **value as usize >= primitive.vertices.len())
    {
        return Err(format!(
            "primitive {index} index {bad} exceeds {} vertices",
            primitive.vertices.len()
        ));
    }
    if let Some(uv1) = &primitive.secondary_tex_coords {
        if uv1.len() != primitive.vertices.len() {
            return Err(format!(
                "primitive {index} secondary UV count does not match vertices"
            ));
        }
        for uv in uv1 {
            finite_slice(uv, &format!("primitive {index} secondary UV"))?;
        }
    }
    for vertex in &primitive.vertices {
        finite_slice(&vertex.position, &format!("primitive {index} position"))?;
        finite_slice(&vertex.normal, &format!("primitive {index} normal"))?;
        finite_slice(&vertex.color, &format!("primitive {index} color"))?;
        finite_slice(&vertex.uv, &format!("primitive {index} UV"))?;
        finite_slice(&vertex.joints, &format!("primitive {index} joints"))?;
        finite_slice(&vertex.weights, &format!("primitive {index} weights"))?;
        finite_slice(&vertex.tangent, &format!("primitive {index} tangent"))?;
    }
    validate_material(&primitive.material, texture_count, index)
}

fn validate_material(
    material: &Material,
    texture_count: usize,
    primitive: usize,
) -> Result<(), String> {
    for texture in [
        material.base_color_texture,
        material.normal_texture,
        material.metallic_roughness_texture,
        material.emissive_texture,
        material.occlusion_texture,
    ] {
        validate_texture_index(texture, texture_count, primitive)?;
    }
    let bindings = [
        material.transmission.texture,
        material.transmission.thickness_texture,
        material.layered_pbr.clearcoat_texture,
        material.layered_pbr.clearcoat_roughness_texture,
        material.layered_pbr.clearcoat_normal_texture,
        material.layered_pbr.specular_texture,
        material.layered_pbr.specular_color_texture,
        material.layered_pbr.sheen_color_texture,
        material.layered_pbr.sheen_roughness_texture,
        material.layered_pbr.anisotropy_texture,
        material.layered_pbr.iridescence_texture,
        material.layered_pbr.iridescence_thickness_texture,
    ];
    for binding in bindings.into_iter().flatten() {
        validate_texture_index(binding.texture, texture_count, primitive)?;
        finite_slice(&binding.transform.offset, "texture transform offset")?;
        finite_slice(&binding.transform.scale, "texture transform scale")?;
        if !binding.transform.rotation.is_finite() {
            return Err(format!(
                "primitive {primitive} has non-finite texture rotation"
            ));
        }
    }
    let factors = [
        material.metallic_factor,
        material.roughness_factor,
        material.alpha_cutoff,
        material.transmission.factor,
        material.transmission.ior,
        material.transmission.thickness_factor,
        material.transmission.attenuation_distance,
        material.transmission.baked_thickness_scale,
        material.layered_pbr.clearcoat_factor,
        material.layered_pbr.clearcoat_roughness_factor,
        material.layered_pbr.clearcoat_normal_scale,
        material.layered_pbr.specular_factor,
        material.layered_pbr.ior,
        material.layered_pbr.sheen_roughness_factor,
        material.layered_pbr.anisotropy_strength,
        material.layered_pbr.anisotropy_rotation,
        material.layered_pbr.iridescence_factor,
        material.layered_pbr.iridescence_ior,
        material.layered_pbr.iridescence_thickness_minimum,
        material.layered_pbr.iridescence_thickness_maximum,
    ];
    // Positive infinity is the authored glTF default for no attenuation.
    if factors
        .into_iter()
        .enumerate()
        .any(|(index, value)| !value.is_finite() && !(index == 6 && value == f32::INFINITY))
    {
        return Err(format!(
            "primitive {primitive} has non-finite material factors"
        ));
    }
    finite_slice(&material.emissive_factor, "emissive factor")?;
    finite_slice(
        &material.transmission.attenuation_color,
        "attenuation color",
    )?;
    finite_slice(
        &material.layered_pbr.specular_color_factor,
        "specular color",
    )?;
    finite_slice(&material.layered_pbr.sheen_color_factor, "sheen color")?;
    if let Some(value) = material.specular_glossiness_factor {
        finite_slice(&value, "specular-glossiness factor")?;
    }
    Ok(())
}

fn validate_texture_index(
    index: Option<u32>,
    texture_count: usize,
    primitive: usize,
) -> Result<(), String> {
    if index.is_some_and(|value| value as usize > texture_count) {
        return Err(format!(
            "primitive {primitive} references absent texture {index:?}"
        ));
    }
    Ok(())
}

fn validate_animation(animation: &AnimationArchive) -> Result<(), String> {
    let joint_count = animation
        .skeleton
        .as_ref()
        .map_or(0, |value| value.joints.len());
    if let Some(skeleton) = &animation.skeleton {
        for root in &skeleton.root_joints {
            if *root as usize >= joint_count {
                return Err(format!("skeleton root joint {root} is absent"));
            }
        }
        for (index, joint) in skeleton.joints.iter().enumerate() {
            finite_matrix(&joint.inverse_bind, &format!("joint {index} inverse bind"))?;
            finite_slice(
                &joint.rest_translation,
                &format!("joint {index} translation"),
            )?;
            finite_slice(&joint.rest_rotation, &format!("joint {index} rotation"))?;
            finite_slice(&joint.rest_scale, &format!("joint {index} scale"))?;
            if joint
                .children
                .iter()
                .any(|child| *child as usize >= joint_count)
            {
                return Err(format!("joint {index} references an absent child"));
            }
        }
    }
    if animation
        .reference_rest_rotations
        .as_ref()
        .is_some_and(|values| values.len() != joint_count)
    {
        return Err("reference rest rotations do not match skeleton joints".to_string());
    }
    for (clip_index, clip) in animation.clips.iter().enumerate() {
        if !clip.duration.is_finite() || clip.duration < 0.0 {
            return Err(format!("animation clip {clip_index} has invalid duration"));
        }
        for channel in &clip.channels {
            if channel.joint_index as usize >= joint_count {
                return Err(format!(
                    "animation clip {clip_index} references absent joint {}",
                    channel.joint_index
                ));
            }
            validate_timestamps(&channel.timestamps, "primary")?;
            if channel.translations.len() > channel.timestamps.len() {
                return Err(
                    "animation translation values exceed the primary timestamp track".to_string(),
                );
            }
            validate_track(
                &channel.rotation_timestamps,
                channel.rotations.len(),
                "rotation",
            )?;
            validate_track(&channel.scale_timestamps, channel.scales.len(), "scale")?;
            for value in &channel.translations {
                finite_slice(value, "animation translation")?;
            }
            for value in &channel.rotations {
                finite_slice(value, "animation rotation")?;
            }
            for value in &channel.scales {
                finite_slice(value, "animation scale")?;
            }
        }
    }
    Ok(())
}

fn validate_track(timestamps: &[f32], values: usize, label: &str) -> Result<(), String> {
    if timestamps.len() != values {
        return Err(format!("animation {label} timestamps do not match values"));
    }
    validate_timestamps(timestamps, label)
}

fn validate_timestamps(timestamps: &[f32], label: &str) -> Result<(), String> {
    if timestamps
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || timestamps.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(format!("animation {label} timestamps are invalid"));
    }
    Ok(())
}

fn finite_matrix(matrix: &[[f32; 4]; 4], label: &str) -> Result<(), String> {
    for row in matrix {
        finite_slice(row, label)?;
    }
    Ok(())
}

fn finite_slice(values: &[f32], label: &str) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} contains a non-finite value"));
    }
    Ok(())
}

fn validate_logical_id(id: &str) -> Result<(), String> {
    if id.is_empty() || id.len() > 512 || !id.is_ascii() || id.contains('\\') || id.starts_with('/')
    {
        return Err(format!("invalid cooked texture logical ID {id:?}"));
    }
    if id
        .split('/')
        .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(format!("invalid cooked texture logical ID {id:?}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> SceneArchive {
        SceneArchive {
            source_geometry_sha256: [7; 32],
            bbox_min: [-1.0; 3],
            bbox_max: [1.0; 3],
            primitives: vec![Primitive {
                vertices: vec![
                    Vertex {
                        position: [0.0, 0.0, 0.0],
                        normal: [0.0, 1.0, 0.0],
                        color: [1.0; 4],
                        uv: [0.0; 2],
                        joints: [0.0; 4],
                        weights: [0.0; 4],
                        tangent: [0.0; 4],
                    },
                    Vertex {
                        position: [1.0, 0.0, 0.0],
                        normal: [0.0, 1.0, 0.0],
                        color: [1.0; 4],
                        uv: [1.0, 0.0],
                        joints: [0.0; 4],
                        weights: [0.0; 4],
                        tangent: [0.0; 4],
                    },
                    Vertex {
                        position: [0.0, 0.0, 1.0],
                        normal: [0.0, 1.0, 0.0],
                        color: [1.0; 4],
                        uv: [0.0, 1.0],
                        joints: [0.0; 4],
                        weights: [0.0; 4],
                        tangent: [0.0; 4],
                    },
                ],
                secondary_tex_coords: None,
                indices: vec![0, 1, 2],
                material: Material {
                    base_color_texture: None,
                    normal_texture: None,
                    metallic_roughness_texture: None,
                    specular_glossiness_factor: None,
                    emissive_texture: None,
                    occlusion_texture: None,
                    metallic_factor: 0.0,
                    roughness_factor: 1.0,
                    emissive_factor: [0.0; 3],
                    alpha_mode: AlphaMode::Opaque,
                    alpha_cutoff: 0.0,
                    alpha_coverage_mips: false,
                    double_sided: false,
                    transmission: Transmission {
                        authored: false,
                        factor: 0.0,
                        texture: None,
                        ior_authored: false,
                        ior: 1.5,
                        volume_authored: false,
                        thickness_factor: 0.0,
                        thickness_texture: None,
                        attenuation_distance: f32::INFINITY,
                        attenuation_color: [1.0; 3],
                        thickness_source: ThicknessSource::Unavailable,
                        baked_thickness_scale: 1.0,
                    },
                    layered_pbr: LayeredPbr {
                        clearcoat_authored: false,
                        clearcoat_factor: 0.0,
                        clearcoat_roughness_factor: 0.0,
                        clearcoat_texture: None,
                        clearcoat_roughness_texture: None,
                        clearcoat_normal_texture: None,
                        clearcoat_normal_scale: 1.0,
                        specular_authored: false,
                        specular_factor: 1.0,
                        specular_texture: None,
                        specular_color_factor: [1.0; 3],
                        specular_color_texture: None,
                        ior_authored: false,
                        ior: 1.5,
                        sheen_authored: false,
                        sheen_color_factor: [0.0; 3],
                        sheen_color_texture: None,
                        sheen_roughness_factor: 0.0,
                        sheen_roughness_texture: None,
                        anisotropy_authored: false,
                        anisotropy_strength: 0.0,
                        anisotropy_rotation: 0.0,
                        anisotropy_texture: None,
                        iridescence_authored: false,
                        iridescence_factor: 0.0,
                        iridescence_texture: None,
                        iridescence_ior: 1.3,
                        iridescence_thickness_minimum: 100.0,
                        iridescence_thickness_maximum: 400.0,
                        iridescence_thickness_texture: None,
                    },
                },
            }],
            placements: vec![Placement {
                primitive: 0,
                transform: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                cast_shadow: true,
                source: Some(PrimitiveSource {
                    mesh_index: 0,
                    primitive_index: 0,
                    placement_index: 0,
                }),
            }],
            textures: Vec::new(),
            animation: None,
            diagnostics: SceneDiagnostics::default(),
        }
    }

    #[test]
    fn scene_round_trip_is_deterministic_and_hashed() {
        let archive = fixture();
        let first = encode_scene(&archive).unwrap();
        let second = encode_scene(&archive).unwrap();
        assert_eq!(first, second);
        assert_eq!(decode_scene(&first).unwrap().archive, archive);
        let mut damaged = first;
        *damaged.last_mut().unwrap() ^= 0x40;
        assert_eq!(
            decode_scene(&damaged).unwrap_err(),
            "cooked scene payload hash mismatch"
        );
    }

    #[test]
    fn incompatible_version_requests_a_recook() {
        let mut bytes = encode_scene(&fixture()).unwrap();
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert!(decode_scene(&bytes).unwrap_err().contains("recook"));
    }
}
