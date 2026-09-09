//! Source-free `.bscene` conversion into Bloom's runtime model structures.

use crate::models::{
    AnimMixer, AnimationChannel, AnimationData, JointData, MaterialAlphaMode, MaterialLayeredPbr,
    MaterialTextureBinding, MaterialTextureTransform, MaterialThicknessSource,
    MaterialTransmission, MeshData, ModelAnimation, ModelData, ModelManager, ModelPrimitiveSource,
    SkeletonData,
};
use crate::renderer::Vertex3D;
use bloom_scene_format as format;
use std::sync::Arc;

pub struct PreparedCookedScene {
    decoded: format::DecodedScene,
}

pub struct CookedScene {
    pub model: ModelData,
    pub animation: Option<ModelAnimation>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct InstalledCookedScene {
    pub model: f64,
    pub animation: Option<f64>,
}

pub fn prepare_cooked_scene(bytes: &[u8]) -> Result<PreparedCookedScene, String> {
    Ok(PreparedCookedScene {
        decoded: format::decode_scene(bytes)?,
    })
}

impl ModelManager {
    /// Install a validated source-free scene after its indexed texture
    /// dependencies have been uploaded. Handles follow dependency order;
    /// no glTF parser or source path participates.
    pub fn load_cooked_scene(
        &mut self,
        file_data: &[u8],
        texture_handles: &[u32],
    ) -> Result<InstalledCookedScene, String> {
        let cooked = prepare_cooked_scene(file_data)?.finish(texture_handles)?;
        let model = self.models.alloc(cooked.model);
        let animation = cooked
            .animation
            .map(|animation| self.animations.alloc(animation));
        Ok(InstalledCookedScene { model, animation })
    }
}

impl PreparedCookedScene {
    pub fn texture_dependencies(&self) -> &[format::TextureDependency] {
        &self.decoded.archive.textures
    }

    pub fn payload_sha256(&self) -> [u8; 32] {
        self.decoded.payload_sha256
    }

    pub fn finish(self, texture_handles: &[u32]) -> Result<CookedScene, String> {
        let archive = self.decoded.archive;
        if archive.textures.len() != texture_handles.len() {
            return Err(format!(
                "cooked scene requires {} textures, received {} handles",
                archive.textures.len(),
                texture_handles.len()
            ));
        }
        let primitives = archive
            .primitives
            .into_iter()
            .map(|primitive| convert_primitive(primitive, texture_handles).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        let mut meshes = Vec::with_capacity(archive.placements.len());
        let mut transforms = Vec::with_capacity(archive.placements.len());
        let mut shadows = Vec::with_capacity(archive.placements.len());
        let mut sources = Vec::with_capacity(archive.placements.len());
        for placement in archive.placements {
            meshes.push(Arc::clone(&primitives[placement.primitive as usize]));
            transforms.push(placement.transform);
            shadows.push(placement.cast_shadow);
            sources.push(placement.source.map(|source| ModelPrimitiveSource {
                mesh_index: source.mesh_index,
                primitive_index: source.primitive_index,
                placement_index: source.placement_index,
            }));
        }
        Ok(CookedScene {
            model: ModelData {
                meshes,
                mesh_transforms: transforms,
                mesh_cast_shadows: shadows,
                mesh_sources: sources,
                source_geometry_sha256: Some(archive.source_geometry_sha256),
                bbox_min: archive.bbox_min,
                bbox_max: archive.bbox_max,
            },
            animation: archive.animation.map(convert_animation),
        })
    }
}

fn convert_primitive(
    primitive: format::Primitive,
    texture_handles: &[u32],
) -> Result<MeshData, String> {
    let material = primitive.material;
    Ok(MeshData {
        vertices: primitive
            .vertices
            .into_iter()
            .map(|vertex| Vertex3D {
                position: vertex.position,
                normal: vertex.normal,
                color: vertex.color,
                uv: vertex.uv,
                joints: vertex.joints,
                weights: vertex.weights,
                tangent: vertex.tangent,
            })
            .collect(),
        secondary_tex_coords: primitive.secondary_tex_coords,
        indices: primitive.indices,
        texture_idx: remap_texture(material.base_color_texture, texture_handles)?,
        normal_texture_idx: remap_texture(material.normal_texture, texture_handles)?,
        metallic_roughness_texture_idx: remap_texture(
            material.metallic_roughness_texture,
            texture_handles,
        )?,
        specular_glossiness_factor: material.specular_glossiness_factor,
        emissive_texture_idx: remap_texture(material.emissive_texture, texture_handles)?,
        occlusion_texture_idx: remap_texture(material.occlusion_texture, texture_handles)?,
        metallic_factor: material.metallic_factor,
        roughness_factor: material.roughness_factor,
        emissive_factor: material.emissive_factor,
        alpha_mode: convert_alpha(material.alpha_mode),
        alpha_cutoff: material.alpha_cutoff,
        alpha_coverage_mips: material.alpha_coverage_mips,
        double_sided: material.double_sided,
        transmission: convert_transmission(material.transmission, texture_handles)?,
        layered_pbr: convert_layered(material.layered_pbr, texture_handles)?,
    })
}

fn remap_texture(index: Option<u32>, handles: &[u32]) -> Result<Option<u32>, String> {
    match index {
        None => Ok(None),
        Some(0) => Ok(Some(0)),
        Some(index) => handles
            .get(index as usize - 1)
            .copied()
            .map(Some)
            .ok_or_else(|| format!("cooked material texture {index} is absent")),
    }
}

fn convert_binding(
    binding: format::TextureBinding,
    handles: &[u32],
) -> Result<MaterialTextureBinding, String> {
    Ok(MaterialTextureBinding {
        source_texture_index: binding.source_texture_index,
        source_image_index: binding.source_image_index,
        runtime_texture_idx: remap_texture(binding.texture, handles)?,
        transform: MaterialTextureTransform {
            offset: binding.transform.offset,
            rotation: binding.transform.rotation,
            scale: binding.transform.scale,
            tex_coord: binding.transform.tex_coord,
        },
    })
}

fn map_binding(
    binding: Option<format::TextureBinding>,
    handles: &[u32],
) -> Result<Option<MaterialTextureBinding>, String> {
    binding
        .map(|binding| convert_binding(binding, handles))
        .transpose()
}

fn convert_transmission(
    value: format::Transmission,
    handles: &[u32],
) -> Result<MaterialTransmission, String> {
    Ok(MaterialTransmission {
        authored: value.authored,
        factor: value.factor,
        texture: map_binding(value.texture, handles)?,
        ior_authored: value.ior_authored,
        ior: value.ior,
        volume_authored: value.volume_authored,
        thickness_factor: value.thickness_factor,
        thickness_texture: map_binding(value.thickness_texture, handles)?,
        attenuation_distance: value.attenuation_distance,
        attenuation_color: value.attenuation_color,
        thickness_source: match value.thickness_source {
            format::ThicknessSource::Unavailable => MaterialThicknessSource::Unavailable,
            format::ThicknessSource::Authored => MaterialThicknessSource::Authored,
            format::ThicknessSource::Approximated => MaterialThicknessSource::Approximated,
        },
        baked_thickness_scale: value.baked_thickness_scale,
    })
}

fn convert_layered(
    value: format::LayeredPbr,
    handles: &[u32],
) -> Result<MaterialLayeredPbr, String> {
    Ok(MaterialLayeredPbr {
        clearcoat_authored: value.clearcoat_authored,
        clearcoat_factor: value.clearcoat_factor,
        clearcoat_roughness_factor: value.clearcoat_roughness_factor,
        clearcoat_texture: map_binding(value.clearcoat_texture, handles)?,
        clearcoat_roughness_texture: map_binding(value.clearcoat_roughness_texture, handles)?,
        clearcoat_normal_texture: map_binding(value.clearcoat_normal_texture, handles)?,
        clearcoat_normal_scale: value.clearcoat_normal_scale,
        specular_authored: value.specular_authored,
        specular_factor: value.specular_factor,
        specular_texture: map_binding(value.specular_texture, handles)?,
        specular_color_factor: value.specular_color_factor,
        specular_color_texture: map_binding(value.specular_color_texture, handles)?,
        ior_authored: value.ior_authored,
        ior: value.ior,
        sheen_authored: value.sheen_authored,
        sheen_color_factor: value.sheen_color_factor,
        sheen_color_texture: map_binding(value.sheen_color_texture, handles)?,
        sheen_roughness_factor: value.sheen_roughness_factor,
        sheen_roughness_texture: map_binding(value.sheen_roughness_texture, handles)?,
        anisotropy_authored: value.anisotropy_authored,
        anisotropy_strength: value.anisotropy_strength,
        anisotropy_rotation: value.anisotropy_rotation,
        anisotropy_texture: map_binding(value.anisotropy_texture, handles)?,
        iridescence_authored: value.iridescence_authored,
        iridescence_factor: value.iridescence_factor,
        iridescence_texture: map_binding(value.iridescence_texture, handles)?,
        iridescence_ior: value.iridescence_ior,
        iridescence_thickness_minimum: value.iridescence_thickness_minimum,
        iridescence_thickness_maximum: value.iridescence_thickness_maximum,
        iridescence_thickness_texture: map_binding(value.iridescence_thickness_texture, handles)?,
    })
}

fn convert_alpha(value: format::AlphaMode) -> MaterialAlphaMode {
    match value {
        format::AlphaMode::Opaque => MaterialAlphaMode::Opaque,
        format::AlphaMode::Mask => MaterialAlphaMode::Mask,
        format::AlphaMode::Blend => MaterialAlphaMode::Blend,
    }
}

fn convert_animation(value: format::AnimationArchive) -> ModelAnimation {
    let skeleton = value.skeleton.map(|skeleton| {
        Arc::new(SkeletonData {
            joints: skeleton
                .joints
                .into_iter()
                .map(|joint| JointData {
                    inverse_bind: joint.inverse_bind,
                    children: joint
                        .children
                        .into_iter()
                        .map(|value| value as usize)
                        .collect(),
                    name: joint.name,
                    rest_translation: joint.rest_translation,
                    rest_rotation: joint.rest_rotation,
                    rest_scale: joint.rest_scale,
                })
                .collect(),
            root_joints: skeleton
                .root_joints
                .into_iter()
                .map(|value| value as usize)
                .collect(),
        })
    });
    let animations = Arc::new(
        value
            .clips
            .into_iter()
            .map(|clip| AnimationData {
                channels: clip
                    .channels
                    .into_iter()
                    .map(|channel| AnimationChannel {
                        joint_index: channel.joint_index as usize,
                        timestamps: channel.timestamps,
                        translations: channel.translations,
                        rotation_timestamps: channel.rotation_timestamps,
                        rotations: channel.rotations,
                        scale_timestamps: channel.scale_timestamps,
                        scales: channel.scales,
                    })
                    .collect(),
                duration: clip.duration,
                name: clip.name,
            })
            .collect(),
    );
    let joint_count = skeleton.as_ref().map_or(0, |value| value.joints.len());
    ModelAnimation {
        skeleton,
        animations,
        ref_rest_rotations: value.reference_rest_rotations.map(Arc::new),
        joint_matrices: vec![identity(); joint_count],
        mixer: AnimMixer::default(),
        joint_world: vec![identity(); joint_count],
        mask_weights: vec![0.0; joint_count],
        mask_cached_root: -1,
    }
}

fn identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
