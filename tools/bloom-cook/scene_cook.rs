//! Convert Bloom's staged glTF import into the shared source-free scene format.

use crate::texture_cook::{PreparedTexture, TextureSettings};
use bloom_scene_format as format;
use bloom_shared::models as runtime;
use bloom_shared::staging::StagedModel;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub(crate) const SCENE_RECIPE_VERSION: u32 = 1;

pub(crate) struct PreparedScene {
    pub(crate) archive: format::SceneArchive,
    pub(crate) textures: Vec<PreparedTexture>,
    pub(crate) sanitation: SceneSanitation,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SceneSanitation {
    pub(crate) non_finite_position_vertices: u64,
    pub(crate) non_finite_attribute_vertices: u64,
    pub(crate) dropped_triangles: u64,
    pub(crate) dropped_primitives: u64,
    pub(crate) dropped_placements: u64,
}

pub(crate) fn prepare_scene(input: &Path, logical_id: &str) -> Result<PreparedScene, String> {
    let source =
        std::fs::read(input).map_err(|error| format!("read {}: {error}", input.display()))?;
    let staged = runtime::load_gltf_staged_from_source_path(&source, input)
        .ok_or_else(|| format!("stage source scene {}", input.display()))?;
    let animation = runtime::load_gltf_animation(&source)
        .filter(|animation| animation.skeleton.is_some() || !animation.animations.is_empty());
    convert_scene(staged, animation, logical_id)
}

fn convert_scene(
    staged: StagedModel,
    animation: Option<runtime::ModelAnimation>,
    logical_id: &str,
) -> Result<PreparedScene, String> {
    let StagedModel { model, textures } = staged;
    let mut prepared_textures = Vec::with_capacity(textures.len());
    let mut texture_dependencies = Vec::with_capacity(textures.len());
    for (index, texture) in textures.into_iter().enumerate() {
        let settings = TextureSettings::from_semantics(
            texture.is_normal,
            texture.is_srgb,
            texture.alpha_coverage_reference,
        );
        let width = texture.width;
        let height = texture.height;
        let alpha_coverage_reference = texture.alpha_coverage_reference;
        let prepared = PreparedTexture::from_rgba(width, height, texture.data, settings)?;
        texture_dependencies.push(format::TextureDependency {
            logical_id: format!("{logical_id}/textures/{:04}", index + 1),
            source_sha256: prepared.source_sha256,
            width,
            height,
            is_normal: texture.is_normal,
            is_srgb: texture.is_srgb,
            alpha_coverage_reference,
        });
        prepared_textures.push(prepared);
    }

    let mut sanitation = SceneSanitation::default();
    let mut unique = HashMap::<usize, Option<u32>>::new();
    let mut primitives = Vec::new();
    let mut placements = Vec::with_capacity(model.meshes.len());
    for (index, mesh) in model.meshes.iter().enumerate() {
        let identity = Arc::as_ptr(mesh) as usize;
        let primitive = if let Some(primitive) = unique.get(&identity) {
            *primitive
        } else {
            let converted = convert_primitive(mesh, &mut sanitation);
            let primitive = converted
                .map(|converted| -> Result<u32, String> {
                    let index = u32::try_from(primitives.len())
                        .map_err(|_| "cooked scene primitive count exceeds u32".to_string())?;
                    primitives.push(converted);
                    Ok(index)
                })
                .transpose()?;
            unique.insert(identity, primitive);
            primitive
        };
        let Some(primitive) = primitive else {
            sanitation.dropped_placements = sanitation.dropped_placements.saturating_add(1);
            continue;
        };
        let transform = model.mesh_transform(index);
        if transform.iter().flatten().any(|value| !value.is_finite()) {
            sanitation.dropped_placements = sanitation.dropped_placements.saturating_add(1);
            continue;
        }
        placements.push(format::Placement {
            primitive,
            transform,
            cast_shadow: model.mesh_cast_shadow(index),
            source: model
                .mesh_source(index)
                .map(|source| format::PrimitiveSource {
                    mesh_index: source.mesh_index,
                    primitive_index: source.primitive_index,
                    placement_index: source.placement_index,
                }),
        });
    }
    let source_geometry_sha256 = model
        .source_geometry_sha256
        .ok_or("staged glTF is missing its source-geometry closure hash")?;
    let (bbox_min, bbox_max) = cooked_bounds(&primitives, &placements)?;
    let archive = format::SceneArchive {
        source_geometry_sha256,
        bbox_min,
        bbox_max,
        primitives,
        placements,
        textures: texture_dependencies,
        animation: animation.map(convert_animation).transpose()?,
        diagnostics: format::SceneDiagnostics {
            non_finite_position_vertices: sanitation.non_finite_position_vertices,
            non_finite_attribute_vertices: sanitation.non_finite_attribute_vertices,
            dropped_triangles: sanitation.dropped_triangles,
            dropped_primitives: sanitation.dropped_primitives,
            dropped_placements: sanitation.dropped_placements,
        },
    };
    format::validate_scene(&archive)?;
    Ok(PreparedScene {
        archive,
        textures: prepared_textures,
        sanitation,
    })
}

fn convert_primitive(
    mesh: &runtime::MeshData,
    sanitation: &mut SceneSanitation,
) -> Option<format::Primitive> {
    let invalid_positions = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.position.iter().any(|value| !value.is_finite()))
        .collect::<Vec<_>>();
    sanitation.non_finite_position_vertices = sanitation
        .non_finite_position_vertices
        .saturating_add(invalid_positions.iter().filter(|value| **value).count() as u64);
    let indices = mesh
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter(|triangle| {
            let invalid = triangle.iter().any(|index| {
                invalid_positions
                    .get(*index as usize)
                    .copied()
                    .unwrap_or(true)
            });
            if invalid {
                sanitation.dropped_triangles = sanitation.dropped_triangles.saturating_add(1);
            }
            !invalid
        })
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    if indices.is_empty() {
        sanitation.dropped_primitives = sanitation.dropped_primitives.saturating_add(1);
        return None;
    }
    let vertices = mesh
        .vertices
        .iter()
        .zip(invalid_positions)
        .map(|(vertex, invalid_position)| {
            let mut repaired = false;
            let mut finite = |values: &[f32], fallback: &[f32]| {
                std::array::from_fn::<_, 4, _>(|index| {
                    values
                        .get(index)
                        .copied()
                        .filter(|value| value.is_finite())
                        .unwrap_or_else(|| {
                            repaired = true;
                            fallback[index]
                        })
                })
            };
            let normal4 = finite(&vertex.normal, &[0.0, 1.0, 0.0, 0.0]);
            let color = finite(&vertex.color, &[1.0; 4]);
            let uv4 = finite(&vertex.uv, &[0.0; 4]);
            let joints = finite(&vertex.joints, &[0.0; 4]);
            let weights = finite(&vertex.weights, &[0.0; 4]);
            let tangent = finite(&vertex.tangent, &[0.0; 4]);
            if repaired {
                sanitation.non_finite_attribute_vertices =
                    sanitation.non_finite_attribute_vertices.saturating_add(1);
            }
            format::Vertex {
                position: if invalid_position {
                    [0.0; 3]
                } else {
                    vertex.position
                },
                normal: [normal4[0], normal4[1], normal4[2]],
                color,
                uv: [uv4[0], uv4[1]],
                joints,
                weights,
                tangent,
            }
        })
        .collect();
    let secondary_tex_coords = mesh.secondary_tex_coords.as_ref().map(|values| {
        values
            .iter()
            .map(|uv| {
                uv.map(|value| {
                    if value.is_finite() {
                        value
                    } else {
                        sanitation.non_finite_attribute_vertices =
                            sanitation.non_finite_attribute_vertices.saturating_add(1);
                        0.0
                    }
                })
            })
            .collect()
    });
    Some(format::Primitive {
        vertices,
        secondary_tex_coords,
        indices,
        material: format::Material {
            base_color_texture: mesh.texture_idx,
            normal_texture: mesh.normal_texture_idx,
            metallic_roughness_texture: mesh.metallic_roughness_texture_idx,
            specular_glossiness_factor: mesh.specular_glossiness_factor,
            emissive_texture: mesh.emissive_texture_idx,
            occlusion_texture: mesh.occlusion_texture_idx,
            metallic_factor: mesh.metallic_factor,
            roughness_factor: mesh.roughness_factor,
            emissive_factor: mesh.emissive_factor,
            alpha_mode: match mesh.alpha_mode {
                runtime::MaterialAlphaMode::Opaque => format::AlphaMode::Opaque,
                runtime::MaterialAlphaMode::Mask => format::AlphaMode::Mask,
                runtime::MaterialAlphaMode::Blend => format::AlphaMode::Blend,
            },
            alpha_cutoff: mesh.alpha_cutoff,
            alpha_coverage_mips: mesh.alpha_coverage_mips,
            double_sided: mesh.double_sided,
            transmission: convert_transmission(mesh.transmission),
            layered_pbr: convert_layered(mesh.layered_pbr),
        },
    })
}

fn cooked_bounds(
    primitives: &[format::Primitive],
    placements: &[format::Placement],
) -> Result<([f32; 3], [f32; 3]), String> {
    let mut minimum = [f32::MAX; 3];
    let mut maximum = [f32::MIN; 3];
    for placement in placements {
        let primitive = &primitives[placement.primitive as usize];
        for index in &primitive.indices {
            let position = primitive.vertices[*index as usize].position;
            let matrix = &placement.transform;
            let world = [
                matrix[0][0] * position[0]
                    + matrix[1][0] * position[1]
                    + matrix[2][0] * position[2]
                    + matrix[3][0],
                matrix[0][1] * position[0]
                    + matrix[1][1] * position[1]
                    + matrix[2][1] * position[2]
                    + matrix[3][1],
                matrix[0][2] * position[0]
                    + matrix[1][2] * position[1]
                    + matrix[2][2] * position[2]
                    + matrix[3][2],
            ];
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(world[axis]);
                maximum[axis] = maximum[axis].max(world[axis]);
            }
        }
    }
    if placements.is_empty() || minimum.iter().any(|value| !value.is_finite()) {
        return Err("scene sanitation removed every finite placement".to_string());
    }
    Ok((minimum, maximum))
}

fn convert_transform(value: runtime::MaterialTextureTransform) -> format::TextureTransform {
    format::TextureTransform {
        offset: value.offset,
        rotation: value.rotation,
        scale: value.scale,
        tex_coord: value.tex_coord,
    }
}

fn convert_binding(value: runtime::MaterialTextureBinding) -> format::TextureBinding {
    format::TextureBinding {
        source_texture_index: value.source_texture_index,
        source_image_index: value.source_image_index,
        texture: value.runtime_texture_idx,
        transform: convert_transform(value.transform),
    }
}

fn convert_transmission(value: runtime::MaterialTransmission) -> format::Transmission {
    format::Transmission {
        authored: value.authored,
        factor: value.factor,
        texture: value.texture.map(convert_binding),
        ior_authored: value.ior_authored,
        ior: value.ior,
        volume_authored: value.volume_authored,
        thickness_factor: value.thickness_factor,
        thickness_texture: value.thickness_texture.map(convert_binding),
        attenuation_distance: value.attenuation_distance,
        attenuation_color: value.attenuation_color,
        thickness_source: match value.thickness_source {
            runtime::MaterialThicknessSource::Unavailable => format::ThicknessSource::Unavailable,
            runtime::MaterialThicknessSource::Authored => format::ThicknessSource::Authored,
            runtime::MaterialThicknessSource::Approximated => format::ThicknessSource::Approximated,
        },
        baked_thickness_scale: value.baked_thickness_scale,
    }
}

fn convert_layered(value: runtime::MaterialLayeredPbr) -> format::LayeredPbr {
    format::LayeredPbr {
        clearcoat_authored: value.clearcoat_authored,
        clearcoat_factor: value.clearcoat_factor,
        clearcoat_roughness_factor: value.clearcoat_roughness_factor,
        clearcoat_texture: value.clearcoat_texture.map(convert_binding),
        clearcoat_roughness_texture: value.clearcoat_roughness_texture.map(convert_binding),
        clearcoat_normal_texture: value.clearcoat_normal_texture.map(convert_binding),
        clearcoat_normal_scale: value.clearcoat_normal_scale,
        specular_authored: value.specular_authored,
        specular_factor: value.specular_factor,
        specular_texture: value.specular_texture.map(convert_binding),
        specular_color_factor: value.specular_color_factor,
        specular_color_texture: value.specular_color_texture.map(convert_binding),
        ior_authored: value.ior_authored,
        ior: value.ior,
        sheen_authored: value.sheen_authored,
        sheen_color_factor: value.sheen_color_factor,
        sheen_color_texture: value.sheen_color_texture.map(convert_binding),
        sheen_roughness_factor: value.sheen_roughness_factor,
        sheen_roughness_texture: value.sheen_roughness_texture.map(convert_binding),
        anisotropy_authored: value.anisotropy_authored,
        anisotropy_strength: value.anisotropy_strength,
        anisotropy_rotation: value.anisotropy_rotation,
        anisotropy_texture: value.anisotropy_texture.map(convert_binding),
        iridescence_authored: value.iridescence_authored,
        iridescence_factor: value.iridescence_factor,
        iridescence_texture: value.iridescence_texture.map(convert_binding),
        iridescence_ior: value.iridescence_ior,
        iridescence_thickness_minimum: value.iridescence_thickness_minimum,
        iridescence_thickness_maximum: value.iridescence_thickness_maximum,
        iridescence_thickness_texture: value.iridescence_thickness_texture.map(convert_binding),
    }
}

fn convert_animation(value: runtime::ModelAnimation) -> Result<format::AnimationArchive, String> {
    let skeleton = value
        .skeleton
        .as_ref()
        .map(|skeleton| -> Result<_, String> {
            Ok(format::Skeleton {
                joints: skeleton
                    .joints
                    .iter()
                    .map(|joint| -> Result<_, String> {
                        Ok(format::Joint {
                            inverse_bind: joint.inverse_bind,
                            children: joint
                                .children
                                .iter()
                                .map(|index| {
                                    u32::try_from(*index).map_err(|_| {
                                        "animation joint index exceeds u32".to_string()
                                    })
                                })
                                .collect::<Result<_, _>>()?,
                            name: joint.name.clone(),
                            rest_translation: joint.rest_translation,
                            rest_rotation: joint.rest_rotation,
                            rest_scale: joint.rest_scale,
                        })
                    })
                    .collect::<Result<_, _>>()?,
                root_joints: skeleton
                    .root_joints
                    .iter()
                    .map(|index| {
                        u32::try_from(*index)
                            .map_err(|_| "animation root joint index exceeds u32".to_string())
                    })
                    .collect::<Result<_, _>>()?,
            })
        })
        .transpose()?;
    let clips = value
        .animations
        .iter()
        .map(|clip| -> Result<_, String> {
            Ok(format::AnimationClip {
                channels: clip
                    .channels
                    .iter()
                    .map(|channel| {
                        Ok(format::AnimationChannel {
                            joint_index: u32::try_from(channel.joint_index).map_err(|_| {
                                "animation channel joint index exceeds u32".to_string()
                            })?,
                            timestamps: channel.timestamps.clone(),
                            translations: channel.translations.clone(),
                            rotation_timestamps: channel.rotation_timestamps.clone(),
                            rotations: channel.rotations.clone(),
                            scale_timestamps: channel.scale_timestamps.clone(),
                            scales: channel.scales.clone(),
                        })
                    })
                    .collect::<Result<_, String>>()?,
                duration: clip.duration,
                name: clip.name.clone(),
            })
        })
        .collect::<Result<_, _>>()?;
    Ok(format::AnimationArchive {
        skeleton,
        clips,
        reference_rest_rotations: value
            .ref_rest_rotations
            .as_ref()
            .map(|rotations| rotations.as_ref().clone()),
    })
}
