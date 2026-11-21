//! glTF morph targets into `MorphTargets` (see `morph_targets.rs`), and
//! `weights` animation channels into `MorphTrack`s (see `anim_morph.rs`).
//!
//! Every loader calls the same two functions, so the three model paths and
//! the animation path agree on the slot numbering: slots are the distinct
//! target NAMES of the file (`mesh.extras.targetNames`), in the order they
//! first appear over the meshes in file order. A character whose body mesh
//! and five face primitives all carry "jawOpen" has one "jawOpen" slot, and
//! its weight drives all six. Unnamed targets are `"<mesh name or index>#n"`.
//!
//! Accessors are read through the gltf crate's iterators, which resolve
//! sparse accessors (Blender writes shape keys sparse), with or without a
//! base buffer view. Deltas below `MORPH_DELTA_EPSILON` are dropped, so a
//! target that moves nothing on a primitive costs nothing there.

use super::*;
use crate::morph_targets::{MorphTargetsBuilder, MAX_MORPH_SLOTS};

/// The file's slot names and, per glTF mesh, the slot of each target.
pub(super) struct MorphSlots {
    pub names: Arc<Vec<String>>,
    pub per_mesh: Vec<Vec<u32>>,
}

fn target_names(mesh: &gltf::Mesh<'_>) -> Vec<String> {
    let count = mesh
        .primitives()
        .map(|p| p.morph_targets().count())
        .max()
        .unwrap_or(0);
    let authored: Vec<String> = mesh
        .extras()
        .as_ref()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw.get()).ok())
        .and_then(|extras| {
            extras.get("targetNames").and_then(|names| {
                names.as_array().map(|a| {
                    a.iter()
                        .map(|n| n.as_str().unwrap_or("").to_string())
                        .collect()
                })
            })
        })
        .unwrap_or_default();
    let mesh_label = mesh
        .name()
        .map(str::to_string)
        .unwrap_or_else(|| mesh.index().to_string());
    (0..count)
        .map(|i| match authored.get(i) {
            Some(name) if !name.is_empty() => name.clone(),
            _ => format!("{mesh_label}#{i}"),
        })
        .collect()
}

pub(super) fn morph_slots(gltf: &gltf::Gltf) -> MorphSlots {
    let mut names: Vec<String> = Vec::new();
    let mut per_mesh = Vec::new();
    for mesh in gltf.meshes() {
        let mut slots = Vec::new();
        for name in target_names(&mesh) {
            let slot = match names.iter().position(|n| *n == name) {
                Some(i) => i,
                None if names.len() < MAX_MORPH_SLOTS => {
                    names.push(name);
                    names.len() - 1
                }
                None => {
                    log::warn!("bloom: more than {MAX_MORPH_SLOTS} morph target names; '{name}' is ignored");
                    usize::MAX
                }
            };
            slots.push(slot.min(u32::MAX as usize) as u32);
        }
        per_mesh.push(slots);
    }
    MorphSlots {
        names: Arc::new(names),
        per_mesh,
    }
}

impl MorphSlots {
    /// Read one primitive's targets. `vertices` are the loader's finished
    /// vertices (to tell skinned from rigid ones); skinned deltas are scaled by
    /// `skin_vertex_scale` exactly as the loader scaled their positions.
    pub(super) fn read<'a, 's, F>(
        &self,
        reader: &gltf::mesh::Reader<'a, 's, F>,
        vertices: &[Vertex3D],
        skin_vertex_scale: f32,
        mesh_index: usize,
    ) -> Option<Arc<crate::morph_targets::MorphTargets>>
    where
        F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
    {
        let slots = self;
        let mesh_slots = slots.per_mesh.get(mesh_index)?;
        let scale_skinned = (skin_vertex_scale - 1.0).abs() > 0.01;
        let mut builder = MorphTargetsBuilder::new(vertices.len());
        let mut any = false;
        for (target, (positions, normals, tangents)) in reader.read_morph_targets().enumerate() {
            let Some(&slot) = mesh_slots.get(target) else {
                continue;
            };
            if slot == u32::MAX {
                continue;
            }
            let positions: Option<Vec<[f32; 3]>> = positions.map(|i| i.collect());
            let normals: Option<Vec<[f32; 3]>> = normals.map(|i| i.collect());
            let tangents: Option<Vec<[f32; 3]>> = tangents.map(|i| i.collect());
            any = true;
            builder.add_target(
                slot,
                positions.as_deref(),
                normals.as_deref(),
                tangents.as_deref(),
                |v, mut d| {
                    let w = vertices[v].weights;
                    if scale_skinned && w[0] + w[1] + w[2] + w[3] > 0.01 {
                        for c in &mut d[0..3] {
                            *c *= skin_vertex_scale;
                        }
                    }
                    d
                },
            );
        }
        if !any {
            return None;
        }
        builder.build(slots.names.clone()).map(Arc::new)
    }
}

/// A `weights` channel of an animation, as a track over model slots.
/// `None` when the channel is not a weights channel or its node has no mesh.
pub(super) fn read_weight_channel<'a, 's, F>(
    channel: &gltf::animation::Channel<'a>,
    get_buffer_data: F,
    slots: &MorphSlots,
) -> Option<crate::anim_morph::MorphTrack>
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    use gltf::animation::util::ReadOutputs;
    if channel.target().property() != gltf::animation::Property::MorphTargetWeights {
        return None;
    }
    let mesh = channel.target().node().mesh()?;
    let mesh_slots = slots.per_mesh.get(mesh.index())?;
    let n = mesh_slots.len();
    if n == 0 {
        return None;
    }
    let reader = channel.reader(get_buffer_data);
    let times: Vec<f32> = reader.read_inputs()?.collect();
    let raw: Vec<f32> = match reader.read_outputs()? {
        ReadOutputs::MorphTargetWeights(w) => w.into_f32().collect(),
        _ => return None,
    };
    let interpolation = channel.sampler().interpolation();
    let values = if interpolation == gltf::animation::Interpolation::CubicSpline {
        // in-tangent, value, out-tangent per key: keep the values.
        raw.chunks(n * 3)
            .flat_map(|key| key[n..(2 * n).min(key.len())].iter().copied())
            .collect()
    } else {
        raw
    };
    if values.len() < times.len() * n {
        return None;
    }
    Some(crate::anim_morph::MorphTrack {
        slots: mesh_slots.clone(),
        times,
        values,
        step: interpolation == gltf::animation::Interpolation::Step,
    })
}

#[cfg(test)]
#[path = "models_gltf_morph_tests.rs"]
mod tests;
