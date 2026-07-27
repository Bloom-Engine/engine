//! Import-only validator for canonical layered-material glTF assets.
//!
//! Usage:
//! `cargo run --example validate_layered_gltf --features models3d -- \
//!    sheen=/path/to/CompareSheen.glb \
//!    anisotropy=/path/to/AnisotropyStrengthTest.glb \
//!    iridescence=/path/to/CompareIridescence.glb`

use bloom_shared::models::load_gltf_staged;
use std::path::Path;
use std::process::ExitCode;

fn has_sheen(material: bloom_shared::models::MaterialLayeredPbr) -> bool {
    material.sheen_authored
        && material
            .sheen_color_factor
            .iter()
            .any(|value| value.is_finite() && *value > 0.0)
}

fn has_anisotropy(material: bloom_shared::models::MaterialLayeredPbr) -> bool {
    material.anisotropy_authored
        && material.anisotropy_strength.is_finite()
        && material.anisotropy_strength > 0.0
}

fn has_iridescence(material: bloom_shared::models::MaterialLayeredPbr) -> bool {
    material.iridescence_authored
        && material.iridescence_factor.is_finite()
        && material.iridescence_factor > 0.0
        && material.iridescence_ior.is_finite()
        && material.iridescence_ior >= 1.0
        && (material.iridescence_thickness_maximum > 0.0
            || (material.iridescence_thickness_texture.is_some()
                && material.iridescence_thickness_minimum > 0.0))
}

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1).peekable();
    if arguments.peek().is_none() {
        eprintln!("expected one or more KIND=PATH arguments");
        return ExitCode::from(2);
    }

    let mut failed = false;
    for argument in arguments {
        let Some((kind, path)) = argument.split_once('=') else {
            eprintln!("{argument}: expected KIND=PATH");
            failed = true;
            continue;
        };
        if !matches!(kind, "sheen" | "anisotropy" | "iridescence") {
            eprintln!("{argument}: KIND must be sheen, anisotropy, or iridescence");
            failed = true;
            continue;
        }
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!("{}: {error}", Path::new(path).display());
                failed = true;
                continue;
            }
        };
        let Some(staged) = load_gltf_staged(&bytes) else {
            eprintln!("{}: import failed", Path::new(path).display());
            failed = true;
            continue;
        };
        let model = &staged.model;

        let sheen_meshes = model
            .meshes
            .iter()
            .filter(|mesh| has_sheen(mesh.layered_pbr))
            .count();
        let anisotropy_meshes = model
            .meshes
            .iter()
            .filter(|mesh| has_anisotropy(mesh.layered_pbr))
            .count();
        let iridescence_meshes = model
            .meshes
            .iter()
            .filter(|mesh| has_iridescence(mesh.layered_pbr))
            .count();
        let authored_tangents = model
            .meshes
            .iter()
            .flat_map(|mesh| &mesh.vertices)
            .filter(|vertex| {
                let tangent = vertex.tangent;
                tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2] > 1e-6
                    && tangent[3].abs() > 0.5
            })
            .count();
        let expected_count = match kind {
            "sheen" => sheen_meshes,
            "anisotropy" => anisotropy_meshes,
            "iridescence" => iridescence_meshes,
            _ => unreachable!(),
        };
        if expected_count == 0 {
            eprintln!(
                "{}: imported {} meshes but found no active {kind} material",
                Path::new(path).display(),
                model.meshes.len(),
            );
            failed = true;
            continue;
        }
        println!(
            "{}: meshes={} sheen={} anisotropy={} iridescence={} \
             authored_tangent_vertices={} bounds={:?}..{:?}",
            Path::new(path).display(),
            model.meshes.len(),
            sheen_meshes,
            anisotropy_meshes,
            iridescence_meshes,
            authored_tangents,
            model.bbox_min,
            model.bbox_max,
        );
        for (texture_index, texture) in staged.textures.iter().enumerate() {
            let mut minimum = [u8::MAX; 4];
            let mut maximum = [u8::MIN; 4];
            let mut sum = [0u64; 4];
            for texel in texture.data.chunks_exact(4) {
                for channel in 0..4 {
                    minimum[channel] = minimum[channel].min(texel[channel]);
                    maximum[channel] = maximum[channel].max(texel[channel]);
                    sum[channel] += u64::from(texel[channel]);
                }
            }
            let texel_count = u64::from(texture.width) * u64::from(texture.height);
            let mean = sum.map(|value| value as f64 / texel_count as f64);
            println!(
                "  texture[{}]: {}x{} rgba_min={minimum:?} rgba_max={maximum:?} \
                 rgba_mean={mean:?} normal={} coverage_reference={:?}",
                texture_index + 1,
                texture.width,
                texture.height,
                texture.is_normal,
                texture.alpha_coverage_reference,
            );
        }
        for (mesh_index, mesh) in model.meshes.iter().enumerate() {
            let material = mesh.layered_pbr;
            let mut normal_minimum = [f32::INFINITY; 3];
            let mut normal_maximum = [f32::NEG_INFINITY; 3];
            let mut position_minimum = [f32::INFINITY; 3];
            let mut position_maximum = [f32::NEG_INFINITY; 3];
            for vertex in &mesh.vertices {
                for channel in 0..3 {
                    normal_minimum[channel] = normal_minimum[channel].min(vertex.normal[channel]);
                    normal_maximum[channel] = normal_maximum[channel].max(vertex.normal[channel]);
                    position_minimum[channel] =
                        position_minimum[channel].min(vertex.position[channel]);
                    position_maximum[channel] =
                        position_maximum[channel].max(vertex.position[channel]);
                }
            }
            let center = [
                (position_minimum[0] + position_maximum[0]) * 0.5,
                (position_minimum[1] + position_maximum[1]) * 0.5,
                (position_minimum[2] + position_maximum[2]) * 0.5,
            ];
            let mut radial_normal_dot = [f32::INFINITY, f32::NEG_INFINITY, 0.0];
            for vertex in &mesh.vertices {
                let radial = [
                    vertex.position[0] - center[0],
                    vertex.position[1] - center[1],
                    vertex.position[2] - center[2],
                ];
                let radial_length =
                    (radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2]).sqrt();
                if radial_length > 1e-6 {
                    let dot = (radial[0] * vertex.normal[0]
                        + radial[1] * vertex.normal[1]
                        + radial[2] * vertex.normal[2])
                        / radial_length;
                    radial_normal_dot[0] = radial_normal_dot[0].min(dot);
                    radial_normal_dot[1] = radial_normal_dot[1].max(dot);
                    radial_normal_dot[2] += dot;
                }
            }
            radial_normal_dot[2] /= mesh.vertices.len().max(1) as f32;
            println!(
                "  mesh[{mesh_index}]: metallic={:.5} roughness={:.5} \
                 base_texture={:?} mr_texture={:?} iridescence_active={} \
                 position_range={position_minimum:?}..{position_maximum:?} \
                 normal_range={normal_minimum:?}..{normal_maximum:?} \
                 radial_normal_dot[min,max,mean]={radial_normal_dot:?} \
                 iridescence={{authored={}, factor={:.5}, ior={:.5}, \
                 thickness_nm={:.5}..{:.5}, factor_texture={:?}, \
                 thickness_texture={:?}}}",
                mesh.metallic_factor,
                mesh.roughness_factor,
                mesh.texture_idx,
                mesh.metallic_roughness_texture_idx,
                has_iridescence(material),
                material.iridescence_authored,
                material.iridescence_factor,
                material.iridescence_ior,
                material.iridescence_thickness_minimum,
                material.iridescence_thickness_maximum,
                material.iridescence_texture,
                material.iridescence_thickness_texture,
            );
        }
    }

    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
