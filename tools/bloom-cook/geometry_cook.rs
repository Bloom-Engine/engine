//! glTF-to-meshlet cooking and command-line reporting.

use crate::geometry_format::{
    decode_geometry, encode_geometry, encode_geometry_with_vertex_encoding, geometry_source_sha256,
    hex_hash, measure_vertex_error, CompatibilityReason, CompatibilityRecord, DEFAULT_PAGE_BYTES,
};
use crate::geometry_quantization::VertexEncoding;
use crate::hierarchy::{
    build_meshlet_hierarchy, build_spatial_leaf_meshlets, offset_relations, order_for_streaming,
    HierarchyStats,
};
use crate::meshlet::{build_leaf_meshlets, Meshlet, MeshletLimits, StaticPrimitive, StaticVertex};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

pub(crate) const GEOMETRY_RECIPE_VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CookOptions {
    meshlet_limits: MeshletLimits,
    page_bytes: u32,
    hierarchy_levels: u32,
    vertex_encoding: VertexEncoding,
}

impl Default for CookOptions {
    fn default() -> Self {
        Self {
            meshlet_limits: MeshletLimits::default(),
            page_bytes: DEFAULT_PAGE_BYTES,
            hierarchy_levels: 0,
            vertex_encoding: VertexEncoding::Float32,
        }
    }
}

#[derive(Debug)]
struct GeometrySource {
    source_sha256: [u8; 32],
    primitives: Vec<StaticPrimitive>,
    compatibility: Vec<CompatibilityRecord>,
    source_meshes: usize,
    source_primitives: usize,
    eligible_triangles: u64,
}

pub(crate) struct PreparedGeometry {
    options: CookOptions,
    source: GeometrySource,
}

pub(crate) struct CookedGeometry {
    pub bytes: Vec<u8>,
    pub report: serde_json::Value,
}

impl PreparedGeometry {
    pub fn source_sha256(&self) -> [u8; 32] {
        self.source.source_sha256
    }

    pub fn build_key_sha256(&self) -> [u8; 32] {
        geometry_build_key_sha256(
            self.source.source_sha256,
            self.options.meshlet_limits.max_vertices,
            self.options.meshlet_limits.max_triangles,
            self.options.page_bytes,
            self.options.hierarchy_levels,
            self.options.vertex_encoding,
        )
    }

    pub fn settings_json(&self) -> serde_json::Value {
        json!({
            "hierarchy_levels": self.options.hierarchy_levels,
            "max_triangles_per_meshlet": self.options.meshlet_limits.max_triangles,
            "max_vertices_per_meshlet": self.options.meshlet_limits.max_vertices,
            "page_budget_bytes": self.options.page_bytes,
            "vertex_format": self.options.vertex_encoding.label(),
        })
    }

    pub fn expected_format_version(&self) -> u32 {
        match self.options.vertex_encoding {
            VertexEncoding::Float32 => crate::geometry_format::VERSION,
            VertexEncoding::Quantized => crate::geometry_format::QUANTIZED_VERSION,
        }
    }
}

pub(crate) fn geometry_build_key_sha256(
    source_sha256: [u8; 32],
    max_vertices: u32,
    max_triangles: u32,
    page_bytes: u32,
    hierarchy_levels: u32,
    vertex_encoding: VertexEncoding,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"bloom-geometry-recipe\0");
    hasher.update(GEOMETRY_RECIPE_VERSION.to_le_bytes());
    hasher.update(source_sha256);
    hasher.update(max_vertices.to_le_bytes());
    hasher.update(max_triangles.to_le_bytes());
    hasher.update(page_bytes.to_le_bytes());
    hasher.update(hierarchy_levels.to_le_bytes());
    hasher.update(
        match vertex_encoding {
            VertexEncoding::Float32 => 1u32,
            VertexEncoding::Quantized => 2u32,
        }
        .to_le_bytes(),
    );
    hasher.finalize().into()
}

pub(crate) fn prepare_geometry(input: &Path, flags: &[String]) -> Result<PreparedGeometry, String> {
    Ok(PreparedGeometry {
        options: parse_options(flags)?,
        source: load_geometry_source(input)?,
    })
}

pub fn cook_geometry_command(
    input: &Path,
    output: &Path,
    flags: &[String],
) -> Result<String, String> {
    let prepared = prepare_geometry(input, flags)?;
    let mut cooked = cook_prepared_geometry(input, prepared)?;
    write_atomically(output, &cooked.bytes)?;
    cooked.report["output"] = json!(output.display().to_string());
    serde_json::to_string_pretty(&cooked.report)
        .map_err(|error| format!("serialize geometry report: {error}"))
}

pub(crate) fn cook_prepared_geometry(
    input: &Path,
    prepared: PreparedGeometry,
) -> Result<CookedGeometry, String> {
    let options = prepared.options;
    let source = prepared.source;
    let mut meshlets = Vec::<Meshlet>::new();
    let mut hierarchy = HierarchyStats::default();
    for primitive in &source.primitives {
        let leaves = if options.hierarchy_levels == 0 {
            build_leaf_meshlets(primitive, options.meshlet_limits)?
        } else {
            build_spatial_leaf_meshlets(primitive, options.meshlet_limits)?
        };
        if options.hierarchy_levels == 0 {
            meshlets.extend(leaves);
        } else {
            let base = meshlets.len();
            let (mut primitive_hierarchy, stats) =
                build_meshlet_hierarchy(leaves, options.meshlet_limits, options.hierarchy_levels)?;
            offset_relations(&mut primitive_hierarchy, base)?;
            hierarchy.merge(stats);
            meshlets.extend(primitive_hierarchy);
        }
    }
    if options.hierarchy_levels != 0 {
        order_for_streaming(&mut meshlets)?;
    }
    let bytes = encode_geometry_with_vertex_encoding(
        &meshlets,
        &source.compatibility,
        source.source_sha256,
        options.page_bytes,
        options.vertex_encoding,
    )?;
    let archive = decode_geometry(&bytes)?;
    let quantization = measure_vertex_error(&meshlets, &bytes)?;
    let float32_archive = if options.vertex_encoding == VertexEncoding::Quantized {
        Some(decode_geometry(&encode_geometry(
            &meshlets,
            &source.compatibility,
            source.source_sha256,
            options.page_bytes,
        )?)?)
    } else {
        None
    };
    let baseline_payload_bytes = float32_archive
        .as_ref()
        .map_or_else(|| archive.payload_bytes(), |value| value.payload_bytes());
    let baseline_root_page_bytes = float32_archive.as_ref().map_or_else(
        || archive.coarse_root_page_bytes(),
        |value| value.coarse_root_page_bytes(),
    );
    let payload_reduction_bytes = baseline_payload_bytes.saturating_sub(archive.payload_bytes());
    let root_reduction_bytes =
        baseline_root_page_bytes.saturating_sub(archive.coarse_root_page_bytes());
    let reduction_percent = |reduction: u64, baseline: u64| {
        if baseline == 0 {
            0.0
        } else {
            reduction as f64 * 100.0 / baseline as f64
        }
    };

    let report = json!({
        "schema": "bloom-geometry-cook-report-v1",
        "input": input.display().to_string(),
        "output": serde_json::Value::Null,
        "format_version": archive.format_version,
        "source_sha256": hex_hash(archive.source_sha256),
        "payload_sha256": hex_hash(archive.payload_sha256),
        "source": {
            "meshes": source.source_meshes,
            "primitives": source.source_primitives,
            "eligible_triangles": source.eligible_triangles,
        },
        "cooked": {
            "meshlets": archive.clusters.len(),
            "triangles": archive.triangle_count(),
            "pages": archive.pages.len(),
            "payload_bytes": archive.payload_bytes(),
            "page_budget_bytes": archive.page_budget_bytes,
            "maximum_page_bytes": archive.maximum_page_bytes(),
            "vertex_encoding": {
                "name": archive.vertex_encoding.label(),
                "stride_bytes": archive.vertex_encoding.stride(),
                "float32_baseline_payload_bytes": baseline_payload_bytes,
                "payload_reduction_bytes": payload_reduction_bytes,
                "payload_reduction_percent":
                    reduction_percent(payload_reduction_bytes, baseline_payload_bytes),
                "float32_baseline_root_page_bytes": baseline_root_page_bytes,
                "root_page_reduction_bytes": root_reduction_bytes,
                "root_page_reduction_percent":
                    reduction_percent(root_reduction_bytes, baseline_root_page_bytes),
                "max_position_absolute_error": quantization.max_position_absolute_error,
                "max_position_cluster_relative_error":
                    quantization.max_position_cluster_relative_error,
                "max_normal_angular_error_degrees":
                    quantization.max_normal_angular_error_degrees,
                "max_tangent_angular_error_degrees":
                    quantization.max_tangent_angular_error_degrees,
                "max_uv_absolute_error": quantization.max_uv_absolute_error,
                "max_color_absolute_error": quantization.max_color_absolute_error,
                "max_tangent_handedness_error":
                    quantization.max_tangent_handedness_error,
            },
            "max_vertices_per_meshlet": options.meshlet_limits.max_vertices,
            "max_triangles_per_meshlet": options.meshlet_limits.max_triangles,
            "leaf_hierarchy_only": options.hierarchy_levels == 0,
            "hierarchy": {
                "requested_levels": options.hierarchy_levels,
                "leaf_clusters": hierarchy.leaf_clusters,
                "leaf_triangles": hierarchy.leaf_triangles,
                "leaf_payload_bytes": hierarchy.leaf_payload_bytes,
                "parent_clusters": hierarchy.parent_clusters,
                "root_clusters": hierarchy.root_clusters,
                "root_triangles": hierarchy.root_triangles,
                "maximum_level": hierarchy.maximum_level,
                "maximum_absolute_error": hierarchy.maximum_error,
                "root_payload_bytes": hierarchy.root_payload_bytes,
                "root_pages": archive.coarse_root_page_count(),
                "root_resident_page_bytes": archive.coarse_root_page_bytes(),
                "root_clusters_by_level": hierarchy.root_clusters_by_level,
                "root_payload_bytes_by_level": hierarchy.root_payload_bytes_by_level,
            },
        },
        "compatibility": compatibility_json(&archive.compatibility),
        "shipping_runtime_changes": {
            "passes": 0,
            "draws": 0,
            "buffers": 0,
            "shader_branches": 0,
        }
    });
    Ok(CookedGeometry { bytes, report })
}

pub fn inspect_geometry_command(input: &Path) -> Result<String, String> {
    let bytes = std::fs::read(input).map_err(|error| format!("read {input:?}: {error}"))?;
    let archive = decode_geometry(&bytes)?;
    serde_json::to_string_pretty(&json!({
        "schema": "bloom-geometry-inspect-report-v1",
        "input": input.display().to_string(),
        "format_version": archive.format_version,
        "vertex_encoding": archive.vertex_encoding.label(),
        "vertex_stride_bytes": archive.vertex_encoding.stride(),
        "file_bytes": bytes.len(),
        "source_sha256": hex_hash(archive.source_sha256),
        "payload_sha256": hex_hash(archive.payload_sha256),
        "meshlets": archive.clusters.len(),
        "triangles": archive.triangle_count(),
        "pages": archive.pages.len(),
        "payload_bytes": archive.payload_bytes(),
        "page_budget_bytes": archive.page_budget_bytes,
        "maximum_page_bytes": archive.maximum_page_bytes(),
        "coarse_root_pages": archive.coarse_root_page_count(),
        "coarse_root_page_bytes": archive.coarse_root_page_bytes(),
        "compatibility": compatibility_json(&archive.compatibility),
        "validation": "pass",
    }))
    .map_err(|error| format!("serialize geometry inspection: {error}"))
}

fn compatibility_json(records: &[CompatibilityRecord]) -> serde_json::Value {
    serde_json::Value::Array(
        records
            .iter()
            .map(|record| {
                json!({
                    "mesh": record.mesh_index,
                    "primitive": record.primitive_index,
                    "reason": record.reason.label(),
                    "detail": record.detail,
                })
            })
            .collect(),
    )
}

fn parse_options(flags: &[String]) -> Result<CookOptions, String> {
    let mut options = CookOptions::default();
    let mut index = 0;
    while index < flags.len() {
        let flag = &flags[index];
        let value = flags
            .get(index + 1)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        if flag == "--vertex-format" {
            options.vertex_encoding = match value.as_str() {
                "float32" => VertexEncoding::Float32,
                "quantized32" => VertexEncoding::Quantized,
                _ => {
                    return Err(format!(
                        "--vertex-format must be float32 or quantized32, got {value:?}"
                    ))
                }
            };
            index += 2;
            continue;
        }
        let parsed = value
            .parse::<u32>()
            .map_err(|_| format!("{flag} requires an unsigned integer, got {value:?}"))?;
        match flag.as_str() {
            "--max-vertices" => options.meshlet_limits.max_vertices = parsed,
            "--max-triangles" => options.meshlet_limits.max_triangles = parsed,
            "--page-bytes" => options.page_bytes = parsed,
            "--hierarchy-levels" => options.hierarchy_levels = parsed,
            _ => return Err(format!("unknown geometry option {flag:?}")),
        }
        index += 2;
    }
    options.meshlet_limits.validate()?;
    if options.hierarchy_levels > 16 {
        return Err(format!(
            "geometry hierarchy levels must be in 0..=16, got {}",
            options.hierarchy_levels
        ));
    }
    // The format writer owns the exact power-of-two/range validation. This
    // dry call keeps CLI errors local without duplicating that contract.
    if options.page_bytes == 0 {
        return Err("geometry page budget must be greater than zero".to_string());
    }
    Ok(options)
}

fn load_geometry_source(path: &Path) -> Result<GeometrySource, String> {
    let source_bytes = std::fs::read(path).map_err(|error| format!("read {path:?}: {error}"))?;
    let mut gltf = gltf::Gltf::from_slice(&source_bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let buffers = gltf::import_buffers(&gltf.document, path.parent(), gltf.blob.take())
        .map_err(|error| format!("load buffers for {}: {error}", path.display()))?;
    let skinned_meshes: HashSet<usize> = gltf
        .nodes()
        .filter(|node| node.skin().is_some())
        .filter_map(|node| node.mesh().map(|mesh| mesh.index()))
        .collect();

    let buffer_slices = buffers
        .iter()
        .map(|buffer| buffer.0.as_slice())
        .collect::<Vec<_>>();
    let source_sha256 = geometry_source_sha256(&source_bytes, &buffer_slices);

    let mut primitives = Vec::new();
    let mut compatibility = Vec::new();
    let mut source_primitives = 0usize;
    let mut eligible_triangles = 0u64;
    let source_meshes = gltf.meshes().len();

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            source_primitives += 1;
            let mesh_index = mesh.index() as u32;
            let primitive_index = primitive.index() as u32;
            let material = primitive.material();
            let material_index = material.index().map(|index| index as u32);
            let base_color_factor = runtime_base_color_factor(&material);

            let compatibility_reason = if primitive.mode() != gltf::mesh::Mode::Triangles {
                Some((
                    CompatibilityReason::NonTriangleTopology,
                    mode_code(primitive.mode()),
                ))
            } else if skinned_meshes.contains(&mesh.index()) {
                Some((CompatibilityReason::Skinned, 0))
            } else if primitive.morph_targets().next().is_some() {
                Some((CompatibilityReason::MorphTargets, 0))
            } else if material.alpha_mode() == gltf::material::AlphaMode::Blend {
                Some((
                    CompatibilityReason::AlphaBlend,
                    material_index.unwrap_or(u32::MAX),
                ))
            } else {
                None
            };
            if let Some((reason, detail)) = compatibility_reason {
                compatibility.push(CompatibilityRecord {
                    mesh_index,
                    primitive_index,
                    reason,
                    detail,
                });
                continue;
            }

            let reader = primitive
                .reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or_else(|| {
                    format!("mesh {mesh_index} primitive {primitive_index} is missing POSITION")
                })?
                .collect();
            if positions.is_empty() {
                return Err(format!(
                    "mesh {mesh_index} primitive {primitive_index} has no positions"
                ));
            }
            let indices: Vec<u32> = match reader.read_indices() {
                Some(indices) => indices.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            if indices.is_empty() || !indices.len().is_multiple_of(3) {
                return Err(format!(
                    "mesh {mesh_index} primitive {primitive_index} has {} indices, \
                     expected a non-empty triangle list",
                    indices.len()
                ));
            }
            if let Some(index) = indices
                .iter()
                .find(|index| **index as usize >= positions.len())
            {
                return Err(format!(
                    "mesh {mesh_index} primitive {primitive_index} index {index} \
                     exceeds {} positions",
                    positions.len()
                ));
            }

            let normals = match reader.read_normals() {
                Some(normals) => collect_attribute(
                    normals,
                    positions.len(),
                    mesh_index,
                    primitive_index,
                    "NORMAL",
                )?,
                None => generate_normals(&positions, &indices),
            };
            let tangents = match reader.read_tangents() {
                Some(tangents) => collect_attribute(
                    tangents,
                    positions.len(),
                    mesh_index,
                    primitive_index,
                    "TANGENT",
                )?,
                None => vec![[0.0; 4]; positions.len()],
            };
            let uv0 = match reader.read_tex_coords(0) {
                Some(uvs) => collect_attribute(
                    uvs.into_f32(),
                    positions.len(),
                    mesh_index,
                    primitive_index,
                    "TEXCOORD_0",
                )?,
                None => vec![[0.0; 2]; positions.len()],
            };
            let uv1 = match reader.read_tex_coords(1) {
                Some(uvs) => collect_attribute(
                    uvs.into_f32(),
                    positions.len(),
                    mesh_index,
                    primitive_index,
                    "TEXCOORD_1",
                )?,
                None => vec![[0.0; 2]; positions.len()],
            };
            let colors = match reader.read_colors(0) {
                Some(colors) => collect_attribute(
                    colors.into_rgba_f32(),
                    positions.len(),
                    mesh_index,
                    primitive_index,
                    "COLOR_0",
                )?,
                None => vec![[1.0; 4]; positions.len()],
            };
            let vertices = positions
                .into_iter()
                .zip(normals)
                .zip(tangents)
                .zip(uv0)
                .zip(uv1)
                .zip(colors)
                .map(
                    |(((((position, normal), tangent), uv0), uv1), color)| StaticVertex {
                        position,
                        normal,
                        tangent,
                        uv0,
                        uv1,
                        color: multiply_rgba(color, base_color_factor),
                    },
                )
                .collect();
            eligible_triangles += (indices.len() / 3) as u64;
            primitives.push(StaticPrimitive {
                mesh_index,
                primitive_index,
                material_index,
                double_sided: material.double_sided(),
                alpha_masked: material.alpha_mode() == gltf::material::AlphaMode::Mask,
                vertices,
                indices,
            });
        }
    }

    if source_primitives == 0 {
        return Err(format!("{} contains no mesh primitives", path.display()));
    }
    Ok(GeometrySource {
        source_sha256,
        primitives,
        compatibility,
        source_meshes,
        source_primitives,
        eligible_triangles,
    })
}

fn collect_attribute<T>(
    values: impl Iterator<Item = T>,
    expected: usize,
    mesh_index: u32,
    primitive_index: u32,
    semantic: &str,
) -> Result<Vec<T>, String> {
    let values: Vec<_> = values.collect();
    if values.len() != expected {
        return Err(format!(
            "mesh {mesh_index} primitive {primitive_index} {semantic} count {} \
             does not match POSITION count {expected}",
            values.len()
        ));
    }
    Ok(values)
}

fn generate_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0; 3]; positions.len()];
    for triangle in indices.as_chunks::<3>().0 {
        let a = positions[triangle[0] as usize];
        let b = positions[triangle[1] as usize];
        let c = positions[triangle[2] as usize];
        let face = cross3(sub3(b, a), sub3(c, a));
        for index in triangle {
            normals[*index as usize] = add3(normals[*index as usize], face);
        }
    }
    for normal in &mut normals {
        let length = dot3(*normal, *normal).sqrt();
        *normal = if length > 1e-20 {
            mul3(*normal, length.recip())
        } else {
            [0.0, 1.0, 0.0]
        };
    }
    normals
}

/// Match the runtime glTF loader's vertex-color contract. Bloom carries the
/// material base/diffuse factor in `Vertex3D::color`; the global material
/// record intentionally contains only texture/scalar response state. Virtual
/// vertices therefore have to bake the same factor or residency changes the
/// authored color.
fn runtime_base_color_factor(material: &gltf::Material<'_>) -> [f32; 4] {
    let pbr = material.pbr_metallic_roughness();
    if pbr.base_color_texture().is_none() {
        if let Some(spec_gloss) = material.pbr_specular_glossiness() {
            let diffuse = spec_gloss.diffuse_factor();
            if spec_gloss.specular_glossiness_texture().is_some() {
                return diffuse;
            }
            return specgloss_to_metalrough(diffuse, spec_gloss.specular_factor()).0;
        }
    }
    pbr.base_color_factor()
}

fn multiply_rgba(lhs: [f32; 4], rhs: [f32; 4]) -> [f32; 4] {
    [
        lhs[0] * rhs[0],
        lhs[1] * rhs[1],
        lhs[2] * rhs[2],
        lhs[3] * rhs[3],
    ]
}

/// Khronos' reference specular-glossiness to metallic-roughness base-color
/// conversion, kept byte-for-byte equivalent to the runtime importer.
fn specgloss_to_metalrough(diffuse: [f32; 4], specular: [f32; 3]) -> ([f32; 4], f32) {
    let dielectric_specular = 0.04_f32;
    let epsilon = 1e-6_f32;
    let one_minus_dielectric = 1.0 - dielectric_specular;
    let diffuse_max = diffuse[0].max(diffuse[1]).max(diffuse[2]);
    let specular_max = specular[0].max(specular[1]).max(specular[2]);
    let a = dielectric_specular;
    let b = diffuse_max * one_minus_dielectric / dielectric_specular.max(epsilon) + specular_max
        - 2.0 * dielectric_specular;
    let c = dielectric_specular - specular_max;
    let discriminant = (b * b - 4.0 * a * c).max(0.0);
    let metallic = if specular_max < dielectric_specular {
        0.0
    } else {
        (((-b + discriminant.sqrt()) / (2.0 * a)).clamp(0.0, 1.0)).min(1.0)
    };
    let diffuse_branch_scale =
        one_minus_dielectric / (1.0 - metallic * dielectric_specular).max(epsilon);
    let metal_weight = metallic * metallic;
    let lerp = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;
    (
        [
            lerp(diffuse[0] * diffuse_branch_scale, specular[0], metal_weight).clamp(0.0, 1.0),
            lerp(diffuse[1] * diffuse_branch_scale, specular[1], metal_weight).clamp(0.0, 1.0),
            lerp(diffuse[2] * diffuse_branch_scale, specular[2], metal_weight).clamp(0.0, 1.0),
            diffuse[3],
        ],
        metallic,
    )
}

fn mode_code(mode: gltf::mesh::Mode) -> u32 {
    match mode {
        gltf::mesh::Mode::Points => 0,
        gltf::mesh::Mode::Lines => 1,
        gltf::mesh::Mode::LineLoop => 2,
        gltf::mesh::Mode::LineStrip => 3,
        gltf::mesh::Mode::Triangles => 4,
        gltf::mesh::Mode::TriangleStrip => 5,
        gltf::mesh::Mode::TriangleFan => 6,
    }
}

pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("output path {} has no UTF-8 file name", path.display()))?;
    let unique = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| format!("system clock precedes Unix epoch: {error}"))?
            .as_nanos()
    );
    let temporary = path.with_file_name(format!(".{file_name}.{unique}.tmp"));
    let write_result = (|| {
        let mut file = std::fs::File::create(&temporary)
            .map_err(|error| format!("create {}: {error}", temporary.display()))?;
        file.write_all(bytes)
            .map_err(|error| format!("write {}: {error}", temporary.display()))?;
        file.sync_all()
            .map_err(|error| format!("flush {}: {error}", temporary.display()))?;
        match std::fs::rename(&temporary, path) {
            Ok(()) => Ok(()),
            Err(first_error) if path.exists() => {
                // Unix replaces atomically in the first branch. Windows does
                // not replace an existing destination, so preserve the old
                // artifact as a rollback target while installing the fully
                // flushed temporary.
                let backup = path.with_file_name(format!(".{file_name}.{unique}.previous"));
                std::fs::rename(path, &backup).map_err(|backup_error| {
                    format!(
                        "replace {} after rename error {first_error}: \
                         could not preserve prior artifact: {backup_error}",
                        path.display()
                    )
                })?;
                match std::fs::rename(&temporary, path) {
                    Ok(()) => {
                        std::fs::remove_file(&backup).map_err(|error| {
                            format!(
                                "installed {} but could not remove backup {}: {error}",
                                path.display(),
                                backup.display()
                            )
                        })?;
                        Ok(())
                    }
                    Err(install_error) => {
                        let restore = std::fs::rename(&backup, path);
                        Err(format!(
                            "install {} failed: {install_error}; prior artifact restore: {}",
                            path.display(),
                            if restore.is_ok() { "pass" } else { "failed" }
                        ))
                    }
                }
            }
            Err(error) => Err(format!(
                "install {} as {}: {error}",
                temporary.display(),
                path.display()
            )),
        }
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    write_result
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul3(value: [f32; 3], factor: f32) -> [f32; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_options_are_strict_and_bounded() {
        let flags = vec![
            "--max-vertices".to_string(),
            "32".to_string(),
            "--max-triangles".to_string(),
            "48".to_string(),
            "--page-bytes".to_string(),
            "32768".to_string(),
            "--hierarchy-levels".to_string(),
            "6".to_string(),
        ];
        let options = parse_options(&flags).unwrap();
        assert_eq!(options.meshlet_limits.max_vertices, 32);
        assert_eq!(options.meshlet_limits.max_triangles, 48);
        assert_eq!(options.page_bytes, 32768);
        assert_eq!(options.hierarchy_levels, 6);
        assert_eq!(options.vertex_encoding, VertexEncoding::Float32);
        let quantized =
            parse_options(&["--vertex-format".to_string(), "quantized32".to_string()]).unwrap();
        assert_eq!(quantized.vertex_encoding, VertexEncoding::Quantized);
        assert!(
            parse_options(&["--vertex-format".to_string(), "packed-magic".to_string(),])
                .unwrap_err()
                .contains("float32 or quantized32")
        );
        assert!(parse_options(&["--surprise".to_string(), "1".to_string()])
            .unwrap_err()
            .contains("unknown"));
        assert!(parse_options(&["--max-vertices".to_string()])
            .unwrap_err()
            .contains("requires"));
    }

    #[test]
    fn missing_normals_are_generated_deterministically() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let a = generate_normals(&positions, &[0, 1, 2]);
        let b = generate_normals(&positions, &[0, 1, 2]);
        assert_eq!(a, b);
        assert_eq!(a, vec![[0.0, 0.0, 1.0]; 3]);
    }

    #[test]
    fn minimal_glb_loads_and_cooks_without_images() {
        let bytes = minimal_triangle_glb();
        let path = std::env::temp_dir().join(format!(
            "bloom-cook-geometry-{}-{}.glb",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, bytes).unwrap();
        let source = load_geometry_source(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(source.source_meshes, 1);
        assert_eq!(source.source_primitives, 1);
        assert_eq!(source.eligible_triangles, 1);
        assert_eq!(source.primitives.len(), 1);
        assert_eq!(source.primitives[0].vertices[0].normal, [0.0, 0.0, 1.0]);
        let meshlets =
            build_leaf_meshlets(&source.primitives[0], MeshletLimits::default()).unwrap();
        let encoded = encode_geometry(
            &meshlets,
            &source.compatibility,
            source.source_sha256,
            DEFAULT_PAGE_BYTES,
        )
        .unwrap();
        assert_eq!(decode_geometry(&encoded).unwrap().triangle_count(), 1);
    }

    #[test]
    fn cooked_vertex_colors_include_runtime_material_factor() {
        let factor = [0.5, 0.25, 0.75, 0.8];
        let bytes = minimal_triangle_glb_with_material(Some(factor));
        let path = std::env::temp_dir().join(format!(
            "bloom-cook-color-factor-{}-{}.glb",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, bytes).unwrap();
        let source = load_geometry_source(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(source.primitives[0]
            .vertices
            .iter()
            .all(|vertex| vertex.color == factor));
    }

    #[test]
    fn atomic_output_can_replace_an_existing_artifact() {
        let path = std::env::temp_dir().join(format!(
            "bloom-cook-atomic-{}-{}.bgeo",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_atomically(&path, b"first").unwrap();
        write_atomically(&path, b"second").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"second");
        let _ = std::fs::remove_file(path);
    }

    fn minimal_triangle_glb() -> Vec<u8> {
        minimal_triangle_glb_with_material(None)
    }

    fn minimal_triangle_glb_with_material(base_color_factor: Option<[f32; 4]>) -> Vec<u8> {
        let mut binary = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            binary.extend_from_slice(&value.to_le_bytes());
        }
        for index in [0u16, 1, 2] {
            binary.extend_from_slice(&index.to_le_bytes());
        }
        binary.resize(binary.len().div_ceil(4) * 4, 0);
        let material_json = base_color_factor.map_or_else(String::new, |factor| {
            format!(
                "\"materials\":{},",
                serde_json::json!([{
                    "pbrMetallicRoughness": { "baseColorFactor": factor }
                }])
            )
        });
        let primitive_material = if base_color_factor.is_some() {
            r#", "material":0"#
        } else {
            ""
        };
        let json = format!(
            r#"{{
                "asset":{{"version":"2.0"}},
                "buffers":[{{"byteLength":{}}}],
                "bufferViews":[
                    {{"buffer":0,"byteOffset":0,"byteLength":36}},
                    {{"buffer":0,"byteOffset":36,"byteLength":6}}
                ],
                "accessors":[
                    {{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3",
                      "min":[0,0,0],"max":[1,1,0]}},
                    {{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}}
                ],
                {material_json}
                "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1{primitive_material}}}]}}],
                "nodes":[{{"mesh":0}}],
                "scenes":[{{"nodes":[0]}}],
                "scene":0
            }}"#,
            binary.len()
        );
        let mut json = json.into_bytes();
        json.resize(json.len().div_ceil(4) * 4, b' ');
        let total_length = 12 + 8 + json.len() + 8 + binary.len();
        let mut glb = Vec::with_capacity(total_length);
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total_length as u32).to_le_bytes());
        glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x4e4f_534au32.to_le_bytes());
        glb.extend_from_slice(&json);
        glb.extend_from_slice(&(binary.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x004e_4942u32.to_le_bytes());
        glb.extend_from_slice(&binary);
        glb
    }
}
