//! Versioned static-vertex payload encoding for cooked virtual geometry.
//!
//! The float32 encoding is the byte-exact v1 contract. The quantized encoding
//! is an explicit v2 opt-in: cluster-local UNORM16 positions, octahedral
//! SNORM16 directions, f16 UVs, UNORM8 color, and SNORM16 tangent handedness.
//! A tangent-valid bit preserves the established all-zero missing-tangent
//! sentinel exactly.

use crate::geometry_format::ClusterRecord;
use crate::meshlet::{Meshlet, StaticVertex};
use half::f16;

pub use bloom_geometry_format::VertexEncoding;
#[cfg(test)]
pub(crate) use bloom_geometry_format::QUANTIZED_VERTEX_BYTES;
const QUANTIZED_TANGENT_VALID: u16 = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QuantizationStats {
    pub max_position_absolute_error: f32,
    pub max_position_cluster_relative_error: f32,
    pub max_normal_angular_error_degrees: f32,
    pub max_tangent_angular_error_degrees: f32,
    pub max_uv_absolute_error: f32,
    pub max_color_absolute_error: f32,
    pub max_tangent_handedness_error: f32,
}

pub fn encoded_meshlet_bytes(meshlet: &Meshlet, encoding: VertexEncoding) -> usize {
    meshlet.vertices.len() * encoding.stride() as usize + meshlet.local_indices.len()
}

pub fn encode_vertex(
    output: &mut Vec<u8>,
    vertex: &StaticVertex,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    encoding: VertexEncoding,
) -> Result<(), String> {
    match encoding {
        VertexEncoding::Float32 => {
            for value in vertex
                .position
                .iter()
                .chain(vertex.normal.iter())
                .chain(vertex.tangent.iter())
                .chain(vertex.uv0.iter())
                .chain(vertex.uv1.iter())
                .chain(vertex.color.iter())
            {
                output.extend_from_slice(&value.to_le_bytes());
            }
        }
        VertexEncoding::Quantized => {
            for axis in 0..3 {
                push_u16(
                    output,
                    encode_position(vertex.position[axis], bounds_min[axis], bounds_max[axis])?,
                );
            }
            let normal = encode_octahedral(vertex.normal, "normal")?;
            push_i16(output, normal[0]);
            push_i16(output, normal[1]);

            let tangent_length = length3([vertex.tangent[0], vertex.tangent[1], vertex.tangent[2]]);
            let tangent_valid = tangent_length > 1.0e-20;
            let tangent = if tangent_valid {
                encode_octahedral(
                    [vertex.tangent[0], vertex.tangent[1], vertex.tangent[2]],
                    "tangent",
                )?
            } else {
                if vertex.tangent[3] != 0.0 {
                    return Err("zero-length quantized tangent has non-zero handedness".to_string());
                }
                [0, 0]
            };
            push_i16(output, tangent[0]);
            push_i16(output, tangent[1]);

            for (label, value) in [
                ("uv0.x", vertex.uv0[0]),
                ("uv0.y", vertex.uv0[1]),
                ("uv1.x", vertex.uv1[0]),
                ("uv1.y", vertex.uv1[1]),
            ] {
                let encoded = f16::from_f32(value);
                if !encoded.is_finite() {
                    return Err(format!(
                        "quantized vertex {label} value {value} exceeds finite f16 range"
                    ));
                }
                push_u16(output, encoded.to_bits());
            }
            for (channel, value) in vertex.color.into_iter().enumerate() {
                if !(0.0..=1.0).contains(&value) {
                    return Err(format!(
                        "quantized vertex color channel {channel} is outside 0..=1: {value}"
                    ));
                }
                output.push((value * 255.0).round() as u8);
            }
            if !(-1.0..=1.0).contains(&vertex.tangent[3]) {
                return Err(format!(
                    "quantized tangent handedness is outside -1..=1: {}",
                    vertex.tangent[3]
                ));
            }
            push_i16(output, encode_snorm16(vertex.tangent[3]));
            push_u16(
                output,
                if tangent_valid {
                    QUANTIZED_TANGENT_VALID
                } else {
                    0
                },
            );
            push_u16(output, 0);
        }
    }
    Ok(())
}

pub fn measure(
    meshlets: &[Meshlet],
    clusters: &[ClusterRecord],
    payload: &[u8],
    encoding: VertexEncoding,
) -> Result<QuantizationStats, String> {
    if meshlets.len() != clusters.len() {
        return Err(format!(
            "quantization source/cluster count mismatch: {} vs {}",
            meshlets.len(),
            clusters.len()
        ));
    }
    let mut stats = QuantizationStats::default();
    for (cluster_index, (meshlet, cluster)) in meshlets.iter().zip(clusters).enumerate() {
        if meshlet.vertices.len() != cluster.vertex_count as usize {
            return Err(format!(
                "quantization source vertex count mismatch for cluster {cluster_index}"
            ));
        }
        let vertex_start = usize::try_from(cluster.vertex_offset)
            .map_err(|_| format!("cluster {cluster_index} vertex offset exceeds host space"))?;
        for (vertex_index, source) in meshlet.vertices.iter().enumerate() {
            let offset = vertex_start
                .checked_add(vertex_index * cluster.vertex_stride as usize)
                .ok_or_else(|| format!("cluster {cluster_index} vertex offset overflow"))?;
            let decoded = decode_vertex(payload, offset, cluster, encoding)?;
            for axis in 0..3 {
                let error = (source.position[axis] - decoded.position[axis]).abs();
                stats.max_position_absolute_error = stats.max_position_absolute_error.max(error);
                let extent = cluster.aabb_max[axis] - cluster.aabb_min[axis];
                if extent > 0.0 {
                    stats.max_position_cluster_relative_error = stats
                        .max_position_cluster_relative_error
                        .max(error / extent);
                }
            }
            stats.max_normal_angular_error_degrees = stats
                .max_normal_angular_error_degrees
                .max(angle_degrees(source.normal, decoded.normal)?);

            let source_tangent = [source.tangent[0], source.tangent[1], source.tangent[2]];
            if length3(source_tangent) > 1.0e-20 {
                stats.max_tangent_angular_error_degrees =
                    stats.max_tangent_angular_error_degrees.max(angle_degrees(
                        source_tangent,
                        [decoded.tangent[0], decoded.tangent[1], decoded.tangent[2]],
                    )?);
            }
            for channel in 0..2 {
                stats.max_uv_absolute_error = stats
                    .max_uv_absolute_error
                    .max((source.uv0[channel] - decoded.uv0[channel]).abs())
                    .max((source.uv1[channel] - decoded.uv1[channel]).abs());
            }
            for channel in 0..4 {
                stats.max_color_absolute_error = stats
                    .max_color_absolute_error
                    .max((source.color[channel] - decoded.color[channel]).abs());
            }
            stats.max_tangent_handedness_error = stats
                .max_tangent_handedness_error
                .max((source.tangent[3] - decoded.tangent[3]).abs());
        }
    }
    Ok(stats)
}

fn decode_vertex(
    payload: &[u8],
    offset: usize,
    cluster: &ClusterRecord,
    encoding: VertexEncoding,
) -> Result<StaticVertex, String> {
    match encoding {
        VertexEncoding::Float32 => {
            let mut values = [0.0f32; 18];
            for (index, value) in values.iter_mut().enumerate() {
                *value = read_f32(payload, offset + index * 4)?;
                if !value.is_finite() {
                    return Err("float32 payload contains NaN/Inf".to_string());
                }
            }
            Ok(StaticVertex {
                position: values[0..3].try_into().unwrap(),
                normal: values[3..6].try_into().unwrap(),
                tangent: values[6..10].try_into().unwrap(),
                uv0: values[10..12].try_into().unwrap(),
                uv1: values[12..14].try_into().unwrap(),
                color: values[14..18].try_into().unwrap(),
            })
        }
        VertexEncoding::Quantized => {
            let flags = read_u16(payload, offset + 28)?;
            let reserved = read_u16(payload, offset + 30)?;
            if flags & !QUANTIZED_TANGENT_VALID != 0 || reserved != 0 {
                return Err("quantized payload has unknown flags or non-zero padding".to_string());
            }
            let mut position = [0.0; 3];
            for (axis, value) in position.iter_mut().enumerate() {
                *value = decode_position(
                    read_u16(payload, offset + axis * 2)?,
                    cluster.aabb_min[axis],
                    cluster.aabb_max[axis],
                );
            }
            let normal = decode_octahedral([
                read_i16(payload, offset + 6)?,
                read_i16(payload, offset + 8)?,
            ]);
            let tangent = if flags & QUANTIZED_TANGENT_VALID != 0 {
                let direction = decode_octahedral([
                    read_i16(payload, offset + 10)?,
                    read_i16(payload, offset + 12)?,
                ]);
                [
                    direction[0],
                    direction[1],
                    direction[2],
                    decode_snorm16(read_i16(payload, offset + 26)?),
                ]
            } else {
                if read_i16(payload, offset + 10)? != 0
                    || read_i16(payload, offset + 12)? != 0
                    || read_i16(payload, offset + 26)? != 0
                {
                    return Err(
                        "missing-tangent payload does not preserve the zero sentinel".to_string(),
                    );
                }
                [0.0; 4]
            };
            let uv0 = [
                decode_f16(payload, offset + 14)?,
                decode_f16(payload, offset + 16)?,
            ];
            let uv1 = [
                decode_f16(payload, offset + 18)?,
                decode_f16(payload, offset + 20)?,
            ];
            let color_bytes = payload
                .get(offset + 22..offset + 26)
                .ok_or("quantized color is truncated")?;
            let color = [
                color_bytes[0] as f32 / 255.0,
                color_bytes[1] as f32 / 255.0,
                color_bytes[2] as f32 / 255.0,
                color_bytes[3] as f32 / 255.0,
            ];
            let vertex = StaticVertex {
                position,
                normal,
                tangent,
                uv0,
                uv1,
                color,
            };
            if vertex
                .position
                .iter()
                .chain(vertex.normal.iter())
                .chain(vertex.tangent.iter())
                .chain(vertex.uv0.iter())
                .chain(vertex.uv1.iter())
                .chain(vertex.color.iter())
                .any(|value| !value.is_finite())
            {
                return Err("quantized payload decodes to NaN/Inf".to_string());
            }
            Ok(vertex)
        }
    }
}

fn encode_position(value: f32, minimum: f32, maximum: f32) -> Result<u16, String> {
    if !value.is_finite() || !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
        return Err("quantized position or bounds are invalid".to_string());
    }
    if maximum == minimum {
        if value != minimum {
            return Err("position lies outside a zero-extent cluster bound".to_string());
        }
        return Ok(0);
    }
    let normalized = (value - minimum) / (maximum - minimum);
    if !(-1.0e-5..=1.00001).contains(&normalized) {
        return Err("position lies outside its cluster bound".to_string());
    }
    Ok((normalized.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16)
}

fn decode_position(value: u16, minimum: f32, maximum: f32) -> f32 {
    if maximum == minimum {
        minimum
    } else {
        minimum + (maximum - minimum) * (value as f32 / u16::MAX as f32)
    }
}

fn encode_octahedral(value: [f32; 3], label: &str) -> Result<[i16; 2], String> {
    let length = length3(value);
    if !length.is_finite() || length <= 1.0e-20 {
        return Err(format!("quantized {label} has zero or invalid length"));
    }
    let n = [value[0] / length, value[1] / length, value[2] / length];
    let inverse_l1 = (n[0].abs() + n[1].abs() + n[2].abs()).recip();
    let mut p = [n[0] * inverse_l1, n[1] * inverse_l1];
    if n[2] < 0.0 {
        p = [
            (1.0 - p[1].abs()) * sign_not_zero(p[0]),
            (1.0 - p[0].abs()) * sign_not_zero(p[1]),
        ];
    }
    Ok([encode_snorm16(p[0]), encode_snorm16(p[1])])
}

fn decode_octahedral(value: [i16; 2]) -> [f32; 3] {
    let x = decode_snorm16(value[0]);
    let y = decode_snorm16(value[1]);
    let mut n = [x, y, 1.0 - x.abs() - y.abs()];
    if n[2] < 0.0 {
        let old_x = n[0];
        n[0] = (1.0 - n[1].abs()) * sign_not_zero(old_x);
        n[1] = (1.0 - old_x.abs()) * sign_not_zero(n[1]);
    }
    let inverse_length = length3(n).recip();
    [
        n[0] * inverse_length,
        n[1] * inverse_length,
        n[2] * inverse_length,
    ]
}

fn angle_degrees(a: [f32; 3], b: [f32; 3]) -> Result<f32, String> {
    let a = a.map(f64::from);
    let b = b.map(f64::from);
    let a_length = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let b_length = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if a_length <= 1.0e-20 || b_length <= 1.0e-20 {
        return Err("cannot measure angular error for a zero vector".to_string());
    }
    let cosine =
        ((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (a_length * b_length)).clamp(-1.0, 1.0);
    Ok(cosine.acos().to_degrees() as f32)
}

fn length3(value: [f32; 3]) -> f32 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

fn sign_not_zero(value: f32) -> f32 {
    if value < 0.0 {
        -1.0
    } else {
        1.0
    }
}

fn encode_snorm16(value: f32) -> i16 {
    (value.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
}

fn decode_snorm16(value: i16) -> f32 {
    (value as f32 / i16::MAX as f32).clamp(-1.0, 1.0)
}

fn decode_f16(payload: &[u8], offset: usize) -> Result<f32, String> {
    let value = f16::from_bits(read_u16(payload, offset)?);
    if !value.is_finite() {
        return Err("quantized f16 payload contains NaN/Inf".to_string());
    }
    Ok(value.to_f32())
}

fn push_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_i16(output: &mut Vec<u8>, value: i16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn read_u16(payload: &[u8], offset: usize) -> Result<u16, String> {
    let bytes = payload
        .get(offset..offset + 2)
        .ok_or("quantized payload is truncated")?;
    Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_i16(payload: &[u8], offset: usize) -> Result<i16, String> {
    let bytes = payload
        .get(offset..offset + 2)
        .ok_or("quantized payload is truncated")?;
    Ok(i16::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_f32(payload: &[u8], offset: usize) -> Result<f32, String> {
    let bytes = payload
        .get(offset..offset + 4)
        .ok_or("float32 payload is truncated")?;
    Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster() -> ClusterRecord {
        ClusterRecord {
            mesh_index: 0,
            primitive_index: 0,
            material_index: None,
            flags: 0,
            page_index: 0,
            vertex_count: 3,
            triangle_count: 1,
            lod_level: 0,
            vertex_offset: 0,
            index_offset: QUANTIZED_VERTEX_BYTES as u64 * 3,
            aabb_min: [-1.0, 2.0, 4.0],
            aabb_max: [3.0, 2.0, 8.0],
            sphere_center: [1.0, 2.0, 6.0],
            sphere_radius: 3.0,
            normal_cone_axis: [0.0, 0.0, 1.0],
            normal_cone_cutoff: 1.0,
            geometric_error: 0.0,
            parent: u32::MAX,
            parent_count: 0,
            first_child: u32::MAX,
            child_count: 0,
            vertex_stride: QUANTIZED_VERTEX_BYTES,
        }
    }

    fn vertex() -> StaticVertex {
        StaticVertex {
            position: [0.25, 2.0, 7.5],
            normal: [-0.25, 0.5, -0.829_156_2],
            tangent: [0.0; 4],
            uv0: [1.25, -2.5],
            uv1: [0.125, 4.0],
            color: [0.25, 0.5, 0.75, 1.0],
        }
    }

    #[test]
    fn quantized_decoder_preserves_degenerate_axes_and_missing_tangents() {
        let cluster = cluster();
        let source = vertex();
        let mut payload = Vec::new();
        encode_vertex(
            &mut payload,
            &source,
            cluster.aabb_min,
            cluster.aabb_max,
            VertexEncoding::Quantized,
        )
        .unwrap();
        let decoded = decode_vertex(&payload, 0, &cluster, VertexEncoding::Quantized).unwrap();

        assert_eq!(decoded.position[1], source.position[1]);
        assert_eq!(decoded.tangent, [0.0; 4]);
        assert!((decoded.position[0] - source.position[0]).abs() < 0.000_1);
        assert!(angle_degrees(decoded.normal, source.normal).unwrap() < 0.01);
    }

    #[test]
    fn quantized_decoder_rejects_noncanonical_bits_and_nonfinite_uvs() {
        let cluster = cluster();
        let mut payload = Vec::new();
        encode_vertex(
            &mut payload,
            &vertex(),
            cluster.aabb_min,
            cluster.aabb_max,
            VertexEncoding::Quantized,
        )
        .unwrap();

        let mut unknown_flags = payload.clone();
        unknown_flags[28..30].copy_from_slice(&2u16.to_le_bytes());
        assert!(
            decode_vertex(&unknown_flags, 0, &cluster, VertexEncoding::Quantized)
                .unwrap_err()
                .contains("unknown flags")
        );

        payload[14..16].copy_from_slice(&f16::INFINITY.to_bits().to_le_bytes());
        assert!(
            decode_vertex(&payload, 0, &cluster, VertexEncoding::Quantized)
                .unwrap_err()
                .contains("NaN/Inf")
        );
    }
}
