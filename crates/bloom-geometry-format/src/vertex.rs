use crate::types::{ClusterRecord, VertexEncoding, QUANTIZED_TANGENT_VALID};
use crate::wire::{read_i16, read_u16, read_u32};
use half::f16;

pub(crate) fn validate_cluster_vertices(
    cluster_index: usize,
    cluster: &ClusterRecord,
    payload: &[u8],
    encoding: VertexEncoding,
) -> Result<(), String> {
    if cluster.vertex_stride != encoding.stride() {
        return Err(format!(
            "cluster {cluster_index} vertex stride {} does not match {} encoding stride {}",
            cluster.vertex_stride,
            encoding.label(),
            encoding.stride()
        ));
    }
    let vertex_start = usize::try_from(cluster.vertex_offset)
        .map_err(|_| format!("cluster {cluster_index} vertex offset exceeds host space"))?;
    for vertex_index in 0..cluster.vertex_count as usize {
        let offset = vertex_start
            .checked_add(vertex_index * cluster.vertex_stride as usize)
            .ok_or_else(|| format!("cluster {cluster_index} vertex offset overflow"))?;
        validate_vertex(payload, offset, cluster, encoding).map_err(|error| {
            format!("cluster {cluster_index} vertex {vertex_index} is invalid: {error}")
        })?;
    }
    Ok(())
}

fn validate_vertex(
    payload: &[u8],
    offset: usize,
    cluster: &ClusterRecord,
    encoding: VertexEncoding,
) -> Result<(), String> {
    match encoding {
        VertexEncoding::Float32 => {
            for component in 0..18 {
                let value = f32::from_bits(read_u32(
                    payload,
                    offset + component * 4,
                    "float32 payload",
                )?);
                if !value.is_finite() {
                    return Err("float32 payload contains NaN/Inf".to_string());
                }
            }
        }
        VertexEncoding::Quantized => validate_quantized_vertex(payload, offset, cluster)?,
    }
    Ok(())
}

fn validate_quantized_vertex(
    payload: &[u8],
    offset: usize,
    cluster: &ClusterRecord,
) -> Result<(), String> {
    let flags = read_u16(payload, offset + 28, "quantized flags")?;
    let reserved = read_u16(payload, offset + 30, "quantized padding")?;
    if flags & !QUANTIZED_TANGENT_VALID != 0 || reserved != 0 {
        return Err("quantized payload has unknown flags or non-zero padding".to_string());
    }
    for axis in 0..3 {
        let encoded = read_u16(payload, offset + axis * 2, "quantized position payload")?;
        let minimum = cluster.aabb_min[axis];
        let maximum = cluster.aabb_max[axis];
        let decoded = if maximum == minimum {
            minimum
        } else {
            minimum + (maximum - minimum) * (encoded as f32 / u16::MAX as f32)
        };
        if !decoded.is_finite() {
            return Err("quantized payload decodes to NaN/Inf".to_string());
        }
    }
    for uv_offset in [14, 16, 18, 20] {
        let value = f16::from_bits(read_u16(
            payload,
            offset + uv_offset,
            "quantized f16 payload",
        )?);
        if !value.is_finite() {
            return Err("quantized f16 payload contains NaN/Inf".to_string());
        }
    }
    if flags & QUANTIZED_TANGENT_VALID == 0
        && (read_i16(payload, offset + 10, "quantized tangent")? != 0
            || read_i16(payload, offset + 12, "quantized tangent")? != 0
            || read_i16(payload, offset + 26, "quantized handedness")? != 0)
    {
        return Err("missing-tangent payload does not preserve the zero sentinel".to_string());
    }
    Ok(())
}
