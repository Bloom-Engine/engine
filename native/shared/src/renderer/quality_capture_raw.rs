//! Exact attachment bytes for cross-backend diagnostics, without PNG transforms.

use super::{QualityReadback, ReadbackKind, Renderer};
use std::path::Path;

fn enabled() -> bool {
    std::env::var("BLOOM_QUALITY_RAW").is_ok_and(|value| value == "1")
}

impl Renderer {
    /// Queue named diagnostics and, when explicitly requested by the tool,
    /// the existing MRT readback. Both execute after the measured window.
    pub fn request_quality_capture(&mut self, directory: String) {
        if enabled() {
            self.pending_mrt_capture_dir = Some(
                Path::new(&directory)
                    .join("mrt")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        self.pending_quality_capture_dir = Some(directory);
    }
}

fn packed_rows(data: &[u8], row_bytes: usize, pitch: usize, height: usize) -> Option<Vec<u8>> {
    if row_bytes > pitch || data.len() < pitch.checked_mul(height)? {
        return None;
    }
    Some(
        data.chunks_exact(pitch)
            .take(height)
            .flat_map(|row| row[..row_bytes].iter().copied())
            .collect(),
    )
}

pub(super) fn write_intermediate(directory: &Path, readback: &QualityReadback, data: &[u8]) {
    if !enabled() {
        return;
    }
    let (format, bytes_per_pixel) = match readback.kind {
        ReadbackKind::Hdr => ("rgba16float", 8),
        ReadbackKind::Depth => ("depth32float", 4),
        ReadbackKind::Rgba8 => ("rgba8unorm", 4),
    };
    let Some(bytes) = packed_rows(
        data,
        readback.width as usize * bytes_per_pixel,
        readback.padded_bytes_per_row as usize,
        readback.height as usize,
    ) else {
        eprintln!(
            "bloom: invalid raw intermediate layout for '{}'",
            readback.name
        );
        return;
    };
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    let metadata = format!(
        "{{\"schema\":\"bloom-raw-intermediate-v1\",\"name\":\"{}\",\"format\":\"{format}\",\"width\":{},\"height\":{},\"bytes_per_pixel\":{bytes_per_pixel},\"byte_count\":{},\"row_order\":\"top-to-bottom\",\"endianness\":\"little\",\"fnv1a64\":\"{hash:016x}\"}}\n",
        readback.name, readback.width, readback.height, bytes.len(),
    );
    let directory = directory.join("raw");
    let result = std::fs::create_dir_all(&directory)
        .and_then(|()| std::fs::write(directory.join(format!("{}.raw", readback.name)), bytes))
        .and_then(|()| std::fs::write(directory.join(format!("{}.json", readback.name)), metadata));
    if let Err(error) = result {
        eprintln!(
            "bloom: raw intermediate '{}' write failed: {error}",
            readback.name
        );
    }
}

#[cfg(test)]
mod tests {
    use super::packed_rows;

    #[test]
    fn raw_rows_exclude_gpu_padding_and_reject_truncated_layouts() {
        let mut bytes = vec![0x99; 512];
        bytes[..4].copy_from_slice(&1.0_f32.to_le_bytes());
        bytes[256..260].copy_from_slice(&0.5_f32.to_le_bytes());
        assert_eq!(
            packed_rows(&bytes, 4, 256, 2).unwrap(),
            [1.0_f32.to_le_bytes(), 0.5_f32.to_le_bytes()].concat(),
        );
        assert!(packed_rows(&bytes[..511], 4, 256, 2).is_none());
        assert!(packed_rows(&bytes, 257, 256, 2).is_none());
    }
}
