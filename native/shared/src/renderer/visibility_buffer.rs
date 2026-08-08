//! Packed visibility-buffer contract for the #27 qualification path.
//!
//! This module does not enable a shipping render path. It locks the 8-byte
//! target ABI and reconstruction math that an opt-in A/B implementation will
//! use. The existing forward MRT remains authoritative until total frame cost
//! and image parity pass on every required capability tier.

pub(crate) const VISIBILITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
pub(crate) const VISIBILITY_BYTES_PER_PIXEL: u64 = 8;
pub(crate) const INVALID_DRAW_ID: u32 = u32::MAX;
pub(crate) const FRONT_FACE_BIT: u32 = 1 << 31;
pub(crate) const PRIMITIVE_ID_MASK: u32 = FRONT_FACE_BIT - 1;

/// One visibility-buffer texel. The second word reserves its high bit for the
/// rasterized face orientation and leaves 31 bits for the primitive index.
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct VisibilityRecord {
    pub draw_id: u32,
    pub primitive_and_face: u32,
}

impl VisibilityRecord {
    pub(crate) const BACKGROUND: Self = Self {
        draw_id: INVALID_DRAW_ID,
        primitive_and_face: u32::MAX,
    };

    pub(crate) const fn encode(
        draw_id: u32,
        primitive_id: u32,
        front_facing: bool,
    ) -> Option<Self> {
        if draw_id == INVALID_DRAW_ID || primitive_id > PRIMITIVE_ID_MASK {
            return None;
        }
        Some(Self {
            draw_id,
            primitive_and_face: primitive_id | if front_facing { FRONT_FACE_BIT } else { 0 },
        })
    }

    pub(crate) const fn decode(self) -> Option<(u32, u32, bool)> {
        if self.draw_id == INVALID_DRAW_ID {
            return None;
        }
        Some((
            self.draw_id,
            self.primitive_and_face & PRIMITIVE_ID_MASK,
            (self.primitive_and_face & FRONT_FACE_BIT) != 0,
        ))
    }
}

/// Exact allocation size of the packed visibility target, excluding backend
/// row/heap alignment that must be reported separately by the runtime A/B.
pub(crate) const fn target_bytes(width: u32, height: u32) -> Option<u64> {
    match (width as u64).checked_mul(height as u64) {
        Some(pixels) => pixels.checked_mul(VISIBILITY_BYTES_PER_PIXEL),
        None => None,
    }
}

/// Stable machine-readable contract included in renderer diagnostics even
/// while the experimental path is disabled.
pub(crate) fn contract_json() -> String {
    let format_name = match VISIBILITY_FORMAT {
        wgpu::TextureFormat::Rg32Uint => "rg32uint",
        _ => "invalid",
    };
    let background = VisibilityRecord::BACKGROUND;
    let max_record = VisibilityRecord::encode(0, PRIMITIVE_ID_MASK, true)
        .expect("the visibility ABI maximum must remain encodable");
    debug_assert_eq!(background.decode(), None);
    debug_assert_eq!(
        max_record.decode(),
        Some((0, PRIMITIVE_ID_MASK, true))
    );
    format!(
        concat!(
            "{{\"format\":\"{}\",\"bytes_per_pixel\":{},",
            "\"invalid_draw_id\":{},\"primitive_bits\":31,",
            "\"front_face_bits\":1,\"shipping_enabled\":false,",
            "\"native_1080p_bytes\":{},\"reconstruction_wgsl_bytes\":{},",
            "\"activation\":\"opt-in A/B qualification required\"}}"
        ),
        format_name,
        VISIBILITY_BYTES_PER_PIXEL,
        INVALID_DRAW_ID,
        target_bytes(1_920, 1_080).expect("1080p visibility allocation is bounded"),
        RECONSTRUCTION_WGSL.len(),
    )
}

pub(crate) const RECONSTRUCTION_WGSL: &str =
    include_str!("../../shaders/visibility_buffer/reconstruct.wgsl");

#[cfg(test)]
fn screen_barycentrics(point: [f32; 2], triangle: [[f32; 2]; 3]) -> Option<[f32; 3]> {
    let edge = |a: [f32; 2], b: [f32; 2], p: [f32; 2]| {
        (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
    };
    let area = edge(triangle[1], triangle[2], triangle[0]);
    if area.abs() <= 1.0e-12 {
        return None;
    }
    Some([
        edge(triangle[1], triangle[2], point) / area,
        edge(triangle[2], triangle[0], point) / area,
        edge(triangle[0], triangle[1], point) / area,
    ])
}

#[cfg(test)]
fn perspective_barycentrics(point: [f32; 2], clip: [[f32; 4]; 3]) -> Option<[f32; 3]> {
    if clip.iter().any(|vertex| vertex[3].abs() <= 1.0e-12) {
        return None;
    }
    let ndc = [
        [clip[0][0] / clip[0][3], clip[0][1] / clip[0][3]],
        [clip[1][0] / clip[1][3], clip[1][1] / clip[1][3]],
        [clip[2][0] / clip[2][3], clip[2][1] / clip[2][3]],
    ];
    let linear = screen_barycentrics(point, ndc)?;
    let weighted = [
        linear[0] / clip[0][3],
        linear[1] / clip[1][3],
        linear[2] / clip[2][3],
    ];
    let sum = weighted[0] + weighted[1] + weighted[2];
    if sum.abs() <= 1.0e-12 {
        return None;
    }
    Some([weighted[0] / sum, weighted[1] / sum, weighted[2] / sum])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn packed_record_is_exactly_one_rg32uint_texel() {
        assert_eq!(std::mem::size_of::<VisibilityRecord>(), 8);
        assert_eq!(std::mem::align_of::<VisibilityRecord>(), 4);
        assert_eq!(VISIBILITY_BYTES_PER_PIXEL, 8);
        assert_eq!(VISIBILITY_FORMAT, wgpu::TextureFormat::Rg32Uint);
        assert_eq!(target_bytes(1_920, 1_080), Some(16_588_800));
        assert_eq!(target_bytes(u32::MAX, u32::MAX), None);

        let report = contract_json();
        assert!(report.starts_with("{\"format\":\"rg32uint\""));
        assert!(report.contains("\"native_1080p_bytes\":16588800"));
        assert!(report.contains("\"shipping_enabled\":false"));
    }

    #[test]
    fn ids_and_face_orientation_round_trip_without_background_collision() {
        for (draw, primitive, front) in [
            (0, 0, false),
            (17, 42, true),
            (u32::MAX - 1, PRIMITIVE_ID_MASK, false),
        ] {
            let encoded = VisibilityRecord::encode(draw, primitive, front).unwrap();
            assert_eq!(encoded.decode(), Some((draw, primitive, front)));
        }
        assert_eq!(VisibilityRecord::BACKGROUND.decode(), None);
        assert_eq!(VisibilityRecord::encode(INVALID_DRAW_ID, 0, true), None);
        assert_eq!(VisibilityRecord::encode(0, FRONT_FACE_BIT, true), None);
    }

    #[test]
    fn perspective_reconstruction_matches_vertices_and_known_depth_weighting() {
        let clip = [
            [-1.0, -1.0, 0.2, 1.0],
            [2.0, -2.0, 0.4, 2.0],
            [0.0, 4.0, 0.8, 4.0],
        ];
        for (point, expected) in [
            ([-1.0, -1.0], [1.0, 0.0, 0.0]),
            ([1.0, -1.0], [0.0, 1.0, 0.0]),
            ([0.0, 1.0], [0.0, 0.0, 1.0]),
        ] {
            let actual = perspective_barycentrics(point, clip).unwrap();
            for lane in 0..3 {
                assert_close(actual[lane], expected[lane]);
            }
        }

        let center = perspective_barycentrics([0.0, -1.0 / 3.0], clip).unwrap();
        assert_close(center[0], 4.0 / 7.0);
        assert_close(center[1], 2.0 / 7.0);
        assert_close(center[2], 1.0 / 7.0);
        assert_close(center.iter().sum(), 1.0);
    }

    #[test]
    fn shared_reconstruction_header_parses_and_keeps_the_cpu_abi_constants() {
        wgpu::naga::front::wgsl::parse_str(RECONSTRUCTION_WGSL)
            .unwrap_or_else(|error| panic!("visibility reconstruction WGSL failed: {error:?}"));
        assert!(RECONSTRUCTION_WGSL
            .contains("const BLOOM_VISIBILITY_FRONT_FACE_BIT: u32 = 0x80000000u"));
        assert!(RECONSTRUCTION_WGSL.contains("fn bloom_perspective_barycentrics("));
    }
}
