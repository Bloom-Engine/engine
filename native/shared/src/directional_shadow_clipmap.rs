use super::{VSM_CLIP_LEVELS, VSM_VIRTUAL_PAGES_PER_AXIS};

const CACHE_KEY_WORDS: usize = 14;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct DirectionalClipmapCacheKey {
    origin_pages: [i32; 2],
    stable_vp_bits: [u32; CACHE_KEY_WORDS],
}

impl DirectionalClipmapCacheKey {
    /// Return the virtual-address shift that keeps old physical pages fixed in
    /// world space. A different basis, scale, or depth projection cannot
    /// safely scroll and returns `None` for full level invalidation.
    pub(crate) fn scroll_from(self, previous: Self) -> Option<[i32; 2]> {
        (self.stable_vp_bits == previous.stable_vp_bits).then_some([
            previous.origin_pages[0].saturating_sub(self.origin_pages[0]),
            self.origin_pages[1].saturating_sub(previous.origin_pages[1]),
        ])
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct DirectionalClipmapProjection {
    pub(crate) level_vps: [[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize],
    pub(crate) cache_keys: [DirectionalClipmapCacheKey; VSM_CLIP_LEVELS as usize],
}

/// Camera-centered directional clip levels with page-stable light-space origins.
///
/// Each level covers the sphere selected by the established CSM split, plus
/// enough guard for one virtual-page origin step and receiver bias. Moving
/// within a virtual page keeps the matrix byte-identical; crossing a page
/// rebases that level while missing pages continue to use CSM.
pub(crate) fn projection(
    light_dir: [f32; 3],
    camera_pos: [f32; 3],
    splits: [f32; VSM_CLIP_LEVELS as usize],
    scene_bounds: Option<([f32; 3], [f32; 3])>,
) -> DirectionalClipmapProjection {
    let d = normalize_or(light_dir, [0.0, 1.0, 0.0]);
    let up_hint = if d[1].abs() > 0.99 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let right = normalize_or(cross(up_hint, d), [1.0, 0.0, 0.0]);
    let ortho_up = cross(d, right);

    let levels = std::array::from_fn(|level| {
        let radius = ((splits[level].max(0.5) * 1.1) * 16.0).ceil() / 16.0;
        let page_world = 2.0 * radius / f32::from(VSM_VIRTUAL_PAGES_PER_AXIS);
        let origin_pages = [
            page_origin(dot(camera_pos, right), page_world),
            page_origin(dot(camera_pos, ortho_up), page_world),
        ];
        let light_x = origin_pages[0] as f32 * page_world;
        let light_y = origin_pages[1] as f32 * page_world;
        let light_z = snap(dot(camera_pos, d), 2.0);

        let mut back = radius;
        let mut far = radius;
        if let Some((bmin, bmax)) = scene_bounds {
            for corner in 0..8 {
                let point = [
                    if corner & 1 == 0 { bmin[0] } else { bmax[0] },
                    if corner & 2 == 0 { bmin[1] } else { bmax[1] },
                    if corner & 4 == 0 { bmin[2] } else { bmax[2] },
                ];
                // Planar clipmap origin is orthogonal to the light direction
                // by construction. Keep depth fitting explicitly independent
                // from it so floating-point cancellation cannot spuriously
                // turn a page scroll into a full depth-projection change.
                let along = dot(point, d) - light_z;
                back = back.max(along);
                far = far.max(-along);
            }
        }
        back = (back / 2.0).ceil() * 2.0;
        far = (far / 2.0).ceil() * 2.0;
        // Equivalent to look_at(center + d * back, center, up_hint), written
        // analytically so the Z translation contains no planar-origin roundoff.
        let view = [
            [right[0], ortho_up[0], d[0], 0.0],
            [right[1], ortho_up[1], d[1], 0.0],
            [right[2], ortho_up[2], d[2], 0.0],
            [-light_x, -light_y, -(light_z + back), 1.0],
        ];
        let projection =
            crate::renderer::mat4_ortho(-radius, radius, -radius, radius, 0.0, back + far);
        let vp = crate::renderer::mat4_multiply(projection, view);
        (
            vp,
            DirectionalClipmapCacheKey {
                origin_pages,
                stable_vp_bits: stable_vp_bits(vp),
            },
        )
    });
    DirectionalClipmapProjection {
        level_vps: levels.map(|level| level.0),
        cache_keys: levels.map(|level| level.1),
    }
}

#[cfg(test)]
pub(crate) fn level_vps(
    light_dir: [f32; 3],
    camera_pos: [f32; 3],
    splits: [f32; VSM_CLIP_LEVELS as usize],
    scene_bounds: Option<([f32; 3], [f32; 3])>,
) -> [[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize] {
    projection(light_dir, camera_pos, splits, scene_bounds).level_vps
}

fn page_origin(value: f32, page_world: f32) -> i32 {
    if !value.is_finite() || !page_world.is_finite() || page_world <= 0.0 {
        return 0;
    }
    (value / page_world).floor() as i32
}

fn snap(value: f32, step: f32) -> f32 {
    (value / step).floor() * step
}

fn stable_vp_bits(vp: [[f32; 4]; 4]) -> [u32; CACHE_KEY_WORDS] {
    std::array::from_fn(|index| {
        if index < 12 {
            vp[index / 4][index % 4].to_bits()
        } else {
            vp[3][index - 10].to_bits()
        }
    })
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length = dot(value, value).sqrt();
    if length > 1.0e-6 {
        scale3(value, length.recip())
    } else {
        fallback
    }
}

fn scale3(value: [f32; 3], scale: f32) -> [f32; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPLITS: [f32; VSM_CLIP_LEVELS as usize] = [16.0, 40.0, 80.0];

    #[test]
    fn sub_page_camera_motion_keeps_every_level_byte_stable() {
        let first = level_vps([0.4, 0.8, 0.2], [2.0, 3.0, 4.0], SPLITS, None);
        let second = level_vps([0.4, 0.8, 0.2], [2.01, 3.0, 4.0], SPLITS, None);
        assert_eq!(first, second);
    }

    #[test]
    fn crossing_near_page_origin_rebases_near_level() {
        let first = projection([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], SPLITS, None);
        let second = projection([0.0, 1.0, 0.0], [1.2, 0.0, 0.0], SPLITS, None);
        assert_ne!(first.level_vps[0], second.level_vps[0]);
        assert_eq!(
            second.cache_keys[0].scroll_from(first.cache_keys[0]),
            Some([0, 1]),
        );
    }

    #[test]
    fn cache_scroll_keeps_a_world_point_on_the_same_physical_page() {
        let first = projection([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], SPLITS, None);
        let second = projection([0.0, 1.0, 0.0], [1.2, 0.0, 0.0], SPLITS, None);
        let page = |vp: &[[f32; 4]; 4]| {
            let clip = crate::renderer::mat4_mul_vec4(vp, &[0.0, 0.0, 0.0, 1.0]);
            let axis = f32::from(VSM_VIRTUAL_PAGES_PER_AXIS);
            [
                ((clip[0] / clip[3] * 0.5 + 0.5) * axis).floor() as i32,
                ((1.0 - (clip[1] / clip[3] * 0.5 + 0.5)) * axis).floor() as i32,
            ]
        };
        let old_page = page(&first.level_vps[0]);
        let new_page = page(&second.level_vps[0]);
        let scroll = second.cache_keys[0]
            .scroll_from(first.cache_keys[0])
            .unwrap();
        assert_eq!(new_page, [old_page[0] + scroll[0], old_page[1] + scroll[1]],);
    }

    #[test]
    fn arbitrary_light_basis_accepts_a_planar_page_scroll() {
        let light = normalize_or([0.4, 0.8, 0.2], [0.0, 1.0, 0.0]);
        let right = normalize_or(cross([0.0, 1.0, 0.0], light), [1.0, 0.0, 0.0]);
        let bounds = ([-8.0, -3.0, -12.0], [14.0, 9.0, 7.0]);
        let first = projection(light, [0.0, 0.0, 0.0], SPLITS, Some(bounds));
        let second = projection(light, scale3(right, 1.2), SPLITS, Some(bounds));
        assert!(second.cache_keys[0]
            .scroll_from(first.cache_keys[0])
            .is_some());
    }

    #[test]
    fn depth_or_light_changes_reject_page_scroll() {
        let first = projection([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], SPLITS, None);
        let depth_changed = projection([0.0, 1.0, 0.0], [0.0, 2.1, 0.0], SPLITS, None);
        let light_changed = projection([0.1, 1.0, 0.0], [0.0, 0.0, 0.0], SPLITS, None);
        assert_eq!(
            depth_changed.cache_keys[0].scroll_from(first.cache_keys[0]),
            None,
        );
        assert_eq!(
            light_changed.cache_keys[0].scroll_from(first.cache_keys[0]),
            None,
        );
    }

    #[test]
    fn light_direction_magnitude_cannot_change_a_clipmap() {
        let first = level_vps([0.0, 1.0, 0.0], [2.0, 3.0, 4.0], SPLITS, None);
        let second = level_vps([0.0, 8.0, 0.0], [2.0, 3.0, 4.0], SPLITS, None);
        assert_eq!(first, second);
    }

    #[test]
    fn scene_depth_bounds_remain_inside_every_level() {
        let bounds = ([-4.0, -8.0, -12.0], [9.0, 11.0, 15.0]);
        let vps = level_vps([0.4, 0.8, 0.2], [2.0, 3.0, 4.0], SPLITS, Some(bounds));
        for vp in vps {
            for corner in 0..8 {
                let point = [
                    if corner & 1 == 0 {
                        bounds.0[0]
                    } else {
                        bounds.1[0]
                    },
                    if corner & 2 == 0 {
                        bounds.0[1]
                    } else {
                        bounds.1[1]
                    },
                    if corner & 4 == 0 {
                        bounds.0[2]
                    } else {
                        bounds.1[2]
                    },
                    1.0,
                ];
                let clip = crate::renderer::mat4_mul_vec4(&vp, &point);
                let depth = clip[2] / clip[3];
                assert!((-1.0e-5..=1.0 + 1.0e-5).contains(&depth));
            }
        }
    }
}
