use super::{VSM_CLIP_LEVELS, VSM_VIRTUAL_PAGES_PER_AXIS};

/// Camera-centered directional clip levels with page-stable light-space origins.
///
/// Each level covers the sphere selected by the established CSM split, plus
/// enough guard for one virtual-page origin step and receiver bias. Moving
/// within a virtual page keeps the matrix byte-identical; crossing a page
/// rebases that level while missing pages continue to use CSM.
pub(crate) fn level_vps(
    light_dir: [f32; 3],
    camera_pos: [f32; 3],
    splits: [f32; VSM_CLIP_LEVELS as usize],
    scene_bounds: Option<([f32; 3], [f32; 3])>,
) -> [[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize] {
    let d = normalize_or(light_dir, [0.0, 1.0, 0.0]);
    let up_hint = if d[1].abs() > 0.99 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let right = normalize_or(cross(up_hint, d), [1.0, 0.0, 0.0]);
    let ortho_up = cross(d, right);

    std::array::from_fn(|level| {
        let radius = ((splits[level].max(0.5) * 1.1) * 16.0).ceil() / 16.0;
        let page_world = 2.0 * radius / f32::from(VSM_VIRTUAL_PAGES_PER_AXIS);
        let snap = |value: f32, step: f32| (value / step).floor() * step;
        let light_x = snap(dot(camera_pos, right), page_world);
        let light_y = snap(dot(camera_pos, ortho_up), page_world);
        let light_z = snap(dot(camera_pos, d), 2.0);
        let center = add3(
            add3(scale3(right, light_x), scale3(ortho_up, light_y)),
            scale3(d, light_z),
        );

        let mut back = radius;
        let mut far = radius;
        if let Some((bmin, bmax)) = scene_bounds {
            for corner in 0..8 {
                let point = [
                    if corner & 1 == 0 { bmin[0] } else { bmax[0] },
                    if corner & 2 == 0 { bmin[1] } else { bmax[1] },
                    if corner & 4 == 0 { bmin[2] } else { bmax[2] },
                ];
                let along = dot(sub3(point, center), d);
                back = back.max(along);
                far = far.max(-along);
            }
        }
        back = (back / 2.0).ceil() * 2.0;
        far = (far / 2.0).ceil() * 2.0;
        let light_pos = add3(center, scale3(d, back));
        let view = crate::renderer::mat4_look_at(light_pos, center, up_hint);
        let projection =
            crate::renderer::mat4_ortho(-radius, radius, -radius, radius, 0.0, back + far);
        crate::renderer::mat4_multiply(projection, view)
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

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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
        let first = level_vps([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], SPLITS, None);
        let second = level_vps([0.0, 1.0, 0.0], [1.2, 0.0, 0.0], SPLITS, None);
        assert_ne!(first[0], second[0]);
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
