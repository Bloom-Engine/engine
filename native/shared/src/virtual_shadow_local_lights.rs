use super::{VSM_MAX_LOCAL_SHADOW_LIGHTS, VSM_MAX_LOCAL_SHADOW_REQUESTS};

const LOCAL_SHADOW_NEAR_MIN: f32 = 0.02;
const LOCAL_SHADOW_NEAR_MAX: f32 = 0.25;

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct LocalShadowRequest {
    pub light_index: u16,
    pub position: [f32; 3],
    pub range: f32,
    pub intensity: f32,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct LocalShadowAdmissionStats {
    pub submitted: u16,
    pub visible: u16,
    pub admitted: u16,
    pub visibility_rejected: u16,
    pub budget_suppressed: u16,
}

fn finite_bounds(request: LocalShadowRequest) -> ([f32; 3], [f32; 3]) {
    (
        [
            request.position[0] - request.range,
            request.position[1] - request.range,
            request.position[2] - request.range,
        ],
        [
            request.position[0] + request.range,
            request.position[1] + request.range,
            request.position[2] + request.range,
        ],
    )
}

fn distance_to_influence(request: LocalShadowRequest, camera: [f32; 3]) -> f32 {
    let dx = request.position[0] - camera[0];
    let dy = request.position[1] - camera[1];
    let dz = request.position[2] - camera[2];
    (dx * dx + dy * dy + dz * dz).sqrt() - request.range
}

/// Six right-handed cube-face view projections using WebGPU's `[0, 1]`
/// depth convention. Face order is shared with the shader's major-axis
/// selector: +X, -X, +Y, -Y, +Z, -Z.
pub(super) fn face_vps(request: LocalShadowRequest) -> [[[f32; 4]; 4]; 6] {
    let position = request.position;
    let near = (request.range * 0.002)
        .clamp(LOCAL_SHADOW_NEAR_MIN, LOCAL_SHADOW_NEAR_MAX)
        .min(request.range * 0.5);
    let far = request.range.max(near + f32::EPSILON);
    let reciprocal_depth = 1.0 / (near - far);
    let projection = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, far * reciprocal_depth, -1.0],
        [0.0, 0.0, near * far * reciprocal_depth, 0.0],
    ];
    let directions_and_up = [
        ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
        ([0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
    ];
    std::array::from_fn(|face| {
        let (direction, up) = directions_and_up[face];
        let center = [
            position[0] + direction[0],
            position[1] + direction[1],
            position[2] + direction[2],
        ];
        crate::renderer::mat4_multiply(
            projection,
            crate::renderer::mat4_look_at(position, center, up),
        )
    })
}

/// Deterministic, camera-visible admission for local shadow address spaces.
///
/// The submission ceiling and selected-light budget are independent: an app
/// can submit the full 256-light lighting ABI while only the five most
/// relevant visible shadow requests compete for the shared 30-page cube-face
/// footprint. Ties end at the stable per-frame light index.
pub(super) fn admit(
    requests: &[LocalShadowRequest],
    camera: [f32; 3],
    camera_planes: &[[f32; 4]; 6],
) -> (Vec<LocalShadowRequest>, LocalShadowAdmissionStats) {
    let submitted = requests.len().min(VSM_MAX_LOCAL_SHADOW_REQUESTS);
    let mut visible: Vec<_> = requests[..submitted]
        .iter()
        .copied()
        .filter(|request| {
            let (minimum, maximum) = finite_bounds(*request);
            !crate::scene::aabb_outside_frustum(camera_planes, minimum, maximum)
        })
        .collect();
    visible.sort_by(|left, right| {
        distance_to_influence(*left, camera)
            .total_cmp(&distance_to_influence(*right, camera))
            .then_with(|| {
                let left_energy = left.intensity * left.range * left.range;
                let right_energy = right.intensity * right.range * right.range;
                right_energy.total_cmp(&left_energy)
            })
            .then_with(|| left.light_index.cmp(&right.light_index))
    });
    let visible_count = visible.len();
    visible.truncate(VSM_MAX_LOCAL_SHADOW_LIGHTS);
    let admitted = visible.len();
    (
        visible,
        LocalShadowAdmissionStats {
            submitted: submitted as u16,
            visible: visible_count as u16,
            admitted: admitted as u16,
            visibility_rejected: (submitted - visible_count) as u16,
            budget_suppressed: (visible_count - admitted) as u16,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(index: u16, x: f32, intensity: f32) -> LocalShadowRequest {
        LocalShadowRequest {
            light_index: index,
            position: [x, 0.0, -5.0],
            range: 1.0,
            intensity,
        }
    }

    fn open_planes() -> [[f32; 4]; 6] {
        [
            [1.0, 0.0, 0.0, 100.0],
            [-1.0, 0.0, 0.0, 100.0],
            [0.0, 1.0, 0.0, 100.0],
            [0.0, -1.0, 0.0, 100.0],
            [0.0, 0.0, 1.0, 100.0],
            [0.0, 0.0, -1.0, 100.0],
        ]
    }

    #[test]
    fn one_hundred_visible_requests_are_submitted_but_page_budget_bounded() {
        let requests: Vec<_> = (0..100)
            .map(|index| request(index, f32::from(index) * 0.01, 1.0))
            .collect();
        let (admitted, stats) = admit(&requests, [0.0; 3], &open_planes());
        assert_eq!(stats.submitted, 100);
        assert_eq!(stats.visible, 100);
        assert_eq!(stats.admitted, VSM_MAX_LOCAL_SHADOW_LIGHTS as u16);
        assert_eq!(stats.budget_suppressed, 95);
        assert_eq!(admitted.len() * super::super::VSM_LOCAL_FACES as usize, 30);
    }

    #[test]
    fn invisible_requests_consume_no_page_budget() {
        let mut planes = open_planes();
        planes[0] = [1.0, 0.0, 0.0, -10.0];
        let requests = vec![request(0, 0.0, 1.0), request(1, 1.0, 1.0)];
        let (admitted, stats) = admit(&requests, [0.0; 3], &planes);
        assert!(admitted.is_empty());
        assert_eq!(stats.visibility_rejected, 2);
        assert_eq!(stats.budget_suppressed, 0);
    }

    #[test]
    fn nearest_influence_wins_then_energy_then_stable_index() {
        let requests = vec![
            request(7, 3.0, 1.0),
            request(4, 3.0, 2.0),
            request(2, 1.0, 0.1),
            request(8, 3.0, 2.0),
            request(9, 4.0, 50.0),
            request(10, 5.0, 50.0),
        ];
        let (admitted, _) = admit(&requests, [0.0; 3], &open_planes());
        let indices: Vec<_> = admitted.iter().map(|request| request.light_index).collect();
        assert_eq!(indices, [2, 4, 8, 7, 9]);
    }

    #[test]
    fn submission_never_reads_beyond_the_public_light_ceiling() {
        let requests: Vec<_> = (0..300)
            .map(|index| request(index as u16, 0.0, 1.0))
            .collect();
        let (_, stats) = admit(&requests, [0.0; 3], &open_planes());
        assert_eq!(stats.submitted, VSM_MAX_LOCAL_SHADOW_REQUESTS as u16);
    }

    fn ndc(vp: &[[f32; 4]; 4], point: [f32; 3]) -> [f32; 3] {
        let clip = crate::renderer::mat4_mul_vec4(vp, &[point[0], point[1], point[2], 1.0]);
        [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]]
    }

    #[test]
    fn cube_faces_center_each_major_axis_with_webgpu_depth() {
        let request = LocalShadowRequest {
            light_index: 0,
            position: [2.0, 3.0, 4.0],
            range: 10.0,
            intensity: 1.0,
        };
        let directions = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for (face, vp) in face_vps(request).iter().enumerate() {
            let point = [
                request.position[0] + directions[face][0],
                request.position[1] + directions[face][1],
                request.position[2] + directions[face][2],
            ];
            let projected = ndc(vp, point);
            assert!(projected[0].abs() < 1.0e-5, "face {face}: {projected:?}");
            assert!(projected[1].abs() < 1.0e-5, "face {face}: {projected:?}");
            assert!(
                (0.0..=1.0).contains(&projected[2]),
                "face {face}: {projected:?}"
            );
        }
    }
}
