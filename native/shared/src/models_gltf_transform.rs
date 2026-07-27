//! Column-major glTF node-transform helpers.
//!
//! Keeping point, direction, and normal transforms distinct is load-bearing:
//! points use the full affine matrix, tangents use its linear part, and normals
//! use the inverse transpose of that linear part.

/// Transform a 3D point by a column-major 4x4 matrix (w = 1).
pub(super) fn mat4_transform_point(m: &[[f32; 4]; 4], p: &[f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

/// Transform a tangent/direction by the linear part of a column-major 4x4.
pub(super) fn mat4_transform_direction(m: &[[f32; 4]; 4], v: &[f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// Transform a direction by a column-major 3x3 matrix.
pub(super) fn mat3_transform_vec(m: &[[f32; 3]; 3], v: &[f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// Inverse transpose of the 3x3 rotation/scale part of a column-major 4x4.
///
/// A singular transform has no well-defined normal matrix. Identity is the
/// conservative fallback: it retains authored normals instead of introducing
/// infinities or NaNs.
pub(super) fn mat4_inverse_transpose_3x3(m: &[[f32; 4]; 4]) -> [[f32; 3]; 3] {
    // Row-major names for the linear block while the source remains
    // column-major: A = [[a,b,c], [d,e,f], [g,h,i]].
    let a = m[0][0];
    let b = m[1][0];
    let c = m[2][0];
    let d = m[0][1];
    let e = m[1][1];
    let f = m[2][1];
    let g = m[0][2];
    let h = m[1][2];
    let i = m[2][2];

    // Adjugate entries grouped as columns of A^-1.
    let inv00 = e * i - f * h;
    let inv10 = f * g - d * i;
    let inv20 = d * h - e * g;
    let inv01 = c * h - b * i;
    let inv11 = a * i - c * g;
    let inv21 = b * g - a * h;
    let inv02 = b * f - c * e;
    let inv12 = c * d - a * f;
    let inv22 = a * e - b * d;

    let det = a * inv00 + b * inv10 + c * inv20;
    if det.abs() < 1e-10 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let inv_det = 1.0 / det;

    // Columns of (A^-1)^T are rows of A^-1. The previous implementation
    // returned the columns of A^-1 itself, applying the opposite rotation to
    // every normal on a rotated glTF node.
    [
        [inv00 * inv_det, inv01 * inv_det, inv02 * inv_det],
        [inv10 * inv_det, inv11 * inv_det, inv12 * inv_det],
        [inv20 * inv_det, inv21 * inv_det, inv22 * inv_det],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec3_close(actual: [f32; 3], expected: [f32; 3]) {
        for channel in 0..3 {
            assert!(
                (actual[channel] - expected[channel]).abs() < 1e-6,
                "channel {channel}: actual={actual:?}, expected={expected:?}"
            );
        }
    }

    #[test]
    fn pure_rotation_moves_directions_and_normals_the_same_way() {
        // +90 degrees around X, column-major.
        let matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [3.0, 4.0, 5.0, 1.0],
        ];
        let direction = mat4_transform_direction(&matrix, &[0.0, 1.0, 0.0]);
        let normal_matrix = mat4_inverse_transpose_3x3(&matrix);
        let normal = mat3_transform_vec(&normal_matrix, &[0.0, 1.0, 0.0]);
        assert_vec3_close(direction, [0.0, 0.0, 1.0]);
        assert_vec3_close(normal, direction);
        assert_vec3_close(
            mat4_transform_point(&matrix, &[0.0, 1.0, 0.0]),
            [3.0, 4.0, 6.0],
        );
    }

    #[test]
    fn non_uniform_scale_uses_distinct_tangent_and_normal_matrices() {
        // +90 degrees around Z after scale (2, 3, 4), column-major.
        let matrix = [
            [0.0, 2.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_vec3_close(
            mat4_transform_direction(&matrix, &[1.0, 0.0, 0.0]),
            [0.0, 2.0, 0.0],
        );
        assert_vec3_close(
            mat3_transform_vec(&mat4_inverse_transpose_3x3(&matrix), &[1.0, 0.0, 0.0]),
            [0.0, 0.5, 0.0],
        );
    }

    #[test]
    fn singular_normal_matrix_falls_back_without_non_finite_values() {
        let singular = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(
            mat4_inverse_transpose_3x3(&singular),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
    }
}
