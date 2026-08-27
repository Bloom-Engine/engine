pub(super) fn finite_affine(model: [[f32; 4]; 4]) -> bool {
    model.iter().flatten().all(|value| value.is_finite())
        && model[0][3].abs() <= 1.0e-6
        && model[1][3].abs() <= 1.0e-6
        && model[2][3].abs() <= 1.0e-6
        && (model[3][3] - 1.0).abs() <= 1.0e-6
}

pub(super) fn normal_rows_and_cone_safety(
    model: [[f32; 4]; 4],
) -> Option<([[f32; 4]; 3], bool, bool)> {
    if !finite_affine(model) {
        return None;
    }
    let a00 = model[0][0];
    let a01 = model[1][0];
    let a02 = model[2][0];
    let a10 = model[0][1];
    let a11 = model[1][1];
    let a12 = model[2][1];
    let a20 = model[0][2];
    let a21 = model[1][2];
    let a22 = model[2][2];
    let cofactors = [
        [
            a11 * a22 - a12 * a21,
            a12 * a20 - a10 * a22,
            a10 * a21 - a11 * a20,
        ],
        [
            a02 * a21 - a01 * a22,
            a00 * a22 - a02 * a20,
            a01 * a20 - a00 * a21,
        ],
        [
            a01 * a12 - a02 * a11,
            a02 * a10 - a00 * a12,
            a00 * a11 - a01 * a10,
        ],
    ];
    let determinant = a00 * cofactors[0][0] + a01 * cofactors[0][1] + a02 * cofactors[0][2];
    if !determinant.is_finite() || determinant.abs() <= 1.0e-12 {
        return None;
    }
    let inverse_determinant = determinant.recip();
    let normal_rows = std::array::from_fn(|row| {
        [
            cofactors[row][0] * inverse_determinant,
            cofactors[row][1] * inverse_determinant,
            cofactors[row][2] * inverse_determinant,
            0.0,
        ]
    });

    let columns = [[a00, a10, a20], [a01, a11, a21], [a02, a12, a22]];
    let squared = columns.map(|column| dot3(column, column));
    let scale2 = squared.into_iter().fold(0.0, f32::max);
    let tolerance = scale2.max(1.0) * 1.0e-4;
    let cone_safe = squared
        .iter()
        .all(|length2| (*length2 - scale2).abs() <= tolerance)
        && dot3(columns[0], columns[1]).abs() <= tolerance
        && dot3(columns[0], columns[2]).abs() <= tolerance
        && dot3(columns[1], columns[2]).abs() <= tolerance;
    Some((normal_rows, cone_safe, determinant < 0.0))
}

pub(super) const fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
