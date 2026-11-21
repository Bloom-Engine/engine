//! Static far sun-shadow layer.
//!
//! A large world (a city seen from a hill or from the air) wants sun shadows to
//! the horizon, but stretching the camera-fitted cascades over kilometres both
//! blurs the near shadows and redraws the whole world whenever any caster or the
//! camera moves. With the far layer on, the third cascade stops following the
//! camera: it is one fixed orthographic map over a box the game names, drawn
//! from the static casters only, and redrawn at most once per refresh interval
//! (the sun, the box or the static caster set having changed). The first two
//! cascades split the ordinary shadow distance between them; receivers beyond
//! it sample the far map. Every receiver shader already selects cascade 2
//! beyond the second split and reads its fit radius from the third split, so
//! no shader changes: the third split carries the far map's half-extent,
//! scaled by `FAR_RECEIVER_OFFSET`.

/// The receivers push themselves 1.5 shadow texels off their surface (and bias
/// their depth by one), sized from the cascade's fit radius. A far texel is
/// metres wide: 1.5 of them lifted the ground under a 7 m house above its roof
/// and the far shadows all but vanished. Half of it keeps sun-facing walls free
/// of acne in the Nineveh test views and lets the shadows of houses land.
pub const FAR_RECEIVER_OFFSET: f32 = 0.5;

/// The far layer as the game set it (`Renderer::set_far_shadow`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FarShadow {
    /// Centre of the covered box on the ground plane (x, z), metres.
    pub center: [f32; 2],
    /// Half the covered rectangle's extent along x and along z, metres.
    pub half: [f32; 2],
    /// Lowest and highest caster height to cover, metres.
    pub y_range: [f32; 2],
    /// Minimum frames between two redraws of the map. The first draw, and a
    /// draw after the box itself changes, happen at once.
    pub refresh_frames: u32,
}

/// Bookkeeping of what the far map holds.
#[derive(Default)]
pub struct FarShadowState {
    /// The projection the far map was last drawn with; receivers keep
    /// sampling through it until the next redraw.
    pub rendered_vp: Option<[[f32; 4]; 4]>,
    /// Frames since the last redraw.
    pub frames_since: u32,
    /// Per-node shadow uniforms for the far map, which holds far more casters
    /// than a cascade's region of the shared buffer (`SHADOW_MAX_NODES`).
    /// Grown on demand; (buffer, bind group, capacity in nodes).
    pub uniforms: Option<(wgpu::Buffer, wgpu::BindGroup, usize)>,
    /// How many casters the last redraw drew (statistics).
    pub last_casters: usize,
    /// How many redraws so far (statistics).
    pub redraws: u64,
}

impl FarShadowState {
    /// A uniform buffer and bind group holding at least `nodes` per-node slots.
    pub fn uniforms_for(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        nodes: usize,
    ) -> (&wgpu::Buffer, &wgpu::BindGroup) {
        let need = nodes.max(1);
        let grow = match &self.uniforms {
            Some((_, _, cap)) => *cap < need,
            None => true,
        };
        if grow {
            let cap = need.next_power_of_two().max(1024);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("far_shadow_uniform_buf"),
                size: (crate::shadows::SHADOW_UNIFORM_STRIDE as usize * cap) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("far_shadow_uniform_bg"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(std::mem::size_of::<
                            crate::shadows::ShadowUniforms,
                        >() as u64),
                    }),
                }],
            });
            self.uniforms = Some((buffer, bind_group, cap));
        }
        let (buffer, bind_group, _) = self.uniforms.as_ref().expect("far uniforms just made");
        (buffer, bind_group)
    }
}

/// The far map's view-projection for a normalised light direction `d`
/// (surface toward light) and the light-plane basis `right`, `up` the cascades
/// use. Returns the VP and the larger half-extent of the ortho window (the
/// value the receiver shaders take as the cascade's fit radius).
pub fn far_vp(
    far: &FarShadow,
    d: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    up_hint: [f32; 3],
) -> ([[f32; 4]; 4], f32) {
    let y_mid = 0.5 * (far.y_range[0] + far.y_range[1]);
    let y_half = 0.5 * (far.y_range[1] - far.y_range[0]).abs();
    let center = [far.center[0], y_mid, far.center[1]];
    // Half-extents of the box (half, y_half, half) projected on each axis.
    let extent =
        |a: [f32; 3]| a[0].abs() * far.half[0] + a[1].abs() * y_half + a[2].abs() * far.half[1];
    // Whole metres, so float noise in the light direction cannot move the map.
    let hx = extent(right).ceil();
    let hy = extent(up).ceil();
    let hz = extent(d).ceil() + 1.0;
    let eye = [
        center[0] + d[0] * hz,
        center[1] + d[1] * hz,
        center[2] + d[2] * hz,
    ];
    let view = crate::renderer::mat4_look_at(eye, center, up_hint);
    let proj = crate::renderer::mat4_ortho(-hx, hx, -hy, hy, 0.0, 2.0 * hz);
    (crate::renderer::mat4_multiply(proj, view), hx.max(hy))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
        let v = [p[0], p[1], p[2], 1.0];
        let mut o = [0.0f32; 4];
        for (r, out) in o.iter_mut().enumerate() {
            *out = (0..4).map(|c| m[c][r] * v[c]).sum();
        }
        [o[0] / o[3], o[1] / o[3], o[2] / o[3]]
    }

    #[test]
    fn far_vp_covers_the_box() {
        let far = FarShadow {
            center: [100.0, -200.0],
            half: 3000.0,
            y_range: [-20.0, 120.0],
            refresh_frames: 30,
        };
        let d = {
            let v = [0.62f32, 0.55, 0.56];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let up_hint = [0.0, 1.0, 0.0];
        let right = {
            let r = [
                up_hint[1] * d[2] - up_hint[2] * d[1],
                up_hint[2] * d[0] - up_hint[0] * d[2],
                up_hint[0] * d[1] - up_hint[1] * d[0],
            ];
            let l = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            [r[0] / l, r[1] / l, r[2] / l]
        };
        let up = [
            d[1] * right[2] - d[2] * right[1],
            d[2] * right[0] - d[0] * right[2],
            d[0] * right[1] - d[1] * right[0],
        ];
        let (vp, fit) = far_vp(&far, d, right, up, up_hint);
        assert!(fit >= 3000.0);
        for &x in &[-2900.0f32, 3100.0] {
            for &z in &[-3200.0f32, 2800.0] {
                for &y in &[-20.0f32, 120.0] {
                    let n = apply(vp, [x, y, z]);
                    assert!(
                        n[0].abs() <= 1.0 && n[1].abs() <= 1.0,
                        "{:?} -> {:?}",
                        (x, y, z),
                        n
                    );
                    assert!(
                        (0.0..=1.0).contains(&n[2]) || (-1.0..=1.0).contains(&n[2]),
                        "depth {:?}",
                        n
                    );
                }
            }
        }
    }
}

impl crate::renderer::Renderer {
    /// Turn the static far shadow layer on over the rectangle of half-extents
    /// `half_x`, `half_z` metres centred on (`center_x`, `center_z`), covering
    /// casters between `y_min` and `y_max`, redrawn at most every
    /// `refresh_frames` frames; or off with a half-extent `<= 0`. See
    /// `crate::far_shadow`.
    #[allow(clippy::too_many_arguments)]
    pub fn set_far_shadow(
        &mut self,
        center_x: f32,
        center_z: f32,
        half_x: f32,
        half_z: f32,
        y_min: f32,
        y_max: f32,
        refresh_frames: u32,
    ) {
        let next = if half_x.is_finite() && half_z.is_finite() && half_x > 0.0 && half_z > 0.0 {
            Some(FarShadow {
                center: [center_x, center_z],
                half: [half_x, half_z],
                y_range: [y_min.min(y_max), y_min.max(y_max)],
                refresh_frames: refresh_frames.max(1),
            })
        } else {
            None
        };
        let box_changed = match (self.shadow_map.far, next) {
            (Some(a), Some(b)) => {
                a.center != b.center || a.half != b.half || a.y_range != b.y_range
            }
            (None, None) => false,
            _ => true,
        };
        self.shadow_map.far = next;
        if box_changed {
            // The new box is drawn at once; the cascades re-split.
            self.shadow_map.far_state.rendered_vp = None;
            self.shadow_map.invalidate();
        }
    }

    /// Far-layer statistics: 0 redraws so far, 1 casters in the last redraw,
    /// 2 frames since it.
    pub fn far_shadow_stat(&self, which: u32) -> f64 {
        let s = &self.shadow_map.far_state;
        match which {
            0 => s.redraws as f64,
            1 => s.last_casters as f64,
            _ => s.frames_since as f64,
        }
    }
}
