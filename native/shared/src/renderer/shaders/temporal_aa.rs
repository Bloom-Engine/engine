//! Temporal antialiasing and reconstruction shader.

/// TAA shader. Reads `composed_rt` (scene HDR + post-effects + fog +
/// shafts already merged upstream) and performs only temporal
/// reprojection with neighborhood clamp, blending against the
/// history RT. For static scenes the blend converges in ~10 frames
/// to a fully sub-pixel-resolved image.
pub(in crate::renderer) const TAA_SHADER_WGSL: &str = concat!(
    include_str!("taa_reconstruction.wgsl"),
    "
struct TaaParams {
    /// abs(x) = blend factor (current-frame weight); sign(x) = whether the
    /// unjittered camera transform moved; yz = the CURRENT frame's
    /// jitter as a composed-texture UV offset (see the unjitter note at the
    /// current-frame sample); abs(w) = render scale, with positive sign for
    /// perspective depth and negative sign for orthographic depth.
    params: vec4<f32>,
    /// Inverse of the current-frame view-projection matrix —
    /// reconstructs world-space position for history reprojection.
    inv_vp: mat4x4<f32>,
    /// Previous-frame view-projection — projects world pos into
    /// history UV.
    prev_vp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u: TaaParams;
@group(0) @binding(1) var composed_tex: texture_2d<f32>;
@group(0) @binding(2) var composed_samp: sampler;
@group(0) @binding(3) var history_tex: texture_2d<f32>;
@group(0) @binding(4) var history_samp: sampler;
@group(0) @binding(5) var depth_tex: texture_depth_2d;
@group(0) @binding(6) var depth_samp: sampler;
@group(0) @binding(7) var velocity_tex: texture_2d<f32>;
@group(0) @binding(8) var velocity_samp: sampler;
@group(0) @binding(9) var history_depth_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct TaaOut {
    @location(0) color: vec4<f32>,
    /// R = geometric depth; G packs persistent history confidence and the
    /// independent detail lock, with a negative encoded range retaining prior
    /// reactive coverage.
    @location(1) provenance_history: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid & 1u) * 4u) - 1.0;
    let y = f32((vid >> 1u) * 4u) - 1.0;
    var out: VsOut;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// RGB <-> YCoCg conversions. Reversible, linear, cheap (no matrix
// multiply). Used by the TAA neighborhood clamp so we can bound
// history's luma (Y) against the source neighborhood's statistical
// range while leaving chroma (Co, Cg) alone — the per-channel RGB
// clamp was causing chromatic sparkle on grazing-angle stone.
fn rgb_to_ycocg(c: vec3<f32>) -> vec3<f32> {
    let Co = c.r - c.b;
    let tmp = c.b + Co * 0.5;
    let Cg = c.g - tmp;
    let Y  = tmp + Cg * 0.5;
    return vec3<f32>(Y, Co, Cg);
}
fn ycocg_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let tmp = c.x - c.z * 0.5;
    let g   = c.z + tmp;
    let b   = tmp - c.y * 0.5;
    let r   = c.y + b;
    return vec3<f32>(r, g, b);
}

// History lives at output resolution. A bilinear lookup is exact while a
// reprojected coordinate remains on an output texel centre, but under camera
// or object motion it averages four already-filtered history pixels. Repeating
// that every frame progressively removes texture and silhouette detail. Keep
// one history fetch, but compress its bilinear phase continuously toward the
// nearest completed output sample. This retains more detail without the hard
// half-texel transitions of point history sampling. Renderer-known camera
// motion selects this path; object-only reprojection retains its established
// behavior until it has an independently qualified phase policy. The
// stationary path is
// byte-for-byte the original lookup so settled temporal supersampling is not
// disturbed. Later variance, depth, and reactive guards still decide whether
// a moving sample belongs to the current surface.
fn sample_history_reprojected(uv: vec2<f32>, camera_moving: bool) -> vec4<f32> {
    if (!camera_moving) {
        return textureSampleLevel(history_tex, history_samp, uv, 0.0);
    }
    let dims = vec2<f32>(textureDimensions(history_tex));
    let history_pixel = uv * dims - vec2<f32>(0.5);
    let base = floor(history_pixel);
    let phase = fract(history_pixel);
    let compressed_phase = phase * phase * (vec2<f32>(3.0) - 2.0 * phase);
    let compressed_uv = (base + compressed_phase + vec2<f32>(0.5)) / dims;
    return textureSampleLevel(history_tex, history_samp, compressed_uv, 0.0);
}

@fragment
fn fs_main(in: VsOut) -> TaaOut {
    // composed_tex already carries HDR + SSR + SSGI*albedo + bloom +
    // fog + shafts — TAA only needs to reproject history and blend.
    // Alpha carries `indirect_weight` (see scene_compose) which the
    // composite pass reads to apply AO only to indirect-dominated
    // pixels; pass it through blended with the colour so history
    // stays consistent.
    // Both current reconstruction and neighborhood statistics operate in the
    // same input texture domain. Compute its size/reciprocal once per output
    // pixel and share it; the previous shader repeated the dimension queries
    // and reciprocal divisions in both stages.
    let input_size = vec2<f32>(textureDimensions(composed_tex));
    let input_texel = 1.0 / input_size;
    var current_weight = abs(u.params.x);
    let camera_moving = u.params.x < 0.0;
    let reconstruction_scale = clamp(abs(u.params.w), 0.5, 1.0);
    // The material-aware variant replaces this compile-time constant. The
    // ordinary lazy topology consumes prior reactive provenance immediately.
    let preserve_reactive_history = false;

    // Fractional reconstruction reads the current color at the jitter-corrected
    // source phase. Select depth and velocity from that same input position;
    // using the unshifted output UV could cross a low-resolution texel boundary
    // and make reprojection/provenance belong to a different surface sample.
    // Native and half-scale TAA deliberately keep their established geometry
    // lookups byte-for-byte. The default 0.75 tier owns this correction during
    // real camera motion, where the moving native-reference corpus qualifies
    // it independently from settled and legacy-half reconstruction.
    let reconstruction_geometry_phase =
        reconstruction_scale >= 0.75 && reconstruction_scale < 0.95 && camera_moving;
    var geometry_uv = in.uv;
    if (reconstruction_geometry_phase) {
        geometry_uv = geometry_uv + u.params.yz;
    }

    // Closest-depth velocity dilation. Sampling the low-resolution velocity
    // buffer linearly mixed foreground motion with zero/background motion at
    // silhouettes. The resulting half-vector reprojected into neither
    // surface, producing translucent gray trails while turning the camera.
    // Select the closest sample in a bounded 3x3 input footprint so thin
    // foreground geometry owns the history lookup around its edge.
    let depth_dims = vec2<i32>(textureDimensions(depth_tex));
    let depth_max_coord = depth_dims - vec2<i32>(1);
    let center_coord = clamp(
        vec2<i32>(floor(geometry_uv * vec2<f32>(depth_dims))),
        vec2<i32>(0),
        depth_max_coord,
    );
    var closest_coord = center_coord;
    var depth = textureLoad(depth_tex, center_coord, 0);
    let dilation_offsets = array<vec2<i32>, 4>(
        vec2<i32>(-1, 0), vec2<i32>(1, 0),
        vec2<i32>(0, -1), vec2<i32>(0, 1),
    );
    for (var i = 0; i < 4; i = i + 1) {
        let coord = clamp(center_coord + dilation_offsets[i], vec2<i32>(0), depth_max_coord);
        let candidate_depth = textureLoad(depth_tex, coord, 0);
        if (candidate_depth < depth) {
            depth = candidate_depth;
            closest_coord = coord;
        }
    }
    var selected_depth_uv = in.uv;
    if (reconstruction_geometry_phase) {
        selected_depth_uv =
            (vec2<f32>(closest_coord) + vec2<f32>(0.5)) * input_texel;
    }
    let ndc = vec4<f32>(
        selected_depth_uv.x * 2.0 - 1.0,
        (1.0 - selected_depth_uv.y) * 2.0 - 1.0,
        depth,
        1.0,
    );
    let world_h = u.inv_vp * ndc;
    let world = world_h.xyz / world_h.w;

    // Persist a geometric depth key beside color history. Perspective clip-W
    // is positive linear view distance; orthographic clip-W is constant, so
    // that projection stores NDC depth instead. Sky gets an explicit far key.
    let perspective = u.params.w > 0.0;
    let current_depth_key = select(
        depth,
        select(1.0 / max(abs(world_h.w), 0.000001), 10000.0, depth >= 0.9999),
        perspective,
    );
    let prev_world_clip = u.prev_vp * vec4<f32>(world, 1.0);
    var expected_prev_depth = select(
        prev_world_clip.z / max(abs(prev_world_clip.w), 0.000001),
        prev_world_clip.w,
        perspective,
    );

    let vel = textureLoad(velocity_tex, closest_coord, 0).rg;
    var velocity_divergence = 0.0;
    if (any(closest_coord != center_coord)) {
        let center_vel = textureLoad(velocity_tex, center_coord, 0).rg;
        velocity_divergence = length(vel - center_vel);
    }
    let vel_len = length(vel);
    let motion_alpha = smoothstep(0.0005, 0.008, vel_len);

    var prev_uv: vec2<f32>;
    if (depth >= 0.9999) {
        // Sky / far plane: the positional reconstruction divides by a
        // near-zero w and reprojects sky pixels onto arbitrary scene
        // points — the luma-only history clamp then locks that wrong
        // chroma in forever (uniform green/red sky tint). The sky is at
        // infinity, so reproject the view DIRECTION instead: exact under
        // camera rotation, translation-invariant by definition.
        let dir = world_h.xyz; // w ~ 0 at the far plane: xyz IS the direction
        let prev_clip = u.prev_vp * vec4<f32>(dir, 0.0);
        if (prev_clip.w > 0.00001) {
            let prev_ndc = prev_clip.xyz / prev_clip.w;
            prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 1.0 - (prev_ndc.y * 0.5 + 0.5));
        } else {
            prev_uv = in.uv;
        }
        expected_prev_depth = 10000.0;
    } else {
        // Geometry always owns a velocity texel, and a zero vector is a
        // meaningful result for a static surface. Falling back to prev_vp
        // when velocity was zero reintroduced the previous projection jitter
        // into an already unjittered full-resolution history surface. Each
        // static frame then sampled history at a different subpixel offset and
        // progressively averaged fine texture detail into a broad blur.
        prev_uv = vec2<f32>(in.uv.x - vel.x, in.uv.y + vel.y);
    }

    // Keep a static native-resolution sample on its OUTPUT pixel. The velocity
    // reference projection already reapplies the current jitter (EN-022), so
    // static history maps to that same pixel; fully undoing jitter here
    // low-pass filtered native labels and texture grain before accumulation.
    // Fractional-resolution reconstruction keeps the established alignment,
    // since one input pixel covers more than one output pixel. Blend only over
    // the final five percent of render scale so custom near-native scales do
    // not cross a hard sampling discontinuity. During actual motion restore
    // full current-frame alignment immediately; it bounds slow-pan phase crawl
    // while reprojection supplies the temporal sample location. Sky has no
    // geometry velocity, so include its actual directional reprojection
    // distance in the same motion classification.
    let reprojection_motion = max(vel_len, length(prev_uv - in.uv));
    var jitter_alignment = 1.0;
    // Uniform branch: fractional tiers keep the former shader path (and its
    // cost) exactly. Only near-native tiers pay for the motion classifier.
    if (reconstruction_scale > 0.95) {
        let static_jitter_alignment =
            1.0 - smoothstep(0.95, 1.0, reconstruction_scale);
        // The static path is deliberately narrow: even a very slow camera pan
        // must use the established aligned sample rather than intermittently
        // mixing between sharp and aligned phases across the image.
        let alignment_motion = smoothstep(0.0000001, 0.000001, reprojection_motion);
        jitter_alignment = mix(static_jitter_alignment, 1.0, alignment_motion);
    }
    let src_uv = in.uv + u.params.yz * jitter_alignment;
    var current_sample: vec4<f32>;
    var center_rgb: vec3<f32>;
    var fractional_coverage = 1.0;
    var fractional_mean = vec3<f32>(0.0);
    var fractional_stddev = vec3<f32>(0.0);
    var current_feature_lock = 0.0;
    var use_fractional_statistics = false;
    // This is uniform across the draw. Keep native reconstruction and its
    // already-qualified material response byte-for-byte unchanged.
    if (
        reconstruction_scale >= 0.5 &&
        reconstruction_scale < 0.95 &&
        current_weight < 0.999
    ) {
        let reconstructed = sample_fractional_lanczos2(
            src_uv,
            input_size,
            input_texel,
            reconstruction_scale,
        );
        current_sample = reconstructed.value;
        center_rgb = reconstructed.center.rgb;
        fractional_mean = reconstructed.mean;
        fractional_stddev = reconstructed.stddev;
        // Seed a persistent detail lock from statistics already required by
        // rectification. Restrict it to real camera motion: settled output is
        // byte-for-byte unchanged, while moving high-frequency detail can
        // retain history through a transient source-phase excursion.
        current_feature_lock = select(
            0.0,
            1.0,
            camera_moving &&
                reconstructed.stddev.x > max(abs(reconstructed.mean.x) * 0.05, 0.002),
        );
        // At half scale, the Lanczos taps move substantially with source
        // phase. Reusing their moments as a history-clamp neighborhood makes
        // the clamp breathe as the Halton phase advances. Keep history
        // validation on the stable output-footprint cross below 0.75 while
        // retaining the sharper sample itself.
        use_fractional_statistics = reconstruction_scale >= 0.75;
        // Preserve the separable kernel's phase coverage through accumulation.
        // At 0.75 its average raw weight is approximately 0.74 over a jitter
        // cycle; normalizing around that mean keeps Bloom's authored temporal
        // window while low-coverage phases contribute proportionally less.
        // The 0.5 path freezes coverage at the measured mean over all sixteen
        // Halton phases and both output-pixel parities. Per-phase coverage
        // changes are useful at 0.75, but at half scale they modulate alpha
        // enough to make a slow pan alternate between sharp and soft frames.
        fractional_coverage = select(
            0.784,
            clamp(reconstructed.weight / 0.74, 0.25, 2.0),
            reconstruction_scale >= 0.75,
        );
        current_weight = 1.0 - pow(1.0 - current_weight, fractional_coverage);
    } else {
        // History resets and the four-frame bootstrap retain the qualified
        // cubic. A single Lanczos phase is intentionally sharp; it is only
        // representative once temporal accumulation can combine phases.
        current_sample = sample_catmull_rom(src_uv, input_size, input_texel);
        center_rgb = textureSampleLevel(composed_tex, composed_samp, src_uv, 0.0).rgb;
    }
    let current = current_sample.rgb;
    let current_w = current_sample.a;

    var history = current;
    var history_w = current_w;
    var history_depth = current_depth_key;
    var history_confidence = 0.0;
    var history_reactive = 0.0;
    var history_feature_lock = 0.0;
    let history_in_bounds =
        prev_uv.x >= 0.0 && prev_uv.x <= 1.0 &&
        prev_uv.y >= 0.0 && prev_uv.y <= 1.0;
    // Reset/bootstrapping frames must not sample old color or provenance.
    // Even a final blend weight of one cannot make stale inputs harmless:
    // floating-point interpolation and confidence locks can retain their effect.
    let history_usable = history_in_bounds && current_weight < 0.999;
    if (history_usable) {
        let h_sample = sample_history_reprojected(prev_uv, camera_moving);
        history = h_sample.rgb;
        history_w = h_sample.a;
        let history_depth_dims = vec2<i32>(textureDimensions(history_depth_tex));
        let history_depth_coord = clamp(
            vec2<i32>(floor(prev_uv * vec2<f32>(history_depth_dims))),
            vec2<i32>(0),
            history_depth_dims - vec2<i32>(1),
        );
        let history_provenance = textureLoad(history_depth_tex, history_depth_coord, 0).rg;
        history_depth = history_provenance.r;
        let temporal_history = unpack_temporal_history(history_provenance.g);
        // Negative provenance came from a reactive frame. The ordinary path
        // resets its confidence and detail lock, consuming current color if the
        // last reactive contributor disappeared with its lazy topology.
        let history_payload_usable = select(
            1.0 - temporal_history.reactive,
            1.0,
            preserve_reactive_history,
        );
        history_confidence = temporal_history.confidence * history_payload_usable;
        history_feature_lock = temporal_history.feature_lock * history_payload_usable;
        history_reactive = temporal_history.reactive *
            select(0.0, 1.0, preserve_reactive_history);
    }

    // Reject history whose geometric provenance no longer matches the world
    // point reprojected into the previous camera. This is the missing guard
    // that color variance cannot provide: a pale counter and pale wall can be
    // statistically similar while belonging to different surfaces, creating
    // the translucent gray haze seen during camera motion.
    // Variance clamp in YCoCg (Karis 2014). Per-channel RGB min/max
    // clamping was producing chromatic sparkle on the stone floor
    // at grazing angles: high-frequency normal-map specular makes
    // each jittered frame's Cg/Co vary significantly, and clamping
    // each channel independently lets history's chroma get pinned
    // to whatever the current frame's specific Cg/Co range was.
    // Clamping only the *luma* axis (Y) preserves chroma stability
    // across frames; the 1σ variance range is a statistical clamp
    // that absorbs single-pixel outliers without collapsing to a
    // hard min/max bound.
    // Define the clipping neighborhood from the OUTPUT-pixel footprint mapped
    // into input texels. At 0.75 scale the old one-input-texel offsets mixed a
    // 33% wider spatial neighborhood into every history decision and softened
    // detail after confidence converged. Using the exact output footprint at
    // 0.75 and above remains stable in the slow-pan corpus. Lower tiers keep
    // the proven 0.8-input-pixel floor, where adjacent output positions are
    // too strongly correlated for a tighter statistical window. Packing scale
    // into the existing projection flag adds no uniform bytes, bindings,
    // samples, or passes.
    let statistics_footprint = select(
        0.80,
        reconstruction_scale,
        reconstruction_scale >= 0.75,
    );
    let statistics_texel = input_texel * statistics_footprint;
    // Keep the center statistical sample bilinear. The reconstructed current
    // contains the cubic filter's negative-lobe response; feeding that into
    // the variance estimate makes the clamp breathe with the reconstruction
    // phase even at native scale. This lookup existed in the baseline path,
    // so retaining it does not add a performance cost versus shipped TAA.
    var mean: vec3<f32>;
    var stddev: vec3<f32>;
    if (use_fractional_statistics) {
        mean = fractional_mean;
        stddev = fractional_stddev;
    } else {
        var m1 = rgb_to_ycocg(center_rgb);
        var m2 = m1 * m1;
        let statistics_offsets = array<vec2<f32>, 4>(
            vec2<f32>(-1.0, 0.0), vec2<f32>(1.0, 0.0),
            vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0),
        );
        for (var i = 0; i < 4; i = i + 1) {
            let s_uv = src_uv + statistics_offsets[i] * statistics_texel;
            let s_rgb = textureSampleLevel(composed_tex, composed_samp, s_uv, 0.0).rgb;
            let s = rgb_to_ycocg(s_rgb);
            m1 = m1 + s;
            m2 = m2 + s * s;
        }
        let n_samples = 5.0;
        mean = m1 / n_samples;
        // A five-sample cross measures 3/5 of the first-order variance measured by
        // the former 3x3 grid over the same radius (2/5 versus 2/3 per axis).
        // Correct that known sampling bias so the history clip preserves the same
        // linear-ramp bandwidth while spending the four saved reads on the exact
        // cubic's diagonal lobes.
        let variance =
            max(m2 / n_samples - mean * mean, vec3<f32>(0.0)) * (5.0 / 3.0);
        stddev = sqrt(variance);
    }

    // Motion-aware γ + alpha. At rest γ=1.25 lets sub-pixel jitter
    // history through for smooth accumulation. Under any camera
    // motion γ collapses fast to 0.25 — forces reprojected
    // history within a quarter-sigma of the neighborhood mean,
    // which is tight enough to reject the 'dark column in
    // history, bright wall in current' case that the wider band
    // let slip. alpha ramps to 0.85 at the same time so remaining
    // history contributes only 15 %.
    let gamma = mix(1.25, 0.25, motion_alpha);
    let y_min = mean.x - gamma * stddev.x;
    let y_max = mean.x + gamma * stddev.x;

    let history_ycocg = rgb_to_ycocg(history);
    let history_y_clamped = clamp(history_ycocg.x, y_min, y_max);
    // Chroma is clamped too, but at 3x the luma band (flicker fix).
    // Fully unclamped chroma let stale history colour bleed through on
    // high-contrast edges — green terrain fringing crawling along cloud
    // and canopy silhouettes during camera motion. The loose band keeps
    // the anti-sparkle intent of the luma-only design (a hard
    // per-channel clamp caused chromatic sparkle on grazing stone)
    // while bounding gross cross-object colour bleed.
    let c_gamma = gamma * 3.0;
    let co_clamped = clamp(history_ycocg.y,
        mean.y - c_gamma * stddev.y, mean.y + c_gamma * stddev.y);
    let cg_clamped = clamp(history_ycocg.z,
        mean.z - c_gamma * stddev.z, mean.z + c_gamma * stddev.z);
    let clamped_history = ycocg_to_rgb(vec3<f32>(history_y_clamped, co_clamped, cg_clamped));

    // Color-change rejection has two deliberately different operating
    // ranges. During motion, a tight clamped-luma test helps flush a sample
    // that reprojected onto the wrong side of a textured silhouette. At rest,
    // however, jitter phases are SUPPOSED to differ inside the source
    // footprint: treating a perfectly ordinary 0.6-sigma texture excursion as
    // a disocclusion forced the low-resolution current sample into most Bistro
    // pixels every frame and prevented temporal super-resolution from ever
    // accumulating its sub-pixel samples.
    let history_dist = abs(history_y_clamped - mean.x);

    // A stationary scene can still change illumination/emissive state or
    // inherit poisoned chroma, so it is not safe to disable color rejection
    // outright. Collapse luma and half-weighted chroma into one max-norm
    // distance. This deliberately avoids a chroma vector length (sqrt) and
    // multiple smoothstep evaluations in the full-output-resolution pass.
    // Coherent lighting/material changes on a flat surface cross the broad
    // band immediately; valid jittered texture samples stay inside it and are
    // variance-clipped before blending.
    let raw_color_delta = abs(history_ycocg - mean);
    let gross_color_dist = max(
        raw_color_delta.x,
        max(raw_color_delta.y, raw_color_delta.z) * 0.5,
    );
    let gross_color_sigma = max(
        stddev.x,
        max(stddev.y, stddev.z) * 0.5,
    );
    let color_dist = mix(gross_color_dist, history_dist, motion_alpha);
    let reject_lo = mix(
        max(gross_color_sigma * 3.0, 0.02),
        stddev.x * 0.25,
        motion_alpha,
    );
    let reject_hi = mix(
        max(gross_color_sigma * 6.0, 0.06),
        max(stddev.x, 0.0001),
        motion_alpha,
    );
    let disocclusion = smoothstep(reject_lo, reject_hi, color_dist);

    // Depth provenance needs a screen gradient to tolerate a real geometric
    // edge. Static color-change classification needs the same quad footprint
    // below. Take both derivatives as one vector pair so the successful narrow
    // phase-strip detector does not add another derivative pair to full-screen
    // TAA.
    let temporal_gradients =
        abs(dpdx(vec2<f32>(expected_prev_depth, disocclusion))) +
        abs(dpdy(vec2<f32>(expected_prev_depth, disocclusion)));
    let depth_base_tolerance = 0.02 + abs(expected_prev_depth) * 0.005;
    let depth_tolerance = max(
        depth_base_tolerance,
        min(temporal_gradients.x * 2.0, depth_base_tolerance * 4.0),
    );
    let depth_error = abs(history_depth - expected_prev_depth);
    let raw_depth_disocclusion = select(
        0.0,
        smoothstep(depth_tolerance, depth_tolerance * 2.0, depth_error),
        history_in_bounds,
    );

    // A stationary projection still jitters the low-resolution depth raster.
    // Thin geometry and grazing, finely tessellated surfaces can therefore
    // alternate which depth owns an output pixel even though their temporal
    // color footprint remains compatible. Treating that coverage change as a
    // geometric disocclusion repeatedly discarded settled history and made
    // native and fractional Bistro detail flicker with a zero motion buffer.
    //
    // Suppress depth-only rejection only for the exact static case. Camera
    // motion keeps the renderer-owned camera flag; object motion keeps its
    // velocity or the closest-depth divergence signal; topology/material
    // changes still use reactive and broad color rejection. This adds no
    // samples or resources and does not weaken moving silhouettes.
    let static_zero_velocity = !camera_moving
        && vel_len < 0.0000001
        && velocity_divergence < 0.0000001;
    let jitter_coverage_compatible = gross_color_dist <= reject_hi;
    let depth_disocclusion = select(
        raw_depth_disocclusion,
        0.0,
        static_zero_velocity && jitter_coverage_compatible,
    );

    // Divergent neighbor motion marks a silhouette/disocclusion footprint.
    // Prefer the current frame there even when both individual vectors are
    // small, rather than allowing a long-lived cross-surface history average.
    let divergence_alpha = smoothstep(0.00025, 0.003, velocity_divergence);
    // Moving high-frequency reconstruction must not accumulate a long-lived
    // phase lag. Its existing detail classifier is already paid for by the
    // fractional rectification footprint; give those pixels a modest current
    // floor while leaving smooth surfaces and every stationary path unchanged.
    let feature_motion_floor = current_feature_lock *
        select(0.0, 0.15, vel_len >= 0.0005);
    let motion_ramped = max(
        max(mix(current_weight, 0.85, motion_alpha), divergence_alpha),
        feature_motion_floor,
    );
    // Reactive coverage is injected by the material-aware TAA variant. Keep a
    // concrete zero in the base shader so confidence policy is identical for
    // both variants and diagnostics can report the real persistent lock.
    let current_reactive = 0.0;
    let reactive = 0.0;
    // Confidence represents geometric/material continuity. Do not erase that
    // persistent fact for a luma-only neighborhood excursion: sub-pixel normal
    // and texture detail can cross the color-disocclusion threshold at rest,
    // and repeatedly unlocking those valid pixels turns detail into shimmer.
    // Color disocclusion still raises alpha below; it simply does not destroy
    // the longer-lived geometric lock.
    let temporal_rejection = max(depth_disocclusion, max(divergence_alpha, reactive));

    // A confidence value of 1 represents sixteen compatible samples. At rest,
    // enforce the unbiased running-average weight 1/(N+1) until lock; after
    // lock, the authored steady-state alpha takes over. Under even sub-pixel
    // motion the established motion policy already bounds stale history, so
    // disable bootstrap before 0.04 output pixels/frame rather than turning a
    // safe resolve into visibly noisier current samples during a slow pan.
    let history_sample_count = history_confidence * 16.0;
    let bootstrap_running_alpha = select(
        1.0,
        fractional_coverage / (history_sample_count + fractional_coverage),
        history_usable,
    );
    let bootstrap_static = 1.0 - smoothstep(0.00001, 0.0001, vel_len);
    let bootstrap_alpha = mix(current_weight, bootstrap_running_alpha, bootstrap_static);
    // A locked native pixel sees the same finite jitter phases repeat. The
    // authored 10% current weight makes them shimmer even when a slow pan has
    // valid, coherent reprojection. Keep every geometric/reactive guard and
    // the static 24-frame window. Compatible color excursions retain that
    // window; changes outside the band still unlock, while gradual lighting
    // converges normally. Coherent slow motion uses the qualified 8.5% cap.
    let settled_coherent_lock = select(
        0.0,
        1.0,
        history_confidence >= 0.999 &&
        (!camera_moving || reconstruction_scale > 0.95) &&
        reprojection_motion < 0.00025 &&
        jitter_coverage_compatible &&
        current_weight >= 0.095 &&
        temporal_rejection <= 0.01,
    );
    // A zero-velocity finite-depth surface is not disoccluding merely because
    // the next finite jitter phase shades a different point inside the same
    // output pixel. Restrict this protection to a narrow color-disocclusion
    // footprint: ordinary static pixels keep the authored 24-frame window,
    // broad lighting/material changes keep normal color rejection, and the
    // procedural sky remains free to animate without geometry velocity.
    let settled_static_phase_candidate = select(
        0.0,
        1.0,
        history_confidence >= 0.999 &&
        static_zero_velocity &&
        depth < 0.9999 &&
        current_weight >= 0.095 &&
        temporal_rejection <= 0.01,
    );
    let rejected_color_phase = select(0.0, 1.0, disocclusion >= 0.10);
    let narrow_color_phase =
        rejected_color_phase * smoothstep(0.05, 0.50, temporal_gradients.y);
    let settled_static_phase_lock = settled_static_phase_candidate * narrow_color_phase;
    let settled_lock = max(settled_coherent_lock, settled_static_phase_lock);
    let static_current_cap = mix(0.041666667, 0.015625, settled_static_phase_lock);
    let settled_current_cap = select(static_current_cap, 0.085, camera_moving);
    let color_motion_ramped = max(motion_ramped, disocclusion);
    let settled_motion_ramped = mix(
        color_motion_ramped,
        min(color_motion_ramped, settled_current_cap),
        settled_lock,
    );
    let settled_bootstrap_alpha = mix(
        bootstrap_alpha,
        min(bootstrap_alpha, settled_current_cap),
        settled_lock,
    );
    let alpha = max(
        max(settled_motion_ramped, max(depth_disocclusion, reactive)),
        settled_bootstrap_alpha,
    );
    let accepted_history = select(0.0, 1.0 - temporal_rejection, history_usable);
    let next_history_confidence =
        min(history_confidence + 1.0 / 16.0, 1.0) * accepted_history;
    // Keep structural lifetime independent from ordinary color confidence.
    // Existing geometric/reactive acceptance, the broad color band, and a
    // stationary camera kill a stale lock; compatible moving detail seeds or
    // renews it. The lock only protects rectification, leaving the qualified
    // accumulation alpha intact.
    let feature_shading_stable = gross_color_dist <= reject_hi * 4.0;
    let protected_feature_lock = select(
        0.0,
        history_feature_lock,
        camera_moving && accepted_history >= 0.99 && feature_shading_stable,
    );
    let next_feature_lock = max(
        select(current_feature_lock, 0.0, current_reactive > 0.01),
        protected_feature_lock,
    );
    let rectification_lock = max(settled_static_phase_lock, protected_feature_lock);
    let stable_history = mix(clamped_history, history, rectification_lock);
    var blended = mix(stable_history, current, alpha);
    // The temporal average suppresses some of the reconstruction's
    // source-phase residual. On a settled stationary surface, feed a bounded
    // part of that already-computed current-vs-linear residual through the
    // temporal update: tying it to alpha lets history accumulate the detail
    // instead of exposing one source phase at full strength. During camera
    // motion, provenance now follows the same jitter-corrected input texel as
    // color; retain only two percent of the bounded current residual to avoid
    // replacing that stability with history lag. Object-only motion keeps its
    // established response. Render-scale changes rebuild the targets and
    // reset history, so policies cannot splice across one accumulation epoch.
    // This adds no samples and vanishes at native scale.
    let settled_static = history_confidence
        * (1.0 - motion_alpha)
        * select(1.0, 0.0, camera_moving);
    let fractional_reconstruction = 1.0 - smoothstep(0.95, 1.0, reconstruction_scale);
    let reconstruction_detail = clamp(
        current - center_rgb,
        vec3<f32>(-0.08),
        vec3<f32>(0.08),
    );
    let reconstruction_detail_weight = select(
        0.20,
        3.0 * alpha,
        reconstruction_scale >= 0.75,
    );
    // Retain the qualified two-percent moving residual on every fractional
    // surface. Pixels already admitted by the production high-frequency
    // detail classifier receive a bounded additional four percent. This
    // targets alpha-cutout foliage and authored microtexture without changing
    // smooth surfaces, adding samples, or widening the classifier.
    let moving_reconstruction_detail_weight = select(
        0.0,
        0.02 + 0.04 * current_feature_lock,
        reconstruction_geometry_phase,
    );
    blended = max(
        blended + reconstruction_detail *
            ((reconstruction_detail_weight * settled_static +
              moving_reconstruction_detail_weight) * fractional_reconstruction),
        vec3<f32>(0.0),
    );
    let blended_w = mix(history_w, current_w, alpha);
    return TaaOut(
        vec4<f32>(blended, blended_w),
        vec2<f32>(
            current_depth_key,
            pack_temporal_history(next_history_confidence, 0.0, next_feature_lock),
        ),
    );
}
",
);
