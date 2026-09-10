//! World-surface probe history accumulation.

/// Probe temporal accumulator. Directional samples are integrated only within
/// the current frame. Sixteen complete phase estimates are retained after
/// world-surface reprojection and averaged as a finite ring, so changing the
/// angular sampling phase cannot reinterpret old radiance as a different
/// direction. Repeated complete phases establish a stationary estimate without
/// clamping it to a noisy current-frame neighborhood.
pub(in crate::renderer) const SSGI_PROBE_TEMPORAL_WGSL: &str = "
struct TemporalParams {
    // x = phase-ring reciprocal (1/16),
    // y = force_refresh (1 → seed every phase with current),
    // z = grid_w, w = grid_h
    params: vec4<f32>,
    // x = half_w, y = half_h, z = tile_size, w = projection p00
    size: vec4<f32>,
    // x = hardware ray provenance/confidence available,
    // y = current angular phase [0, 15],
    // z = short output-blend current weight, w = camera moving.
    confidence: vec4<f32>,
    // x = world-cache capacity, y = static scene/lighting signature,
    // z = monotonic probe frame, w = cache writes allowed.
    world_cache: vec4<u32>,
};

struct ProbeWorldCacheEntry {
    key: atomic<u32>,
    sequence: atomic<u32>,
    padding: vec2<u32>,
    world_pos: vec4<f32>,
    normal: vec4<f32>,
    diffuse: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: TemporalParams;
@group(0) @binding(1) var radiance_in: texture_3d<f32>;
@group(0) @binding(2) var history_in: texture_3d<f32>;
@group(0) @binding(3) var history_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var<storage, read_write> probes: array<ProbeHeader>;
@group(0) @binding(5) var velocity_tex: texture_2d<f32>;
@group(0) @binding(6) var<storage, read_write> world_cache: array<ProbeWorldCacheEntry>;

fn world_cache_mix(hash_in: u32, value: u32) -> u32 {
    var hash = (hash_in ^ value) * 0x45d9f3bu;
    hash = (hash ^ (hash >> 16u)) * 0x45d9f3bu;
    return hash ^ (hash >> 16u);
}

fn world_cache_key(world_pos: vec3<f32>, normal: vec3<f32>) -> u32 {
    let quantized_pos = vec3<i32>(round(world_pos * 100.0));
    let encoded_normal = vec2<u32>(round(oct_encode(normal) * 255.0));
    var hash = max(u.world_cache.y, 1u);
    hash = world_cache_mix(hash, bitcast<u32>(quantized_pos.x));
    hash = world_cache_mix(hash, bitcast<u32>(quantized_pos.y));
    hash = world_cache_mix(hash, bitcast<u32>(quantized_pos.z));
    hash = world_cache_mix(hash, encoded_normal.x | (encoded_normal.y << 8u));
    return max(hash, 1u);
}

// The trace stores eight equal-weight samples drawn from the receiver's cosine
// hemisphere. Average those current samples and publish the temporally
// filtered diffuse convolution through ProbeHeader.
var<workgroup> diffuse_radiance: array<vec3<f32>, 64>;
var<workgroup> diffuse_luminance: array<f32, 64>;
var<workgroup> confidence_error_samples: array<f32, 64>;
var<workgroup> independent_estimator_error: f32;
var<workgroup> current_integrated_shared: vec3<f32>;
var<workgroup> phase_ring_settled_shared: u32;
var<workgroup> world_cache_hit_shared: u32;
var<workgroup> world_cache_diffuse_shared: vec3<f32>;
var<workgroup> reprojected_history_probe: u32;
var<workgroup> reprojected_history_valid: u32;

@compute @workgroup_size(8, 2, 1)
fn cs_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let grid_w = u32(u.params.z);
    let grid_h = u32(u.params.w);
    if (wg.x >= grid_w || wg.y >= grid_h) { return; }

    let probe_index = wg.y * grid_w + wg.x;
    let coord = vec3<i32>(i32(wg.x), i32(wg.y), i32(lid.y * PROBE_OCT_SIZE + lid.x));
    let lane = lid.y * PROBE_OCT_SIZE + lid.x;
    // Trace owns eight cosine-hemisphere rays. The other eight lanes own the
    // remaining integrated-phase slots; they never read stale trace layers.
    var current_sample = vec4<f32>(0.0);
    if (lane < PROBE_TRACE_RAYS) {
        let raw_current_sample = textureLoad(radiance_in, coord, 0);
        current_sample = vec4<f32>(
            bounded_probe_history(raw_current_sample.rgb),
            max(raw_current_sample.a, 0.0),
        );
    }
    let curr = current_sample.rgb;
    confidence_error_samples[lane] = 0.0;

    if (lane == 0u) {
        reprojected_history_probe = probe_index;
        reprojected_history_valid = 0u;
        let current_world_pos = probes[probe_index].world_pos;
        let current_normal = probes[probe_index].normal;
        if (current_world_pos.w >= 0.5) {
            let current_phase = u32(round(u.confidence.y)) & 15u;
            let placement_motion = select(0.0, 1.0, u.confidence.w > 0.5);
            let current_jitter = probe_lattice_jitter(current_phase) * placement_motion;
            let current_uv = (
                vec2<f32>(wg.xy) * u.size.z +
                u.size.z * (vec2<f32>(0.5) + current_jitter)
            ) / u.size.xy;
            let velocity_size = vec2<i32>(textureDimensions(velocity_tex));
            let velocity_coord = clamp(
                vec2<i32>(current_uv * vec2<f32>(velocity_size)),
                vec2<i32>(0),
                velocity_size - vec2<i32>(1),
            );
            let velocity = textureLoad(velocity_tex, velocity_coord, 0).xy;
            // Velocity stores current-minus-previous NDC. UV's Y axis is
            // flipped, matching the established TAA and SSR reprojection.
            let previous_uv = vec2<f32>(
                current_uv.x - velocity.x,
                current_uv.y + velocity.y,
            );
            if (all(previous_uv >= vec2<f32>(0.0)) &&
                all(previous_uv <= vec2<f32>(1.0))) {
                let previous_phase = (current_phase + 15u) & 15u;
                // A start/stop transition can differ by at most half a tile;
                // the existing 3x3 same-plane search below covers it. During
                // continuous motion this is the exact prior lattice phase.
                let previous_jitter =
                    probe_lattice_jitter(previous_phase) * placement_motion;
                let previous_grid_position =
                    previous_uv * u.size.xy / u.size.z -
                    vec2<f32>(0.5) - previous_jitter;
                let previous_grid_center = vec2<i32>(
                    floor(previous_grid_position + vec2<f32>(0.5)),
                );

                // The nearest prior grid sample can sit half a tile from
                // this surface point. Derive a world-space acceptance
                // radius from that footprint instead of using a fixed
                // tolerance that collapses as resolution/depth changes.
                let probe_world_spacing =
                    2.0 * max(current_normal.w, 0.1) * u.size.z /
                    max(abs(u.size.w) * u.size.x, 0.0001);
                let maximum_world_shift = 0.05 + probe_world_spacing * 0.9;
                // A lateral screen-probe shift may span most of one footprint,
                // but movement through the surface normal must stay within a
                // thin same-plane slab. Otherwise parallel foreground detail
                // can donate its bright history to the wall behind it.
                let maximum_plane_shift = 0.025 + probe_world_spacing * 0.08;
                var best_score = 1e30;
                for (var dy = -1; dy <= 1; dy = dy + 1) {
                    for (var dx = -1; dx <= 1; dx = dx + 1) {
                        let candidate_xy = previous_grid_center + vec2<i32>(dx, dy);
                        if (candidate_xy.x < 0 || candidate_xy.y < 0 ||
                            candidate_xy.x >= i32(grid_w) || candidate_xy.y >= i32(grid_h)) {
                            continue;
                        }
                        let candidate_index =
                            u32(candidate_xy.y) * grid_w + u32(candidate_xy.x);
                        let previous_world_pos = probes[candidate_index].previous_world_pos;
                        let previous_normal = probes[candidate_index].previous_normal;
                        if (!probe_history_geometry_values_valid(
                            current_world_pos,
                            current_normal,
                            previous_world_pos,
                            previous_normal,
                            maximum_world_shift,
                            maximum_plane_shift,
                        )) {
                            continue;
                        }
                        let world_shift = distance(
                            current_world_pos.xyz,
                            previous_world_pos.xyz,
                        );
                        let normal_penalty = 1.0 - clamp(dot(
                            current_normal.xyz,
                            previous_normal.xyz,
                        ), 0.0, 1.0);
                        let score = world_shift + normal_penalty * maximum_world_shift;
                        if (score < best_score) {
                            best_score = score;
                            reprojected_history_probe = candidate_index;
                            reprojected_history_valid = 1u;
                        }
                    }
                }
            }
        }
    }
    workgroupBarrier();

    // Read a fully converged value previously published for this exact world
    // surface. The sequence is sampled on both sides of the payload so a
    // concurrent refresh can never expose a torn record.
    if (u.world_cache.x > 0u) {
        if (lane == 0u) {
            world_cache_hit_shared = 0u;
            world_cache_diffuse_shared = vec3<f32>(0.0);
            let current_world_pos = probes[probe_index].world_pos;
            let current_normal = probes[probe_index].normal;
            if (current_world_pos.w >= 0.5) {
                let key = world_cache_key(current_world_pos.xyz, current_normal.xyz);
                let first_slot = key % u.world_cache.x;
                for (var attempt = 0u; attempt < 8u; attempt = attempt + 1u) {
                    let slot = (first_slot + attempt) % u.world_cache.x;
                    if (atomicLoad(&world_cache[slot].key) != key) { continue; }
                    let sequence_before = atomicLoad(&world_cache[slot].sequence);
                    if (sequence_before == 0u || (sequence_before & 1u) != 0u) { continue; }
                    let cached_world_pos = world_cache[slot].world_pos;
                    let cached_normal = world_cache[slot].normal;
                    let cached_diffuse = world_cache[slot].diffuse.rgb;
                    let sequence_after = atomicLoad(&world_cache[slot].sequence);
                    if (sequence_before == sequence_after &&
                        distance(current_world_pos.xyz, cached_world_pos.xyz) <= 0.035 &&
                        dot(current_normal.xyz, cached_normal.xyz) >= 0.98) {
                        world_cache_hit_shared = 1u;
                        world_cache_diffuse_shared = bounded_probe_history(cached_diffuse);
                        break;
                    }
                }
            }
        }
        workgroupBarrier();
    } else if (lane == 0u) {
        world_cache_hit_shared = 0u;
        world_cache_diffuse_shared = vec3<f32>(0.0);
    }

    // Direction samples change phase every frame. They are current-frame
    // Monte-Carlo samples, not temporal history slots. Accumulating them by
    // octel would assign old radiance to a new direction and turn a sparse
    // bright hit into a persistent projector-shaped strip.
    diffuse_radiance[lane] = curr;
    diffuse_luminance[lane] = select(
        0.0,
        dot(curr, vec3<f32>(0.2126, 0.7152, 0.0722)),
        lane < PROBE_TRACE_RAYS,
    );
    workgroupBarrier();
    if (lane < 4u) {
        diffuse_luminance[lane] =
            diffuse_luminance[lane] + diffuse_luminance[lane + 4u];
    }
    workgroupBarrier();
    if (lane < 2u) {
        diffuse_luminance[lane] =
            diffuse_luminance[lane] + diffuse_luminance[lane + 2u];
    }
    workgroupBarrier();
    if (lane == 0u) {
        diffuse_luminance[0] =
            diffuse_luminance[0] + diffuse_luminance[1];
    }
    workgroupBarrier();

    // One fixed ray that happens to intersect a tiny bright texture or lamp
    // can represent far more solid angle than the source covers. Prevent that
    // quadrature outlier from becoming a whole probe streak. A 5x mean cap
    // preserves broad sky/sun fields and only winsorizes energy too
    // concentrated for this sampling density.
    let ray_luminance = dot(
        diffuse_radiance[lane],
        vec3<f32>(0.2126, 0.7152, 0.0722),
    );
    let mean_luminance = diffuse_luminance[0] / f32(PROBE_TRACE_RAYS);
    let solid_angle_cap = mean_luminance * 5.0;
    let angular_outlier = max(
        ray_luminance - mean_luminance * 2.5,
        0.0,
    ) / (0.05 + ray_luminance);
    if (lane < PROBE_TRACE_RAYS) {
        confidence_error_samples[lane] = max(
            confidence_error_samples[lane],
            angular_outlier * angular_outlier,
        );
        if (ray_luminance > solid_angle_cap && ray_luminance > 0.0) {
            diffuse_radiance[lane] =
                diffuse_radiance[lane] * (solid_angle_cap / ray_luminance);
        }
    }
    workgroupBarrier();
    if (lane == 0u) {
        var sample_sum = vec3<f32>(0.0);
        for (var ray = 0u; ray < PROBE_TRACE_RAYS; ray = ray + 1u) {
            sample_sum = sample_sum + diffuse_radiance[ray];
        }
        let sample_mean = sample_sum / f32(PROBE_TRACE_RAYS);
        var estimator_variance = 0.0;
        for (var ray = 0u; ray < PROBE_TRACE_RAYS; ray = ray + 1u) {
            let delta = diffuse_radiance[ray] - sample_mean;
            estimator_variance = estimator_variance + dot(delta, delta);
        }
        // Relative standard error of the eight-ray mean drives only the
        // geometry-aware reconstruction footprint. Temporal phase accumulation
        // supplies 128 angular samples per receiver over a complete cycle.
        independent_estimator_error =
            sqrt(estimator_variance / 64.0) /
            (0.05 + length(sample_mean));
        current_integrated_shared = bounded_probe_history(sample_mean);
        probes[probe_index].current_diffuse = vec4<f32>(current_integrated_shared, 1.0);
    }
    workgroupBarrier();

    // Layers 32..47 are a ring of complete, equal-weight diffuse estimates,
    // one for each angular phase.
    // Layers 0..7 remain current directional samples for capture-only
    // diagnostics. Reproject the integrated ring as a
    // unit onto the compatible prior world surface; never read a directional
    // lane as history. Invalid history seeds the whole ring with the current
    // estimate, avoiding both stale light and a dark 16-frame warm-up.
    if (lane < 16u) {
        let phase_slot = lane;
        let current_phase = u32(u.confidence.y) & 15u;
        var phase_integrated = current_integrated_shared;
        if (u.confidence.x < 0.5) {
            // Match the retained phases' precision before comparison and
            // reduction. Texture stores may round differently from pack on
            // different adapters; an already representable value is exact.
            phase_integrated = vec3<f32>(
                unpack2x16float(pack2x16float(phase_integrated.xy)),
                unpack2x16float(pack2x16float(vec2<f32>(phase_integrated.z, 0.0))).x,
            );
        }
        // A seeded slot supplies stable first-frame output but is not a real
        // sample of this angular phase. Its owner.w remains zero until that
        // phase is traced at the current surface.
        var phase_owner = vec4<f32>(probes[probe_index].world_pos.xyz, 0.0);
        if (phase_slot == current_phase && probes[probe_index].world_pos.w >= 0.5) {
            if (u.confidence.x > 0.5) {
                // A hardware phase becomes publishable only after the
                // TLAS/card-light field is coherent. Existing pre-coherence
                // slots age out over the following complete 16-phase cycle.
                phase_owner.w = select(0.0, 1.0, u.world_cache.w != 0u);
            } else {
                // A software phase is stationary only after the same surface
                // reproduces its previous complete estimate at rgba16float
                // storage precision. A changed phase immediately revokes that
                // state, retaining the responsive current-neighborhood clamp.
                phase_owner.w = 1.0;
                if (u.params.y <= 0.5 && reprojected_history_valid != 0u) {
                    let history_xy = vec2<i32>(
                        i32(reprojected_history_probe % grid_w),
                        i32(reprojected_history_probe / grid_w),
                    );
                    let old_phase = textureLoad(history_in,
                        vec3<i32>(history_xy, i32(32u + phase_slot)), 0);
                    let old_owner = textureLoad(history_in,
                        vec3<i32>(history_xy, i32(48u + phase_slot)), 0);
                    let same_radiance = all(old_phase.rgb == phase_integrated);
                    if (old_owner.w >= 1.0 && same_radiance &&
                        distance(old_owner.xyz, phase_owner.xyz) <= 0.05) {
                        phase_owner.w = 2.0;
                    }
                }
            }
        }
        if (u.params.y <= 0.5 && reprojected_history_valid != 0u &&
            phase_slot != current_phase) {
            let history_x = i32(reprojected_history_probe % grid_w);
            let history_y = i32(reprojected_history_probe / grid_w);
            let history_layer = i32(32u + phase_slot);
            phase_integrated = bounded_probe_history(textureLoad(
                history_in,
                vec3<i32>(history_x, history_y, history_layer),
                0,
            ).rgb);
            phase_owner = textureLoad(
                history_in,
                vec3<i32>(history_x, history_y, i32(48u + phase_slot)),
                0,
            );
        }
        diffuse_radiance[32u + phase_slot] = phase_integrated;
        // The luminance reduction is dead after current integration. Reuse
        // its existing 64 floats for sixteen vec4 phase owners instead of
        // increasing workgroup memory and reducing occupancy on software GPUs.
        let owner_base = phase_slot * 4u;
        diffuse_luminance[owner_base] = phase_owner.x;
        diffuse_luminance[owner_base + 1u] = phase_owner.y;
        diffuse_luminance[owner_base + 2u] = phase_owner.z;
        diffuse_luminance[owner_base + 3u] = phase_owner.w;
        textureStore(
            history_out,
            vec3<i32>(i32(wg.x), i32(wg.y), i32(32u + phase_slot)),
            vec4<f32>(phase_integrated, 1.0),
        );
    }
    workgroupBarrier();

    if (lane == 0u) {
        phase_ring_settled_shared = 1u;
        let current_world_pos = probes[probe_index].world_pos;
        if (current_world_pos.w < 0.5) {
            phase_ring_settled_shared = 0u;
        } else {
            for (var phase = 0u; phase < 16u; phase = phase + 1u) {
                let owner_base = phase * 4u;
                let owner = vec4<f32>(
                    diffuse_luminance[owner_base],
                    diffuse_luminance[owner_base + 1u],
                    diffuse_luminance[owner_base + 2u],
                    diffuse_luminance[owner_base + 3u],
                );
                // Owner positions are stored in rgba16float, whose absolute
                // precision is about 1.6 cm around a 20 m Bistro coordinate.
                let required_owner = select(1.5, 0.5, u.confidence.x > 0.5);
                if (owner.w < required_owner ||
                    distance(owner.xyz, current_world_pos.xyz) > 0.05) {
                    phase_ring_settled_shared = 0u;
                }
            }
        }
    }
    workgroupBarrier();

    if (lane == 0u) {
        var phase_sum = vec3<f32>(0.0);
        for (var phase = 0u; phase < 16u; phase = phase + 1u) {
            phase_sum = phase_sum + diffuse_radiance[32u + phase];
        }
        let phase_integrated = bounded_probe_history(phase_sum * u.params.x);
        var integrated = phase_integrated;
        probes[probe_index].current_diffuse.w = f32(phase_ring_settled_shared);
        if (u.confidence.x < 0.5 && u.params.y <= 0.5 &&
            reprojected_history_valid != 0u && phase_ring_settled_shared == 0u) {
            // Screen/SDF traces can still change at Hi-Z silhouettes even
            // after their angular ring is complete. Preserve the established
            // short output blend on those approximate backends. Hardware
            // queries are deterministic for a fixed world receiver and use
            // the phase-stationary ring directly.
            let previous_integrated = bounded_probe_history(
                probes[reprojected_history_probe].previous_diffuse.rgb,
            );
            integrated = mix(
                previous_integrated,
                phase_integrated,
                u.confidence.z,
            );
        }
        // A returning surface can recover its fully converged value while the
        // screen-owned angular ring refills. Never publish transient path
        // samples back to the cache: all sixteen phase owners must first agree
        // with this receiver.
        if (phase_ring_settled_shared == 0u && world_cache_hit_shared != 0u) {
            integrated = world_cache_diffuse_shared;
        }
        // Confidence is populated below on the hardware path. Zero is the
        // well-converged default for Hi-Z/SDF; one would incorrectly request
        // the widest reconstruction footprint every frame.
        probes[probe_index].diffuse = vec4<f32>(integrated, 0.0);

        if (phase_ring_settled_shared != 0u && u.world_cache.w != 0u &&
            u.world_cache.x > 0u) {
            let current_world_pos = probes[probe_index].world_pos;
            let current_normal = probes[probe_index].normal;
            let key = world_cache_key(current_world_pos.xyz, current_normal.xyz);
            let first_slot = key % u.world_cache.x;
            var destination = u.world_cache.x;
            for (var attempt = 0u; attempt < 8u; attempt = attempt + 1u) {
                let slot = (first_slot + attempt) % u.world_cache.x;
                let existing_key = atomicLoad(&world_cache[slot].key);
                if (existing_key == key) {
                    // The completed phase ring is stationary within this
                    // scene/light signature. Keep the first coherent value
                    // immutable so cache hits never race a redundant refresh.
                    destination = u.world_cache.x;
                    break;
                }
                if (existing_key == 0u) {
                    let claim = atomicCompareExchangeWeak(
                        &world_cache[slot].key,
                        0u,
                        key,
                    );
                    if (claim.exchanged || claim.old_value == key) {
                        destination = slot;
                        break;
                    }
                }
            }
            if (destination < u.world_cache.x) {
                let sequence_odd = u.world_cache.z * 2u + 1u;
                let sequence_even = sequence_odd + 1u;
                let old_sequence = atomicLoad(&world_cache[destination].sequence);
                if ((old_sequence & 1u) == 0u) {
                    let lock = atomicCompareExchangeWeak(
                        &world_cache[destination].sequence,
                        old_sequence,
                        sequence_odd,
                    );
                    if (lock.exchanged) {
                        world_cache[destination].world_pos = current_world_pos;
                        world_cache[destination].normal = current_normal;
                        world_cache[destination].diffuse = vec4<f32>(
                            phase_integrated,
                            1.0,
                        );
                        atomicStore(
                            &world_cache[destination].sequence,
                            sequence_even,
                        );
                    }
                }
            }
        }
    }

    workgroupBarrier();
    if (lane < 4u) {
        confidence_error_samples[lane] = confidence_error_samples[lane] + confidence_error_samples[lane + 4u];
    }
    workgroupBarrier();
    if (lane < 2u) {
        confidence_error_samples[lane] = confidence_error_samples[lane] + confidence_error_samples[lane + 2u];
    }
    workgroupBarrier();
    if (lane == 0u && u.confidence.x > 0.5 &&
        probes[probe_index].world_pos.w >= 0.5) {
        let rms_disagreement = sqrt(
            (confidence_error_samples[0] + confidence_error_samples[1]) /
            f32(PROBE_TRACE_RAYS),
        );
        let confidence_error = max(
            rms_disagreement,
            independent_estimator_error,
        );
        // `diffuse.w` is not sampled as radiance. Preserve the uncertainty for
        // the bounded spatial footprint and capture-only diagnostics.
        probes[probe_index].diffuse.w = confidence_error;
    }

    // Preserve current samples for capture-only diagnostics. They are never
    // interpreted as matching temporal directions by the production path.
    // Layers 32..47 were written above by the integrated phase ring.
    if (lane < PROBE_TRACE_RAYS) {
        textureStore(history_out, coord, current_sample);
    }
    if (lane < 16u) {
        let owner_base = lane * 4u;
        textureStore(
            history_out,
            vec3<i32>(i32(wg.x), i32(wg.y), i32(48u + lane)),
            vec4<f32>(
            diffuse_luminance[owner_base],
            diffuse_luminance[owner_base + 1u],
            diffuse_luminance[owner_base + 2u],
            diffuse_luminance[owner_base + 3u],
            ),
        );
    }
}

// Filter the completed irradiance estimate in probe space, after every probe
// has published its current result. Doing this in a second tiny dispatch avoids
// cross-workgroup races and costs only one invocation per 8x8 half-resolution
// tile. The geometry-clamped reconstruction occupies layer zero of history_out.
// ProbeHeader.previous_diffuse retains unfiltered temporal RGB plus a scalar
// reconstruction energy ratio for fallback resolve. This dispatch reads the
// prior diffuse/current values, not that destination field, so the write has no
// cross-workgroup race and spatial filtering never feeds back into history.
@compute @workgroup_size(8, 8, 1)
fn cs_spatial(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_w = u32(u.params.z);
    let grid_h = u32(u.params.w);
    if (gid.x >= grid_w || gid.y >= grid_h) { return; }

    let center_index = gid.y * grid_w + gid.x;
    let center = probes[center_index];
    let output_coord = vec3<i32>(i32(gid.x), i32(gid.y), 0);
    if (center.world_pos.w < 0.5) {
        probes[center_index].previous_diffuse = vec4<f32>(0.0);
        textureStore(history_out, output_coord, vec4<f32>(0.0));
        return;
    }

    let confidence_error = max(center.diffuse.w, 0.0);
    // Hardware's complete 128-direction phase ring is intrinsically
    // low-frequency. Let the full geometry-clamped 5x5 footprint own it;
    // retaining a portion of the screen-tiled centre leaves a camera-fixed
    // lattice. Approximate Hi-Z/SDF histories remain variance-adaptive: their
    // current-neighborhood clamp changes every angular phase, so forcing the
    // widest filter there amplifies rather than suppresses settled variation.
    // This is still a tiny probe-domain dispatch (one invocation per 8x8
    // half-res tile), and the normal/plane tests below keep every contribution
    // on the same surface and preserve its boundaries.
    let hardware_history = u.confidence.x > 0.5;
    let radius = select(
        select(1, 2, confidence_error > 0.08),
        2,
        hardware_history,
    );
    let filter_strength = select(
        clamp(0.30 + confidence_error * 4.0, 0.30, 1.0),
        1.0,
        hardware_history,
    );
    let probe_world_spacing =
        2.0 * max(center.normal.w, 0.1) * u.size.z /
        max(abs(u.size.w) * u.size.x, 0.0001);
    let plane_sigma = 0.02 + probe_world_spacing * 0.16;

    var accum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    var current_first = vec3<f32>(0.0);
    var current_second = vec3<f32>(0.0);
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            if (abs(dx) > radius || abs(dy) > radius) { continue; }
            let sample_xy = vec2<i32>(gid.xy) + vec2<i32>(dx, dy);
            if (sample_xy.x < 0 || sample_xy.y < 0 ||
                sample_xy.x >= i32(grid_w) || sample_xy.y >= i32(grid_h)) {
                continue;
            }
            let sample_index = u32(sample_xy.y) * grid_w + u32(sample_xy.x);
            let sample = probes[sample_index];
            if (sample.world_pos.w < 0.5) { continue; }

            let normal_similarity = clamp(
                dot(center.normal.xyz, sample.normal.xyz),
                0.0,
                1.0,
            );
            if (normal_similarity < 0.85) { continue; }
            let world_delta = sample.world_pos.xyz - center.world_pos.xyz;
            let plane_error = max(
                abs(dot(world_delta, center.normal.xyz)),
                abs(dot(world_delta, sample.normal.xyz)),
            );
            if (plane_error > plane_sigma * 2.5) { continue; }

            let offset2 = f32(dx * dx + dy * dy);
            let spatial_weight = exp(-0.5 * offset2 / 1.96);
            let plane_weight = exp(
                -0.5 * plane_error * plane_error /
                max(plane_sigma * plane_sigma, 0.000001),
            );
            let weight = spatial_weight * plane_weight * pow(normal_similarity, 12.0);
            // Spatially reconstruct the world-reprojected integral, not the
            // raw 32-ray sample set. Filtering current samples here discarded
            // most of the temporal estimator every frame, so camera motion
            // exposed a fresh screen-tile pattern even when history had found
            // the same wall. The final neighborhood clamp below still bounds
            // every retained value by current-frame evidence.
            accum = accum + sample.diffuse.rgb * weight;
            weight_sum = weight_sum + weight;
            let current_neighbor = bounded_probe_history(sample.current_diffuse.rgb);
            current_first = current_first + current_neighbor * weight;
            current_second = current_second + current_neighbor * current_neighbor * weight;
        }
    }

    // Approximate Hi-Z/SDF histories still need a current-frame safety bound.
    // Bound them by the current neighborhood's mean and spread rather than its
    // hard min/max. A sparse estimate of a bright
    // nearby source (the sun-lit Bistro awnings under their façade) is
    // binomially noisy per probe, so a min/max clamp repeatedly crushed the
    // converged EMA toward whichever realization the current frame produced —
    // the red bounce pumped in and out as the camera moved. With variance
    // bounds, a noisy-but-consistent neighborhood keeps its converged mean,
    // while genuinely stale history (disocclusion ghosts, lighting changes)
    // still gets pulled to current evidence because agreement between
    // neighbors shrinks the spread toward zero. Hardware ray queries already
    // maintain a geometry-reprojected finite phase ring; clamping that stable
    // 128-direction estimate to the fresh eight-ray screen cells nearly
    // doubles its measured motion error and reintroduces the probe lattice.
    // The ring seeds current on disocclusion and refreshes one complete phase
    // per frame, so the hardware path must not apply this second estimator.
    // Keep the software-path floor relative to
    // HDR signal scale: the former absolute 0.005 allowance exceeded the
    // complete indirect signal on many Bistro facade probes and therefore
    // admitted old path-dependent light without clipping it at all.
    var history_clamped = center.diffuse.rgb;
    if (u.confidence.x < 0.5 && center.current_diffuse.w < 0.5 && weight_sum > 0.0001) {
        let current_mean = current_first / weight_sum;
        let current_sigma = sqrt(max(
            current_second / weight_sum - current_mean * current_mean,
            vec3<f32>(0.0),
        ));
        let slack = current_sigma
            + abs(current_mean) * 0.02
            + vec3<f32>(0.0001);
        history_clamped = clamp(
            center.diffuse.rgb,
            current_mean - slack,
            current_mean + slack,
        );
    }
    let spatial = select(
        history_clamped,
        accum / max(weight_sum, 0.0001),
        weight_sum > 0.0001,
    );
    let reconstructed = bounded_probe_history(mix(
        history_clamped,
        spatial,
        filter_strength,
    ));
    // This is the authoritative state consumed next frame. Previously only
    // `reconstructed` was clamped while `ProbeHeader.diffuse` kept the raw EMA;
    // a bright card hit could therefore remain hidden for seconds and become
    // visible again when a later current-frame bound happened to include it.
    let luminance_weights = vec3<f32>(0.2126, 0.7152, 0.0722);
    let history_luminance = dot(history_clamped, luminance_weights);
    let reconstructed_luminance = dot(reconstructed, luminance_weights);
    let reconstructed_energy_ratio = select(
        1.0,
        clamp(reconstructed_luminance / history_luminance, 0.0, 4.0),
        history_luminance > 0.000001,
    );
    probes[center_index].previous_diffuse =
        vec4<f32>(history_clamped, reconstructed_energy_ratio);
    textureStore(history_out, output_coord, vec4<f32>(reconstructed, 1.0));
}
";
