# Temporal history contract

This document is the authoritative convention for renderer motion and history
work. It starts the common contract required by issue #135; individual effects
may use different storage or filters, but they must not reinterpret motion or
history lifetime.

## Motion vectors

- The velocity target stores current minus previous position in UV-sized units:
  `(current_ndc.xy - previous_ndc.xy) * 0.5`.
- NDC Y points up while texture UV Y points down. A consumer therefore
  reconstructs the previous texture coordinate as
  `vec2(current_uv.x - velocity.x, current_uv.y + velocity.y)`.
- The previous projection used for velocity carries the current frame's jitter
  on top of the previous unjittered projection. Static geometry consequently
  writes exact zero velocity instead of a jitter-phase delta.
- Rigid, skinned, and procedurally displaced draws must evaluate their previous
  position with the matching previous transform/deformation state. A deliberate
  zero velocity means the consumer must use camera/depth reprojection or reject
  history; it must not be treated as proof that the surface was stationary.
- Sky reprojection uses a direction (`w = 0`), not a finite reconstructed
  position.

## History validity

A temporal history has a separate validity state. A global frame counter is not
a substitute: effects can be enabled, disabled, resized, or suspended at
different times.

Invalid history must be replaced by the current filtered sample before ordinary
temporal blending begins. A ping-pong index advances only after its producing
pass wrote a valid current history.

History becomes invalid when any input changes in a way its rejection model
cannot represent, including:

- target creation or resize/render-scale change;
- effect enable/disable;
- switching between raster ownership and path-traced frame ownership;
- a parameter change that changes the history's radiance domain;
- an explicit camera cut/teleport reset.

## SSR implementation

SSR now owns `ssr_history_valid` independently of TAA:

- invalid history uses temporal alpha `1.0`, replacing stale or zero storage;
- valid history uses alpha `0.1`;
- the established roughness fade keeps wide-lobe surfaces on prefiltered IBL
  while preserving SSR ownership for valid smooth reflections;
- march thickness is evaluated against the same explicit-LOD-0 depth sample
  used to declare the hit, avoiding a coarse-Hi-Z footprint mismatch;
- accepted hit radiance is unchanged through 8 linear luminance units and
  bounded there before it can poison temporal history;
- resize, SSR toggles, SSR-strength changes, and PT-mode transitions invalidate
  it;
- frames owned by PT neither write nor advance SSR history;
- the existing two HDR history images, shader, pass count, and steady-state
  blend are unchanged.

Telemetry exposes `temporal_history.ssr_valid` and `ssr_index`. Unit and
headless-GPU tests pin initialization and every transition above.

The hit bound adds no texture fetch, allocation, pass, draw, or bind group.
It is a luminance dot/select/scale only on accepted hits. A more aggressive
rough-dielectric cutoff was qualification-tested and rejected because it
changed an existing transparent-GI receiver's material response; the
established IBL/SSR ownership curve remains matched on both sides.

## SSGI implementation

SSGI now owns `probe_history_valid` independently of TAA:

- invalid history uses the existing force-refresh route (`alpha = 1.0`);
- force refresh assigns current radiance directly, avoiding the undefined
  `invalid_history * 0` side of a nominal `mix(..., alpha = 1)`;
- valid history keeps the existing variance-adaptive four-frame EMA;
- non-finite half-float history, trace input, and trace output are replaced
  component-wise with zero; ordinary finite radiance is unchanged;
- degenerate depth-derived normals use finite fallback directions during probe
  placement and resolve, preventing NaN normals from rejecting every probe;
- resize, SSGI toggles, intensity/radius changes, transparent-GI route changes,
  and PT ownership invalidate it;
- suppressed frames neither preserve validity nor advance the probe ping-pong;
- the existing two 3D history images, production pass count, and steady-state
  filter remain unchanged.

Telemetry exposes `temporal_history.ssgi_probe_valid` and `ssgi_probe_index`.

## TAA/TSR implementation

TAA/TSR now separates history validity from its Halton jitter counter:

- invalid history keeps the established current-frame weight `1.0`;
- valid history keeps the existing four-frame warmup and render-scale-aware
  steady blend;
- resize, render-scale changes, and TAA toggles invalidate and reset storage;
- transitions between raster and path-traced scene color invalidate history,
  including progressive PT changing ownership without a mode change;
- the ping-pong advances only after the TAA pass wrote its current target.

Telemetry exposes `temporal_history.taa_valid`, `taa_index`, and
`taa_pt_owned`.

## Auto-exposure implementation

Auto-exposure now tracks whether its 1×1 history was written in the current
enable epoch:

- the first valid frame uses the reserved negative-rate seed signal, replacing
  history from the current histogram instead of a stale disabled-era value;
- later frames keep the authored adaptation rate exactly;
- either toggle invalidates and resets the ping-pong;
- the ping-pong advances only after the exposure pass writes, and a skipped
  producer falls back to the last valid slot (or manual exposure if none).

Telemetry exposes `temporal_history.exposure_valid` and `exposure_index`.

## Path-tracing implementation

PT history validity remains the path tracer's own sample count, which the
kernel already receives as `size.w`:

- every off/progressive/realtime mode transition resets the sample count,
  ping-pong index, deterministic sequence, and ownership;
- a zero sample count suppresses reprojection and now also suppresses
  disocclusion neighbor seeding, so retained buffer bytes cannot resurrect
  history after a toggle, camera cut, seed change, or diagnostic reset;
- buffers are retained when compatible and recreated only when trace-grid size
  changes, preserving the established steady-state memory and pass cost.

Telemetry exposes `temporal_history.pt_samples` and `pt_index`.

## Camera cuts and discontinuities

Games call `bloom_reset_temporal_history()` before the next 3D camera after a
camera cut, teleport, discontinuous FOV/projection change, or world load. The
same operation is available as `Renderer::reset_temporal_history()`.

The reset invalidates TAA/TSR, SSAO, SSR, SSGI, PT, and auto-exposure without
reallocating their storage. When the next camera begins, it is pinned as its
own previous camera; material, skin-palette, and retained-scene transform
history are also pinned for that frame. This prevents a cut from creating
full-screen velocity or motion-blur streaks while every temporal filter seeds
from current data.

Telemetry exposes `camera_cut_pending`, `camera_cut_active`, `ssao_frames`, and
`ssao_index` under `temporal_history`.

## Cached-model motion

Ordinary cached-model draws now pair each stable submission slot with its
previous model transform and compose `prev_mvp` from the common
jitter-cancelled previous camera. The model handle is part of the pairing, so
a different model entering a reused slot seeds from its current transform
instead of inheriting a departed object's velocity. Explicit camera cuts clear
the pairing before the next draw. Skinned cached models continue to use their
keyed current/previous joint palettes because those matrices already contain
world locomotion.

This reuses the existing per-draw uniform and motion target: GPU memory, render
passes, draws, bind groups, readbacks, and shader branches added are all zero.
CPU storage is two grow-on-demand entries of one `u64` handle plus one 4x4
matrix per live cached instance; its allocated capacity is reported as
`cached_model_motion_cpu_capacity_bytes`. The current entry count, zero GPU
bytes, and zero added passes are reported alongside it.

## Per-pixel TAA/TSR diagnostics

`captureDebugIntermediates(directory)` now adds four surface-resolution PNGs
without changing the production TAA shader or its output:

- `taa-rejection-reason.png`: gray = invalid-history seed, red = reprojected
  UV outside the prior frame, cyan = reactive coverage, magenta =
  disocclusion, yellow = neighborhood clamp, blue = motion weighting, and
  green = accepted history;
- `taa-motion.png`: red/green encode signed velocity around 0.5 and blue
  encodes magnitude;
- `taa-reprojected-uv.png`: red/green encode the previous-frame UV and blue is
  one only when that coordinate is valid;
- `taa-temporal-confidence.png`: red = local luma variance, green = history
  clamp magnitude, and blue = the retained-history contribution.

The maps are produced by one separate four-target pass only for a native debug
capture, after the measured quality window. Normal frames execute no extra
GPU pass, bind group, shader branch, or texture allocation; the CPU only checks
the existing pending-capture flag. The four RGBA8 targets cost exactly
`surface_width * surface_height * 16` bytes during capture; readback cost is
four 256-byte-row-aligned RGBA8 buffers. Both are released after PNG encoding,
so persistent diagnostic memory is zero. These values and the one-pass count
are reported under `temporal_history.diagnostic_*`.

Native quality captures also include `ssr-raw.png` and `ssr.png` for the
quarter-resolution stochastic march and filtered history. Their accompanying
`*.metrics.json` files retain raw HDR evidence: finite/non-finite counts,
mean/max/p99/p99.9 luminance, alpha hit coverage, and isolated local outliers.
An isolated outlier is over 4 linear luminance and over four times every 3x3
neighbor. These diagnostics reuse the two production SSR targets and add no
render or diagnostic pass by themselves.

When SSR is active, the same request also emits
`ssr-rejection-reason.png` and `ssr-temporal-confidence.png` from a
capture-only pass that reuses the production temporal bind group. The reason
palette shares TAA's gray seed, red off-screen, magenta invalid-history, yellow
neighborhood-clamp, and green accepted-history meanings. Confidence RGB shares
the local-variation, clamp-magnitude, and retained-history convention. The two
half-resolution RGBA8 targets and one pass exist only for the capture and are
released after readback; normal frames and production SSR output are
unchanged. Telemetry includes their exact capture texture/readback byte counts,
one diagnostic pass, and zero persistent bytes for the complete SSR capture.

When SSGI is active, a capture also emits `ssgi-rejection-reason.png` and
`ssgi-temporal-confidence.png`. SSGI history lives in a 3D
probe-X/probe-Y/octahedral-texel domain, so the capture-only compute pass
flattens each probe's 8x8 octahedral slab into a 2D atlas. Gray marks invalid
or freshly seeded probes, magenta marks invalid or adaptively refreshed
radiance, and green marks retained history. Confidence RGB contains local
radiance variation, current-radiance strength, and retained-history
contribution. Screen-space categories such as off-screen reprojection and
motion weighting are intentionally absent because they do not exist in this
representation.

The two RGBA8 atlases and their readback buffers exist only for the native
capture and are released after encoding. Normal frames add no diagnostic pass,
texture, bind group, or persistent allocation. Telemetry reports
`ssgi_diagnostic_persistent_bytes = 0`, exact capture texture/readback bytes,
one capture-only pass, and whether the temporary resources are live.

Native captures now include the resolved half-resolution `ssgi.png` plus
`ssgi.metrics.json`. The raw HDR metrics gate finite output and nonzero
indirect radiance after the SDF/card backend has warmed up, independently of
the display tonemap. Marking the existing SSGI target as `COPY_SRC` adds no
normal-frame pass or allocation.

Realtime path tracing emits `pt-rejection-reason.png`, `pt-motion.png`,
`pt-reprojected-uv.png`, and `pt-temporal-confidence.png` at its trace-grid
resolution. One capture-only compute pass reads the current and previous SVGF
moment buffers, current irradiance/variance, depth, velocity, and the existing
PT matrices. It reproduces the denoiser's bilinear depth validation and
footprint-retention decision without writing production history.

The PT reason palette uses gray for seed/sky, red for off-screen reprojection,
magenta for invalid or depth-rejected history, cyan for a retained
subpixel-surface flip, blue for accepted motion-vector reprojection, and green
for accepted matrix/static history. Motion and reprojected UV follow the shared
encoding. Confidence RGB stores variance heat, normalized 32-frame history
length, and retained-history contribution. The four RGBA8 textures, bind
group, pipeline, and readbacks exist only for a requested native capture and
are released after encoding. `pt_diagnostic_*` telemetry reports zero
persistent bytes, exact temporary bytes, one capture pass, and resource
lifetime.

## Temporal sequence gates

The headless GPU corpus now evaluates a sequence rather than only a final
still. It isolates TAA/TSR from unrelated temporal effects and covers:

- a camera cut combined with a discontinuous FOV change;
- a 1.2-radian fast rotation followed by 24 stationary recovery frames;
- an eight-frame subpixel pan;
- retained opaque rigid motion;
- retained transparent motion through reactive TAA coverage;
- cached skinned locomotion plus joint-palette deformation;
- cached alpha-tested card translation and rotation;
- emissive geometry plus a local light switching both on and off;
- retained opaque physical-transmission translation and rotation;
- a dark interior with an extreme bright opening and a smooth-reflection
  negative control;
- a `1.0 -> 0.5` render-scale step;
- a `320x192 -> 256x256` target resize.

The cut frame must be byte-identical to a fresh-history frame at the same
camera. Fast-rotation recovery is compared with the mean of a settled
16-frame Halton cycle: mean RGB error must contract within four frames,
`>32/255` coherent outliers must cover at most 2% then, and severe
`>64/255` trails must remain below 0.5% after at most four frames. The settled
cycle's mean variation is bounded to 2 RGB levels. Slow-pan pairwise variation
is bounded to 4 RGB levels with at most 3% coherent outliers.

The rigid, reactive, skinned, alpha-tested, and emissive content sequences use
the same recovery gate and include a negative control requiring visibly
different stable states. A motion trail may cover at most 2% of pixels after
four frames, severe `>64/255` residue must settle below 0.5% within four
frames, and the settled jitter cycle must vary by no more than 2 RGB levels.
The skinned sequence uses the production cached-model draw, keyed previous
palette, world locomotion, and two-joint deformation paths. The foliage card
uses the production cached alpha-mask/coverage-mip route and additionally
requires at least 250 pixels of captured nonzero object velocity, preventing
depth rejection from hiding a broken motion-vector path. The emissive sequence
checks both dark-to-bright and bright-to-dark radiance convergence on
stationary geometry.

The physical-refraction sequence requires at least 250 pixels of both captured
nonzero velocity and reactive rejection coverage. Because fully reactive
transmission deliberately consumes the current refracted sample rather than
accumulating it, recovery is compared against the same Halton phase one
16-sample cycle later. Severe residue must remain below 0.5% within four frames,
coherent outliers may cover at most 2% after four frames, and the settled cycle
must still vary by no more than 2 RGB levels. This distinguishes real history
lag from the intended subpixel sample pattern.

The dark-interior SSR gate requires finite raw march/history values, no
isolated HDR fireflies under the documented local rule, a populated reflection
buffer, and a visible SSR-on/off delta from the smooth-reflection control.

The SSGI gate warms a retained emissive receiver scene through the production
Hi-Z-to-SDF backend transition, then requires a finite, nonzero resolved HDR
target and real retained probe history. It also verifies the probe-domain
reason/confidence dimensions and the zero-persistent-memory capture contract.

The realtime PT gate orbits the deterministic retained ray-query scene for 24
frames, then requires finite nonzero HDR output, accepted history, valid
reprojection, and accumulated SVGF confidence. It verifies all four diagnostic
dimensions and the zero-persistent-memory capture contract in one frame.

A second realtime PT sequence moves retained rigid geometry between two
visibly distinct poses after 24 settled frames. The transition must write at
least 100 motion texels, retain overlapping history, and classify at least 90%
of moving texels as retained, depth-rejected, or footprint-retained. Severe
trails must settle within four frames, coherent frame-four outliers must stay
below 2%, and stable stochastic flicker must remain below 2 RGB levels.

Render-scale changes must produce a first frame byte-identical to a freshly
seeded history at the new scale. Resize changes must have no `>32/255`
outliers against a fresh target-size seed, with mean RGB error at most 0.5 and
maximum channel error at most 32.

Each run reports temporal SSIM, mean variation, coherent and severe outlier
fractions, ghost-trail duration, and the diagnostic rejection-reason ratio.
These are relative sequence gates, so they do not require a GPU-family-specific
still-image baseline. The capture-only diagnostic test also requires real
off-screen, reactive, and neighborhood-clamp classifications from the
production TAA inputs, preventing an unexercised diagnostic shader from
silently passing.

## Remaining #135 work

Complete the PT-specific lighting-change and reset sequence coverage on
required hardware runners, including the goldens owned by #127 and the
timing/memory qualification owned by #128.
