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
- resize, SSR toggles, SSR-strength changes, and PT-mode transitions invalidate
  it;
- frames owned by PT neither write nor advance SSR history;
- the existing two HDR history images, shader, pass count, and steady-state
  blend are unchanged.

Telemetry exposes `temporal_history.ssr_valid` and `ssr_index`. Unit and
headless-GPU tests pin initialization and every transition above.

## SSGI implementation

SSGI now owns `probe_history_valid` independently of TAA:

- invalid history uses the existing force-refresh route (`alpha = 1.0`);
- valid history keeps the existing variance-adaptive four-frame EMA;
- resize, SSGI toggles, intensity/radius changes, transparent-GI route changes,
  and PT ownership invalidate it;
- suppressed frames neither preserve validity nor advance the probe ping-pong;
- the existing two 3D history images, shader, pass count, and steady-state
  filter are unchanged.

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

## Remaining #135 work

The next slice should add the common camera-cut/FOV-change reset API, then
continue with per-pixel rejection diagnostics and the sequence-based motion
corpus.
