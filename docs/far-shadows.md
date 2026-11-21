# Static far sun shadows

`setFarShadow(centerX, centerZ, halfX, halfZ, yMin, yMax, refreshFrames)` gives a large, mostly static world sun shadows
to the horizon (a city seen from a hill or from the air), which stretching the camera cascades over kilometres cannot do
cheaply: they blur, and they redraw the world whenever the camera or any caster moves.

With it on, the third cascade stops following the camera. It is one fixed orthographic map over the given rectangle,
drawn from the static casters only, redrawn at most once every `refreshFrames` frames when the sun, the box or the static
casters changed; between redraws receivers sample it through the projection it was drawn with. The first two cascades
split `setShadowDistance` between them, and moving casters keep their shadows there. The receiver shaders already take
cascade 2 beyond the second split, so no shader changed; the third split carries the far map's fit radius, scaled by
`FAR_RECEIVER_OFFSET` (half the usual normal offset: a far texel is metres wide, and the usual 1.5 texels lifted the ground
under a house above its roof).

The far map has its own per-node uniform buffer, so it is not limited to a cascade's 1,792 static slots.

Measured on the Nineveh city (`city-view --farshadow 1`, about 2,000 to 6,000 casters in the map, M1 Max at 1600 x 900, a
loaded machine): steady state within noise of off (main pass 1.96 ms against 1.94 to 1.99); one redraw about 8 ms of GPU,
so at 30 frames it averages to nothing measurable in a fly-through, and redrawing every frame costs that 8 ms each frame.
Sharpness is the cascade map size over the rectangle as the sun sees it (`setShadowMapSize`).

Code: `native/shared/src/far_shadow.rs`, the cascade split in `shadows.rs::compute_cascade_vps`, the redraw gate and the
uniform buffer in `renderer/shadow_pass.rs`.
