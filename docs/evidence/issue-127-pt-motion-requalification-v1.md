# Issue #127 path-tracing motion requalification v1

Bloom renderer revision `09ad0b755af9f10083712327d7f0edb1d88f228b` passes
the complete Metal path-tracing oracle and the focused realtime temporal
corpus after the SSGI resolve pass-tail checkpoint. No path-tracing golden was
updated.

## Canonical hardware oracle

The exact ignored hardware gate ran on Apple M1 Max / Metal with ray query
required, so a structured skip could not masquerade as a pass:

```sh
BLOOM_REQUIRE_RAY_QUERY=1 cargo test --release \
  --manifest-path native/shared/Cargo.toml \
  --test golden_render qualify_pt_oracle_hardware \
  -- --ignored --exact --nocapture
```

The gate keeps one device alive for three progressive runs, three
realtime-camera-motion runs, and both negative controls.

| Mode | Repeats | Mean RGBA | Outlier pixels | Max error | SSIM | Render times |
|---|---:|---:|---:|---:|---:|---:|
| Progressive, 300 frames | 3 | 0.101040 | 0.038147% | 69 | 0.998132777 | 699, 582, 629 ms |
| Realtime motion, 48 frames | 3 | 0.040993 | 0.012207% | 48 | 0.999404554 | 126, 137, 111 ms |

Every metric is identical across all three repeats in each mode. Saved normal
diagnostics cover accumulated/denoised output, raw radiance, depth, normal,
albedo, sun visibility, motion, history length, and variance. Inspection shows
coherent geometry and illumination without the former black workgroup regions
or block trails.

The seeded controls both failed for the intended reason:

- BRDF-energy fault: mean difference 5.299 exceeded the 4.0 limit;
- reprojection fault: mean difference 6.441 exceeded 6.0 and 6.0760% coherent
  outlier pixels exceeded the 1.0% limit.

The negative controls reached SSIM 0.995500 and 0.910582 respectively, proving
that the accepted tolerance still rejects both transport and temporal-history
regressions.

## Focused realtime temporal corpus

Four additional supported-adapter tests ran serially with captures enabled:

```sh
BLOOM_REQUIRE_RAY_QUERY=1 BLOOM_KEEP_TEMPORAL_DIAGNOSTICS=1 \
  cargo test --release --manifest-path native/shared/Cargo.toml \
  --test golden_render realtime_path_tracing \
  -- --nocapture --test-threads=1
```

- Simultaneous SVGF capture: 10,984 accepted-history texels, 10,989 valid
  reprojections, 10,989 accumulated texels, zero non-finite HDR pixels, and
  maximum luminance 0.3492 over 16,384 pixels.
- Rigid motion: visible change mean 5.2560, zero severe-trail frames, zero
  frame-four coherent outliers, stable flicker 0.1426, and 855 moving texels.
  Of those, 45 retained valid reprojected history and 810 explicitly rejected
  it; no moving texel was unclassified.
- Reset and PT off/on: both reproduce the fresh warmed seed byte-for-byte
  (`cut_max=0`, `toggle_max=0`) with zero non-seed history texels.
- Lighting on/off: both converge without reset or lag; frame-12 coherent
  outliers are zero in both directions.

The focused result is four passed, zero failed. Together with the canonical
oracle, this covers camera motion, retained rigid-object motion, disocclusion,
reset ownership, dynamic lighting, temporal capture, and negative controls on
the current Metal renderer.

## Remaining acceptance item

Issue #127 remains open because its contract also requires one DX12 or Vulkan
ray-query hardware result. GitHub-hosted runners expose no suitable GPU and the
repository currently has no self-hosted runner. Static shader validation and
structured skips are not substituted for that hardware gate.
