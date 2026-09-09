# Issue #132 directional VSM contact-detail qualification v1

This evidence qualifies the opt-in fixture and semantic gate introduced at
`c1131d02dd788982b7580dd0e8e53539130f706d` on an Apple M1 Max using
Metal and Bloom's native high-end profile. This checkpoint changes no
production renderer path or ordinary example behavior.

## Oracle

`quality-motion --vsm-contact-detail` creates a 19 by 14 field of
0.055-metre-wide rigid posts above a neutral ground receiver. Their long,
closely spaced directional shadows become sub-pixel lines with distance. The
fixture uses a fixed camera, light, exposure, native Ultra render scale, and
TAA. It is isolated from the established skinned/alpha fixture:

- the flag alone selects a 1280 by 720 logical viewport;
- the Fox and Sponza curtain are hidden only for this fixture;
- the 266 colored posts cast and receive shadows;
- the ground remains receive-only;
- without the flag, the existing 800 by 450 motion fixture is unchanged.

After 120 warm-up frames the VSM control had all 224 demanded pages resident,
zero dirty/pending pages, 224 cache hits, zero misses/denials/evictions, and no
fallback. This ensures the comparison measures settled page resolution rather
than page arrival.

Reproduction:

```sh
perry compile examples/quality-motion/main.ts -o examples/quality-motion/main

BLOOM_VSM=1 ./examples/quality-motion/main \
  --vsm-contact-detail \
  --quality-run 120 120 0.016666667 \
  /tmp/vsm-detail.png \
  /tmp/vsm-detail.json

./examples/quality-motion/main \
  --vsm-contact-detail \
  --quality-run 120 120 0.016666667 \
  /tmp/csm-detail.png \
  /tmp/csm-detail.json

python3 tools/quality/shadow_detail.py \
  --vsm /tmp/vsm-detail.png \
  --csm /tmp/csm-detail.png \
  --output /tmp/vsm-contact-detail.json
```

## Automated semantic gate

The dependency-free gate decodes RGB/RGBA8 PNGs with the existing governed
PNG decoder. It evaluates a fixed normalized ground region and uses the
intersection of pixels whose RGB chroma is at most seven code values in both
captures. This excludes the colored post geometry, sky, and chromatic TAA
silhouettes while selecting the same 398,641 neutral ground pixels.

For each selected pixel whose four central-difference neighbors are also in
the intersection, the gate measures maximum horizontal/vertical luminance
gradient. It requires:

- at least 5 times as many gradients above 0.10;
- at least 1.35 times the 99th-percentile edge magnitude;
- at least 0.004 more 95th-to-5th-percentile shadow contrast.

Synthetic tests prove a sharp-line candidate passes, the reversed
sharp-versus-blurred control fails, and colored geometry is excluded.

The real result across 349,133 common edge samples was:

| Metric | CSM | VSM | VSM / delta |
| --- | ---: | ---: | ---: |
| Strong edge pixels | 161 | 8,110 | 50.37x |
| Edge p99 | 0.071697 | 0.116095 | 1.619x |
| Shadow contrast | 0.284005 | 0.291565 | +0.007560 |

All five VSM images were byte-identical with SHA-256
`81c6d840a275ee4fe3ff9fbeb87a5a1e21b23bb1c087a4e24c024c2f778b0bde`.
All five CSM images were byte-identical with SHA-256
`a5367b679f8ae11f42b6b28faf750290afecbef39fd0f065296f440da6e2c79c`.

The gate is intentionally narrow: it proves directional contact-detail
retention in this fixed geometry region. It does not claim that a sharper edge
metric alone approves general shadow softness, light types, motion, foliage,
or art direction.

## Measured memory and performance

Five runs per mode were counterbalanced. Each used 120 warm-up and 120
measured frames. Median-of-run metrics were:

| Domain | CSM control | VSM | Delta |
| --- | ---: | ---: | ---: |
| Wall frame mean | 13.791750 ms | 13.418323 ms | -2.71% |
| CPU frame mean | 8.454446 ms | 7.962276 ms | -5.82% |
| CPU frame p50 | 3.758875 ms | 3.635167 ms | -3.29% |
| GPU frame mean | 50.589569 ms | 50.603689 ms | +0.014120 ms / +0.03% |
| GPU frame p50 | 49.070917 ms | 47.649169 ms | -2.90% |
| Render-total CPU | 4.264043 ms | 4.031751 ms | -5.45% |

The stable VSM sampling work is visible rather than hidden:
`main_hdr_pass` GPU increased by 0.231049 ms and settled shadow CPU work
increased by 0.049419 ms. Other pass/scheduling variation offset that cost in
whole-frame medians; the lower totals are not claimed as an optimization.
GPU-frame mean is effectively neutral at +0.03%.

VSM retains live CSM as its guaranteed missing-page fallback. Telemetry
reported 19,951,824 VSM pool/metadata bytes plus 51,200 lazy caster-indirect
bytes, for 20,003,024 measured incremental bytes in this fixture. Small,
disabled, and non-qualifying paths remain lazy as established by the prior
milestone.

Because this checkpoint adds only an opt-in example branch and an offline
qualification tool, it introduces zero production passes, shaders, buffers,
bindings, draws, branches, or normal-frame CPU work.

## Regression gates

- The complete `scripts/ci-check.sh --quick` lane passed.
- Three contact-detail semantic tests and their negative controls passed.
- FFI/schema parity, strict Clippy, formatting, file-size ratchet, Web/WASM,
  347 shared tests, 59 runnable GPU goldens, 4 render-target tests, cooker
  tests, quality governance, and all 20 canonical examples passed.
- The contact images are deterministic within each mode.

Machine-readable evidence is in
`docs/evidence/issue-132-contact-detail-v1.json`.
