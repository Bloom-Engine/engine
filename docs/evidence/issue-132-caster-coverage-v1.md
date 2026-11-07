# Issue #132 VSM alpha-tested and skinned caster qualification v1

This evidence qualifies the automated caster oracle implemented at
`862981fd378e03f000109794091382d3d743bc14` on an Apple M1 Max using Metal
and Bloom's native high-end profile. It covers the alpha-tested and skinned
caster acceptance criterion; it does not claim completion of the remaining
multi-light, motion-corpus, or debug-view criteria.

## Oracle and negative controls

The established `quality-motion --vsm-dynamic` fixture contains both caster
classes named by issue #132:

- Sponza's MASK curtain is a visible alpha-tested caster;
- the animated Fox is a skinned caster;
- the ground receives both shadows.

The fixture must run from `examples/quality-motion` so its relative Fox,
Sponza, and HDR asset paths resolve. VSM telemetry makes successful asset
submission observable: the full capture recorded four cutout page draws and
four skinned page draws.

Two controls isolate the actual shadow contribution:

- `--vsm-alpha-no-cast` leaves the MASK curtain visible and receiving shadows
  but removes only its shadow submission. It recorded zero cutout draws while
  retaining four skinned draws.
- an eight-frame pose offset preserves the TAA jitter phase while advancing
  the animated Fox. A ground-only ROI excludes the visible Fox silhouette.

Reproduction:

```sh
cd examples/quality-motion

BLOOM_VSM=1 ./main --vsm-dynamic \
  --quality-run 120 120 0.016666667 \
  /tmp/vsm-casters.png /tmp/vsm-casters.json

BLOOM_VSM=1 ./main --vsm-dynamic --vsm-alpha-no-cast \
  --quality-run 120 120 0.016666667 \
  /tmp/vsm-alpha-control.png /tmp/vsm-alpha-control.json

BLOOM_VSM=1 ./main --vsm-dynamic \
  --quality-run 120 128 0.016666667 \
  /tmp/vsm-skinned-later.png /tmp/vsm-skinned-later.json

cd ../..
python3 tools/quality/vsm_caster_coverage.py \
  --full /tmp/vsm-casters.png \
  --alpha-control /tmp/vsm-alpha-control.png \
  --skinned-later /tmp/vsm-skinned-later.png \
  --telemetry /tmp/vsm-casters.json \
  --alpha-control-telemetry /tmp/vsm-alpha-control.json \
  --skinned-later-telemetry /tmp/vsm-skinned-later.json \
  --output /tmp/vsm-caster-coverage.json
```

## Automated semantic gate

The dependency-free gate uses Bloom's governed standard-library PNG decoder
and fixed normalized ground ROIs. A pixel is changed when luminance differs
by at least 6/255.

The alpha-tested gate requires at least 500 changed pixels, 0.5–75% ROI
coverage, at least 1.5 changed segments per occupied row, and p95 contrast of
at least 0.05. The upper coverage bound and segmentation requirement prevent
an opaque replacement shadow from passing as cutout coverage.

The skinned gate requires at least 500 changed ground pixels, at least 0.5%
ROI coverage, and p95 contrast of at least 0.03. Both full captures must
report active VSM, cutout and skinned page draws, and a current-frame dynamic
overlay page. Malformed or absent telemetry fails closed.

The real 1600 by 900 result was:

| Metric | Alpha-tested control delta | Skinned pose delta |
| --- | ---: | ---: |
| ROI pixels | 71,280 | 51,840 |
| Changed pixels | 19,931 | 4,465 |
| Changed ratio | 27.962% | 8.613% |
| Mean luminance delta | 0.033107 | 0.010765 |
| Changed-pixel p95 | 0.178709 | 0.201384 |
| Segments / occupied row | 4.646 | 3.922 |

Synthetic negative controls prove that unchanged images, missing caster
telemetry, malformed telemetry, and an opaque alpha ROI cannot pass.

## Bounded work and non-regression

The full capture used the existing bounded dynamic overlay:

- 153 overlay pages requested;
- four pages rendered against the hard four-page budget;
- eight total page draws against the hard 64-draw budget;
- 119 pages deferred to live CSM fallback;
- 224 pages resident, 119 dirty, and four pending;
- 19,951,824 VSM pool and metadata bytes.

The checkpoint adds no render pass, shader, resource, binding, page, or draw.
It adds two integer telemetry classifications inside the already opt-in VSM
page draw loop. The default CSM path and ordinary example behavior are
unchanged. The measured full fixture reported 0.198471 ms shadow CPU and
4.083594 ms shadow GPU, including 0.040564 ms CPU and 3.721886 ms GPU for
virtual pages. These are single-run observability values, not optimization
claims.

A repeat full capture preserved the same counters and policy. Its only image
variation was a handful of one-code-value pixels: maximum absolute RGB delta
1/255, zero pixels above 0.02, luminance RMSE 0.000006899, SSIM 1.000000,
OKLab mean delta 0.000000039, and edge RMSE 0.000000051. This is comfortably
below the semantic gate threshold.

## Regression gates

- `scripts/ci-check.sh --quick` passed: contracts, FFI parity, strict Clippy,
  formatting, file-size ratchet, 348 shared tests plus one ignored, headless
  device negotiation, shared Web/WASM, 59 GPU goldens plus two policy
  ignores, four render-target tests, quality governance, and all 20 examples.
- All five caster-oracle unit tests and negative controls passed.
- The real three-capture semantic oracle passed.

Machine-readable evidence is in
`docs/evidence/issue-132-caster-coverage-v1.json`.
