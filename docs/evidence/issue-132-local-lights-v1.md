# Issue #132 bounded local-light VSM evidence v1

This evidence qualifies local point-light virtual shadows on an Apple M1 Max
using Metal and Bloom's native high-end profile.

- Qualified head: `6a70ca7d8fe044f042035fe7ad61b3be012e44c7`
- Hardware capture revision: `a01cf8a`
- Capture resolution: 2560x1440

The commits after the hardware capture do not change the admitted local-light
rendering algorithm. `630cb23` adds a uniform early-out when a frame has no
local requests, `93ebdc3` moves tests out of the runtime source file, and
`6a70ca7` removes one formatting-only blank line. The complete regression lane
passed at the qualified head.

## Public and fail-closed contract

`addShadowedPointLight(...)` accepts a shadow-required point light only when
high-tier VSM is active. It rejects unsupported tiers, Web, path tracing,
disabled shadows, invalid values, and submissions above the fixed limit. A
submitted light starts with zero direct intensity. The shadow pass restores
its intensity only after the request wins deterministic visibility admission
and all six cube faces are resident and clean.

Suppressed and pending shadow-required lights therefore cannot leak as
unshadowed direct lights. They are removed before the retained/immediate point
light loops and before the froxel light upload. Ordinary point lights keep
their established behavior.

## Hard bounds

The opt-in `quality-stress --vsm-local-lights` fixture submitted 128
camera-visible shadow-required point lights:

| Bound | Observed |
| --- | ---: |
| Public submissions | 128 of 256 |
| Visible requests | 128 |
| Admitted lights | 5 of 5 |
| Budget-suppressed lights | 123 |
| Faces per admitted light | 6 |
| Local page footprint | 30 |
| Shared physical residency | 254 of 256 |
| Shared page renders in a frame | 8 of 8 |

Admission is deterministic: nearest influence first, then emitted energy, then
stable submission index. The fixture's admitted lights were indices
0, 1, 2, 3, and 4.

On a cold cache, the existing shared eight-page render budget produced:

- 30 local pages resident, 22 still dirty, and exactly 8 rendered;
- one fully clean local light active and four fail-closed pending;
- 224 directional pages resident/dirty and zero directional page renders;
- no denied allocation and no capacity overflow.

Dirty directional pages continued to use the established live CSM fallback.
On the warm cache, all 30 local pages and all 224 directional pages were clean,
all five admitted local lights were active, and the frame issued zero page
renders. Local pages reuse the existing depth array and existing page render
pass. The enlarged VSM sampling uniform adds 6,176 bytes, bringing total VSM
GPU storage in this fixture to 19,958,000 bytes; no per-light Cartesian work or
unbounded page allocation is created.

When no local request is admitted, a uniform bit exits local sampling before
the dynamically indexed metadata array. This keeps the directional-only VSM
point-light path inert.

## Controlled image oracle

The settled candidate and reference contain the same scene, camera,
directional light, ambient term, and five point lights. The only difference is
that the candidate submits those five lights through local VSM while the
reference submits them as ordinary unshadowed lights.

The comparison changed 0.857964396% of pixels at the 0.02 threshold, localized
behind the occluder grid:

- luminance RMSE: 0.010086859;
- SSIM: 0.994366527;
- maximum absolute channel error: 0.588235319;
- mean OKLab delta: 0.000701895;
- mean edge delta: 0.000410529.

Visual review confirmed the intended point-light occlusion without cube-face
seams, page-shaped blocks, missing bands, or frame-wide color changes.

SHA-256:

- cold VSM frame:
  `2ac7c645ec5bcb51b193308bf610275ef907fe58c6b073ee04343c601d19eee2`
- warm VSM frame:
  `f6226c53cdb82d91552c5ab7a8b04fa84f9427ca617d0a52a10fea013822ac63`
- unshadowed reference:
  `a134f3d0742741f9ac7066c80d2476b490c4991eb64372e982074b5c116f00d7`
- cold telemetry:
  `064e25cb45418835bf01992ff4eda4a4c798ad9a47b2d77e3bce329f7af59c4f`
- warm telemetry:
  `27eae425df1800cfa446a302334babd3b7f79b17317c7bb22fc11d6a023581a5`

Both cold and warm reports passed
`tools/quality/vsm_local_lights.py --min-submitted 100`.

## Regression gates

The complete `scripts/ci-check.sh --quick` lane passed at the qualified head:

- all platform FFI/schema parity and Web arity checks;
- formatting, strict correctness/performance Clippy, and file-size ratchet;
- 355 shared tests passed with 1 ignored;
- the headless negotiated-device test passed;
- 59 GPU goldens passed with 2 hardware-policy tests ignored;
- all 4 render-target tests passed;
- `wasm32-unknown-unknown` Web checks passed;
- 39 quality-governance/oracle tests passed;
- all visual metric, fault-engine, and asset-cooker tests passed;
- all 20 canonical examples were present.

The ordinary many-point-light immediate and clustered golden tests also passed
explicitly after the local-light implementation.

Machine-readable measurements accompany this note in
`docs/evidence/issue-132-local-lights-v1.json`.
