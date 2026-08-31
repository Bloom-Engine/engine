# Issue #132 VSM capability fallback qualification v1

This evidence qualifies the forced lower-tier fallback implemented at
`c34743e4c51177de3638cb2e5173bc5b41e2ce80` on an Apple M1 Max / Metal.
Before this change, `BLOOM_FORCE_RENDER_TIER=baseline BLOOM_VSM=1` still
compiled, allocated, rendered, and sampled VSM: telemetry reported
`active: true`, 256 physical pages, and 19,951,824 GPU bytes. That contradicted
issue #132's architecture contract that middle/baseline tiers use cascaded
shadows.

## Startup policy

Renderer construction now records the accepted capability tier before any
VSM-aware shader, layout, material, or shadow resource is selected. The
policy has three explicit outcomes:

| User request | Accepted tier | Result |
| --- | --- | --- |
| off | any | canonical CSM, `not-requested` |
| on | baseline/modern | canonical CSM, `lower-tier-csm-fallback` |
| on | high-end | VSM with live CSM fallback, `high-tier-vsm` |

The lower-tier result does not merely disable sampling after allocation. It
uses the same VSM-free shader specialization as the CSM control, creates no
VSM resources, walks no page demand, submits no page renders, and allocates no
VSM caster-indirect buffers. The capability table now states this contract.
Both quality telemetry and the public renderer capability report expose user
intent, eligibility, enablement, active state, and selection reason.

## Image and resource oracle

The fixed 266-post contact-detail fixture was captured after 120 warm-up and
120 measured frames at 2560 by 1440:

```sh
BLOOM_FORCE_RENDER_TIER=baseline \
  ./examples/quality-motion/main --vsm-contact-detail \
  --quality-run 120 120 0.016666667 \
  /tmp/baseline-control.png /tmp/baseline-control.json

BLOOM_FORCE_RENDER_TIER=baseline BLOOM_VSM=1 \
  ./examples/quality-motion/main --vsm-contact-detail \
  --quality-run 120 120 0.016666667 \
  /tmp/baseline-request.png /tmp/baseline-request.json

cmp /tmp/baseline-control.png /tmp/baseline-request.png
```

The images are byte-identical. Both have SHA-256
`596a8b611496b56b9f2e5747f95b706effa2b85e136e84e481afb1d0bc987d6f`.
The requested fallback reports:

- selected tier `baseline`;
- `requested: true`, `capability_eligible: false`, `enabled: false`, and
  `active: false`;
- `selection_reason: "lower-tier-csm-fallback"` and `fallback: "csm"`;
- zero physical capacity, GPU bytes, resident/dirty/requested/pending pages,
  receiver work, and page renders;
- zero VSM caster pages, draws, calls, and bytes.

An additional forced-modern request was byte-identical to its forced-modern
CSM control and reported the same zero-work fallback.

## High-tier non-regression

The same 120/120-frame high-tier VSM reproduction retained the previously
qualified SHA-256
`81c6d840a275ee4fe3ff9fbeb87a5a1e21b23bb1c087a4e24c024c2f778b0bde`.
It reported `capability_eligible: true`, `enabled: true`, `active: true`, all
224 demanded pages resident and clean, zero pending pages, and the unchanged
19,951,824-byte VSM allocation. The default-off and eligible high-tier
rendering implementations are otherwise untouched.

## Performance boundary

After selection, the requested lower-tier process and its CSM control execute
the same shader variants and frame branches. The differing request and reason
fields are read only for telemetry. Five counterbalanced 60-warm-up /
180-measured-frame runs confirmed no shadow-path cost: median shadow CPU was
0.087858 ms for the CSM control and 0.088146 ms for the requested fallback, a
0.000288 ms difference within run noise. Median wall, CPU mean, GPU mean, GPU
p50, render CPU, and main-HDR GPU were all lower for the requested fallback;
CPU p50 was 0.141333 ms higher amid much wider process-to-process dispersion,
so no timing improvement is claimed. The structural zero-work and exact-image
oracles are the performance gate.

## Platform and regression gates

- `scripts/ci-check.sh --quick`: pass, including 348 shared tests plus one
  ignored, the headless negotiated-device test, 59 GPU goldens plus two
  policy ignores, four render-target tests, strict correctness/performance
  Clippy, FFI parity, file-size ratchet, shared Web/WASM, quality tooling, and
  all 20 examples.
- Web crate `wasm32-unknown-unknown` check with its supported default features:
  pass.
- iOS `aarch64-apple-ios` check with `models3d,image-extras`: pass.
- Android `aarch64-linux-android` check with `models3d,image-extras`: pass.

The selection module is platform-independent. Web and mobile default to CSM
when no VSM environment request exists; a requested accepted lower tier uses
the same qualified CSM-only construction path.
