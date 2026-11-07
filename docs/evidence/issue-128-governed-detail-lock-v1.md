# Issue #128 governed detail-lock contract v1 evidence

This checkpoint repairs the governed quick-corpus diagnostic contract at
`1938e86c2ecbcdda139d8a76adf95407682114d0`, compared with
`922aff287f42898e94118372f898e0846aba2ec2`. It changes only the quality
manifest, runner allowlist, and their unit contract. Renderer pixels,
shaders, resources, timing, and graph topology are unchanged.

## Root cause

Commit `d8190ea` originally exposed a capture-only
`taa-thin-feature-confidence` classifier. Commit `2e9e744` deliberately
replaced it with `taa-detail-lock`, which reports the production detail seed,
incoming persistent lock, and validated outgoing lock. The renderer and GPU
golden test were migrated, but all eight high-quality corpus cases and the
Python allowlist still required the obsolete filename.

The governed runner therefore emitted the complete current diagnostic set but
failed every affected case for a file the renderer no longer promised. The
fix migrates the governed contract to `taa-detail-lock`; it does not restore
the removed nine-read classifier.

## Exact governed run

```sh
python3 tools/quality/run.py run quick \
  --case pbr-spheres-high \
  --report-only \
  --out /private/tmp/bloom-quality-detail-lock-v2
```

The run emits 30 named intermediates, including
`taa-detail-lock.png`, and reports no missing intermediate. Its sole remaining
failure is the intentional governance boundary: the approved portable
`pbr-spheres-high` baseline has not been installed.

| Measurement | Value |
|---|---:|
| Active fixed timestep | 0.016666667 s |
| Measured frames | 300 |
| Measurement wall | 1720.394042 ms |
| Wall mean | 5.734647 ms |
| CPU mean / p95 | 2.031954 / 2.384414 ms |
| GPU mean / p95 | 12.569622 / 23.123957 ms |

The final image SHA-256 is
`8a4061b51573d9a509b7ecee2e0a37bf5aa5289bef119064af86a54aa5cb057a`.
The detail-lock diagnostic SHA-256 is
`a3292ba0201de59ccd4c99e48698b5edce16926cd9a84bd7db75bcaecfbf8992`.
It is black for this static native-scale case as expected: no fractional
moving-detail lock is active. The populated reconstruction-footprint image
SHA-256 is
`bd38da6b4b6d962430c4a54344f7b9473c9896ab1ac9ff6e7ef88b4ae86e7e8b`.

## Review boundary

A non-mutating initial-baseline review bundle was generated at
`/private/tmp/bloom-quality-detail-lock-review-v2`. Its review manifest has
SHA-256
`4f72fc7ad20d160a585d080f55e8bbf81faf7be96470d611dbc4dbb00a062908`,
records `baseline_state: absent`, and contains the proposed final plus all 30
intermediates. No baseline was installed: installation requires an independent
human reviewer name and an explicit separate command.

The quality build records source commit `1938e86` and one dirty path,
`native/macos/Cargo.lock`, because Cargo refreshes an inherited stale platform
lockfile for the already-committed scene-format dependency. The diagnostic
contract files themselves were clean at capture start; that unrelated lockfile
is deliberately excluded from this checkpoint because the integrated worktree
already owns overlapping package work.

## Validation

```sh
python3 -m unittest tools/quality/test_run.py -v
git diff --check
```

All 12 quality-governance tests pass. The manifest now requires the same name
that `TAA_DIAGNOSTIC_NAMES` and the real-GPU capture test emit. This completes
the named-diagnostic portion of the quick-corpus gate, not issue #128; baseline
approval and the remaining corpus cases stay open.
