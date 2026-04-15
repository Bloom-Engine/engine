# Bloom renderer validation tools

Two companion binaries that close the loop between the Bloom realtime
renderer and a reference ground truth. Used to track whether each
renderer change moves the realtime output closer to or farther from
physically-correct rendering.

## `bloom-reference` — CPU path tracer

Renders a glTF/GLB scene via Monte-Carlo path tracing. Produces
noise-free PNGs that serve as ground truth for the realtime renderer
to be measured against.

**Features** (end of Phase 4):

- BVH accelerated ray/triangle intersection
- PBR BRDF: GGX specular + Burley diffuse, metalness-aware energy split
- Full glTF PBR textures: base color, metallic-roughness, emissive,
  normal, occlusion (all sampled with correct sRGB/linear decoding)
- HDR environment map (.hdr Radiance) IBL with importance sampling
- Explicit directional (sun) light with delta NEE
- Next event estimation with multiple importance sampling (balance
  heuristic) between env map and BRDF sampling
- Multi-bounce path tracing with Russian roulette termination
- Deterministic per-pixel RNG seeding for reproducible renders
- ACES tone mapping + sRGB output, matching the Bloom realtime convention

```shell
cd tools/bloom-reference
cargo build --release
./target/release/bloom-reference \
  --spec ../../examples/renderer-test/specs/helmet.json \
  --out ref.png
```

## `bloom-diff` — pixel comparison

Compares two PNGs (reference vs realtime) and produces quantitative
diff metrics + optional visualization.

**Output**:

- Console: per-channel RMSE, luminance RMSE, max-abs-error, %-pixels
  above tolerance, SSIM.
- `--heatmap PATH`: false-color per-pixel difference visualization.
- `--composite PATH`: 3-up side-by-side (reference | candidate | heatmap).
- Exit code 0/1 based on `--tolerance`, for CI integration.

```shell
cd tools/bloom-diff
cargo build --release
./target/release/bloom-diff \
  --reference ../bloom-reference/ref.png \
  --candidate /path/to/realtime-shot.png \
  --composite diff.png \
  --tolerance 0.05
```

## End-to-end validation workflow

1. **Define a viewpoint** in a shared JSON spec (example:
   `examples/renderer-test/specs/helmet.json`). Both tools read the
   scene path, camera, env map, and resolution from it.

2. **Render the reference**:
   ```
   cd tools/bloom-reference
   ./target/release/bloom-reference \
     --spec ../../examples/renderer-test/specs/helmet.json \
     --out ref.png
   ```

3. **Render the realtime screenshot** (uses `takeScreenshot()` via
   Bloom's FFI, writes a PNG after 30 warmup frames then exits):
   ```
   cd examples/renderer-test
   ./renderer-test \
     --camera 1.8 1.2 2.4 0 0 0 45 \
     --out realtime.png
   ```
   *Note*: the realtime test takes camera args on the CLI (7 floats:
   px py pz tx ty tz fov) because Perry's JSON array indexing has
   backend-level bugs that prevent clean spec-file reads from TS.

4. **Diff**:
   ```
   cd tools/bloom-diff
   ./target/release/bloom-diff \
     --reference ../bloom-reference/ref.png \
     --candidate ../../examples/renderer-test/realtime.png \
     --composite diff.png
   ```

The RMSE / SSIM numbers give an objective answer to "is my renderer
change an improvement?". As the Bloom realtime renderer gains normal
maps, MR textures, HDR IBL etc. through the v2 spec phases, those
numbers should monotonically decrease.

## Multi-camera validation suite (`tools/validate.sh`)

Wraps the workflow above in a script that runs four cameras of the
helmet scene (front, three-quarter, side, top-down) through both
renderers and reports per-view + aggregate metrics:

```
$ tools/validate.sh --width 1024 --height 1024 --spp 128 --bounces 4
view              RMSE      SSIM     %>tol
----              ----      ----     -----
front          0.23286   0.67296    59.06%
threequarter   0.23566   0.67409    62.37%
side           0.30854   0.56995    62.02%
topdown        0.18701   0.66082    68.08%
----              ----      ----     -----
average        0.24102   0.64446    62.88%
```

Renders cache: `tools/validate-out/ref-{view}.png` is reused if
present (delete to force re-render after a reference change). The
realtime captures (`rt-{view}.png`) are always re-rendered so engine
changes are picked up.

Numbers from this suite are higher than single-camera diff at native
resolution because the realtime output is downsampled via `sips` to
match the reference resolution — sips's resampling adds blur that
inflates RMSE. The suite is consistent with itself across runs, so
it's good for regression detection; just don't directly compare its
numbers to single-camera native-res diff.

## PBR material grid (`examples/pbr-spheres/`)

Diagnostic scene: a 5×5 grid of spheres where rows vary metallic
(0 → 1) and columns vary roughness (0 → 1), all sharing a gold base
color and lit purely by the outdoor HDR. Visual-only validation
right now (no comparable bloom-reference path yet — synthetic
scenes don't load through glTF).

```
cd examples/pbr-spheres
perry compile main.ts
./main --camera 0 0 6 0 0 0 45 --out grid.png
```

Reading the output:
- Top row should look like polished/rough chrome (full metal).
- Bottom row should look like gold paint (full dielectric).
- Left column should show sharp env reflections (smooth).
- Right column should show heavily blurred env (rough).
- A clean diagonal gradient between corners means the BRDF, IBL
  prefilter and BRDF LUT are working together correctly.

Useful when changing any material/shading code — visible breakage
is immediate and unambiguous, no diff numbers needed.

**Procedural mesh note**: the sphere mesh is built via `new Array(N)`
+ index assignment, not `.push()`. Perry's current backend has
issues with `.push`-built arrays passed to FFI: `.length` returns
the literal-init size, and the post-push data isn't where the FFI
expects it. Use `createMeshExplicit(verts, vCount, idx, iCount)`
from `bloom/models` and pass counts manually for any procedural
mesh — see `examples/pbr-spheres/main.ts` for the pattern.
