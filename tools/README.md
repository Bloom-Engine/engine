# Bloom renderer validation tools

The canonical regression workflow is
[`tools/quality`](quality/README.md). It owns versioned cameras/assets,
fixed-step uncapped capture, named render-graph evidence, perceptual gates,
GPU/CPU budgets, adapter metadata, negative controls, and baseline governance.

The companion binaries below close the loop between the Bloom realtime
renderer and reference ground truth. The quality harness reuses them; direct
invocation remains useful for investigation and reference generation.

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

For the engine path-tracing golden, use the built-in scene rather than
maintaining a second asset. It mirrors the test's floor slab, six
colored cubes, camera, sun, ambient sky, and deterministic seed, and
writes a machine-readable reproduction record:

```shell
cargo run --release -- \
  --builtin pt-golden \
  --out pt-golden-reference.png \
  --metadata pt-golden-reference.json \
  --width 256 --height 256 --spp 256 --bounces 8 --seed 0 \
  --camera 5 4 7 0 0.5 0 50 \
  --sun-dir 0.5 1 0.3 --sun-intensity 1.2
```

Use this reference to inspect energy, visibility, silhouettes, and
occlusion. Do not gate raw whole-frame RMSE against the GPU golden:
the reference renders analytic sky on camera misses, performs
environment NEE/MIS, and uses a fixed ACES+sRGB display transform,
while the engine PT preserves the raster sky and passes through the
engine HDR/post stack.

### Layered-PBR parameter reference

`bloom-brdf-reference` evaluates the versioned layered-material contracts
without scene, texture, sampling-noise, or tone-mapping variables:

```shell
cargo run --release \
  --manifest-path tools/bloom-reference/Cargo.toml \
  --bin bloom-brdf-reference -- \
  --out tools/bloom-reference/reference/layered-pbr-v1.json
```

The checked-in 48-case matrix includes separate diffuse/specular terms,
directional-light response, MIS PDF, and deterministic white-furnace
reflectance. Tests require exact regeneration plus reciprocal, finite, and
energy-bounded behavior. See
[`docs/layered-pbr.md`](../docs/layered-pbr.md).

Version 2 adds clearcoat and dielectric specular/IOR:

```shell
cargo run --release \
  --manifest-path tools/bloom-reference/Cargo.toml \
  --bin bloom-brdf-reference -- \
  --version 2 \
  --out tools/bloom-reference/reference/layered-pbr-v2.json
```

Its 39 checked cases cover base/default equivalence, water/diamond/zero IOR,
disabled/colored specular, smooth/rough clearcoat, and combined lobes at three
view angles. The tests additionally sweep a larger reciprocal white-furnace
matrix and require pure conductors to remain independent of dielectric
specular controls.

Version 3 adds Charlie sheen and tangent-space anisotropic GGX:

```shell
cargo run --release \
  --manifest-path tools/bloom-reference/Cargo.toml \
  --bin bloom-brdf-reference -- \
  --version 3 \
  --out tools/bloom-reference/reference/layered-pbr-v3.json
```

The 30 checked rows cover sheen roughness, anisotropy strength/rotation,
fabric, clearcoat-over-sheen, and all-lobe combinations at three view angles.
The same tool owns the 128×128 R16F sheen directional-albedo LUT oracle.

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

## Legacy multi-camera exploration (`tools/validate.sh`)

This script predates `tools/quality` and is no longer a merge/qualification
gate. It runs four cameras of the
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
it remains useful for macOS-only visual exploration. Do not use it to certify a
renderer change: its cameras live in the shell script, references are cached,
output is resized through `sips`, and it does not enforce the quality harness's
fixed-step, warm-up, adapter, timing, intermediate, or baseline-review
contracts.

## PBR material grid (`examples/pbr-spheres/`)

Diagnostic scene: a 5×5 grid of spheres where rows vary metallic
(0 → 1) and columns vary roughness (0 → 1), all sharing a gold base
color and lit purely by the outdoor HDR. The versioned CPU parameter matrix
above now supplies a lobe-level numeric reference; image comparison still
requires a shared scene/camera because synthetic scenes do not load through
glTF.

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
