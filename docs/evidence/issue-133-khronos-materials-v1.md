# Issue #133 Khronos material qualification v1

This checkpoint qualifies Bloom's alpha, transmission, volume attenuation,
and transmission-order rendering against four official Khronos
`glTF-Sample-Assets` controls. The final hardware run passed at revision
`b40d28b3f52332fc210f1382fa9cb4efa5450f8f` on the native high-end Metal
profile with hardware ray queries.

## Pinned corpus

The opt-in `tools/quality/khronos_materials.py` runner downloads assets only
from Khronos revision
`2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf`, verifies every GLB hash, and
records the source metadata and license:

| Control | License | GLB SHA-256 |
|---|---|---|
| AlphaBlendModeTest | CC-BY-4.0 | `37c3577d143071b42dd46e9d942b157837eb25c6340112171d7faecaa987b14e` |
| TransmissionTest | CC0-1.0 | `dd9732dae5517f8605ad4324d78b077b969c3e8357c056280d0a4e4b67797d15` |
| AttenuationTest | CC-BY-4.0 | `7ca161b7f8a9e4b2ac1f7f75816b5848bb31f3b4c226c4cb731b487c8809b756` |
| TransmissionOrderTest | CC0-1.0 | `d904b6cd6c83792fd4a4d9ad4f0366bde76a63e347541c465f2ad4c5baf22a21` |

The source repository is
[KhronosGroup/glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets).
The run is reproducible with:

```sh
python3 tools/quality/khronos_materials.py \
  --out tools/quality/out/khronos-materials
```

## Static-node volume correction

The first qualification run exposed a real physical-material defect. Bloom
baked static glTF node transforms into vertex positions but discarded their
scale before evaluating `KHR_materials_volume`. The official AttenuationTest
requires its Node Scale row to deepen absorption as scale increases from 0.25
to 2.0.

At clean pre-fix revision `d2032b0`, median front-face blue-minus-red values
were `[36, 39, 36, 37, 39]`; the scale response was effectively flat. The
corrected import retains the mean baked basis scale separately from the
authored `thicknessFactor`. It is consumed consistently by camera refraction,
transmitted shadows, and transparent GI, while later draw/instance scale
continues to be applied normally.

The corrected ramp is `[18, 29, 36, 41, 42]`. A dependency-free automated
gate now checks that this row has a materially increasing absorption span.
Running the new gate against the clean pre-fix binary fails only
AttenuationTest, proving that it detects the defect it was added to prevent.

An isolated pre/fix image comparison changed only the five Node Scale cubes:

- pre-fix attenuation SHA-256:
  `8643da1ff400f0d3a5829df11798afb925243ff9c7cf6da4039504986b62ab1e`;
- corrected attenuation SHA-256:
  `a2a1b2c3c5a4969e708f09566dbc476a602e007a996152bcd2f857c55c46df50`;
- luminance RMSE: `0.004399808`;
- luminance SSIM: `0.998071194`;
- pixels above a 0.02 tolerance: `2.013852835%`;
- mean OKLab delta: `0.000606760`;
- mean edge delta: `0.000184952`.

The other official controls are byte-exact before and after the correction:

| Control | Capture SHA-256 |
|---|---|
| AlphaBlendModeTest | `6fc0ca56217f87f1edf6e9ff99403b417fe8b0bad3a84747eb9837bbc8e92043` |
| TransmissionTest | `46f840a1700483cfad8b4ab5a674e5f2e6772eca637e891bc5aac03233bee1c0` |
| TransmissionOrderTest | `bc92a884c41b4e349642222620776855190850f1a34373975dcd490ee5e1525b` |

## Final qualification

Every final control rendered twice with identical SHA-256 values, passed its
semantic image gate, produced non-flat/non-black output, and emitted no
supported-field import or validation diagnostics. The qualification tool
records these as review candidates and never installs or approves a baseline;
reference-image approval remains an explicit human action.

The implementation preserves the 96-byte transmission GPU uniform and adds no
render pass, draw, GPU allocation, texture sample, binding, or shader branch.
It adds one CPU material scalar and one bounded scale multiplication when
material uniforms are prepared. The textured loader also moves joint and
weight accessor collection out of the vertex loop, removing repeated
whole-accessor decoding.

Staged and CPU-only loaders now share the same static node-transform,
instancing, normal/tangent, bounds, and volume-thickness contract as the
textured loader. This fixes previously missing scene transforms and instances
in those paths; it does not add work to the existing textured render path.

## Regression gates

- 330 shared tests passed in the complete suite (329 runnable, 1 ignored);
- 59 GPU goldens passed, with 2 hardware-policy cases intentionally ignored;
- all 4 render-target tests passed;
- native release `models3d` and `wasm32-unknown-unknown` Web builds passed;
- FFI/schema parity passed for macOS, Linux, Windows, Android, Apple targets,
  and Web;
- formatting and the file-size ratchet passed;
- 17 quality-governance tests, 3 visual-diff fault tests, and 17 cooker tests
  passed;
- all four final Khronos cases passed twice at the qualified revision.

Machine-readable measurements accompany this note in
`docs/evidence/issue-133-khronos-materials-v1.json`.
