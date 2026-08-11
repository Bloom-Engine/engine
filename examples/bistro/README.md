# Bistro example

The interactive rich profile loads the complete authored exterior: all 2,909
mesh-node placements backed by 551 shared source meshes, with the expanded
original materials and textures:

```sh
./scripts/run-bistro-rich.sh
```

Run that command from the repository root. It builds the example and launches
the complete `assets/bistrox.gltf` scene directly. The sibling `bistro.gltf`
uses `MSFT_texture_dds` indirection and remains available for importer
diagnostics. Extra arguments are forwarded to the example.

Controls: WASD moves, Shift sprints, the mouse looks around, and Tab releases
or captures the cursor.

The default presentation keeps local contrast clear for material and temporal
quality inspection. Use `--fog 1`, `--sun-shafts 1`, and/or `--motion-blur 1`
to opt into the atmospheric and camera-effect qualification modes.

## Exact Godot Bistro reference

`run-godot-reference.sh` converts and launches the exact render content from
Jamsers' `Bistro-Demo-Tweaked` project: its 37 section GLBs, 112 authored
materials, source textures, repeated fill-out props, and hand-authored window
patches. Geometry and textures remain in the source checkout; the converter
adds only one merged glTF document and geometry buffer under `.bloom/`.

Place the source checkout at the default sibling location:

```text
.benchmarks/Bistro-Demo-Tweaked-source
```

Then run, from the Bloom engine repository root:

```sh
./examples/bistro/scripts/run-godot-reference.sh
```

Set `BLOOM_GODOT_BISTRO_SOURCE=/absolute/path/to/Bistro-Demo-Tweaked` to use a
different checkout. Extra arguments are forwarded to the demo. The reference
profile launches fullscreen at native 1.0 render scale with Ultra quality,
the source camera pose and sun direction, full ACES, stable manual exposure,
and the source project's 1.17 output saturation. Calibration overrides are
available as `--manual-exposure`, `--reference-sun`, `--reference-env`, and
`--reference-ambient`; `--reference-sky-sun` calibrates the procedural sky's
solar radiance independently from surface direct lighting.

The Bistro content is licensed CC BY 4.0 and the reference project's code is
MIT; retain the source project's attribution when redistributing its assets.

`tools/quality/prepare_bistro.py` remains available for small deterministic
subsets used by automated captures, but it is no longer the interactive scene.
