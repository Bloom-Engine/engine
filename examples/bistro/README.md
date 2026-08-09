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

`tools/quality/prepare_bistro.py` remains available for small deterministic
subsets used by automated captures, but it is no longer the interactive scene.
