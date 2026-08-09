# Bistro example

The interactive rich profile keeps all 551 unique exterior meshes and the
materials and textures they reference, while avoiding the current loader's
expansion of 2,909 authored mesh nodes:

```sh
./scripts/run-bistro-rich.sh
```

Run that command from the repository root. It deterministically prepares the
scene into the ignored `examples/bistro/.generated/` directory, builds the
example, and launches it. Extra arguments are forwarded to the example.

Controls: WASD moves, Shift sprints, the mouse looks around, and Tab releases
or captures the cursor.

Do not launch `assets/bistro.gltf` directly for routine testing yet. Bloom's
current `ModelData` ABI expands repeated node geometry, which has consumed
about 19 GB of resident memory and exceeded the ray-query geometry buffer
limit in local testing. Native mesh instancing is required for a faithful,
bounded load of all 2,909 authored placements.
