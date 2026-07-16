# world-viewer — the reference consumer of the world format

Loads any `*.world.json` (the shared format documented in
[`docs/world-format.md`](../../docs/world-format.md)) and shows it with a fly
camera, through the **generic path** end to end:

```
loadWorld → instantiateWorld → applyWorldEnvironment (every frame)
```

That path is what makes the format "any game's": terrain, entities, prefab
expansion, water volumes, river ribbons, and lights all spawn from the shared
engine modules with no game code. This example is the conformance harness for
it — if a world looks right here, it looks right in any game built on the same
calls. (Until this example existed, `instantiateWorld` had zero consumers;
every game hand-rolled its own spawn code.)

## Usage

```bash
cd engine/examples/world-viewer
perry compile main.ts -o world-viewer

# modelRef paths in a world file are relative to the game's root, so run from there:
cd ../../../shooter
../engine/examples/world-viewer/world-viewer --world assets/worlds/arena_02.world.json --prefabs assets/prefabs
```

On Windows, `dxcompiler.dll` + `dxil.dll` must be next to the exe or in the
CWD (the game roots that ship them work as-is).

## Controls

WASD move · Q/E down/up · hold right mouse to look · Shift = fast.

Instantiation warnings (missing models, unresolved prefabs) print to stderr and
are counted in the overlay.
