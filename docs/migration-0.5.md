# Migrating to Bloom 0.5

0.5 makes the API consistent in three places where conventions silently
diverged. Each change is breaking on purpose — the old inconsistencies
caused invisible bugs (colors that rendered white, rotations that were
60× too fast). All engine examples are already migrated and serve as
references.

## Surface colors are 0–255 everywhere

`setSceneNodeColor`, `setOutlineColor`, and the color part of
`setSceneNodeWaterMaterial` previously took 0–1 floats — the only places
in the API that did. They now take 0–255 like every `draw*` call and the
`Colors` presets.

```ts
// before                                    // after
setSceneNodeColor(node, 0.75, 0.75, 0.7);    setSceneNodeColor(node, 191, 191, 179);
setSceneNodeColor(node, c.r/255, c.g/255, …) setSceneNodeColor(node, c.r, c.g, c.b, c.a);  // Colors presets now just work
```

Symptom of unmigrated code: scene nodes render almost black (values
divided twice).

**Unchanged:** light colors (`addDirectionalLight`, `addPointLight`)
stay 0–1 floats with a separate intensity — that's the radiometric
convention (Unity and Unreal do the same), and light color × intensity
can meaningfully exceed 1.0. The `*.world.json` format also keeps 0–1
tints (serialized data is versioned separately); the loader converts.

## Angles are degrees everywhere

`drawModelRotated`'s `rotY` was radians; `Camera2D.rotation` was degrees.
Everything user-facing is now degrees (the raylib convention).

```ts
// before                                    // after
drawModelRotated(m, p, 1.0, Math.PI / 2, t); drawModelRotated(m, p, 1.0, 90, t);
```

Symptom of unmigrated code: models spin ~57× faster than intended.

**Unchanged:** physics angular velocity stays radians/sec (SI, matches
Jolt), and quaternions are quaternions.

## `Texture.handle` (was `Texture.id`)

`Texture` was the only resource type whose handle field was named `id`;
`Sound`, `Music`, `Font`, and `Model` all use `handle`.

```ts
// before            // after
myAtlas.id           myAtlas.handle
```

## Also in 0.5 (non-breaking)

- `physics.step(world, dt)` is now fixed-timestep with an accumulator
  and returns the interpolation alpha; `physics.stepVariable` is the
  old exact-dt behavior. See docs/physics.md "Stepping".
- Stale handles (use-after-free/destroy) now fail lookups instead of
  aliasing whatever object reused the slot.
- Most `*Raw` function variants are documented `@internal`. They work
  around an aarch64-Android Perry miscompilation of `obj.field` reads
  feeding f64 FFI args; a few (e.g. `loadMusicRaw`) are actively
  recommended on Android and are staying.
- Coordinate system is now documented at the top of the physics and
  scene modules: right-handed, Y-up, meters, SI units.
