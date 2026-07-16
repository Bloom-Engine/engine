# Physics

Bloom uses **[JoltPhysics 5.5.0](https://github.com/jrouwe/JoltPhysics)** (the engine behind
Horizon Forbidden West and Godot 4's default backend) on native, and
**[JoltPhysics.js 1.0.0](https://github.com/jrouwe/JoltPhysics.js)** on the web target.
Jolt replaced the previous Rapier 3D backend — see ["Why Jolt (and not PhysX or Rapier)"](#why-jolt-and-not-physx-or-rapier) for the rationale.

## Architecture

```
                             ┌─── TypeScript game code ───┐
                             │  src/physics/index.ts      │
                             │    createWorld(), step()…  │
                             └────────────┬───────────────┘
                                          ↓ Perry-FFI (f64 scalars only)
              ┌───────────────────────────┴────────────────────────────┐
              │                                                        │
         NATIVE                                                      WEB
              ↓                                                        ↓
   native/<platform>/src/lib.rs                        native/web/src/lib.rs
   (#[no_mangle] extern "C" fn                         (#[wasm_bindgen] pub fn
    bloom_physics_*)                                    bloom_physics_*)
              ↓ forwards via                                          ↓ wasm_bindgen imports
   define_physics_ffi! macro                          jolt_bridge.js (JS module)
              ↓ calls                                                 ↓ calls
   native/shared/src/physics_jolt.rs                  state.worlds / state.bodies /…
   (JoltPhysics struct, handle registries)                           ↓
              ↓ extern "C" through                                JoltPhysics.js
   native/shared/src/jolt_sys.rs                      (standalone WASM module, ~1 MB)
              ↓ links
   native/third_party/bloom_jolt/          ← C++ shim (~1800 LOC + 500-line header)
              ↓ depends on
   native/third_party/JoltPhysics/         ← Jolt 5.5.0 (git submodule, built via cmake crate)
```

**Why the split?** The C++ Jolt library doesn't target WebAssembly through the Rust
cmake path — you'd need Emscripten + a separate Jolt WASM build. Rather than ship two
build systems, we use Jolt's author's ready-made WASM distribution (JoltPhysics.js)
on web and the native Jolt on every other platform. The TypeScript-facing API is
identical in both cases.

## Build integration

- **Native:** `native/shared/build.rs` invokes the `cmake` crate against
  `native/third_party/bloom_jolt/`. The first build compiles Jolt from source
  (~60 s); subsequent builds link against the cached `libJolt.a` + `libbloom_jolt.a`.
- **WASM:** `build.rs` skips cmake when `CARGO_CFG_TARGET_ARCH == "wasm32"`. The web
  crate's `wasm-bindgen` import of `/jolt_bridge.js` is resolved at `wasm-pack`
  time; the bridge file gets bundled into `pkg/snippets/`. The JS glue
  (`bloom_glue.js`) imports `jolt-physics@1.0.0` and hands the module factory
  to `bloom.bloom_physics_init_jolt()` before any physics call.
- **Feature flag:** physics is behind the `jolt` feature on every platform crate.
  Jolt defaults **ON** everywhere — each platform crate ships
  `default = ["jolt", "models3d", "image-extras"]` so existing games are
  unaffected. Opt *out* with `default-features = false` if a build shouldn't
  pay the ~60 s Jolt first-build cost.

## Supported features

Matches or exceeds UE5's built-in physics surface.

| Tier | Feature | Native | Web |
|---|---|---|---|
| 1 | Box / sphere / capsule / cylinder shapes | ✅ | ✅ |
| 1 | Convex hull / triangle mesh / heightfield / static compound | ✅ | ✅ |
| 1 | Scaled + offset-COM shape wrappers | ✅ | ✅ |
| 1 | Dynamic / kinematic / static bodies | ✅ | ✅ |
| 1 | Forces, impulses, torques (center-of-mass + at-point) | ✅ | ✅ |
| 1 | Linear + angular damping, gravity factor, CCD | ✅ | ✅ |
| 1 | Raycast closest + raycast all | ✅ | ✅ |
| 1 | Shape cast (closest hit) | ✅ | native only |
| 1 | Overlap sphere / box / point | ✅ | ✅ |
| 1 | Fixed / point / hinge / slider / distance constraints | ✅ | ✅ |
| 1 | Contact events (added / persisted / removed) via polled queue | ✅ | ✅ |
| 1 | 16-layer collision matrix, object-layer filtering | ✅ | ✅ (create-time only) |
| 2 | **Character controller** (`CharacterVirtual` — slope + stair handling) | ✅ | ✅ |
| 2 | **Soft bodies** — cloth, rope, jelly (per-vertex pinning via `invMass=0`) | ✅ | ✅ |
| 2 | **Wheeled vehicles** — 4-wheel, ray collision tester, engine + differential | ✅ | ✅ |
| 2 | **Ragdolls** (EN-025) — built at runtime from the skinned skeleton; capsule-per-bone + limited six-DOF joints | ✅ (via `createRagdoll()` in `bloom/models`, `native/shared/src/ragdoll.rs`) | — |

Six-DOF constraints exist in the shim (`bj_constraint_six_dof`) but are
internal-only — ragdoll articulation uses a locked-translation wrapper; there
is no public `bloom_physics_*` export or TS API for them.

Gaps vs. full Jolt API: tracked vehicles, motorcycle controller, public six-DOF
constraints, constraint runtime enable/disable round-tripping on web, body-lock
damping setters on web, raycast world-space normals (currently returns (0,1,0)
— body-lock read is the fix).

## TypeScript API quick-start

```typescript
import * as physics from '@bloom/physics';

// 1. Create a world (once, on game start).
const world = physics.createWorld({ gravity: { x: 0, y: -9.81, z: 0 } });

// 2. Build shapes (reusable — one shape, many bodies).
const ballShape = physics.sphereShape(0.5);
const groundShape = physics.boxShape({ x: 50, y: 0.5, z: 50 });

// 3. Bodies reference shapes + carry motion state.
const ground = physics.createBody(world, groundShape, {
  motionType: physics.MotionType.STATIC,
  position: { x: 0, y: -0.5, z: 0 },
  objectLayer: physics.Layer.NON_MOVING,
});
const ball = physics.createBody(world, ballShape, {
  motionType: physics.MotionType.DYNAMIC,
  position: { x: 0, y: 10, z: 0 },
  restitution: 0.6,
});

// 4. Call optimizeBroadphase once after initial body setup.
physics.optimizeBroadphase(world);

// 5. In your game loop — pass the real frame delta; step() runs the
// simulation at a fixed rate internally (see "Stepping" below):
physics.step(world, deltaTime);
const pos = physics.getBodyPosition(ball);
// ... read positions and render sprites / meshes at those transforms
```

## Stepping

`physics.step(world, deltaTime)` is **fixed-timestep**: it accumulates the
wall-clock delta and advances the solver in whole steps of 1/60 s
(configurable via `setFixedTimestep(world, hz, maxSteps?)`). Variable-size
solver steps feed frame hitches straight into the constraint solver —
tunneling and joint explosions on any slow frame — so the accumulator is
the default, with two protections baked in:

- A single frame's contribution is clamped to 0.25 s (debugger pauses and
  OS hitches produce one slowed-down frame, not minutes of catch-up).
- At most `maxSteps` (default 4) fixed steps run per frame; surplus
  backlog is dropped. The simulation slows down instead of spiraling.

`step` returns the **interpolation alpha**: how far the carried remainder
sits between the last two physics states. Two ways to use it:

```typescript
// Easiest: let the engine blend body transforms for rendering.
physics.setInterpolation(world, true);
runGame((dt) => {
  physics.step(world, dt);
  const p = physics.getBodyPosition(ball); // already smoothed
});

// Manual: interpolate game-side state with the same alpha.
const alpha = physics.step(world, dt);     // or physics.getStepAlpha(world)
```

With interpolation on, `getBodyPosition`/`getBodyRotation` return the
blended state; physics queries (raycasts, overlaps, contacts) always see
the raw simulation state.

Exact-dt stepping is still available as `physics.stepVariable(world, dt)`
for code that drives its own accumulator — it trades stability for
control.

## Character controllers (Tier 2)

For player movement, use `CharacterVirtual` — Jolt's kinematic controller with
slope handling and step climbing.

```typescript
const capsule = physics.capsuleShape(0.5, 0.3);
const character = physics.createCharacter(world, capsule, {
  position: { x: 0, y: 5, z: 0 },
  mass: 70,
  maxSlopeAngleRad: Math.PI / 3,  // 60° walkable
});

// Every frame: set desired horizontal velocity, let gravity handle Y.
const moveInput = readInput();
physics.setCharacterLinearVelocity(character, {
  x: moveInput.x * speed,
  y: physics.getCharacterLinearVelocity(character).y,
  z: moveInput.z * speed,
});
physics.updateCharacter(character, dt, { x: 0, y: -9.81, z: 0 });

if (physics.isCharacterGrounded(character) && jumpPressed) {
  physics.setCharacterLinearVelocity(character, { x: 0, y: 8, z: 0 });
}
```

`updateCharacter` integrates gravity into velocity internally (Jolt's raw
`ExtendedUpdate` does not). This matches UE5/Unity ergonomics.

## Soft bodies (Tier 2)

Cloth, rope, and inflated jelly are all soft bodies with different pressure /
compliance settings.

```typescript
// 3×3 cloth pinned at 4 corners — simple banner / flag.
const vertices: Vec3[] = [];
const inverseMasses: number[] = [];
for (let y = 0; y < 3; y++) {
  for (let x = 0; x < 3; x++) {
    vertices.push({ x: x - 1, y: 5, z: y - 1 });
    const isCorner = (x === 0 || x === 2) && (y === 0 || y === 2);
    inverseMasses.push(isCorner ? 0 : 1);   // 0 = pinned in place
  }
}
const indices = [0,3,1, 1,3,4, 1,4,2, 2,4,5, 3,6,4, 4,6,7, 4,7,5, 5,7,8];

const cloth = physics.createSoftBody(world, {
  vertices, inverseMasses, indices,
  edgeCompliance: 1e-3,   // soft cloth; 1e-5 = stiff, 1e-2 = floppy
  pressure: 0,             // cloth/rope; >0 = balloon/jelly
});

// Each frame, read vertex positions (world-space) and update your mesh VBO:
for (let i = 0; i < physics.softBodyVertexCount(cloth); i++) {
  const pos = physics.getSoftBodyVertex(cloth, i);
  // write to mesh[i] for rendering
}
```

Drop `pressure: 200` for inflatable objects (balloons, jelly cubes). Pin a
vertex at a moving anchor by setting its `invMass` to 0 and calling
`setSoftBodyVertex` each frame.

## Wheeled vehicles (Tier 2)

4-wheel car with rear-wheel drive, front-wheel steering, ray collision tester.

```typescript
// Use offset-COM so wheels dangle below the chassis geometry.
const chassisBox = physics.boxShape({ x: 1, y: 0.2, z: 1.9 });
const chassisShape = physics.offsetCenterOfMassShape(chassisBox, { x: 0, y: -0.6, z: 0 });

const car = physics.createVehicle(world, {
  chassisShape,
  position: { x: 0, y: 2, z: 0 },
  engineMaxTorque: 800,       // Nm
  maxSteerAngleRad: Math.PI / 6,
});

// Every frame:
physics.setVehicleInput(car, throttle, steering, brake, handbrake);
physics.step(world, dt);

// Read transforms for rendering:
const chassisXf = physics.getBodyTransform(physics.getVehicleChassis(car));
for (let i = 0; i < 4; i++) {
  const wheelXf = physics.getWheelTransform(car, i);
  // draw wheel mesh at wheelXf
}
```

### Vehicle tuning notes

Getting a test car to accelerate reliably is **game-specific tuning**, not
something defaults solve. The three knobs that matter:

1. **Chassis geometry.** Use `offsetCenterOfMassShape` to put the chassis box
   *above* the COM. Wheel Y positions (chassis-local) should be `-(chassisHalfHeight + 0.1)`
   or lower so wheels clearly hang below the box. If the chassis shape overlaps
   the ground, the car floats on the box itself and the wheels lose contact.
2. **Mass → suspension.** A 1500 kg car with default suspension (1.5 Hz, damping
   0.5) needs about 0.1 m of compression per wheel under static load. If your
   chassis is heavier, lower the suspension frequency or the car will bottom out.
3. **Tire friction curves.** `WheelSettingsWV` defaults produce ~1.2 peak
   friction coefficient; combined with ground friction 0.8, that's plenty for
   most games. If your car spins its wheels without moving, raise the *ground*
   friction (`BodyConfig.friction = 1.0`), not the wheel's.

The bundled `vehicle_api_smoke` test verifies the FFI chain works (engine RPMs,
wheels spin, API doesn't crash) but intentionally does *not* assert
drive-forward behaviour — that requires per-game tuning.

## Handle lifetime + ownership

All handles are opaque 1-based `number` / `f64` values (0 = invalid). The Rust
side uses `HandleRegistry<T>` for O(1) alloc/free with slot reuse; the web
side uses `Map<number, T>`.

- **Shapes** are refcounted — `releaseShape` decrements. Bodies holding a
  reference keep the shape alive.
- **Bodies / constraints / characters / vehicles** are owned by the world and
  cleaned up when the world is destroyed or you call the matching `destroy*`.
- **Soft bodies** are bodies — destroy with `destroyBody`.
- **Contact events** accumulate every step; `popContacts()` returns and clears
  the queue.

## Threading

- Jolt internally threads broadphase + integration across `num_threads` workers
  (default = CPU count - 1). Safe to use one world per process; multi-world is
  supported but all ops on a given world must happen on the thread that created it.
- The contact listener runs on Jolt's job threads; events push into a
  mutex-guarded queue, drained by the main-thread `popContacts`.
- On web, JoltPhysics.js is single-threaded (the WASM build ships without
  SharedArrayBuffer pthreads support for compatibility). `num_threads` is
  ignored on the web target.

## Scaling / perf

Current target: 60 fps with ~10 000 dynamic bodies on an M-series Mac. Jolt's
advertised ceiling is much higher (~100 000 bodies). If you need to push past
what the defaults allow:

- Bump `WorldConfig.maxBodies` (default 65 536) + `maxBodyPairs` + `maxContactConstraints`.
- Call `optimizeBroadphase(world)` after all static bodies are placed.
- Use compound shapes instead of many child bodies where possible.
- Soft bodies are expensive — typical budget is 1–2 cloth patches of
  256 vertices each at 60 fps.
- Vehicles add 4 ray casts per wheel per step — effectively free for <50 cars.

## File layout

```
native/third_party/
├── JoltPhysics/                       # git submodule, pinned v5.5.0
└── bloom_jolt/
    ├── CMakeLists.txt                 # builds libbloom_jolt.a + libJolt.a
    ├── include/bloom_jolt.h           # C ABI — the contract
    └── src/bloom_jolt.cpp             # Jolt C++ ↔ C ABI translation

native/shared/src/
├── jolt_sys.rs                        # Rust ↔ C extern "C" bindings (+ smoke tests)
└── physics_jolt.rs                    # HandleRegistry-based wrapper +
                                       # define_physics_ffi! macro

src/physics/index.ts                   # TypeScript game-facing API
package.json                           # Perry FFI manifest (121 bloom_physics_* entries)

native/web/
├── jolt_bridge.js                     # JS-side implementation via JoltPhysics.js
├── src/lib.rs                         # wasm_bindgen imports → JS bridge
└── bloom_glue.js                      # imports jolt-physics@1.0.0, hands the
                                       # factory to bloom_physics_init_jolt()
```

## Extending

**Adding a new FFI function** requires 6 edits:

1. `native/third_party/bloom_jolt/include/bloom_jolt.h` — C declaration
2. `native/third_party/bloom_jolt/src/bloom_jolt.cpp` — C++ implementation
3. `native/shared/src/jolt_sys.rs` — Rust `extern "C"` binding
4. `native/shared/src/physics_jolt.rs` — Rust wrapper method + `define_physics_ffi!` macro entry
5. `package.json` — Perry FFI manifest entry (this feeds the web code generator)
6. `src/physics/index.ts` — TypeScript wrapper

For the web target, also add matching code to `native/web/jolt_bridge.js` and
regenerate `native/web/src/lib.rs`'s FFI block from `package.json`.

No platform crates need touching — the `define_physics_ffi!` macro is invoked
once per platform and picks up new entries automatically.

## Why Jolt (and not PhysX or Rapier)

- **Rapier** is a solid open-source option but tops out at rigid-body dynamics;
  no cloth, no vehicles, no character controller with stair climbing.
- **PhysX 5** is excellent but doesn't target WebAssembly — picking it meant
  either losing the web target or maintaining two backends.
- **Jolt** is the engine behind Horizon Forbidden West, actively developed,
  permissively licensed, faster than PhysX on rigid-body benchmarks, and has a
  first-party WASM distribution. UE5 itself switched from PhysX to its own
  Chaos engine; Jolt is the closest third-party equivalent in features + polish.
