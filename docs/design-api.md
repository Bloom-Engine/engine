# API Design — Functions, Not Classes

Bloom's public API is a flat collection of free functions operating on plain-data
interfaces. There are no classes, no inheritance trees, no `this`-bound methods,
and no lifecycle base types to extend. This document records *why*.

The short version: the industry's most-cited performance voices, the architectural
trend in every major engine, and the practical constraints of our Perry FFI all
point the same direction. Classes would be fighting three fights at once.

## The stated rationale (README)

> **Simple API** — Functions, not classes. The entire API fits on a cheatsheet.

That one-liner captures the user-facing benefit. The rest of this document
captures the engineering reasons behind it.

## Three industry camps, one conclusion

Three distinct communities arrived at "fewer classes is better" from different
starting points. The reasoning diverges; the conclusion doesn't.

### 1. The performance / data-oriented camp

Mike Acton (then Engine Director at Insomniac Games, later Unity DOTS) gave the
definitive talk on this at CppCon 2014. The core claim: OOP organizes source
code around *data types* rather than physically grouping fields and arrays for
cache-friendly access. Cache-coherent data layout can yield 10×+ speedups that
class-per-entity models actively work against, because each object scatters its
fields across the heap instead of packing related values contiguously.

Casey Muratori (Handmade Hero, ex-RAD Game Tools) has spent years arguing that
OOP is "a deeply flawed programming methodology." His 2024 talk on the 35-year
history of OOP traces how a specific approach to organizing code "created
decades of unnecessary complexity in software development," and argues for
compression-oriented / data-oriented programming as the alternative.

Jonathan Blow (Jai, Braid, The Witness) has made similar arguments across many
talks: deep class hierarchies are a cost that gameplay code almost never
recovers value from, and the industry has spent too long pretending otherwise.

The shared thesis: **classes optimize for the wrong thing (type identity) at
the cost of the right thing (memory access patterns).**

### 2. The ECS / composition-over-inheritance camp

Every major game engine has, over the last decade, built an escape hatch out of
its own class-based foundation:

- **Unity** shipped **DOTS** (Data-Oriented Technology Stack) with ECS, Burst,
  and Jobs — explicitly because `MonoBehaviour` hits a wall at scale. Engine
  overhead checking every component for `Update()` each frame, no native
  multithreading, and inheritance breaking Unity's message dispatch (only the
  most-derived class receives messages) are all well-documented pain points.
- **Unreal** shipped **Mass**, an ECS-style framework for large-scale
  simulations, alongside its existing `UObject`/`AActor` hierarchy. The
  `UCLASS`/`GENERATED_BODY()` macro boilerplate, the no-multiple-inheritance
  rule, and the pressure to prefer composition-via-members-of-`UObject`-types
  have been recurring community complaints for years.
- **Bevy**, **Amethyst**, and most new Rust-based engines are **ECS from the
  ground up** — no OOP layer to escape in the first place.

The shared thesis: **deep `Player → Character → Entity` trees become
unmaintainable at scale, and components-as-data is the escape hatch.** ECS is
the de-facto consensus architecture for any engine that needs to simulate many
entities efficiently.

### 3. The simplicity / library-design camp

Raylib (the library whose API shape Bloom most directly echoes) is a flat C99
function API designed to be "learned just from a cheatsheet." Its design notes
explicitly emphasize:

- **Accessibility** — no OOP vocabulary needed to start.
- **Portability** — plain C binds cleanly to 60+ other languages; a class API
  does not.
- **Opt-in abstraction** — a separate `raylib-cpp` wrapper exists for users who
  want OOP on top. The core stays functional.

The shared thesis: **a flat function surface is the smallest possible learning
target and the most portable foundation.**

## How Bloom compares to Unreal and Unity

| | **Unreal** (`UObject` / `AActor`) | **Unity** (`MonoBehaviour`) | **Bloom** |
|---|---|---|---|
| Base model | Deep `UObject` inheritance tree, `UCLASS` + `GENERATED_BODY()` macros | Inherit `MonoBehaviour`; engine reflects `Update`/`Start`/etc. per frame | Plain interfaces (`Vec2`, `Texture`, `Sound` as data handles) + free functions |
| Typical complaints | Macro boilerplate, no multiple inheritance, composition encouraged but inheritance structurally required | Per-frame method-lookup overhead, inheritance breaks Unity messages, no native multithreading | — |
| Escape hatch shipped | **Mass** (ECS) for large-scale simulation | **DOTS / ECS / Burst / Jobs** — a whole parallel stack | N/A — started where they're migrating to |
| Language binding | C++ only | C# only | Compiles via Perry to every target; functions map 1:1 to `bloom_*` C ABI |

The observation: both giants have spent years building ECS escape hatches *from
their own class-based foundations.* Bloom skipping classes isn't a contrarian
aesthetic call — it's where the industry has been migrating for a decade.

## The Bloom-specific reason: the Perry FFI boundary

Bloom compiles TypeScript through [Perry](../../perry/perry) (our AOT compiler)
and hands data across an FFI boundary to platform-specific Rust crates. The
boundary has a specific shape, documented in `CLAUDE.md` and `package.json`:

- **~465 `bloom_*` FFI functions** declared in `package.json` under
  `perry.nativeLibrary.functions`.
- **Native platforms** use `#[no_mangle] extern "C"` — a C ABI.
- **Web** uses `#[wasm_bindgen]`; Perry's runtime decodes NaN-boxed args
  (`wrapFfiForI64`) and the JS glue routes strings to `_str` variants.
- **String parameters** are `i64` Perry StringHeader pointers on native, NaN-
  boxed IDs on web.
- **Handles** (textures, sounds, models, physics bodies) are all `i64` / plain
  integers, indexing into per-subsystem registries on the Rust side (e.g.
  `physics_jolt.rs`'s handle registries).

This shape is fundamentally hostile to a class API and fundamentally friendly
to a function API:

- **Free functions map 1:1 to C ABI entries.** A call like
  `drawText(text, x, y, size, color)` is exactly one FFI function with scalar
  arguments. There is no hidden receiver, no vtable, no `this`.
- **Plain-data interfaces map 1:1 to FFI argument lists.** `Texture` is a
  handle plus width/height; it can be passed through the boundary by value or
  reconstructed from a handle. A class with methods would require Perry to
  model vtables across the FFI, which neither the native C ABI nor
  `wasm_bindgen` does cleanly.
- **Handle-based identity is already how the engine is structured.** Every
  subsystem already stores its real state in a Rust-side registry keyed by an
  integer handle — that's the natural representation for a resource owned by
  native code and referenced from TypeScript. Classes on the TS side would be
  a thin vanity layer that still has to hand back to an integer handle at
  every FFI call.

In short: the FFI already forces the engine to be data-oriented on the Rust
side. Making the TypeScript surface match that shape keeps the whole stack
coherent. Wrapping it in classes would add a layer of abstraction that every
single FFI call has to punch back through.

## What this looks like in practice

```typescript
// Data:
interface Vec3    { x: number; y: number; z: number }
interface Texture { handle: number; width: number; height: number }
interface Sound   { handle: number }
interface Model   { handle: number; meshCount: number; materialCount: number; transform: Mat4 }

// Functions operate on data:
const tex  = loadTexture("assets/hero.png");
const snd  = loadSound("assets/jump.wav");
drawTexture(tex, 100, 200, Colors.WHITE);
playSound(snd);
unloadTexture(tex);
```

No `new Texture(...)`. No `hero.draw()`. No `class Enemy extends Entity`. Game
state lives in plain objects; behavior lives in functions that read and mutate
that state. This is the same shape you'd write in C, in Jai, in a Rust ECS, or
in Raylib — and it's the shape Unity and Unreal users drop into whenever they
hit the performance ceiling of the OOP layer.

## What we give up

This section is deliberately here to keep the doc honest.

- **No polymorphism via method dispatch.** If you want different enemy types to
  "update" differently, you dispatch on an enum/tag, not a virtual method.
  This is a feature (all behavior is visible, inspectable, and data-driven)
  but it is a tradeoff.
- **No RAII for engine resources.** Textures, sounds, and models must be
  explicitly unloaded. TypeScript has no destructors, and the FFI boundary
  would not respect them even if it did.
- **No "smart" object APIs that discover methods via IDE autocomplete.** You
  navigate by module (`bloom/textures`, `bloom/audio`) and function name.
  The [cheatsheet](../README.md#modules) is the map.

We've judged these acceptable — and in several cases desirable — given the
performance, portability, and simplicity benefits above.

## References

Performance / data-oriented design:
- [Data-Oriented Design and C++ — Mike Acton, CppCon 2014](https://neil3d.github.io/assets/img/ecs/DOD-Cpp.pdf)
- [Data-oriented design — Wikipedia](https://en.wikipedia.org/wiki/Data-oriented_design)
- [Developing a Data-Oriented Game Engine — Daniel Sefton](https://danielsefton.com/2016/05/developing-a-data-oriented-game-engine-part-1/)
- [The Downfall of Object-Oriented Programming — Casey Muratori](https://gist.ly/youtube-summarizer/the-downfall-of-object-oriented-programming-with-casey-muratori)
- [Programming Community Debates 35-Year OOP Mistake — BigGo News](https://biggo.com/news/202507241923_Programming_Community_Debates_OOP_Mistake)
- [Casey Muratori on OOP — Alejandro M. P.](https://alejandromp.com/development/blog/casey-muratori-about-oop/)
- [Why are we not using Object Oriented Programming? — Handmade Network](https://hero.handmade.network/forums/code-discussion/t/209-why_are_we_not_using_object_oriented_programming)

ECS and composition over inheritance:
- [Entity component system — Wikipedia](https://en.wikipedia.org/wiki/Entity_component_system)
- [Nomad Game Engine Part 2: ECS — Down with inheritance!](https://medium.com/@savas/nomad-game-engine-part-2-ecs-9132829188e5)
- [ECS 1: Inheritance vs Composition — LeatherBee Games](https://leatherbee.org/index.php/2019/09/12/ecs-1-inheritance-vs-composition-and-ecs-background/)

Unity:
- [The Constraints of MonoBehaviour — Roydon, Medium](https://medium.com/@roystharayil/the-constraints-of-monobehaviour-analyzing-its-impact-on-unity-development-9973d9087765)
- [When 100 Enemies Brought My Game to Its Knees (Unity ECS) — Outscal](https://outscal.com/blog/entity-component-system-csharp-guide)

Unreal:
- [Why does Unreal Engine use inheritance? — Epic Developer Community Forums](https://forums.unrealengine.com/t/why-does-unreal-engine-use-inheritance/253750)

Raylib (API-shape prior art):
- [raylib — GitHub](https://github.com/raysan5/raylib)
- [raylib — Wikipedia](https://en.wikipedia.org/wiki/Raylib)
