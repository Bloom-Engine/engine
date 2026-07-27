# Compiled render graph

Bloom's retained 3D renderer executes from an immutable, cached frame plan.
The implementation deliberately preserves the established serial command
order and rendered pixels while making resource lifetime and pass contracts
explicit.

## Runtime contract

`renderer/graph/model.rs` is the declaration API. Textures and buffers use
typed logical handles; each write produces a new version. Imported persistent
and external resources declare their initial/final usage and ownership.
Transient descriptors include extent policy, format, dimension, mip/sample
counts, allowed usage, load policy, and alias class.

`renderer/graph/compiler.rs`:

- validates producers, versions, usage permissions, typed handles, divergent
  writers, and cycles before wgpu validation;
- derives deterministic resource and explicit-dependency edges;
- topologically sorts by declaration ID when multiple orders are valid;
- computes inclusive first/last-use positions and usage/queue transitions;
- assigns compatible, strictly non-overlapping transient lifetimes to
  physical slots; and
- generates a stable plan ID from the complete execution contract.

The live topology is declared in `renderer/graph/frame_plan.rs`. Optional
SSAO, SSR, SSGI (including acceleration/card/SDF/radiance-cache preparation),
bloom, scene-snapshot, and capture work is selected before compilation by
`FramePlanKey`. Uniform-only changes do not affect the key.
`ExecutableGraph` binds only frame-local recording closures to cached pass
positions; it does not rebuild or schedule a declaration graph per frame.

Current backends execute the compiled order on one serial wgpu queue. Queue
capability and transitions are diagnostic metadata, not a second barrier or
multi-queue API.

## Allocation and resize

`TransientPool::prepare_compiled_plan` caches physical texture allocations by
plan ID and exact render/output extents. A stable plan and extent performs no
new physical allocation. An exact resize invalidates the allocation
generation. Persistent resources and temporal histories never enter the
alias set.

The current Ultra Sponza qualification topology has no eligible transient
texture: its post-FX targets are persistent imports with stable views or
temporal/history ownership. It therefore reports 0 unaliased bytes, 0
physical slots, and no meaningful percentage reduction.

The only currently materialized optional transients are the refractive
material's scene-color and scene-depth snapshots. Their lifetimes overlap in
the translucent pass and their exact descriptors differ (`Rgba16Float`
color versus `Depth32Float` depth), so conservative aliasing correctly keeps
two slots. At 1920×1080 they total 24,883,200 bytes and save 0%; at the
800×450 Sponza qualification extent they would total 4,320,000 bytes.
Converting persistent post-FX targets would require a broader view/bind-group
rebinding migration and was excluded from this no-regression change.

Synthetic allocator tests cover the positive case and reduce two compatible,
non-overlapping textures from two physical allocations to one (50%). They
also prove that overlapping or descriptor-incompatible resources never
alias.

## Capture and diagnostics

Set `BLOOM_GRAPH_DEBUG_MARKERS=1` while taking a GPU capture to bracket every
compiled pass with its stable graph name using wgpu debug groups. The opt-in
keeps ordinary release frames free of marker encoding overhead while making
passes visible in RenderDoc, Xcode GPU captures, and PIX-compatible backends.
Qualification readback is a keyed terminal `capture_readback` pass.
It copies the output plus these logical resource names:

- `hdr-scene`
- `scene-depth`
- `shadow-cascade-0`
- `shadow-cascade-1`
- `shadow-cascade-2`

The capture path resolves names explicitly and rejects unknown resources.
Normal frames do not include the capture pass or allocate staging buffers.

Set `BLOOM_GRAPH_DUMP_DIR` to write each distinct plan once:

```sh
BLOOM_GRAPH_DEBUG_MARKERS=1 \
BLOOM_GRAPH_DUMP_DIR=/tmp/bloom-graphs \
  python3 tools/quality/run.py run quick --report-only
```

Each `bloom-frame-<plan-id>.json` contains pass accesses, dependencies,
resource lifetimes, physical slots, and transitions. The adjacent DOT file
is a compact dependency graph.

Native qualification telemetry exposes:

- `compile_count`, `cache_hit_count`, and `cached_plan_count`;
- the current `plan_id` and `pass_count`;
- aliasing state, planned/unaliased transient bytes, and physical slot count.

A 420-frame high-quality PBR run records one compile and 419 cache hits for
the stable normal-frame configuration. The post-measurement capture uses a
separate cached topology; telemetry is intentionally serialized before that
debug-only frame.

## Regression procedure

Run compiler and allocator tests first:

```sh
cd native/shared
cargo test --release renderer::graph
cargo test --release renderer::transient
```

Then run the report-only quality corpus and compare final plus intermediate
outputs to the pre-change evidence. A missing human-approved baseline remains
a qualification failure; `--report-only` does not convert it into a pass.
