# Morph targets (blend shapes)

glTF morph targets on skinned characters, for facial animation (lip sync, blinks, gaze, expressions)
and anything else a rig drives by shapes. EN-074 in `tickets.md`.

## The API (`@bloomengine/engine/models`)

```ts
modelMorphCount(model: Model): number                 // distinct target names of the model
modelMorphIndex(model: Model, name: string): number   // slot of a name, -1 if absent (load time only)
animSetMorphWeight(handle: number, index: number, weight: number): void
animSetMorphWeights(handle: number, first: number, weights: number[]): void  // consecutive slots
animClearMorphs(handle: number): void                 // drop the weights set by the game
animMorphWeight(handle: number, index: number): number // the weight drawn (clips + set, clamped)
```

A frame of a talking face:

```ts
// at load
const face = loadModel("assets/jonah.glb");
const anim = instantiateAnimation(loadModelAnimation("assets/jonah.glb"));
const JAW = modelMorphIndex(face, "jawOpen");
const BLINK_L = modelMorphIndex(face, "eyeBlinkLeft");
// per frame, BEFORE animUpdate
animSetMorphWeight(anim, JAW, jaw);
animSetMorphWeight(anim, BLINK_L, blink);
animUpdate(anim, dt, scale, x, y, z, yaw);
drawModel(face, pos, scale, WHITE);
```

- **Slots are model-level, by name.** A model's slots are the distinct target names
  (`mesh.extras.targetNames`, as Blender writes them) over all its meshes and primitives, in the
  order they first appear in the file. A character whose body mesh and five face primitives (skin,
  mouth, teeth, eyes, lashes) all carry `jawOpen` has ONE `jawOpen` slot, and its weight moves all
  six. An unnamed target is called `"<mesh name or index>#<n>"`. At most 1024 slots.
- **Weights belong to the animation instance** (`instantiateAnimation`), like its pose: N
  characters sharing one model each have their own face. They are drawn with the pose of the next
  `animUpdate` / `animUpdateMatrix` / `updateModelAnimation` (and ragdoll update), so set them before
  it. A weight set stays until set again or `animClearMorphs`.
- **Clips can carry weights too.** glTF `weights` animation channels are read into the clips
  (LINEAR and STEP; CUBICSPLINE through its key values) and the mixer blends them exactly like
  joints: the crossfade from the previous clip, blend partners at the base clip's phase, the masked
  layer by its weight (a layer changes only the slots it animates). A slot a clip does not animate
  counts as 0 there.
- **What is drawn is `clamp(clip + set, -1, 1)`**: the game's weights ADD to the clips' (a lip-sync
  curve over an authored expression clip). `animMorphWeight` returns that.
- **Only skinned models morph** (the weights travel with the pose; a model without a skin logs one
  warning and draws its rest shape). The immediate-mode mesh path and custom-material draws
  (`drawModelWithMaterial`) do not morph.
- The model and its animation should come from the same file (or files with the same target names
  in the same order): clip tracks are numbered by the animation file's name table.

## Loading

All three glTF loaders (`load_gltf_with_textures`, `load_gltf_staged_impl`, `load_gltf`) read every
primitive's POSITION, NORMAL and TANGENT deltas (`models_gltf_morph.rs`). Accessors go through the
gltf crate's iterators, which resolve **sparse accessors** (Blender's `export_try_sparse_sk`), with
or without a base buffer view. Skinned vertices' deltas are scaled by the loader's armature
compensation (`skin_vertex_scale`) as their positions are; deltas on rigid vertices of a skinned
primitive are baked through the node transform (`models_gltf_bake.rs`) as those vertices are.

Deltas whose components are all below 1e-6 are dropped, so a target that moves nothing on a
primitive (an exporter writes every target on every primitive) costs nothing there.

Export normals with the shape keys (Blender: *Shape Key Normals*): without normal deltas a moving
mouth keeps its rest normals. Tangent deltas are used when present.

## Data layout (`morph_targets.rs`)

`MeshData::morph: Option<Arc<MorphTargets>>`, shared by every placement and instance of the mesh:

- `ranges`: per vertex, its range of entries (CSR, `vertex_count + 1` u32). An untouched vertex costs
  4 bytes.
- `entries`: one target's delta on one vertex, 24 bytes: the slot (u32) and position, normal and
  tangent deltas as half floats. Positions are divided by the mesh's largest position delta
  component (`pos_scale`) so they always lie in [-1, 1]: no overflow in any unit, and a precision
  of 2^-11 of the largest delta (10 µm on a 2 cm shape).
- `bound_growth`: per axis, the most any vertex can move with every weight in [-1, 1]. The cached
  mesh's rest bounds grow by it, so culling and shadow casting never lose an open jaw.

A face mesh of 50k vertices with 12k touched by 56 targets (about 20 targets per touched vertex) is
about 200k entries: 5 MB on the CPU and 5 MB on the GPU, against about 30 MB for dense f32 deltas.

## On the GPU (`renderer/morph_pass.rs`)

A skinned cached model keeps one bind-pose vertex buffer shared by all its instances. An instance
whose weights are not all zero gets its **own vertex buffer per morphed mesh**, which a compute pass
rewrites each frame it is drawn: bind pose + Σ wᵢ·Δᵢ for positions; normals and tangents get their
deltas and are renormalised (a zero "no tangent" sentinel stays zero). One thread per vertex; the
entries of each block of 64 vertices are re-laid lane-interleaved at upload (entry k of lane i at
`first + 64k + i`, padded to the block's busiest vertex) so a wave reads them coalesced. Untouched
vertices are copied once, when the buffer is made. Entries whose weight is zero now and was zero
last frame are skipped.

That buffer replaces the shared one as the draw's vertex buffer at every site that draws cached
model commands (`MorphPass::morph_vertex`: main and depth passes, shadows, transmitted shadows,
layered-PBR SSR, refraction, sorted transparency) and as the source of the ray scene's compute skin
(`pt_dynamic_skin.rs`). The vertex shaders' current-position paths are unchanged.

**Motion vectors.** The pass knows last frame's weights (kept per animation handle, like the joint
palettes), so it also writes `previous − current` morphed position of every vertex of the mesh into
one frame-wide Rgba32Float texture, the *morph arena*, bound at `@group(3) @binding(2)` in every
joint bind group. A morphed draw's uniform `misc.w` is `1 + its first texel`; the skinned branch of
`vs_main_scene` adds the displacement to the position it feeds the previous palette. TAA and TSR get
the real velocity of a moving lip (`golden_render/morph_targets.rs` checks that a morph alone
writes velocity and leaves no trail, and that without the history it would not). `misc.w = 0` (every
other draw) reads nothing.

**Pose plumbing.** The morph state travels with the pose: `set_joint_matrices_*` opens a slot in a
queue that runs in step with `pending_skin_groups`, `stage_morph_weights` (called by every posing
FFI right after it) fills it, and `take_staged_skin_offset` pops it for the draw. A draw finds its
morphed buffers by its joint offset, which is unique per posed instance in a frame.

**Costs nothing when unused:** no buffer, job or texel for models without morphs or instances at
zero. The frame an instance returns to zero is morphed once more (its velocity is not zero that
frame); then it draws from the shared buffer again. Its own buffers are freed after 240 frames
unused, or when the model is unloaded.

**Without compute** (the WebGL2 baseline tier) the same is done on the CPU from the cached
bind-pose vertices: the instance buffer and the arena are uploaded each frame. `BLOOM_MORPH_CPU=1`
or `Renderer::set_morph_cpu_fallback(true)` forces it; the golden test compares the two.

## Cost

Measured with `bench_morph_pass_eight_faces` (`cargo test --release --test golden_render
bench_morph -- --ignored --nocapture`, timestamp queries around the pass): 8 characters of 50,176
vertices, 12,000 touched by 56 targets (20 per touched vertex), every weight non-zero and changing
each frame, M1 Max: **0.59 ms GPU for all eight** (0.07 ms a face). With one target per touched
vertex 0.27 ms; with nothing touched 0.13 ms (the arena texels of 400k vertices). Real lip sync
has far fewer non-zero weights than the bench, and zero entries are skipped.

Memory per morphed instance: one vertex buffer of the morphed mesh (96 B a vertex: 4.8 MB for 50k
vertices) and 16 B a vertex of arena while drawn.

## Limits

- Skinned models only (see above); no morphs on immediate meshes or custom-material draws.
- Cooked scenes (`.bscene`) do not hold morphs: `bloom-cook` refuses a source with morph targets
  rather than drop them. `bloom-cook glb-compress` keeps them (it only re-encodes images).
- Two draws of the same animation instance in one frame share its morphed buffers (the last
  weights staged win).
- At most 16M morphed vertices in a frame (`misc.w` is an f32), 1024 slots per model.
- Rigid (unweighted) vertices of a morphed skinned mesh get morph velocity through `u.model`;
  their position comes from the morphed buffer like any other.
