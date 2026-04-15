# Bloom Engine Renderer — v2 Spec (UE5-Tier Target)

**Supersedes (in ambition):** `bloom-renderer-spec.md` — the 12-month "2019 AAA" plan. That document is still correct about the wedge and the taste bottleneck; read §4 of it before this one.

**Target:** As close to UE5 (Lumen + Nanite + VSM + Substrate + TSR) as a small orchestrated-agent team can pragmatically reach. The goal is *not* to beat UE5. The goal is that a technical art director looking at Bloom screenshots cannot name a specific rendering subsystem where Bloom is obviously behind modern AAA.

**Constraint reframe:** Coding agents write nearly every line. The bottlenecks are spec discipline, integration architecture, and human taste review — not engineering hours. So scope is bounded by what can be *specified and reviewed*, not what can be typed. A 24–30 month plan is acceptable. Slipping is acceptable. Shipping 85% of this is a win.

**What we still won't build (v1):** MetaHuman-tier face rigs (animation problem, not rendering), Chaos-class destruction (physics), node-graph material editor (Bloom's TS API *is* the authoring layer — that's the wedge).

---

## 0. Architectural Ground Rules — Locked Day One

Retrofitting any of these costs months. Lock them before any agent writes renderer code.

- **Graphics APIs:** Vulkan + Metal + D3D12 from day one. Agents can parallelize backends; the cost of adding D3D12 later is much higher than building it alongside.
- **Bindless resource model everywhere.** Descriptor heaps / argument buffers / bindless Vulkan from the first commit. No per-draw descriptor set binding. This is the entry ticket for GPU-driven rendering, virtual textures, and ray tracing.
- **GPU-driven rendering.** Frustum + occlusion culling on GPU, LOD selection on GPU, indirect draw submission. CPU submits work lists, not draws.
- **Mesh shaders as the primary geometry path**, with a vertex-shader fallback for non-mesh-shader hardware. Required for virtualized geometry later.
- **Acceleration structure abstraction from day one**, even before we use ray tracing. BLAS/TLAS management, build/refit scheduling, shader binding tables. "RT-ready" is cheap if you bake it in; expensive if you don't.
- **Render graph** with automatic barriers, resource aliasing, async compute scheduling, and multi-queue support. Reference: Granite, Themaister's articles, Frostbite FrameGraph.
- **Hybrid render architecture:** visibility buffer for opaque geometry (this is the path that makes Nanite-equivalent work), clustered forward+ for transparency and MSAA-sensitive passes. Both paths live in the same render graph.
- **Shader language:** Slang. It cross-compiles to HLSL/MSL/SPIR-V/WGSL, has real generics, and Microsoft is backing it. Alternative: HLSL with DXC. Do not write three shader versions by hand.
- **Color pipeline:** linear lighting throughout, sRGB at output, physical EV exposure, ACES + AgX selectable. Locked day one. Non-negotiable.
- **Material system:** Substrate-style layered slabs from day one. *Do not* add clearcoat/sheen/aniso/SSS as discrete BRDF variants later — that path leads to a permutation explosion and a painful rewrite. Build the layered system first and make the "simple PBR" material a trivial single-slab case.
- **Asset import:** glTF 2.0 with the full KHR_materials_* extension set. Custom shipped format deferred to the streaming phase.

---

## 1. Track-Based Phase Plan

The original spec ran phases sequentially because a human was writing it. With agent parallelism, we run *tracks* in parallel with explicit sync points. Each track is specced, implemented by agents on isolated branches, integrated on weekly human-review days.

### The Tracks

| Track | Owns | Parallelism unlock |
|---|---|---|
| **Core** | API abstraction, render graph, shader system, resource streaming | Everything depends on this — must be stable by end of Phase A |
| **Geometry** | Visibility buffer, GPU-driven draws, virtualized geometry, LOD, HLOD, impostors | Independent of lighting tracks once V-buffer format is locked |
| **Shadows** | CSM → VSM, shadow caching, contact shadows | Independent once depth/normal formats are locked |
| **GI** | Baked lightmaps → DDGI → SDF GI → surface cache → HW RT GI | The longest and riskiest track — starts early, layers upward |
| **Reflections** | Cubemaps → SSR → RT reflections | Layers on GI track infrastructure |
| **Temporal** | TAA → TSR-equivalent → DLSS/FSR integration | Dependencies: motion vectors, jittered projection |
| **Materials** | Substrate slabs → strand hair → skin → eyes → water → foliage → decals | Independent of lighting once BRDF API is locked |
| **Post** | Tone mapping → bloom → DoF → motion blur → grading | Independent of everything else |
| **Atmosphere** | Volumetric fog → Hillaire sky → clouds → aerial perspective | Independent; integrates via fog injection and sky IBL |
| **Streaming** | Virtual textures, world partition, HLOD streaming | Depends on bindless and GPU-driven; independent otherwise |
| **VFX** | Particle rendering, GPU sim, ribbons, beams, mesh particles | Independent once material system is stable |
| **Reference** | CPU path tracer, screenshot harness, regression system | Tooling — runs throughout, feeds every acceptance gate |

### Phase A — Foundation (Months 1–4)

**Goal:** Core track stable enough that every other track can start in parallel. Nothing that ships visually; everything that enables scale.

- **Core:** API abstraction (Vulkan + Metal + D3D12), bindless descriptor model, render graph with async compute, Slang shader pipeline, resource streaming skeleton, timeline semaphores / barriers.
- **Reference:** CPU path tracer (Embree-based) producing ground-truth images from glTF scenes. This is the acceptance gate for every later phase — if the path tracer says your GI looks like X, your realtime GI should look like roughly X.
- **Core:** glTF 2.0 importer with KHR_materials_clearcoat, _sheen, _ior, _specular, _transmission, _volume, _anisotropy. Import into Substrate slab representation.

**Exit criteria:** Three backends render the same test scene pixel-identical (tolerance TBD). Render graph schedules an async compute pass correctly. Path tracer produces reference images for the test scene library. Zero visual features shipped yet — that's fine.

### Phase B — Parallel Launch (Months 4–10)

All tracks start. Each produces an initial-but-working version of its subsystem. Expect this phase to be chaotic. Weekly integration days.

- **Geometry:** Visibility buffer opaque path, GPU-driven culling + LOD, mesh preprocessing pipeline (meshoptimizer-based meshlet generation, DAG-based LOD hierarchy ready for virtualized geometry later), HLOD support, basic impostor generation.
- **Shadows:** CSM with stable fitting + PCF Vogel disk filtering, shadow caching for static casters, contact shadows, GTAO.
- **GI (initial):** Offline lightmap baker (path tracer from Reference track, reused), xatlas for UV generation, Open Image Denoise integration, SH9 light probe volumes for dynamic objects, parallax-corrected cubemap reflection probes.
- **Reflections (initial):** Hierarchical Z, SSR with importance sampling for rough surfaces, cubemap blending.
- **Temporal:** Motion vector buffer (including skinned meshes — coordinate with animation system), jittered projection, TAA with neighborhood clipping and disocclusion handling. **Budget 2 months of iteration.** This is the pacing item for all downstream temporal effects.
- **Materials:** Substrate slab evaluator (metal, dielectric, clearcoat, sheen, aniso, thin-film, subsurface as composable layers), energy conservation across layers, multi-scatter compensation.
- **Post:** ACES + AgX tone mapping, histogram auto-exposure, physical bloom (CoD downsample/upsample), vignette, chromatic aberration, film grain, lens dirt.
- **Atmosphere:** Froxel volumetric fog with full light injection (coordinate with light culling), Hillaire sky with precomputed LUTs, aerial perspective, time-of-day system.
- **Streaming:** Virtual texture system foundation (feedback pass, page table, indirection texture, upload path).
- **VFX:** Particle renderer with soft particles, lit particles, GPU simulation backend, ribbon and mesh particle types.

**Exit criteria for Phase B:** End of month 10, you can render a Sponza/Bistro/San Miguel test scene with baked GI, TAA, volumetrics, sky, SSR, and substrate materials, at 1440p 60fps on an RTX 3060. Screenshot diffs against the CPU path tracer are within "close but clearly not dynamic GI" tolerance. This is already ~2019 AAA quality — it's the *foundation* for the UE5-tier work, not the destination.

### Phase C — Virtualized Geometry (Months 8–14, parallel with B tail and D)

**This is the single biggest rock.** Start it as soon as the Geometry track's V-buffer path is stable — late Phase B. Expect 6 months of agent work.

- Meshlet cluster DAG construction offline. Reference: Nanite SIGGRAPH 2021 paper, Bevy's `bevy_pbr/meshlet` module, meshoptimizer, zeux/meshoptimizer cluster building.
- GPU cluster culling: two-pass hi-Z occlusion, cluster-level frustum and backface culling.
- GPU LOD selection: error-metric-driven, DAG traversal on GPU.
- Software rasterizer for sub-pixel triangles (compute shader rasterization). Reference: Nanite paper §5.
- Streaming: cluster pages from disk, LRU residency, page fault handling.
- Impostor fallback for beyond-streaming-range geometry.
- Integration with V-buffer and material resolve pass.

**Exit criteria:** A scene with 10M+ source triangles renders at 60fps, with LOD transitions that are not visible to the eye. Memory budget stays under a configurable cap. Streaming holds up under fast camera motion.

**Fallback if it slips:** Ship the GPU-driven meshlet renderer without the DAG/software-rasterizer Nanite-equivalent parts. You still get the UE4-era "aggressive mesh culling + mesh shaders" win. Layer full Nanite-equivalent as v1.1.

### Phase D — Virtual Shadow Maps + RT Shadows (Months 10–14, parallel with C)

- Page-based VSM: conceptual 16k×16k per light, sparsely resident, page allocation driven by screen-space derivatives.
- Cached pages for static geometry; invalidation on caster movement.
- HW ray-traced shadows as an alternative path, selectable per-light.
- Contact shadows remain as a near-field layer.

**Exit criteria:** Crisp shadows at any distance in outdoor scenes, no cascade popping, hundreds of shadow-casting lights in indoor scenes without the atlas falling apart.

### Phase E — Dynamic GI (Months 12–20)

**The second biggest rock, and the one most likely to slip.** Start with the most pragmatic approach, layer quality upward.

The plan is staged — each stage ships a working dynamic GI system, each adds quality:

**E1 — DDGI + RT Reflections (months 12–15).** Probe grid with RT updates on capable hardware, screen-space near-field, RT reflections for smooth surfaces, SSR fallback for rough. This alone is already better than the baked path. Reference: Majercik et al. DDGI papers, NVIDIA RTXGI SDK.

**E2 — Global SDF + mesh SDFs (months 14–17).** Build and maintain a global distance field of the scene (top-down voxelization + mesh SDFs per object). Used for software-traced GI when RT isn't available, and for soft shadows, particle collision, and cloud shadows. Reference: UE's Distance Field Global Illumination, Claybook GDC talk.

**E3 — Surface cache (months 16–20).** Card-based per-mesh surface parameterization, lit and cached each frame from GI results. This is the Lumen core insight and the thing that makes dynamic GI fast enough to ship. Reference: Lumen SIGGRAPH 2022 course.

**E4 — Radiance caching + final gather (months 18–20).** World-space radiance cache with temporal reuse, final gather per pixel for high-quality indirect. Integrates surface cache + SDF trace + RT trace as three trace backends behind a common interface.

**Exit criteria:** Indoor scene with fully dynamic bounce lighting that matches the CPU path tracer within reasonable tolerance. Moving lights produce correct bounce. Opening a door changes indirect lighting in an adjacent room.

**Fallback if it slips:** Ship E1 + E2 as "Bloom Dynamic GI v1." E3 and E4 become v1.1. Baked lightmaps stay available for scenes that want maximum quality.

### Phase F — TSR-Equivalent Upscaling + DLSS Integration (Months 16–19)

- Custom temporal super-resolution taking material hints (roughness, depth, motion, reactive mask). Reference: UE TSR whitepaper, AMD FSR3 implementation.
- DLSS integration via NVIDIA SDK for NV hardware.
- FSR2/3 integration via AMD SDK for everywhere else.
- The three are selectable; custom TSR is the default and the fallback.

**Exit criteria:** 1080p → 4K upscale is visually indistinguishable from native 4K on still frames and holds up on motion. Thin geometry (foliage, wires) doesn't disintegrate.

### Phase G — Virtual Textures + World Partition + Streaming (Months 18–24)

- Runtime virtual texturing: feedback pass, LRU page cache, async upload, indirection textures.
- Virtual shadow maps integrate with the VT page management (shared infrastructure).
- World partition: cell-based scene streaming, HLOD layers, data layers, async load and unload.
- Texture streaming budget control and quality scaling.

**Exit criteria:** A scene 10x the size of Bistro streams smoothly during camera flythrough, no visible LOD popping, texture budget stays under cap.

### Phase H — Advanced Materials + Characters (Months 20–26)

Parallel subtracks, each independent once the substrate slab system is frozen.

- **Strand hair:** Marschner BCSDF, hair LOD (strands → cards → mesh), hair self-shadowing via deep opacity maps or per-strand shadows, transmittance. Reference: Epic's strand hair talks, TressFX.
- **Advanced skin:** Jimenez separable SSS + dual specular lobe + transmission + wetness + micro-detail normal. Reference: Jimenez SIGGRAPH courses, Penner preintegrated SSS.
- **Eyes:** refraction + caustics + iris parallax + wet meniscus. Reference: Penner eye shader, UE digital human talks.
- **Water:** FFT wave simulation, tessellated surface, SSR + RT refl fallback, refraction, caustics, foam, wet-shore blending.
- **Foliage:** two-sided lighting, wind animation (vertex-shader based), alpha-to-coverage with TAA, SH ambient tinting.
- **Deferred decals:** screen-space projection, compatible with V-buffer pipeline (decal evaluation in material resolve).

**Exit criteria:** A character model with hair, skin, eyes, and clothing renders at parity with a reference UE5 character, side by side, on the same lighting.

### Phase I — VFX System (Months 24–28)

- GPU particle simulation with SDF collision (reuses Phase E SDFs), forces, curl noise, emission from mesh surfaces.
- Lit particles (per-particle GI sampling from radiance cache).
- Mesh particles, ribbons, beams.
- Soft and refractive particles.
- Not shipping: a Niagara-class node graph authoring UI. Bloom's TS API is the authoring layer; agents write the shaders.

**Exit criteria:** A smoke plume, muzzle flash, fluid splash, and sparks effect all render convincingly in the Bloom test scenes.

### Phase J — Tuning + Reference Parity + Profiling (Months 26–30)

- Dedicated human-driven tuning month against AAA reference screenshots. This is non-delegable.
- GPU profiler and frame debugger integration (RenderDoc-compatible markers, in-engine GPU timing view, pipeline stall detection).
- Material validation tests against path tracer for all substrate layer combinations.
- Performance regression suite covering all tracks.
- Final pass on tone mapping curves, bloom response, exposure adaptation, AO darkness, VT/shadow budgets.

**Exit criteria:** Blind-test screenshots against UE5.3 reference scenes. A technical art director should not be able to reliably pick out Bloom vs. UE5 on a rendering-quality basis alone.

---

## 2. Dependency Graph (What Blocks What)

```
Core (A) ─┬─> Geometry ────> V-buffer ──> Virt. Geometry (C)
          ├─> Shadows ─────> CSM ───────> VSM (D)
          ├─> GI ──────────> Baked ─────> DDGI (E1) ──> SDF GI (E2) ──> Surface Cache (E3) ──> Radiance Cache (E4)
          ├─> Reflections ─> Cubemaps ──> SSR ────────> RT Reflections
          ├─> Temporal ────> TAA ────────────────────> TSR (F) ──> DLSS/FSR
          ├─> Materials ───> Substrate ────────────> Hair/Skin/Eyes/Water (H)
          ├─> Post ────────> Tone/Bloom/DoF/MB (runs independently throughout)
          ├─> Atmosphere ──> Fog/Sky/Clouds (runs independently throughout)
          ├─> Streaming ───> VT/WorldPart (G)
          └─> VFX ─────────> Particles/GPU Sim (I)

Reference (path tracer) runs continuously as acceptance gate for all tracks.
```

Parallelism unlocks after Phase A (month 4). Phases C (virtualized geometry) and E (dynamic GI) are the longest poles and run concurrently with most other work.

---

## 3. Agent Orchestration — What Makes This Actually Work

The 12-month spec was right about this and v2 doesn't change the principles. It scales them up.

**Spec-per-subsystem.** Every track entry in Phase B–J needs its own multi-page spec *before agents touch it*. Scope, reference implementations, data formats, acceptance criteria, test scenes. Vague specs produce plausible-looking wrong code. This is the single highest-leverage thing you do as orchestrator.

**Named references per subsystem.** Agents working from real implementations produce code an order of magnitude better than agents working from descriptions. Canonical references:

| Subsystem | Reference |
|---|---|
| Render graph | Granite (Themaister), Frostbite FrameGraph GDC |
| Virtualized geometry | Nanite SIGGRAPH 2021, Bevy meshlet renderer |
| VSM | UE5 VSM whitepapers, Bungie shadow talks |
| DDGI | Majercik et al. papers, NVIDIA RTXGI |
| SDF GI | Claybook GDC, UE DFGI docs |
| Surface cache | Lumen SIGGRAPH 2022 course (Karis et al.) |
| TAA / TSR | Karis HQ temporal supersampling, INSIDE reprojection, UE TSR whitepaper |
| Strand hair | Epic strand hair talks, TressFX |
| Skin | Jimenez separable SSS, Penner preintegrated |
| Volumetric fog | Bartłomiej Wroński Frostbite volumetrics |
| Sky | Hillaire production sky model |
| Clouds | Schneider Horizon: Zero Dawn, Vos Guerrilla |
| AO | XeGTAO reference |
| Upscaling | AMD FSR2/3 source, NVIDIA DLSS SDK |
| Path tracer | pbrt-v4, Embree tutorials |

**Isolated worktrees per agent task.** Every subsystem implementation runs on a dedicated worktree, integrated on weekly human-reviewed merge days. Agents do not share branches during implementation.

**Screenshot regression harness.** Every subsystem change produces before/after screenshots on the test scene library (Sponza, Bistro, San Miguel, Bloom-specific scenes). Regression is detected automatically. This is how you keep 10+ agents from silently breaking each other.

**Reference path tracer as ground truth.** Built in Phase A, used throughout. Realtime GI is measured against it, not against "does it look OK."

**Human review bottlenecks, accepted.** Integration, material tuning, tone mapping, exposure response, AO calibration, final pass quality — all human. Budget one human day per week minimum for integration, one human week per phase for tuning. This doesn't scale with agent count and that's fine.

**The taste wall is real.** Somewhere around month 14–16 everything will look technically correct and visibly "not quite right." Budget a dedicated two-week tuning sprint at that point before pushing into Phase E3/E4 and the character work.

---

## 4. Risk Assessment

Ordered by blast radius.

1. **Phase C (virtualized geometry) slips past month 16.** Mitigation: ship mesh-shader-based meshlet renderer without the full DAG/software-rasterizer stack as fallback. Defer full Nanite-equivalent to v1.1. This is a 70% quality, 40% effort option.
2. **Phase E (dynamic GI) doesn't reach E3/E4.** Mitigation: ship E1+E2 (DDGI + SDF GI) as v1. This is already competitive with early UE5 releases.
3. **TAA is never stable.** Everything temporal downstream breaks. Mitigation: budget 3 months of Phase B on TAA alone if needed. Do not push past Phase B until TAA passes the thin-geometry test.
4. **Render graph design debt from Phase A.** Every later phase pays interest. Mitigation: spend disproportionate month-1 effort on the render graph API. Review the interface *before* implementing. Rewrite in month 2 if needed — cheap now, impossible later.
5. **Substrate material system design debt.** Same shape as render graph. Spend the time up front.
6. **Agent quality on temporal filters and path tracing.** Historically these are the two areas where agent code is weakest — small bugs are invisible in still frames and brutal in motion. Mitigation: human-driven code review for every temporal and GI change, with reference path tracer as ground truth.
7. **Platform-specific backend drift.** Three backends means three places for bugs. Mitigation: cross-backend pixel-identical test at every integration merge.
8. **Scope creep into "everything UE has."** If you find yourself building a node graph material editor, a Blueprint-equivalent, or a Sequencer, stop. Those are not rendering.

---

## 5. What You Actually Get

At month ~14 (end of Phase B + D + early C): a renderer that is unambiguously better than 2019 AAA, with virtualized geometry coming online, VSM live, TAA stable, substrate materials, and full post/volumetrics.

At month ~20 (end of C, D, F; E at E2 or E3): a renderer that is in the UE5-lite tier. Dynamic GI is real, upscaling is competitive, shadows are at parity, geometry scales to millions of triangles.

At month ~28: a renderer that does not obviously lose any subsystem-level comparison against UE5.3. Hair, skin, eyes, water, clouds, GI, reflections, and geometry are all in the same ballpark. Bloom is still missing UE's tooling depth (node-graph materials, Sequencer, Blueprint, MetaHuman rigging) — and that's fine, because Bloom's authoring layer is TypeScript + Perry, which is the actual wedge.

At month 30: tuning pass and reference parity testing. Ship.

---

## 6. Strategic Framing (unchanged from v1)

Perry, native compilation, sub-second iteration, and the TypeScript authoring layer are Bloom's wedge. This spec exists so visuals don't become a *reason to reject* Bloom. That framing still holds even with the expanded ambition — a developer who picks Bloom because their build times dropped from 5 minutes to 5 seconds will be *delighted* by UE5-tier visuals, but won't reject Bloom for lacking them if the rest is compelling.

The difference between v1 and v2 of this spec is not strategic. It's a bet that, given coding agents under tight specs, reaching UE5-tier is not meaningfully harder than reaching 2019-tier — it's just longer. And if the bet is wrong, every phase of v2 ships incremental quality improvements on its own, so slipping Phase E3/E4 or Phase C's Nanite-equivalent parts still leaves Bloom with a renderer that's better than the v1 spec would have produced.

Orient to the floor, not the ceiling: each phase must ship something usable, even if the next phase never happens.
