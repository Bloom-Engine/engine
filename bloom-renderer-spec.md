# Bloom Engine Renderer — 12-Month Spec

**Target:** 2019–2020 AAA visual quality (Gears 5, Days Gone, Rise of the Tomb Raider tier). Not UE5 Lumen/Nanite. Unambiguously professional, beautiful, competitive with ~90% of shipped AAA titles.

**Constraint:** Built primarily by Claude Code agents under human orchestration. Spec quality, review discipline, and integration architecture are the real bottlenecks — not raw engineering hours.

**Non-goals (v1):** Nanite-equivalent virtualized geometry, Lumen-equivalent fully dynamic GI, strand-based hair, MetaHuman-tier characters, hardware ray tracing as a hard requirement.

---

## 0. Architectural Ground Rules

These constrain every downstream decision. Lock them before any agent writes a line of renderer code.

- **Graphics API:** Vulkan + Metal via a thin internal abstraction layer. D3D12 deferred to year 2.
- **Rendering architecture:** Clustered forward+ (not deferred). Justification: better MSAA support, cleaner transparency, mobile-friendly future, modern GPU perf gap is closed.
- **Color pipeline:** Linear-space lighting throughout, sRGB only at final output, ACES or AgX tone mapping, physical exposure in EV stops. **Locked from day one — retrofitting this is a multi-week pain.**
- **BRDF:** GGX specular, Burley diffuse, energy-conserving, multi-scatter compensation (Fdez-Agüera approximation). Metallic-roughness workflow.
- **Asset format:** glTF 2.0 as the primary import format. Native binary format for shipped builds, deferred to month 10+.
- **Render graph:** All passes registered through a render graph abstraction from day one. Non-negotiable — this is what lets you reorder, add, and remove passes without rewriting the world later.
- **Shader language:** WGSL or HLSL with cross-compilation. Pick one and lock it.

---

## 1. Phase Plan

Each phase has: scope, agent task breakdown, integration risks, and acceptance criteria. Phases overlap where dependencies allow — agent parallelism is the whole point.

### Phase 1 — Foundation (Months 1–2)

**Scope:** Clustered forward+ base, color pipeline, PBR BRDF, glTF importer, render graph skeleton, basic shadow mapping.

**Agent task breakdown:**
- Graphics API abstraction layer (Vulkan + Metal backends, command buffer recording, resource management, sync primitives)
- Render graph implementation (pass registration, resource lifetime tracking, automatic barriers)
- Clustered light culling compute shader + CPU-side cluster build
- Forward+ main pass with PBR BRDF
- glTF 2.0 importer (meshes, materials, textures, scene hierarchy, animations stubbed)
- Cascaded shadow maps for directional light (4 cascades, stable fitting, slope-scaled bias)
- Basic shadow atlas for point/spot lights
- Tone mapping pass (ACES + AgX, switchable)
- Test scene loader and screenshot harness

**Integration risks:**
- Vulkan/Metal abstraction is the single highest-risk piece. Easy to over-engineer, easy to under-engineer. Spec it tightly, review the API surface manually before agents implement.
- Render graph design choices propagate everywhere. Reference: Frostbite's "FrameGraph" GDC talk, Granite's render graph implementation.

**Acceptance criteria:** Test scene with 200+ PBR objects, 50+ dynamic lights, directional sun with stable cascade shadows, 60fps at 1440p on RTX 3060-tier hardware. Color pipeline verified against reference renders.

---

### Phase 2 — Shadows and AO (Months 3–4)

**Scope:** Production-quality shadow filtering, GTAO, shadow atlas allocation strategy.

**Agent task breakdown:**
- PCF shadow filtering with Vogel disk sampling, configurable kernel size
- Stable cascade fitting (texel snapping)
- Shadow atlas allocator with importance-based region sizing
- GTAO implementation following XeGTAO reference
- Contact shadows (screen-space ray-marched short shadows)
- Shadow caching for static geometry (only re-render moving casters)

**Integration risks:**
- GTAO requires depth + normal buffers in specific formats. Lock the G-buffer-lite layout before starting.
- Shadow caching needs static/dynamic geometry classification — tie this into the scene system early.

**Acceptance criteria:** Outdoor scene with non-shimmering sun shadows during camera movement, indoor scene with 30+ shadow-casting lights at 60fps, AO that grounds objects without obvious haloing.

---

### Phase 3 — Post Stack and TAA (Months 5–6)

**Scope:** The full post-processing chain and temporal anti-aliasing. **This is the biggest visual leap of the year.**

**Agent task breakdown:**
- Physically based bloom (CoD:AW downsample/upsample chain, 13-tap filter)
- Depth of field with hexagonal/circular bokeh
- Per-object motion vector buffer
- Motion blur (camera + per-object)
- Auto-exposure with histogram metering and adaptation curves
- Vignette, chromatic aberration, film grain, lens dirt
- **TAA implementation:** jittered projection matrix, history reprojection, neighborhood variance clipping, disocclusion handling

**Integration risks:**
- TAA is the hardest piece in the entire 12 months. Budget 4–6 weeks even with agent parallelism. Plan for multiple iterations. Reference: Karis "High Quality Temporal Supersampling," Pedersen "Temporal Reprojection AA in INSIDE."
- TAA depends on motion vectors being correct for *every* dynamic object including skinned meshes — coordinate with whoever builds the animation system.
- Motion vector buffer format and precision affects everything downstream. Lock early.

**Acceptance criteria:** Screenshots that look "cinematic" without further work. TAA stable on thin geometry (foliage, hair-like objects), no ghosting on fast camera motion, no shimmering on specular highlights.

---

### Phase 4 — Reflections and GI (Months 7–8)

**Scope:** SSR, reflection probes, offline lightmap baker, light probe volumes for dynamic objects.

**Agent task breakdown:**
- Hierarchical Z-buffer construction
- Screen-space reflections with importance sampling for rough surfaces
- Temporal SSR accumulation (now possible because TAA infrastructure exists)
- Reflection probe capture pipeline (cubemap rendering, prefiltering for roughness mips)
- Parallax-corrected cubemap blending
- **Offline lightmap baker:** CPU path tracer (Embree-based), lightmap UV generation/packing, denoiser integration (Intel Open Image Denoise)
- Light probe volume placement and SH9 storage
- Runtime SH probe sampling for dynamic objects

**Integration risks:**
- The lightmap baker is essentially a second renderer. It's a major project on its own — easily 6+ weeks of agent work even with good parallelism.
- Lightmap UV generation is a known-hard problem. Consider integrating xatlas rather than rolling your own.
- Decision point at start of phase: confirm we're going baked GI for v1, not DDGI. **Recommendation: stay baked.** DDGI is a year-2 layer.

**Acceptance criteria:** Indoor scene with bounce lighting that matches a path-traced reference within reasonable tolerance. Wet outdoor scenes with believable reflections. Dynamic characters lit consistently with their static surroundings.

---

### Phase 5 — Volumetrics and Sky (Months 9–10)

**Scope:** Volumetric fog, physical sky, volumetric clouds. The "atmosphere" tier.

**Agent task breakdown:**
- Froxel-based volumetric fog (3D texture aligned to view frustum, single raymarch, sampled by main pass)
- Light injection into froxel volume (every shadow-casting light contributes)
- Hillaire sky model (Rayleigh + Mie scattering, precomputed transmittance/multiscatter LUTs)
- Time-of-day system tied to sky model
- Volumetric clouds (Schneider/Vos approach: noise-based density field, raymarched, lit by sun + ambient)
- Aerial perspective LUT for distance fog on opaque geometry

**Integration risks:**
- Volumetric fog needs every light to inject — coordinate with the light culling system. May require a second light list pass.
- Cloud rendering is expensive. Plan the LOD/quality settings from the start.
- Sky needs to feed back into reflection probes for accurate ambient — this couples Phases 4 and 5.

**Acceptance criteria:** Forest scene with god rays through canopy. Mountain vista with believable atmospheric perspective. Day/night cycle that looks right at every hour. Drifting volumetric clouds that cast shadows on the ground.

---

### Phase 6 — Upscaling, Materials, Polish (Months 11–12)

**Scope:** FSR2/3 integration, advanced material features, particles, decals, water, skin shading. The long tail.

**Agent task breakdown:**
- FSR2 or FSR3 integration (use AMD's reference implementation, do not roll your own)
- Material features: clearcoat, sheen, anisotropy, subsurface approximation
- Particle system rendering: soft particles, lit particles, GPU simulation backend
- Deferred decals (screen-space projection)
- Skin shading: Jimenez separable SSS
- Eye shading model
- Water v1: tessellated plane, Gerstner waves, SSR + cubemap fallback, foam approximation
- Foliage rendering: two-sided lighting, wind animation, alpha-to-coverage with TAA
- Final tuning pass against reference screenshots

**Integration risks:**
- Material features bloat shader permutations fast. Use uber-shader with feature flags + compile-time pruning, not separate shaders per combination.
- Water touches everything (reflection, refraction, fog, sky). It's a horizontal feature, not vertical. Budget accordingly.
- The "tuning month" is non-negotiable. Real engines have technical artists tuning defaults for years; we get one month.

**Acceptance criteria:** Screenshots that pass a blind test against 2019–2020 AAA references. No single rendering subsystem is the obvious "this looks worse than Unreal" weak point.

---

## 2. The Agent Orchestration Layer

The thing that makes this plan actually work — and the thing that's unique to your approach.

**Spec discipline.** Every phase needs a written spec at the same level of detail as Phase 1's task breakdown above, *before* any agent starts. Vague specs produce agent code that looks plausible and is subtly wrong, which is worse than no code.

**Reference implementations as ground truth.** For every major subsystem, identify 1–2 open-source reference implementations the agents can study. XeGTAO for AO, AMD FSR for upscaling, Hillaire's sky shader code, Granite's render graph, Bevy's renderer for general patterns. Agents working from references produce vastly better code than agents working from descriptions.

**Test scenes as acceptance gates.** Build a library of test scenes early — Sponza, Bistro, San Miguel, plus custom Bloom-specific scenes. Every phase ends with screenshot diffs against the previous phase and against reference renders. This is how you catch regressions when 6 agents are touching the renderer in parallel.

**Integration windows.** Agents work in parallel on isolated subsystems, but integration is sequential and human-driven. Plan for one integration day per week minimum. This is where you (the human) earn your keep.

**The taste bottleneck.** Agents will produce technically correct renderers that look subtly wrong. Tone mapping curves, bloom intensity, exposure response, AO darkness — these all require human eyes against reference material. Don't try to delegate this.

---

## 3. Honest Risk Assessment

**What can actually break this plan:**

- **TAA.** If TAA isn't solid by end of month 6, everything downstream (SSR, stochastic effects, upscaling) is compromised. This is the single highest-risk subsystem. If it slips, slip the whole plan rather than shipping bad TAA.
- **The lightmap baker.** Building an offline path tracer is a real project. If it's not ready by end of month 8, fall back to a simpler ambient solution (SH probes everywhere, no lightmaps) and ship baked GI in v1.1.
- **Render graph design debt.** If the render graph abstraction is wrong, every phase pays interest. Spend disproportionate time on this in month 1.
- **The taste wall at month 7–8.** Everything will look technically correct but not as good as references. Budget a week of pure tuning at this point or the wall becomes month 11's problem.

**What this plan does NOT get you:**

- Lumen-equivalent dynamic GI
- Nanite-equivalent geometry
- Strand-based hair
- MetaHuman-tier characters
- Hardware ray tracing as a primary path

These are all year-2+ items. The 12-month target is "visuals are not the reason a developer rejects Bloom," not "Bloom out-renders Unreal."

---

## 4. Strategic Framing

The renderer is not Bloom's wedge. Perry, native compilation, sub-second iteration, and the developer experience are the wedge. This spec exists so visuals don't become a *reason to reject* Bloom — not so they become the headline.

A developer who picks Bloom because their build times dropped from 5 minutes to 5 seconds will happily accept "looks like 2019 Unreal." This plan delivers exactly that, in the only way one person realistically can: by orchestrating agents against tight specs rather than typing every line.

The fact that "everyone could do this" but almost no one actually is — that's the moat. Specs like this one are the moat.
