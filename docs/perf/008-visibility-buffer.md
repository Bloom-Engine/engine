# 008 — Visibility buffer replaces 4-MRT G-buffer

**Effort:** staged · **Expected gain:** workload-dependent · **Status:** active qualification

## 2026 design correction and safety gate

The original proposal below incorrectly called `Rgba32Uint` an ~8-byte
target; it is 16 bytes/pixel. Bloom also now has an alpha-aware depth prepass
and Equal-tested main pass, so opaque PBR already shades only the winning
surface. A visibility path therefore cannot claim an overdraw or bandwidth
win from the old estimate.

The implementation contract is now:

- `Rg32Uint` (8 bytes/pixel): full 32-bit draw ID plus a 31-bit primitive ID
  and one front-face bit;
- reconstruct perspective-correct barycentrics from the referenced
  triangle's clip positions rather than storing them or expanding shared
  vertices;
- require the optional WebGPU `primitive-index` capability and preserve the
  existing 96-byte `Vertex3D` storage layout as six packed `vec4<u32>` lanes;
- admit only static opaque/masked shared-arena geometry with Tier-A global
  materials; blend, transmission, layered/custom, skinned, deforming, and
  unsupported content stays on the forward compatibility path;
- keep the forward MRT authoritative and the visibility path opt-in until an
  identical-camera A/B proves total depth/visibility/shading GPU time is no
  worse, image parity passes, and memory is bounded;
- never enable it merely because the visibility raster writes fewer bytes
  than the MRT. The comparison must include every added pass, storage fetch,
  reconstructed output, and transient allocation.

The shared CPU/WGSL ABI and perspective reconstruction live in
`renderer/visibility_buffer.rs`,
`shaders/visibility_buffer/reconstruct.wgsl`, and
`shaders/visibility_buffer/geometry.wgsl`. Hardware readback oracles now prove
ID/winding rasterization, perspective reconstruction, non-zero first-index and
base-vertex addressing, the exact packed `Vertex3D` byte layout, and all 24
reconstructed vertex lanes. The opt-in runtime diagnostic now executes the
depth-equal ID raster and full-screen attribute reconstruction against the
real GPU-driven draw/index/vertex buffers. Production composition follows
only after the completed PBR A/B passes the no-regression gate.

Runtime qualification is explicitly requested before engine attachment:

- `BLOOM_VISIBILITY_BUFFER=validate` runs the private visibility and
  reconstructed-normal passes while the forward image remains unchanged;
- `BLOOM_VISIBILITY_BUFFER=debug` additionally overlays reconstructed normals
  on admitted pixels, leaving compatibility-rendered content intact so holes
  and routing mistakes are visible;
- unset/off requests no `primitive-index` device feature, creates no pipeline,
  texture, or bind group, and records no visibility work.

Both modes expose `visibility_buffer_runtime` in the public renderer capability
report, including admitted/compatibility draw counts, current extent, exact
owned bytes, activation reason, and whether the current frame recorded work.

## Original problem statement (historical)

The `main_hdr_pass` writes four MRTs at the full physical resolution:

| RT | Format | Bytes/pixel |
|---|---|---|
| `hdr_rt`       | Rgba16Float | 8 |
| `material_rt`  | Rg8Unorm    | 2 |
| `velocity_rt`  | Rg16Float   | 4 |
| `albedo_rt`    | Rgba8Unorm  | 4 |

Total: **18 bytes per winning pixel written** by the current main pass. At
1600×900 that is 26 MB per full-surface pass before attachment compression or
backend effects. The old claim that overdraw multiplies all four writes is no
longer valid because the alpha-aware depth prepass rejects hidden fragments.

UE5's Nanite uses a **visibility buffer** instead: store only `(triangle_id,
barycentrics)` (~8 bytes) in the G-buffer, defer material evaluation to the
shading pass. No material sampling happens for hidden pixels. Combined with
depth prepass, every visible pixel shades exactly once.

## Proposed approach

This is a significant refactor — do ticket 005 (depth prepass) first. Then:

1. **Rasterize eligible geometry** into one `Rg32Uint` visibility target,
   storing `(draw_id, primitive_id + front-face bit)`. Material and normal are
   not written; barycentrics are reconstructed in the shading pass.
2. **New shading pass** reads the visibility buffer, fetches the vertex data
   for the referenced triangle, interpolates attributes from barycentrics,
   and evaluates the full PBR shader once per pixel.
3. **MRTs that post-FX consumes** (normal, albedo, material, velocity) can
   either be rebuilt per-pixel in the shading pass OR kept as separate passes.
   Simplest path: the shading pass writes them alongside the final HDR
   colour. That preserves the winning-pixel attachment footprint, so any gain
   must come from the cheaper visibility raster or improved scheduling and
   must be demonstrated by the total-pass A/B.

## Simpler intermediate step (landed)

Bloom already **drops unused MRTs when features are off** on the constrained
`lean_mrt` route:

- `velocity_rt` is only needed when TAA or motion-blur is on.
- `albedo_rt` is only needed when SSGI or SSR is on.
- `material_rt` is needed for SSR and the shadow map sampler stuff.

The constrained scene pipeline uses fewer MRT targets when dependent post-FX
is disabled, reducing its attachment bandwidth without changing the full
quality path.

## References

- Burns & Hunt — "The Visibility Buffer: A Cache-Friendly Approach to
  Deferred Shading" (JCGT 2013)
- UE5 Nanite "Deep dive" (GDC 2022) — visibility buffer + cluster cull
- Activision's "Geometry Rendering Pipeline Architecture at Call of Duty"
  SIGGRAPH talks for visibility-buffer variants

## Acceptance

- Total uncapped depth + visibility + shading time is no worse than forward
  MRT on the admitted workload, and improves materially on the target stress
  scene. Record pass timings rather than inferring a win from format size.
- Same PBR output (SSIM ≥ 0.99 vs baseline).
- Post-FX that consume G-buffer content still work.
- Doesn't regress perf on non-overdraw-heavy scenes.
- Unsupported/skinned/translucent/layered/custom content composes through an
  explicit inspectable compatibility route without holes or ordering changes.

## Notes for the implementer

- Reuse #28's STORAGE-capable shared vertex/index arenas and `GpuDrawRecord`.
  `draw.y` is the first index, `bitcast<i32>(draw.z)` is the base vertex, and
  `draw.w` is the material ID. Do not declare `Vertex3D` with native WGSL
  `vec3` fields: storage alignment would not match Rust's tightly packed
  offsets. Use the checked six-`vec4<u32>` decoder.
- Animated meshes (skinned) need special handling — triangle positions change
  per frame. Either compute-skin to a fixed buffer first, or keep animated
  meshes on the traditional path and use visibility buffer for static only.

## Files likely to change

- `native/shared/src/renderer/` (`mod.rs` + `scene_pass.rs`; the old single
  `renderer.rs` was split) — scene_pipeline, main_hdr_pass, shading
  pass, SSR/SSGI/SSAO inputs.
- `native/shared/src/scene.rs` — mesh_id assignment, vertex buffer layout.

## Activation state

The ticket is under active qualification, but the shipping path remains off.
The old percentage and MB/frame estimates are not activation evidence because
they predate the alpha-aware depth prepass and omit the visibility shading
pass. Enablement requires uncapped captures on at least the representative
discrete and integrated/mobile tiers, with per-pass GPU timestamps, total
transient bytes, full compatibility composition, and governed image diffs.

The low-quality `lean_mrt` intermediate already drops unused material/albedo
attachments on constrained profiles. That remains the safe bandwidth path
for adapters which lack `primitive-index` or do not win the runtime A/B.
