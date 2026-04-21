# 007b — Lumen screen probes (HW trace, ray-query)

**Effort:** 1-2 weeks · **Expected gain:** off-screen occlusion + bleed · **Status:** open

Phase 1b of the [Lumen roadmap](lumen-roadmap.md). Depends on 007-prep and
007a; develops in parallel with 007a, merges after 007a lands so the probe
infrastructure (placement/filter/temporal/resolve) already exists.

## Problem

007a's SW trace only samples radiance from **on-screen** geometry. Rays that
point behind the camera, through walls, or past the screen edges contribute
zero. This is the structural limitation of screen-space GI that full Lumen
solves with SDFs (SW) or BVH traces (HW). 007b delivers the HW half so
Apple / Windows / Linux platforms get correct off-screen GI immediately,
without waiting for the Phase-3 SDF work.

## Approach

Keep the probe infrastructure from 007a. Add a second WGSL trace variant that
replaces the Hi-Z march with `rayQueryProceed`; pick which variant to
pipeline at device init based on adapter features.

### Feature detection — platform crates

At device request time in each `native/<platform>/src/lib.rs`:

```rust
let wants_rt = adapter.features().contains(
    wgpu::Features::EXPERIMENTAL_RAY_QUERY
    | wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
);
let required_features = base_features | if wants_rt { rt_features } else { empty };
```

Expose `hw_rt_enabled: bool` on the renderer struct. Web skips the check
entirely (feature not supported). Platform gates:

| Platform | RT request | Expected runtime |
|---|---|---|
| macOS, iOS, tvOS | yes | enabled on Apple Silicon / A-series post-A13 |
| Windows | yes | enabled on any DXR 1.1 GPU |
| Linux | yes | enabled on RT-capable Vulkan GPUs; fallback on older iGPUs |
| Android | yes | usually fallback; enabled on recent Adreno 7xx / Mali-Immortalis |
| Web | no | always SW |

### BLAS lifecycle — `native/shared/src/models.rs`

- New optional field `blas: Option<wgpu::Blas>` on the mesh struct.
- Built once at model load (`bloom_model_load_glb` or equivalent) from the
  existing vertex + index buffers via `CommandEncoder::build_acceleration_structures`.
- No refit path in Phase 1b — all Sponza meshes are static. Dynamic meshes
  land in a later ticket.

### TLAS lifecycle — `native/shared/src/renderer/mod.rs`

- New fields: `tlas: wgpu::Tlas`, `tlas_package: wgpu::TlasPackage`,
  `tlas_version: u64`, `tlas_instance_data_buffer: wgpu::Buffer`.
- Mirror the `shadow_version` pattern from ticket 004: `tlas_version++` when
  the scene graph changes; rebuild TLAS only when `tlas_version != tlas_built_version`.
- On rebuild: walk the scene graph, emit one TLAS instance per visible mesh
  with its world transform and material index (stored in
  `instance_custom_index`), encode with `encoder.build_acceleration_structures`.

### Per-instance GI data — new storage buffer

`InstanceGIData { albedo: vec3, emissive_luma: f32, normal_ws: vec3, _pad: f32 }`
indexed by `rayQueryGetCommittedIntersection(&rq).instance_custom_index`.
Populated at TLAS build time from each scene node's material. Hit-shading
looks up this buffer — no bindless texture fetch, no per-vertex attribute
interpolation in Phase 1b.

### Hit-lighting-lite — `SSGI_PROBE_TRACE_HW_WGSL`

At each committed intersection:

```wgsl
let inst_data = instance_gi_data[hit.instance_custom_index];
let hit_world = ray_origin + ray_dir * hit.t;
let n = inst_data.normal_ws;
let ndl = max(dot(n, sun_dir_ws), 0.0);
let shadow = sample_sun_shadow(hit_world, sun_vp);   // reuse existing cascade
let sky = sky_irradiance(n);                          // analytic dome
let radiance = inst_data.albedo * (ndl * shadow * sun_color + sky)
             + inst_data.emissive_luma * inst_data.albedo;
```

Flat per-instance normals are fine for GI bounce — the probe filter + temporal
accumulation hide the faceting. Full per-vertex normals + textured albedo come
with Mesh Cards in ticket 013.

### Shader variant selection

`SSGI_PROBE_TRACE_HW_WGSL` coexists with `SSGI_PROBE_TRACE_SW_WGSL`. At
pipeline creation in `Renderer::new`, pick one:

```rust
let trace_shader = if hw_rt_enabled {
    SSGI_PROBE_TRACE_HW_WGSL
} else {
    SSGI_PROBE_TRACE_SW_WGSL
};
```

The `probe_trace_pipeline` layout differs (HW adds `acceleration_structure`
+ `instance_gi_data` bindings), so two `probe_trace_layout` values — pick the
matching one for the bind group. Every other pass stays single-variant.

### Platform Cargo.tomls

Gate RT features per platform so Android + web Cargo.toml don't pull
Vulkan RT layers they can't use:

```toml
# native/macos/Cargo.toml, native/ios/Cargo.toml, native/tvos/Cargo.toml,
# native/windows/Cargo.toml, native/linux/Cargo.toml, native/android/Cargo.toml
[dependencies]
wgpu = { version = "<upgraded>", features = ["ray-tracing"] }  # exact feature name per wgpu version
```

Web crate unchanged.

## Acceptance

- `./examples/intel-sponza/main --quality 3 --ssgi 1 --fps-only 300` ≥ **47.2 fps**
  on a DXR / Metal-RT adapter (same bar as 007a — HW must not be slower than SW
  at the same quality setting).
- **Off-screen bleed visual test**: place camera in a Sponza arch looking at an
  interior wall while a bright window sits behind the camera. HW path shows
  warm bleed on the wall from the off-screen window. SW (007a) path shows
  nothing. Commit before/after PNGs of both.
- **Fallback test**: run on a machine without RT (or force `hw_rt_enabled = false`
  via env var `BLOOM_FORCE_SW_GI=1`). Output matches 007a pixel-for-pixel
  within TAA noise floor.
- `./examples/intel-sponza/main --quality 0 --fps-only 60` hits 60.
- Adapter-lacks-RT path compiles and runs (Android emulator / software
  Vulkan) — no crash, no validation errors, falls back silently.
- `tlas_version` cache hit on stationary camera: TLAS rebuilds exactly once
  per scene change, not per frame. Verify via profiler timestamp on
  `tlas_build_pass`.

## Notes

- BLAS build cost is paid at model load, not per frame. Record this cost in
  the commit so future sessions see the one-time hit.
- Firefly clamp at hit-lighting-lite: cap `radiance` luma at 10 (match 007a's
  SW clamp). HW hits can be very bright if a ray lands directly on a lit
  emissive surface.
- The new `instance_gi_data` buffer must stay in sync with TLAS instance
  order. Rebuild both together; never rebuild one without the other.
- Hit-lighting-lite is known-incomplete. See `lumen-roadmap.md` HW quality
  caveat and ticket 013 (Mesh Cards) for the upgrade path.
