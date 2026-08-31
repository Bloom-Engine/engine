# Capability-tiered material and texture indirection

Issue #130 adds a stable resource-addressing layer without changing the
existing material pixels. Scene and future GPU-driven passes use typed IDs;
the backend selected from the actual `wgpu::Device` decides how those IDs map
to resources.

## Tiers

| Tier | Selection | Resource binding | Draw switching |
|------|-----------|------------------|----------------|
| A | `TEXTURE_BINDING_ARRAY`, non-uniform indexing, and non-zero texture/sampler binding-array limits | One persistent material storage buffer, texture descriptor array, sampler descriptor array, and generation table | No per-material bind-group creation or switch in the GPU-driven opaque path |
| B | At least 16 texture-array layers and 8 sampled textures per stage | Deterministic first-fit pages backed by texture arrays/atlases | Stable-sort by page; switches are bounded by populated page count |
| C | Everything else, or a forced compatibility override | Existing per-material bind groups and ABI v3 | Existing behavior |

The selection uses `device.features()` and `device.limits()`. It never infers a
tier from an operating system name. Native device-creation paths request Tier
A features only when the adapter advertises the complete feature pair. WebGPU
and downlevel mobile adapters therefore select B or C naturally.

`BLOOM_MATERIAL_TIER=A|B|C` is a startup diagnostic override. An override can
lower the detected tier but cannot manufacture unsupported features. At
runtime, `setMaterialBindingTierOverride("auto" | "A" | "B" | "C")` follows
the same rule and reports rejection with `false`.

## Stable IDs and lifetime

`MaterialId`, `TextureId`, `SamplerId`, `MeshId`, and `BufferViewId` are typed
32-bit values:

- bits 0–19: one-based descriptor slot;
- bits 20–31: generation;
- zero: diagnostic fallback.

Retiring a resource immediately makes its ID non-resident, but the owned GPU
resource remains alive until `Queue::on_submitted_work_done` advances the
completion epoch. Only then is the slot reclaimed and its generation bumped.
A stale ID therefore cannot resolve to a later resource that reused the same
slot.

Tier A mirrors texture and sampler generations in a GPU storage buffer. WGSL
checks the generation before descriptor lookup and redirects a mismatch to
descriptor zero. Record zero is a white, rough, non-metallic, non-emissive
fallback; missing normals return tangent-space +Z. This check is required:
CPU-only generational handles do not protect a shader after a descriptor slot
is reused.

All allocation failures are deterministic. They return ID zero, increment the
`limit_fallbacks` counter, and emit an actionable diagnostic rather than
sampling an unrelated resource.

## GPU material record

`GpuMaterialRecord` is 16-byte aligned and contains:

- generation, layered-PBR version/lobe mask, and user-parameter range;
- base color, metallic/roughness, emissive, shading-model, and foliage data;
- typed texture IDs for base color, normal, MR, emissive, occlusion,
  reflection, and Tier B array/page resources;
- typed sampler IDs.

The record remains exactly 176 bytes. `header.y` uses its high eight bits for
the layered-PBR material-record version and its low 24 bits for the feature
mask. Version 1 with mask zero is the base-only default; version-zero legacy
records are normalized to that default on allocation/update. The matching
Tier C/custom record remains 80 bytes and keeps exact version/mask bit
patterns in its previously reserved `foliage_params.zw` lanes. See
`docs/layered-pbr.md` for the versioned lobe assignments.

The storage-buffer record is populated in parallel with the legacy material
state. Tier C continues binding ABI-v3 per-material groups, so introducing the
record has no render-order, shader, or pixel effect.

The shared GPU-driven header is
`native/shared/shaders/material_indirection.wgsl`. It is embedded by
`shader_library` and intentionally separate from `material_abi.wgsl`; custom
legacy materials do not receive a surprise ABI bump.

## Color, normal, mip, and sampler rules

Every resident texture records color space, semantic, mip count, and whether
the view format performs hardware sRGB decoding.

- sRGB content in a linear view is decoded by the WGSL lookup helper.
- sRGB views are not decoded twice.
- normal and metallic/roughness resources are linear.
- a missing/stale normal returns `(0, 0, 1)`.
- HDR resources are tagged linear.
- texture views keep their full mip chain; sampler IDs select filtering
  independently.

Tier B pages group IDs; they do not reinterpret texels. Existing texture-array
creation remains the upload oracle for sRGB/linear formats, mips, and filtering.

## Tier B planning contract

`build_tier_b_dispatch_plan` accepts draws with stable material and texture
IDs plus the adapter page capacity.

1. Duplicate and fallback texture IDs are removed.
2. Materials are placed by deterministic first fit.
3. A material whose unique set exceeds one page is marked for Tier C fallback.
4. Draws are stable-sorted by `(page, original_submission_index)`.
5. `page_switches <= populated_pages`.

The stress test covers 4,096 textures and 10,000 visible draws at 256 entries
per page: 16 pages, 16 switches, no fallback, and identical plans across runs.

## TypeScript diagnostics

`getMaterialBindingCapabilities()` returns a read-only report containing:

- detected, selected, and overridden tier;
- required feature booleans;
- raw adapter limits and effective capacities;
- current resident counts;
- stale/limit fallback counters;
- an actionable diagnostic when an override is rejected.

The same object is embedded under `runtime_paths.material_binding` in quality
telemetry.

## Integration for #27 and #28

GPU-driven consumers should:

1. Store `MaterialId` in instance/visibility records, not a bind group.
2. Store `TextureId`/`SamplerId` in `GpuMaterialRecord`.
3. Select the dispatch plan from `selected_tier`.
4. Bind Tier A's `global_layout`/`global_bind_group` once, or bind each Tier B
   page once per grouped run.
5. Route Tier B overflow and every Tier C draw through the existing material
   dispatcher.
6. Never cache a descriptor index without its generation-bearing typed ID.

The registration and delayed-retirement methods are the streaming hooks;
virtual texture streaming itself remains out of scope.

## Qualification

Focused coverage includes:

- generation churn, double retirement, delayed reclamation, and fallback;
- feature/limit selection and unsupported upward overrides;
- deterministic Tier B paging;
- 4,096-texture/10,000-draw stress;
- a real Tier A GPU bind group with 4,096 resident descriptors;
- sRGB decode and stale-ID rejection after descriptor-slot reuse;
- existing material texture-array GPU tests;
- forced Tier C visual/performance comparison through the standard quality
  corpus.
