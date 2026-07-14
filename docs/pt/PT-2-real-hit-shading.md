# PT-2 — Real hit shading (Tier 2)

Parent: [pt-roadmap.md](pt-roadmap.md). Builds on PT-1.

## Scope

Replace PT-1's flat-normal/card-albedo hit shading with real surface
data at every bounce:

1. **Geometry megabuffer** — every TLAS instance's `Vertex3D` + index
   data concatenated into two storage buffers (raw f32/u32 words,
   zero repack via bytemuck; nodes retain CPU vertices). Per-instance
   window in `InstanceGiData.geo` = (vertex_base, index_base,
   index_count, texture_idx); `index_count == 0` falls back to PT-1
   shading. Rebuilt with the instance-data buffer (grow-only).
2. **Interpolated attributes** — `fetch_hit_attrs` reads the hit
   triangle via `primitive_index`, interpolates normal + UV with
   `barycentrics` (DXR convention: w=1-u-v weights v0), transforms the
   normal by the ray query's `world_to_object` (v·M = (M⁻¹)ᵀ·n).
3. **Texture binding array** — `binding_array<texture_2d<f32>>` in its
   own bind group (wgpu forbids binding arrays next to uniform
   buffers), fixed 256 slots padded with the white texture (no
   PARTIALLY_BOUND needed). Requires TEXTURE_BINDING_ARRAY +
   SAMPLED_..._NON_UNIFORM_INDEXING **and**
   `max_binding_array_elements_per_shader_stage` (defaults to 0 —
   must be requested at device creation). Hit albedo =
   `inst.albedo × pt_textures[geo.w]` sampled at the interpolated UV.
   Without the features the kernel compiles a white-stub variant and
   stays on card albedo.
4. `mat_params` (roughness/metalness) plumbed per instance — GGX
   specular lobe still TODO (M3).

## Status (2026-07-13, AMD 760M / DX12+DXC 1.9)

- Megabuffer + windows verified end-to-end: numeric dump shows correct
  per-pixel t/instance/prim; debug 6 shows smooth interpolated normals
  (leaf cards each with their own direction, smooth terrain); debug 7
  shows real leaf/bark/plaster textures at traced hits.
- Debug 13 sanity: traced t agrees with the G-buffer everywhere a
  proxy matches the visible mesh; expected mismatches only (grass and
  skinned characters are not in the TLAS; tree proxies are unrotated).
- Found + fixed the transposed `inv_vp` upload (see the post-scriptum
  in PT-1's ticket) — this had silently broken all PT-1 transport.

## M3 — GGX (landed b146b6d)

`sample_brdf` ported: VNDF specular + Burley diffuse, Fresnel-weighted
lobe pick, metal/dielectric f0 split. Primary material from the
G-buffer material RT, bounce material from `mat_params`. Verified: the
45 s converged title render is clean (no fireflies, no NaN blowups),
with visibly deeper foliage shading than the pure-Lambert version.

## Open (rolls into PT-3/PT-5)

- Specular NEE (direct highlights from sun/point lights on smooth
  surfaces) — bounce sampling covers reflections but not delta-light
  highlights; diffuse NEE is scaled by (1 − metallic) meanwhile.
- Emissive triangles at hits already work via `albedo × emissive_luma`;
  emissive-as-light-source sampling is PT-4 (ReSTIR) territory.
- Known gaps, accepted: skinned enemies not in the TLAS (no per-frame
  BLAS path yet); tree GI proxies unrotated (bounce-grade accuracy
  only); leaf-card cutouts opaque to rays.
- Point-light NEE + colour-bleed visual + bloom-reference RMSE (carried
  from PT-1 M3) — needs a lit test world + fixed exposure.
