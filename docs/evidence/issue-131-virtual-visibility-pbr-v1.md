# Issue #131 virtual visibility PBR reconstruction v1 evidence

This checkpoint qualifies the unattached virtual-geometry PBR consumer at
revision `9b3130f8b3a11c9c2b80fdd6aa882093e0b9c550`. It proves that a virtual
visibility ID can reconstruct the established scene-shader inputs and compile
against the production renderer's exact four-MRT material and lighting ABI. It
does not register virtual geometry in ordinary frames or claim integrated
Bistro parity.

## Render-ready selection ABI

The selected-cluster record remains 32 bytes. Traversal now writes an absolute
cluster-table index, an absolute physical-page byte base, and the cooked vertex
encoding in the high flag bits. The cluster's formerly reserved final payload
word carries its generation-safe owning mesh ID. Raster and reconstruction
validate that owner before reading raw page data.

This removes the virtual mesh-table storage binding and per-vertex/per-pixel
mesh lookup. The raw visibility vertex stage now uses four storage buffers. The
complete PBR fragment layout uses exactly eight storage buffers: four virtual
geometry buffers, two authoritative material buffers, and two clustered-light
buffers. The record size, indirect command ABI, visibility target, and default
renderer ownership remain unchanged.

A real-GPU multi-mesh negative control registers a Float32 mesh followed by a
quantized mesh. It selects only the second mesh and proves nonzero absolute
cluster addresses, aligned in-range page bases, correct packed encoding,
generation ownership, and quantized decoded vertices. This prevents an
identical single-mesh fixture from hiding cross-mesh aliasing.

## Exact PBR reconstruction

The shared reconstruction source validates the virtual namespace, selected
record, dense instance, cluster ownership, primitive range, and local `u8`
indices before decoding three cooked vertices. It reconstructs:

- perspective-correct barycentrics and current clip position;
- previous clip position from the previous instance and view-projection state;
- current world position and inverse-transpose world normal;
- transformed tangent direction and mirrored-transform handedness;
- UV, vertex color multiplied by instance tint, remapped material ID, and face
  orientation;
- adjacent helper lanes for the authoritative material evaluator's texture
  gradients and specular antialiasing.

The fullscreen virtual consumer calls the same specialized
`shade_main_scene` function used by production scene rendering and returns its
established `SceneOut` attachments. The compatibility consumer discards
virtual IDs, and the virtual consumer discards compatibility IDs, so ownership
is disjoint without a shared shader branch.

## Real Metal oracles

The raw raster test runs traversal, draw emission, visibility rasterization,
and a reconstruction compute probe in one Metal command stream. Every covered
pixel proves valid virtual identity, unit-sum barycentrics, current and previous
clip values, world position, inverse-transpose normal, negative tangent
handedness, UV, tint/color, and material/face state. Covered and background
pixels both exist and no compatibility or background collision is observed.

A separate test constructs a complete headless `Renderer`, specializes the
actual production scene shader, and creates the virtual four-MRT pipeline using
the renderer-owned draw, lighting, global-material, and joint layouts. This is
the test that exposed the original nine-storage-buffer layout as over budget;
the render-ready selected record reduced it to the qualified eight-buffer
contract.

## Regression boundary and gates

The new consumer is explicit and unattached. `Renderer`, `EngineState`, and
`ModelManager` own no virtual pool, selector, emitter, raster, or shading field.
Ordinary frames register no virtual pass and allocate, bind, submit, or shade
no virtual resources. Existing immediate-mode, glTF, compatibility visibility,
and default pixels are therefore unchanged by construction.

The governed quick lane passed in 122 seconds:

- 410 shared unit tests passed and one existing hot-reload test was ignored;
- the negotiated real-device test and all 29 virtual-geometry tests passed;
- 59 golden tests passed and two hardware-specific tests were ignored;
- four render-target tests, two visibility parity tests, and both visibility
  runtime tests passed;
- strict correctness/suspicious/performance Clippy, formatting, FFI/schema
  parity, CI contracts, and file-size governance passed;
- 39 quality-governance tests, three visual fault-engine tests, and 29 cooker
  tests passed;
- the WebAssembly `web` check and canonical example inventory passed.

The three hardware quick scenes also built and captured successfully at the
qualified revision. Their strict result is **not** a visual pass: this checkout
lacks approved portable baselines for PBR Spheres, Damaged Helmet, and Sponza,
so SSIM, OKLab, and edge comparisons were unavailable. No baseline was created
or approved automatically.

## Remaining acceptance work

No #131 acceptance checkbox changes at this checkpoint. Production activation
still needs ownership/routing and attachment composition, a bounded submission
path for adapters without `MULTI_DRAW_INDIRECT_COUNT`, conservative previous-
frame Hi-Z, asynchronous index and streaming feedback, Bistro crack/temporal
captures with approved comparisons, 10-million-triangle stress, and total GPU
timing on integrated and discrete adapters.
