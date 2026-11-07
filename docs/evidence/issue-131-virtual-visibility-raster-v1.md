# Issue #131 virtual visibility raster v1 evidence

This checkpoint qualifies collision-free visibility IDs and the first raw-page
virtual visibility raster at revision
`0ab0629aa78423c73f5e3f1fd77e1681458e98e2`. It proves that four GPU-selected
clusters can write valid virtual draw and primitive identities plus depth from
their cooked resident pages. It does not register a production render path,
shade those IDs, or change ordinary scene pixels.

## Shared visibility namespace

Bloom keeps its existing 8-byte `Rg32Uint` target. Draw word bit 31 now selects
the producer: zero is the existing compatibility visibility path and one is
virtual geometry. The lower 31 bits address the draw within that producer.
The second word remains a 31-bit primitive ID with front-face orientation in
bit 31.

Compatibility IDs in the new lower-31-bit namespace retain their bytes;
formerly accepted high-bit compatibility IDs are now rejected explicitly.
Compatibility and virtual draws cannot alias, and virtual index `0x7fffffff`
is excluded so its encoded word cannot collide with the `0xffffffff`
background sentinel. CPU and shared WGSL encode/decode tests cover both
namespace boundaries, primitive orientation, and the sentinel.

## Raw resident-page raster

`GpuVirtualVisibilityRaster` binds the physical page, mesh, cluster, selected,
and instance buffers owned by the preceding pool/selector/emitter stages. Its
non-indexed vertex shader follows each command's exact `first_instance`, reads
the local `u8` index, decodes the Float32 or quantized cooked vertex, applies
the current instance and view-projection matrices, and writes a namespaced
draw index. The fragment shader adds the primitive and face identity while a
`Less` depth attachment establishes the visible surface.

Masked clusters discard and remain explicitly owned by compatibility rendering
until virtual shading has the authoritative alpha texture, sampler, and cutoff.
Single-sided clusters discard back faces; double-sided clusters retain both
faces and the orientation bit. The 128-byte frame ABI includes current and
previous view-projection matrices so the next shading stage can combine them
with the already-qualified current and previous instance transforms.

Construction checks `PRIMITIVE_INDEX`, `INDIRECT_FIRST_INSTANCE`, five
vertex-stage storage-buffer bindings, producer identity, and namespace
capacity. The only shipping draw entry point uses the emitter's GPU count and
requires `MULTI_DRAW_INDIRECT_COUNT`. Without that feature it returns an
explicit unsupported error and the caller must retain compatibility rendering.
The exact fixed-count path used by the oracle is compiled only for tests; stale
command slots cannot become an accidental shipping fallback.

## Real Metal oracle

The focused 26-test virtual-geometry suite passed. Its render oracle runs
selection, bounded command emission, and raw visibility rasterization in one
real Metal command stream. Four fixture triangles write a 16x16 `Rg32Uint`
target. Readback proves that both covered and background pixels exist, every
covered pixel decodes to the virtual namespace, observed draw indices are
bounded to 0 through 3, every primitive ID is zero, and no covered pixel
decodes as compatibility or background.

## Regression boundary and gates

The raster has no `Renderer`, `EngineState`, `ModelManager`, frame-graph,
ordinary glTF, or FFI owner. The default path therefore constructs no new
buffer, texture, bind group, pipeline, pass, or draw and changes no pixels.

The complete governed quick lane passed in 59 seconds:

- 406 shared unit tests passed and one existing hot-reload test was ignored;
- the negotiated real-device test and all 26 virtual-geometry tests passed;
- 59 golden tests passed and two hardware-specific tests were ignored;
- four render-target tests, two visibility parity tests, and both visibility
  runtime tests passed;
- strict correctness/suspicious/performance Clippy, formatting, FFI/schema
  parity, CI contracts, and file-size governance passed;
- 39 quality-governance tests, three visual fault-engine tests, and 29 cooker
  tests passed;
- baseline wasm, native `models3d`, and wasm `web,models3d` checks passed;
- the no-default dependency graph still contains neither
  `bloom-geometry-format` nor SHA-256.

## Remaining acceptance work

No #131 acceptance checkbox changes at this checkpoint. The next gated slice
is exact PBR reconstruction for virtual IDs: perspective-correct attributes,
current/previous clip positions, inverse-transpose normals, transformed
tangents, tint, remapped materials, face state, and all four established MRTs.
It must compose with compatibility pixels without overlap or holes. Production
activation also still needs a bounded non-count submission path, conservative
previous-frame Hi-Z, asynchronous feedback/streaming, Bistro crack and temporal
captures, 10-million-triangle stress, total GPU timing, and integrated/discrete
adapter qualification.
