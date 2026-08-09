// Bounded fallback for adapters without indirect-count submission. GPU
// compaction writes selected indices into 22 power-of-two triangle bins, so
// this path issues a constant 22 draws and invokes fewer than twice the exact
// number of cluster vertices.

struct VirtualBinnedSelectionIndices { records: array<u32>, };
@group(0) @binding(5) var<storage, read> virtual_binned_selection_indices:
    VirtualBinnedSelectionIndices;

@vertex
fn vs_virtual_visibility_binned(
    @builtin(vertex_index) corner: u32,
    @builtin(instance_index) binned_index: u32,
) -> VirtualVisibilityVertexOut {
    if (binned_index >= arrayLength(&virtual_binned_selection_indices.records)) {
        return bloom_invalid_virtual_vertex();
    }
    return bloom_virtual_visibility_vertex(
        virtual_binned_selection_indices.records[binned_index],
        corner,
    );
}
