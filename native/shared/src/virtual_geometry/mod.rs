//! Opt-in runtime ownership and fixed-budget residency for cooked geometry.
//!
//! A caller must explicitly load a `.bgeo` archive and enable the renderer
//! owner. The ordinary/default renderer retains no virtual-geometry state.

mod asset;
mod decode;
mod draw_emission;
mod gpu_pool;
mod hiz;
mod residency;
pub(crate) mod shading;
#[cfg(not(target_arch = "wasm32"))]
mod store;
mod streaming;
mod traversal;
mod visibility;

pub use asset::{
    ArtifactIdentity, VirtualGeometryAlphaMaskedRoute, VirtualGeometryAsset,
    VirtualGeometryLoadError, VirtualGeometrySourceMeshRoute,
};
pub use bloom_geometry_format::{
    ClusterRecord, CompatibilityRecord, GeometryArchive, PageRecord, VertexEncoding,
};
pub use draw_emission::{
    GpuVirtualBinnedSubmissionState, GpuVirtualDispatchIndirect, GpuVirtualDrawEmissionState,
    GpuVirtualDrawEmitter, GpuVirtualDrawIndirect, VirtualGeometryDrawEmissionError,
    VirtualGeometrySubmissionMode,
};
pub use gpu_pool::{
    GpuPageTransition, GpuVirtualClusterEntry, GpuVirtualGeometryConfig, GpuVirtualGeometryPool,
    GpuVirtualGeometryTelemetry, GpuVirtualMeshEntry, GpuVirtualPageEntry, VirtualGeometryGpuError,
    VirtualMaterialBinding, VirtualMeshId, VirtualPageId, GPU_VIRTUAL_MESH_MATERIALS_BOUND,
    GPU_VIRTUAL_MESH_VALID, GPU_VIRTUAL_PAGE_PINNED, GPU_VIRTUAL_PAGE_RESIDENT,
};
pub use hiz::GpuVirtualHiZTelemetry;
pub(crate) use hiz::VirtualGeometryHiZFrame;
pub use residency::{
    ClusterGroup, PageTransition, ResidencyError, ResidencyTelemetry, ResolvedClusterGroup,
    VirtualGeometryResidency,
};
pub use shading::GpuVirtualVisibilityShading;
#[cfg(not(target_arch = "wasm32"))]
pub use store::{
    ResolvedVirtualGeometryAsset, VirtualGeometryAssetProfile, VirtualGeometrySelectionKind,
    VirtualGeometryStoreConfig, VirtualGeometryStoreError, VirtualGeometryStoreLoader,
    VirtualGeometryStoreRequest, VirtualGeometryStoreRequestPolicy, VirtualGeometryStoreSelection,
    VirtualGeometryStoreTelemetry, VirtualGeometryStoreTicket,
};
pub use streaming::{
    GpuVirtualPageStreamer, GpuVirtualStreamingConfig, GpuVirtualStreamingError,
    GpuVirtualStreamingTelemetry,
};
pub use traversal::{
    GpuSelectedVirtualCluster, GpuVirtualHierarchySelector, GpuVirtualInstance,
    GpuVirtualPageRequest, GpuVirtualPageUse, GpuVirtualTraversalConfig,
    GpuVirtualTraversalCounters, VirtualGeometryTraversalDispatch, VirtualGeometryTraversalError,
    VirtualGeometryView,
};
pub use visibility::{
    GpuVirtualVisibilityFrame, GpuVirtualVisibilityRaster, VirtualGeometryVisibilityError,
};

#[cfg(test)]
mod tests;
