//! Opt-in runtime ownership and fixed-budget residency for cooked geometry.
//!
//! This module has no renderer registration or default-path state. A caller
//! must explicitly load a `.bgeo` archive and create a residency plan.

mod asset;
mod decode;
mod draw_emission;
mod gpu_pool;
mod residency;
mod traversal;

pub use asset::{ArtifactIdentity, VirtualGeometryAsset, VirtualGeometryLoadError};
pub use bloom_geometry_format::{
    ClusterRecord, CompatibilityRecord, GeometryArchive, PageRecord, VertexEncoding,
};
pub use draw_emission::{
    GpuVirtualDispatchIndirect, GpuVirtualDrawEmissionState, GpuVirtualDrawEmitter,
    GpuVirtualDrawIndirect, VirtualGeometryDrawEmissionError,
};
pub use gpu_pool::{
    GpuPageTransition, GpuVirtualClusterEntry, GpuVirtualGeometryConfig, GpuVirtualGeometryPool,
    GpuVirtualGeometryTelemetry, GpuVirtualMeshEntry, GpuVirtualPageEntry, VirtualGeometryGpuError,
    VirtualMaterialBinding, VirtualMeshId, VirtualPageId, GPU_VIRTUAL_MESH_MATERIALS_BOUND,
    GPU_VIRTUAL_MESH_VALID, GPU_VIRTUAL_PAGE_PINNED, GPU_VIRTUAL_PAGE_RESIDENT,
};
pub use residency::{
    ClusterGroup, PageTransition, ResidencyError, ResidencyTelemetry, ResolvedClusterGroup,
    VirtualGeometryResidency,
};
pub use traversal::{
    GpuSelectedVirtualCluster, GpuVirtualHierarchySelector, GpuVirtualInstance,
    GpuVirtualPageRequest, GpuVirtualTraversalConfig, GpuVirtualTraversalCounters,
    VirtualGeometryTraversalDispatch, VirtualGeometryTraversalError, VirtualGeometryView,
};

#[cfg(test)]
mod tests;
