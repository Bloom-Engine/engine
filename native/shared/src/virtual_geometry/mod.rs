//! Opt-in runtime ownership and fixed-budget residency for cooked geometry.
//!
//! This module has no renderer registration or default-path state. A caller
//! must explicitly load a `.bgeo` archive and create a residency plan.

mod asset;
mod gpu_pool;
mod residency;

pub use asset::{ArtifactIdentity, VirtualGeometryAsset, VirtualGeometryLoadError};
pub use bloom_geometry_format::{
    ClusterRecord, CompatibilityRecord, GeometryArchive, PageRecord, VertexEncoding,
};
pub use gpu_pool::{
    GpuPageTransition, GpuVirtualGeometryConfig, GpuVirtualGeometryPool,
    GpuVirtualGeometryTelemetry, GpuVirtualMeshEntry, GpuVirtualPageEntry, VirtualGeometryGpuError,
    VirtualMeshId, VirtualPageId, GPU_VIRTUAL_PAGE_PINNED, GPU_VIRTUAL_PAGE_RESIDENT,
};
pub use residency::{
    ClusterGroup, PageTransition, ResidencyError, ResidencyTelemetry, ResolvedClusterGroup,
    VirtualGeometryResidency,
};

#[cfg(test)]
mod tests;
