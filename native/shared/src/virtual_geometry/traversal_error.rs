use super::{VirtualGeometryGpuError, VirtualMeshId};
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VirtualGeometryTraversalError {
    InvalidConfig,
    DeviceUnsupported,
    DeviceLimitExceeded {
        resource: &'static str,
        requested_bytes: u64,
        maximum_bytes: u64,
    },
    PoolMismatch,
    TooManyInstances {
        requested: usize,
        capacity: u32,
    },
    InvalidView,
    InvalidInstanceTransform {
        instance: u32,
    },
    UnboundMaterials {
        mesh: VirtualMeshId,
    },
    SourceMeshFilterRequired {
        mesh: VirtualMeshId,
    },
    SourceMeshNotVirtual {
        mesh: VirtualMeshId,
        source_mesh_index: u32,
    },
    DispatchLimitExceeded {
        requested: u32,
        maximum: u32,
    },
    Pool(VirtualGeometryGpuError),
}

impl From<VirtualGeometryGpuError> for VirtualGeometryTraversalError {
    fn from(value: VirtualGeometryGpuError) -> Self {
        Self::Pool(value)
    }
}

impl fmt::Display for VirtualGeometryTraversalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(formatter, "invalid virtual-geometry traversal config"),
            Self::DeviceUnsupported => write!(
                formatter,
                "device lacks the storage-buffer or compute limits required by virtual-geometry traversal"
            ),
            Self::DeviceLimitExceeded {
                resource,
                requested_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "virtual-geometry traversal {resource} requires {requested_bytes} bytes but the device limit is {maximum_bytes}"
            ),
            Self::PoolMismatch => write!(
                formatter,
                "virtual-geometry selector was recorded with a different page pool"
            ),
            Self::TooManyInstances {
                requested,
                capacity,
            } => write!(
                formatter,
                "virtual-geometry traversal received {requested} instances but has capacity for {capacity}"
            ),
            Self::InvalidView => write!(formatter, "invalid virtual-geometry camera data"),
            Self::InvalidInstanceTransform { instance } => write!(
                formatter,
                "virtual-geometry instance {instance} has a non-finite or singular transform"
            ),
            Self::UnboundMaterials { mesh } => write!(
                formatter,
                "virtual mesh {} has no complete GPU material binding",
                mesh.raw()
            ),
            Self::SourceMeshFilterRequired { mesh } => write!(
                formatter,
                "multi-source virtual mesh {} requires an explicit source glTF mesh filter",
                mesh.raw()
            ),
            Self::SourceMeshNotVirtual {
                mesh,
                source_mesh_index,
            } => write!(
                formatter,
                "virtual mesh {} has no eligible clusters for source glTF mesh {}",
                mesh.raw(), source_mesh_index
            ),
            Self::DispatchLimitExceeded { requested, maximum } => write!(
                formatter,
                "virtual-geometry traversal needs {requested} workgroups in one dimension but the device limit is {maximum}"
            ),
            Self::Pool(error) => error.fmt(formatter),
        }
    }
}
