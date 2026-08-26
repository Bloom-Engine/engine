use super::ModelData;

/// Stable glTF identity for one drawable model placement. Repeated scene
/// nodes share mesh/primitive identity and differ by `placement_index`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ModelPrimitiveSource {
    pub mesh_index: u32,
    pub primitive_index: u32,
    pub placement_index: u32,
}

impl ModelData {
    pub fn mesh_transform(&self, index: usize) -> [[f32; 4]; 4] {
        self.mesh_transforms
            .get(index)
            .copied()
            .unwrap_or(crate::renderer::IDENTITY_MAT4)
    }

    pub fn mesh_cast_shadow(&self, index: usize) -> bool {
        self.mesh_cast_shadows.get(index).copied().unwrap_or(true)
    }

    pub fn mesh_source(&self, index: usize) -> Option<ModelPrimitiveSource> {
        self.mesh_sources.get(index).copied().flatten()
    }
}

#[cfg(feature = "models3d")]
#[path = "models_virtual_geometry.rs"]
mod virtual_geometry_routing;
#[cfg(feature = "models3d")]
pub use virtual_geometry_routing::{
    ModelVirtualCompatibilityPlacement, ModelVirtualCompatibilityRoute, ModelVirtualGeometryRoute,
    ModelVirtualGeometryRouteError, ModelVirtualPlacement,
};
