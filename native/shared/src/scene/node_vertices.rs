//! A scene node's vertices, read without copying the shared primitive's
//! compact store (models_vertex_store.rs) where a count, a position or a sum
//! is enough.

use super::SceneNode;
use crate::renderer::{Vertex3D, VertexLayout};

impl SceneNode {
    /// Read the node's effective local vertex payload. Imported geometry may
    /// be backed by an immutable shared model primitive.
    ///
    /// Whole vertices: borrowed for geometry the node owns, rebuilt from the
    /// compact store for a shared model primitive. Counts, positions and
    /// normals have their own accessors that never rebuild.
    pub fn vertices(&self) -> std::borrow::Cow<'_, [Vertex3D]> {
        match self.shared_model_mesh.as_ref() {
            Some(mesh) => std::borrow::Cow::Owned(mesh.vertices.to_vertices()),
            None => std::borrow::Cow::Borrowed(self.vertices.as_slice()),
        }
    }

    /// An address that stays the same while the node shows the same
    /// geometry: the shared primitive's, or the node's own vertices'.
    pub(crate) fn geometry_identity(&self) -> usize {
        match self.shared_model_mesh.as_ref() {
            Some(mesh) => std::sync::Arc::as_ptr(mesh) as usize,
            None => self.vertices.as_ptr() as usize,
        }
    }

    /// The format the node's vertices go to the GPU in (vertex_formats.rs).
    pub(crate) fn vertex_layout(&self) -> VertexLayout {
        match self.shared_model_mesh.as_ref() {
            Some(mesh) => VertexLayout::for_store(&mesh.vertices),
            None => VertexLayout::for_vertices(&self.vertices),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.shared_model_mesh
            .as_ref()
            .map_or(self.vertices.len(), |mesh| mesh.vertices.len())
    }

    /// Vertex `i` whole.
    pub fn vertex(&self, i: usize) -> Vertex3D {
        match self.shared_model_mesh.as_ref() {
            Some(mesh) => mesh.vertices.get(i),
            None => self.vertices[i],
        }
    }

    /// Every vertex's position (a slice of the shared primitive's store).
    pub fn positions(&self) -> std::borrow::Cow<'_, [[f32; 3]]> {
        match self.shared_model_mesh.as_ref() {
            Some(mesh) => std::borrow::Cow::Borrowed(mesh.vertices.positions()),
            None => std::borrow::Cow::Owned(self.vertices.iter().map(|v| v.position).collect()),
        }
    }

    /// The sum of every vertex normal, in vertex order.
    pub fn normal_sum(&self) -> [f32; 3] {
        match self.shared_model_mesh.as_ref() {
            Some(mesh) => mesh.vertices.normal_sum(),
            None => {
                let mut sum = [0.0f32; 3];
                for v in &self.vertices {
                    sum[0] += v.normal[0];
                    sum[1] += v.normal[1];
                    sum[2] += v.normal[2];
                }
                sum
            }
        }
    }
}
