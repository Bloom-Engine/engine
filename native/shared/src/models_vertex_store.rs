//! A mesh's CPU vertices, stored by field and losslessly compact.
//!
//! A loaded model keeps its vertices on the CPU after they are uploaded: the
//! software GI field gathers positions from them whenever the scene changes,
//! a scene node made again after its mesh's last node went uploads them
//! again, and picking reads them. As `Vec<Vertex3D>` that is 96 bytes a
//! vertex; a city scene on an iPad held 681 MB of them (2026-11-25). Most of
//! a static mesh's fields are the same for every vertex (colour, joints,
//! weights, often the tangent), so each field is its own column, and a column
//! whose every value is the same holds that value once. A static mesh keeps
//! 32 to 48 bytes a vertex. `to_vertices()` gives back the exact
//! `Vec<Vertex3D>` (bit for bit), for an upload or anything else that needs
//! whole vertices; positions are a slice, with no copy.

use crate::renderer::Vertex3D;

#[derive(Clone, Debug)]
enum Column<T: Copy> {
    Same(T),
    Each(Vec<T>),
}

impl<T: Copy> Column<T> {
    fn get(&self, i: usize) -> T {
        match self {
            Column::Same(v) => *v,
            Column::Each(v) => v[i],
        }
    }
    fn bytes(&self) -> usize {
        match self {
            Column::Same(_) => std::mem::size_of::<T>(),
            Column::Each(v) => v.len() * std::mem::size_of::<T>(),
        }
    }
}

/// One column from `field` of each vertex: a single value when every vertex
/// has the same bits, else every value.
fn column<const N: usize>(
    vertices: &[Vertex3D],
    field: impl Fn(&Vertex3D) -> [f32; N],
) -> Column<[f32; N]> {
    let Some(first) = vertices.first().map(&field) else {
        return Column::Same([0.0; N]);
    };
    let same =
        |a: &[f32; N], b: &[f32; N]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    if vertices.iter().all(|v| same(&field(v), &first)) {
        Column::Same(first)
    } else {
        Column::Each(vertices.iter().map(field).collect())
    }
}

#[derive(Clone, Debug)]
pub struct VertexStore {
    positions: Vec<[f32; 3]>,
    normals: Column<[f32; 3]>,
    colors: Column<[f32; 4]>,
    uvs: Column<[f32; 2]>,
    joints: Column<[f32; 4]>,
    weights: Column<[f32; 4]>,
    tangents: Column<[f32; 4]>,
}

impl Default for VertexStore {
    fn default() -> Self {
        Self::from_vertices(&[])
    }
}

impl VertexStore {
    pub fn from_vertices(vertices: &[Vertex3D]) -> Self {
        Self {
            positions: vertices.iter().map(|v| v.position).collect(),
            normals: column(vertices, |v| v.normal),
            colors: column(vertices, |v| v.color),
            uvs: column(vertices, |v| v.uv),
            joints: column(vertices, |v| v.joints),
            weights: column(vertices, |v| v.weights),
            tangents: column(vertices, |v| v.tangent),
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Every vertex's position (a slice of the store: no copy).
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }

    /// Vertex `i` whole.
    pub fn get(&self, i: usize) -> Vertex3D {
        Vertex3D {
            position: self.positions[i],
            normal: self.normals.get(i),
            color: self.colors.get(i),
            uv: self.uvs.get(i),
            joints: self.joints.get(i),
            weights: self.weights.get(i),
            tangent: self.tangents.get(i),
        }
    }

    pub fn uv(&self, i: usize) -> [f32; 2] {
        self.uvs.get(i)
    }

    pub fn normal(&self, i: usize) -> [f32; 3] {
        self.normals.get(i)
    }

    pub fn weight(&self, i: usize) -> [f32; 4] {
        self.weights.get(i)
    }

    pub fn joint(&self, i: usize) -> [f32; 4] {
        self.joints.get(i)
    }

    /// The sum of every normal, added in vertex order (as a loop over the
    /// whole vertices would, so the result has the same bits).
    pub fn normal_sum(&self) -> [f32; 3] {
        let mut sum = [0.0f32; 3];
        for i in 0..self.len() {
            let n = self.normals.get(i);
            sum[0] += n[0];
            sum[1] += n[1];
            sum[2] += n[2];
        }
        sum
    }

    /// The vertices whole, exactly as they were stored.
    pub fn to_vertices(&self) -> Vec<Vertex3D> {
        (0..self.len()).map(|i| self.get(i)).collect()
    }

    /// Bytes this store holds (for reports).
    pub fn bytes(&self) -> usize {
        self.positions.len() * 12
            + self.normals.bytes()
            + self.colors.bytes()
            + self.uvs.bytes()
            + self.joints.bytes()
            + self.weights.bytes()
            + self.tangents.bytes()
    }
}

impl From<Vec<Vertex3D>> for VertexStore {
    fn from(vertices: Vec<Vertex3D>) -> Self {
        Self::from_vertices(&vertices)
    }
}

impl From<&[Vertex3D]> for VertexStore {
    fn from(vertices: &[Vertex3D]) -> Self {
        Self::from_vertices(vertices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(i: usize, skinned: bool) -> Vertex3D {
        let f = i as f32;
        Vertex3D {
            position: [f, f * 2.0, -f],
            normal: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv: [f * 0.1, 0.5],
            joints: if skinned {
                [f, 1.0, 0.0, 0.0]
            } else {
                [0.0; 4]
            },
            weights: if skinned {
                [0.75, 0.25, 0.0, 0.0]
            } else {
                [0.0; 4]
            },
            tangent: [1.0, 0.0, 0.0, if i % 2 == 0 { 1.0 } else { -1.0 }],
        }
    }

    fn bits(vs: &[Vertex3D]) -> Vec<u32> {
        bytemuck::cast_slice::<Vertex3D, u32>(vs).to_vec()
    }

    #[test]
    fn vertices_come_back_bit_for_bit() {
        for skinned in [false, true] {
            let mut vs: Vec<Vertex3D> = (0..1000).map(|i| v(i, skinned)).collect();
            vs[7].normal = [-0.0, 1.0, f32::NAN]; // signed zero and NaN keep their bits
            let store = VertexStore::from_vertices(&vs);
            assert_eq!(store.len(), vs.len());
            assert_eq!(bits(&store.to_vertices()), bits(&vs));
            assert_eq!(store.positions()[5], vs[5].position);
            assert_eq!(bits(&[store.get(7)]), bits(&[vs[7]]));
        }
        assert!(VertexStore::from_vertices(&[]).to_vertices().is_empty());
    }

    #[test]
    fn a_static_mesh_takes_well_under_half() {
        let vs: Vec<Vertex3D> = (0..1000).map(|i| v(i, false)).collect();
        let store = VertexStore::from_vertices(&vs);
        // positions, uvs and tangents vary; normal, colour, joints and weights are one value each
        assert!(store.bytes() < vs.len() * 40, "{} bytes", store.bytes());
    }
}
