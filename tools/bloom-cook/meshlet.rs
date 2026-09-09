//! Deterministic leaf-meshlet construction for cooked virtual geometry.
//!
//! The builder deliberately preserves source triangle order. More elaborate
//! locality and hierarchy optimization can replace this implementation behind
//! the versioned format after measured GPU/cache experiments; the public
//! engine API is not coupled to these limits.

pub const DEFAULT_MAX_VERTICES: u32 = 64;
pub const DEFAULT_MAX_TRIANGLES: u32 = 124;
pub const MAX_ENCODED_VERTICES: u32 = u8::MAX as u32;
pub use bloom_geometry_format::{
    FLAG_ALPHA_MASKED, FLAG_COARSE_ROOT, FLAG_DOUBLE_SIDED, NO_RELATION,
};
/// The cluster is a root of the cooked hierarchy and must be part of the
/// coarse always-resident set before runtime traversal can be enabled.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    pub color: [f32; 4],
}

impl StaticVertex {
    pub const ENCODED_BYTES: u32 = bloom_geometry_format::FLOAT32_VERTEX_BYTES;
}

#[derive(Clone, Debug)]
pub struct StaticPrimitive {
    pub mesh_index: u32,
    pub primitive_index: u32,
    pub material_index: Option<u32>,
    pub double_sided: bool,
    pub alpha_masked: bool,
    pub vertices: Vec<StaticVertex>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshletLimits {
    pub max_vertices: u32,
    pub max_triangles: u32,
}

impl Default for MeshletLimits {
    fn default() -> Self {
        Self {
            max_vertices: DEFAULT_MAX_VERTICES,
            max_triangles: DEFAULT_MAX_TRIANGLES,
        }
    }
}

impl MeshletLimits {
    pub fn validate(self) -> Result<Self, String> {
        if !(3..=MAX_ENCODED_VERTICES).contains(&self.max_vertices) {
            return Err(format!(
                "meshlet max vertices must be in 3..={MAX_ENCODED_VERTICES}, got {}",
                self.max_vertices
            ));
        }
        if self.max_triangles == 0 {
            return Err("meshlet max triangles must be greater than zero".to_string());
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshletBounds {
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub sphere_center: [f32; 3],
    pub sphere_radius: f32,
    pub normal_cone_axis: [f32; 3],
    /// Minimum dot product between the cone axis and a triangle normal.
    /// `-1` disables cone rejection conservatively.
    pub normal_cone_cutoff: f32,
}

#[derive(Clone, Debug)]
pub struct Meshlet {
    pub mesh_index: u32,
    pub primitive_index: u32,
    pub material_index: Option<u32>,
    pub flags: u32,
    pub vertices: Vec<StaticVertex>,
    pub local_indices: Vec<u8>,
    pub bounds: MeshletBounds,
    /// Leaf clusters have zero geometric error. Hierarchy construction will
    /// assign non-zero parent errors without changing the format.
    pub geometric_error: f32,
    pub lod_level: u32,
    /// First cluster in the atomic parent replacement group.
    pub parent: u32,
    /// Number of clusters in the atomic parent replacement group. Zero when
    /// `parent` is absent.
    pub parent_count: u32,
    /// First cluster in the atomic child group replaced by this cluster and
    /// its siblings.
    pub first_child: u32,
    pub child_count: u32,
}

impl Meshlet {
    pub fn triangle_count(&self) -> u32 {
        (self.local_indices.len() / 3) as u32
    }

    pub fn encoded_payload_bytes(&self) -> usize {
        self.vertices.len() * StaticVertex::ENCODED_BYTES as usize + self.local_indices.len()
    }
}

pub fn build_leaf_meshlets(
    primitive: &StaticPrimitive,
    limits: MeshletLimits,
) -> Result<Vec<Meshlet>, String> {
    let limits = limits.validate()?;
    validate_primitive(primitive)?;

    let mut result = Vec::new();
    let mut source_vertices = Vec::<u32>::with_capacity(limits.max_vertices as usize);
    let mut local_indices = Vec::<u8>::with_capacity(limits.max_triangles as usize * 3);

    for triangle in primitive.indices.as_chunks::<3>().0 {
        let additional_vertices = triangle
            .iter()
            .filter(|index| !source_vertices.contains(index))
            .count() as u32;
        let triangle_count = local_indices.len() as u32 / 3;
        let would_overflow = !local_indices.is_empty()
            && (source_vertices.len() as u32 + additional_vertices > limits.max_vertices
                || triangle_count + 1 > limits.max_triangles);
        if would_overflow {
            result.push(finish_meshlet(primitive, &source_vertices, &local_indices));
            source_vertices.clear();
            local_indices.clear();
        }

        for source_index in triangle {
            let local_index = match source_vertices
                .iter()
                .position(|candidate| candidate == source_index)
            {
                Some(index) => index,
                None => {
                    source_vertices.push(*source_index);
                    source_vertices.len() - 1
                }
            };
            local_indices.push(local_index as u8);
        }
    }

    if !local_indices.is_empty() {
        result.push(finish_meshlet(primitive, &source_vertices, &local_indices));
    }
    Ok(result)
}

pub(crate) fn validate_primitive(primitive: &StaticPrimitive) -> Result<(), String> {
    if primitive.vertices.is_empty() {
        return Err(format!(
            "mesh {} primitive {} has no vertices",
            primitive.mesh_index, primitive.primitive_index
        ));
    }
    if primitive.indices.is_empty() || !primitive.indices.len().is_multiple_of(3) {
        return Err(format!(
            "mesh {} primitive {} index count {} is not a non-empty triangle list",
            primitive.mesh_index,
            primitive.primitive_index,
            primitive.indices.len()
        ));
    }
    for (vertex_index, vertex) in primitive.vertices.iter().enumerate() {
        for (attribute, components) in [
            ("POSITION", vertex.position.as_slice()),
            ("NORMAL", vertex.normal.as_slice()),
            ("TANGENT", vertex.tangent.as_slice()),
            ("TEXCOORD_0", vertex.uv0.as_slice()),
            ("TEXCOORD_1", vertex.uv1.as_slice()),
            ("COLOR_0", vertex.color.as_slice()),
        ] {
            if let Some((component_index, component)) = components
                .iter()
                .enumerate()
                .find(|(_, component)| !component.is_finite())
            {
                return Err(format!(
                    "mesh {} primitive {} vertex {vertex_index} {attribute}[{component_index}]={component:?} contains NaN/Inf",
                    primitive.mesh_index, primitive.primitive_index
                ));
            }
        }
    }
    if let Some(index) = primitive
        .indices
        .iter()
        .find(|index| **index as usize >= primitive.vertices.len())
    {
        return Err(format!(
            "mesh {} primitive {} index {index} exceeds {} vertices",
            primitive.mesh_index,
            primitive.primitive_index,
            primitive.vertices.len()
        ));
    }
    Ok(())
}

fn finish_meshlet(
    primitive: &StaticPrimitive,
    source_indices: &[u32],
    local_indices: &[u8],
) -> Meshlet {
    let vertices: Vec<_> = source_indices
        .iter()
        .map(|index| primitive.vertices[*index as usize])
        .collect();
    let flags = (u32::from(primitive.double_sided) * FLAG_DOUBLE_SIDED)
        | (u32::from(primitive.alpha_masked) * FLAG_ALPHA_MASKED);
    let bounds = calculate_bounds(&vertices, local_indices, primitive.double_sided);
    Meshlet {
        mesh_index: primitive.mesh_index,
        primitive_index: primitive.primitive_index,
        material_index: primitive.material_index,
        flags,
        vertices,
        local_indices: local_indices.to_vec(),
        bounds,
        geometric_error: 0.0,
        lod_level: 0,
        parent: NO_RELATION,
        parent_count: 0,
        first_child: NO_RELATION,
        child_count: 0,
    }
}

fn calculate_bounds(
    vertices: &[StaticVertex],
    local_indices: &[u8],
    double_sided: bool,
) -> MeshletBounds {
    let mut aabb_min = [f32::INFINITY; 3];
    let mut aabb_max = [f32::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            aabb_min[axis] = aabb_min[axis].min(vertex.position[axis]);
            aabb_max[axis] = aabb_max[axis].max(vertex.position[axis]);
        }
    }
    let sphere_center = [
        (aabb_min[0] + aabb_max[0]) * 0.5,
        (aabb_min[1] + aabb_max[1]) * 0.5,
        (aabb_min[2] + aabb_max[2]) * 0.5,
    ];
    let sphere_radius = vertices
        .iter()
        .map(|vertex| length3(sub3(vertex.position, sphere_center)))
        .fold(0.0, f32::max);

    let mut face_normals = Vec::with_capacity(local_indices.len() / 3);
    let mut normal_sum = [0.0; 3];
    for triangle in local_indices.as_chunks::<3>().0 {
        let a = vertices[triangle[0] as usize].position;
        let b = vertices[triangle[1] as usize].position;
        let c = vertices[triangle[2] as usize].position;
        let cross = cross3(sub3(b, a), sub3(c, a));
        let length = length3(cross);
        if length > 1e-20 {
            let normal = mul3(cross, length.recip());
            face_normals.push(normal);
            normal_sum = add3(normal_sum, normal);
        }
    }
    let sum_length = length3(normal_sum);
    let (normal_cone_axis, normal_cone_cutoff) =
        if double_sided || face_normals.is_empty() || sum_length <= 1e-6 {
            ([0.0, 0.0, 1.0], -1.0)
        } else {
            let axis = mul3(normal_sum, sum_length.recip());
            let cutoff = face_normals
                .iter()
                .map(|normal| dot3(axis, *normal))
                .fold(1.0, f32::min)
                .clamp(-1.0, 1.0);
            (axis, cutoff)
        };

    MeshletBounds {
        aabb_min,
        aabb_max,
        sphere_center,
        sphere_radius,
        normal_cone_axis,
        normal_cone_cutoff,
    }
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul3(value: [f32; 3], factor: f32) -> [f32; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length3(value: [f32; 3]) -> f32 {
    dot3(value, value).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(x: f32, y: f32, z: f32) -> StaticVertex {
        StaticVertex {
            position: [x, y, z],
            normal: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv0: [x, y],
            uv1: [x, y],
            color: [1.0; 4],
        }
    }

    fn primitive(vertices: Vec<StaticVertex>, indices: Vec<u32>) -> StaticPrimitive {
        StaticPrimitive {
            mesh_index: 2,
            primitive_index: 3,
            material_index: Some(7),
            double_sided: false,
            alpha_masked: true,
            vertices,
            indices,
        }
    }

    #[test]
    fn partition_preserves_triangle_order_and_limits() {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for triangle in 0..11 {
            let base = vertices.len() as u32;
            vertices.push(vertex(triangle as f32, 0.0, 0.0));
            vertices.push(vertex(triangle as f32, 1.0, 0.0));
            vertices.push(vertex(triangle as f32, 0.0, 1.0));
            indices.extend([base, base + 1, base + 2]);
        }
        let source = primitive(vertices.clone(), indices);
        let meshlets = build_leaf_meshlets(
            &source,
            MeshletLimits {
                max_vertices: 9,
                max_triangles: 4,
            },
        )
        .unwrap();
        assert_eq!(meshlets.len(), 4);
        assert!(meshlets
            .iter()
            .all(|meshlet| meshlet.vertices.len() <= 9 && meshlet.triangle_count() <= 4));

        let reconstructed: Vec<_> = meshlets
            .iter()
            .flat_map(|meshlet| {
                meshlet
                    .local_indices
                    .iter()
                    .map(|index| meshlet.vertices[*index as usize].position)
            })
            .collect();
        let expected: Vec<_> = source
            .indices
            .iter()
            .map(|index| vertices[*index as usize].position)
            .collect();
        assert_eq!(reconstructed, expected);
    }

    #[test]
    fn bounds_and_normal_cone_are_conservative() {
        let source = primitive(
            vec![
                vertex(-1.0, -2.0, 0.0),
                vertex(3.0, -2.0, 0.0),
                vertex(-1.0, 2.0, 0.0),
            ],
            vec![0, 1, 2],
        );
        let meshlet = build_leaf_meshlets(&source, MeshletLimits::default()).unwrap()[0].clone();
        assert_eq!(meshlet.bounds.aabb_min, [-1.0, -2.0, 0.0]);
        assert_eq!(meshlet.bounds.aabb_max, [3.0, 2.0, 0.0]);
        assert_eq!(meshlet.bounds.sphere_center, [1.0, 0.0, 0.0]);
        assert!((meshlet.bounds.sphere_radius - 2.0_f32.sqrt() * 2.0).abs() < 1e-6);
        assert_eq!(meshlet.bounds.normal_cone_axis, [0.0, 0.0, 1.0]);
        assert_eq!(meshlet.bounds.normal_cone_cutoff, 1.0);
        assert_eq!(meshlet.flags, FLAG_ALPHA_MASKED);
    }

    #[test]
    fn double_sided_meshlets_disable_cone_rejection() {
        let mut source = primitive(
            vec![
                vertex(0.0, 0.0, 0.0),
                vertex(1.0, 0.0, 0.0),
                vertex(0.0, 1.0, 0.0),
            ],
            vec![0, 1, 2],
        );
        source.double_sided = true;
        let meshlet = &build_leaf_meshlets(&source, MeshletLimits::default()).unwrap()[0];
        assert_eq!(meshlet.bounds.normal_cone_cutoff, -1.0);
        assert_ne!(meshlet.flags & FLAG_DOUBLE_SIDED, 0);
    }

    #[test]
    fn invalid_source_is_rejected_before_indexing() {
        let mut source = primitive(vec![vertex(0.0, 0.0, 0.0); 3], vec![0, 1, 4]);
        assert!(build_leaf_meshlets(&source, MeshletLimits::default())
            .unwrap_err()
            .contains("exceeds"));
        source.indices = vec![0, 1];
        assert!(build_leaf_meshlets(&source, MeshletLimits::default())
            .unwrap_err()
            .contains("triangle list"));
        source.indices = vec![0, 1, 2];
        source.vertices[1].position[0] = f32::NAN;
        let error = build_leaf_meshlets(&source, MeshletLimits::default()).unwrap_err();
        assert!(error.contains("vertex 1 POSITION[0]=NaN"), "{error}");
        assert!(error.contains("NaN/Inf"), "{error}");
    }
}
