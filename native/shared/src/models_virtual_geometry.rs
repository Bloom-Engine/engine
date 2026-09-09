use super::{ModelData, ModelPrimitiveSource};
use crate::renderer::mat4_multiply;
use crate::virtual_geometry::{
    CompatibilityRecord, GeometryArchive, GpuVirtualInstance, VirtualGeometryAsset,
    VirtualGeometryTraversalError, VirtualMeshId,
};
use bloom_geometry_format::FLAG_ALPHA_MASKED;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// One ordinary glTF node placement that has at least one virtual-eligible
/// primitive. The source-mesh filter prevents that placement from traversing
/// any other mesh stored in the shared `.bgeo` archive.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelVirtualPlacement {
    pub source_mesh_index: u32,
    pub placement_index: u32,
    pub model_transform: [[f32; 4]; 4],
    pub cast_shadow: bool,
}

/// One drawable placement retained by the established renderer, including the
/// cooker's inspectable reason for excluding its primitive from virtualization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelVirtualCompatibilityPlacement {
    pub model_mesh_index: usize,
    pub source: ModelPrimitiveSource,
    pub route: ModelVirtualCompatibilityRoute,
}

/// Why one source primitive remains on Bloom's established cached renderer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelVirtualCompatibilityRoute {
    Cooked(CompatibilityRecord),
    /// The archive contains static clusters, but virtual visibility does not
    /// yet own the material's exact alpha-coverage test.
    AlphaMaskedVisibility {
        material_index: Option<u32>,
    },
}

/// Complete, fail-closed split for one runtime model and its exact cooked
/// source closure. Callers submit `virtual_placements` through filtered
/// `GpuVirtualInstance`s and draw every `compatibility_placement` normally.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelVirtualGeometryRoute {
    pub virtual_placements: Vec<ModelVirtualPlacement>,
    pub compatibility_placements: Vec<ModelVirtualCompatibilityPlacement>,
}

impl ModelVirtualGeometryRoute {
    pub fn compatibility_model_mesh_indices(&self) -> Vec<usize> {
        self.compatibility_placements
            .iter()
            .map(|placement| placement.model_mesh_index)
            .collect()
    }

    /// Build the exact virtual half of this route for one outer model
    /// placement. Source-node transforms remain shared with the ordinary
    /// cached renderer, while the source-mesh filter prevents other meshes in
    /// the same `.bgeo` archive from being traversed for each placement.
    pub fn virtual_instances(
        &self,
        mesh: VirtualMeshId,
        first_instance_id: u32,
        model: [[f32; 4]; 4],
        previous_model: [[f32; 4]; 4],
        tint: [f32; 4],
    ) -> Result<Vec<GpuVirtualInstance>, ModelVirtualGeometryRouteError> {
        let mut instances = Vec::with_capacity(self.virtual_placements.len());
        for (offset, placement) in self.virtual_placements.iter().enumerate() {
            let offset = u32::try_from(offset)
                .map_err(|_| ModelVirtualGeometryRouteError::InstanceIdOverflow)?;
            let instance_id = first_instance_id
                .checked_add(offset)
                .ok_or(ModelVirtualGeometryRouteError::InstanceIdOverflow)?;
            instances.push(GpuVirtualInstance::with_source_mesh_render_state(
                mesh,
                placement.source_mesh_index,
                instance_id,
                mat4_multiply(model, placement.model_transform),
                mat4_multiply(previous_model, placement.model_transform),
                tint,
            )?);
        }
        Ok(instances)
    }
}

impl ModelData {
    /// Correlate a loaded glTF model with an exact `.bgeo` source closure.
    /// Every drawable primitive must appear in exactly one cooked partition;
    /// incomplete loader metadata, source mismatch, or an unlisted primitive
    /// rejects the complete route before either renderer path is submitted.
    pub fn route_virtual_geometry(
        &self,
        asset: &VirtualGeometryAsset,
    ) -> Result<ModelVirtualGeometryRoute, ModelVirtualGeometryRouteError> {
        let source_geometry_sha256 = self
            .source_geometry_sha256
            .ok_or(ModelVirtualGeometryRouteError::MissingSourceClosure)?;
        if source_geometry_sha256 != asset.archive().source_sha256 {
            return Err(ModelVirtualGeometryRouteError::SourceClosureMismatch);
        }
        build_route(self, asset.archive())
    }
}

fn build_route(
    model: &ModelData,
    archive: &GeometryArchive,
) -> Result<ModelVirtualGeometryRoute, ModelVirtualGeometryRouteError> {
    if model.mesh_sources.len() != model.meshes.len()
        || model.mesh_transforms.len() != model.meshes.len()
        || model.mesh_cast_shadows.len() != model.meshes.len()
    {
        return Err(
            ModelVirtualGeometryRouteError::IncompletePlacementMetadata {
                meshes: model.meshes.len(),
                sources: model.mesh_sources.len(),
                transforms: model.mesh_transforms.len(),
                cast_shadows: model.mesh_cast_shadows.len(),
            },
        );
    }

    let eligible = archive
        .clusters
        .iter()
        .filter(|cluster| cluster.flags & FLAG_ALPHA_MASKED == 0)
        .map(|cluster| (cluster.mesh_index, cluster.primitive_index))
        .collect::<BTreeSet<_>>();
    let alpha_masked = archive
        .clusters
        .iter()
        .filter(|cluster| cluster.flags & FLAG_ALPHA_MASKED != 0)
        .map(|cluster| {
            (
                (cluster.mesh_index, cluster.primitive_index),
                cluster.material_index,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let compatibility = archive
        .compatibility
        .iter()
        .map(|record| ((record.mesh_index, record.primitive_index), *record))
        .collect::<BTreeMap<_, _>>();
    let mut seen_primitives = BTreeSet::new();
    let mut virtual_placements = BTreeMap::<(u32, u32), ModelVirtualPlacement>::new();
    let mut compatibility_placements = Vec::new();

    for model_mesh_index in 0..model.meshes.len() {
        let source = model
            .mesh_source(model_mesh_index)
            .ok_or(ModelVirtualGeometryRouteError::MissingPrimitiveSource { model_mesh_index })?;
        let primitive = (source.mesh_index, source.primitive_index);
        seen_primitives.insert(primitive);
        let is_virtual = eligible.contains(&primitive);
        let compatibility_route = compatibility.get(&primitive).copied();
        let alpha_masked_material = alpha_masked.get(&primitive).copied();
        match (is_virtual, compatibility_route, alpha_masked_material) {
            (true, None, None) => {
                let key = (source.mesh_index, source.placement_index);
                let placement = ModelVirtualPlacement {
                    source_mesh_index: source.mesh_index,
                    placement_index: source.placement_index,
                    model_transform: model.mesh_transforms[model_mesh_index],
                    cast_shadow: model.mesh_cast_shadows[model_mesh_index],
                };
                if let Some(previous) = virtual_placements.get(&key) {
                    if previous != &placement {
                        return Err(ModelVirtualGeometryRouteError::InconsistentPlacement {
                            source_mesh_index: source.mesh_index,
                            placement_index: source.placement_index,
                        });
                    }
                } else {
                    virtual_placements.insert(key, placement);
                }
            }
            (false, Some(route), None) => {
                compatibility_placements.push(ModelVirtualCompatibilityPlacement {
                    model_mesh_index,
                    source,
                    route: ModelVirtualCompatibilityRoute::Cooked(route),
                });
            }
            (false, None, Some(material_index)) => {
                compatibility_placements.push(ModelVirtualCompatibilityPlacement {
                    model_mesh_index,
                    source,
                    route: ModelVirtualCompatibilityRoute::AlphaMaskedVisibility { material_index },
                });
            }
            _ => {
                return Err(ModelVirtualGeometryRouteError::UnroutedPrimitive {
                    mesh_index: source.mesh_index,
                    primitive_index: source.primitive_index,
                });
            }
        }
    }

    for &(mesh_index, primitive_index) in eligible
        .iter()
        .chain(compatibility.keys())
        .chain(alpha_masked.keys())
    {
        if !seen_primitives.contains(&(mesh_index, primitive_index)) {
            return Err(
                ModelVirtualGeometryRouteError::ArchivePrimitiveMissingFromModel {
                    mesh_index,
                    primitive_index,
                },
            );
        }
    }

    Ok(ModelVirtualGeometryRoute {
        virtual_placements: virtual_placements.into_values().collect(),
        compatibility_placements,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelVirtualGeometryRouteError {
    MissingSourceClosure,
    SourceClosureMismatch,
    IncompletePlacementMetadata {
        meshes: usize,
        sources: usize,
        transforms: usize,
        cast_shadows: usize,
    },
    MissingPrimitiveSource {
        model_mesh_index: usize,
    },
    UnroutedPrimitive {
        mesh_index: u32,
        primitive_index: u32,
    },
    ArchivePrimitiveMissingFromModel {
        mesh_index: u32,
        primitive_index: u32,
    },
    InconsistentPlacement {
        source_mesh_index: u32,
        placement_index: u32,
    },
    InstanceIdOverflow,
    Traversal(VirtualGeometryTraversalError),
}

impl From<VirtualGeometryTraversalError> for ModelVirtualGeometryRouteError {
    fn from(value: VirtualGeometryTraversalError) -> Self {
        Self::Traversal(value)
    }
}

impl fmt::Display for ModelVirtualGeometryRouteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSourceClosure => write!(
                formatter,
                "model loader did not resolve the complete virtual-geometry source closure"
            ),
            Self::SourceClosureMismatch => write!(
                formatter,
                "model and virtual-geometry archive have different source closures"
            ),
            Self::IncompletePlacementMetadata {
                meshes,
                sources,
                transforms,
                cast_shadows,
            } => write!(
                formatter,
                "model placement metadata is incomplete: {meshes} meshes, {sources} sources, {transforms} transforms, {cast_shadows} shadow flags"
            ),
            Self::MissingPrimitiveSource { model_mesh_index } => write!(
                formatter,
                "model mesh placement {model_mesh_index} has no glTF primitive identity"
            ),
            Self::UnroutedPrimitive {
                mesh_index,
                primitive_index,
            } => write!(
                formatter,
                "glTF mesh {mesh_index} primitive {primitive_index} is not in exactly one cooked route"
            ),
            Self::ArchivePrimitiveMissingFromModel {
                mesh_index,
                primitive_index,
            } => write!(
                formatter,
                "cooked glTF mesh {mesh_index} primitive {primitive_index} is absent from the runtime model"
            ),
            Self::InconsistentPlacement {
                source_mesh_index,
                placement_index,
            } => write!(
                formatter,
                "glTF mesh {source_mesh_index} placement {placement_index} has inconsistent primitive transforms or shadow intent"
            ),
            Self::InstanceIdOverflow => write!(
                formatter,
                "virtual-geometry placement IDs overflowed the 32-bit instance namespace"
            ),
            Self::Traversal(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ModelVirtualGeometryRouteError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{MaterialAlphaMode, MaterialLayeredPbr, MaterialTransmission, MeshData};
    use crate::renderer::{Vertex3D, IDENTITY_MAT4};
    use bloom_geometry_format::{
        ClusterRecord, CompatibilityReason, PageRecord, VertexEncoding, FLAG_COARSE_ROOT,
        NO_RELATION, VERSION,
    };
    use std::sync::Arc;

    fn mesh() -> Arc<MeshData> {
        Arc::new(MeshData {
            vertices: vec![Vertex3D::default(); 3],
            secondary_tex_coords: None,
            indices: vec![0, 1, 2],
            texture_idx: None,
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
            specular_glossiness_factor: None,
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            emissive_factor: [0.0; 3],
            alpha_mode: MaterialAlphaMode::Opaque,
            alpha_cutoff: 0.0,
            alpha_coverage_mips: false,
            double_sided: false,
            transmission: MaterialTransmission::default(),
            layered_pbr: MaterialLayeredPbr::default(),
        })
    }

    fn cluster() -> ClusterRecord {
        ClusterRecord {
            mesh_index: 0,
            primitive_index: 0,
            material_index: None,
            flags: FLAG_COARSE_ROOT,
            page_index: 0,
            vertex_count: 3,
            triangle_count: 1,
            lod_level: 0,
            vertex_offset: 0,
            index_offset: 216,
            aabb_min: [0.0; 3],
            aabb_max: [1.0; 3],
            sphere_center: [0.5; 3],
            sphere_radius: 1.0,
            normal_cone_axis: [0.0, 0.0, 1.0],
            normal_cone_cutoff: -1.0,
            geometric_error: 0.0,
            parent: NO_RELATION,
            parent_count: 0,
            first_child: NO_RELATION,
            child_count: 0,
            vertex_stride: 72,
        }
    }

    fn archive(source: [u8; 32]) -> GeometryArchive {
        GeometryArchive {
            format_version: VERSION,
            vertex_encoding: VertexEncoding::Float32,
            source_sha256: source,
            payload_sha256: [0; 32],
            page_budget_bytes: 4096,
            file_payload_offset: 0,
            clusters: vec![cluster()],
            pages: vec![PageRecord {
                payload_offset: 0,
                payload_bytes: 219,
                first_cluster: 0,
                cluster_count: 1,
                sha256: [0; 32],
            }],
            compatibility: vec![CompatibilityRecord {
                mesh_index: 0,
                primitive_index: 1,
                reason: CompatibilityReason::AlphaBlend,
                detail: 4,
            }],
        }
    }

    fn model(source: [u8; 32]) -> ModelData {
        let mut second_placement = IDENTITY_MAT4;
        second_placement[3][0] = 4.0;
        ModelData {
            meshes: vec![mesh(), mesh(), mesh(), mesh()],
            mesh_transforms: vec![
                IDENTITY_MAT4,
                IDENTITY_MAT4,
                second_placement,
                second_placement,
            ],
            mesh_cast_shadows: vec![true, true, false, false],
            mesh_sources: vec![
                Some(ModelPrimitiveSource {
                    mesh_index: 0,
                    primitive_index: 0,
                    placement_index: 0,
                }),
                Some(ModelPrimitiveSource {
                    mesh_index: 0,
                    primitive_index: 1,
                    placement_index: 0,
                }),
                Some(ModelPrimitiveSource {
                    mesh_index: 0,
                    primitive_index: 0,
                    placement_index: 1,
                }),
                Some(ModelPrimitiveSource {
                    mesh_index: 0,
                    primitive_index: 1,
                    placement_index: 1,
                }),
            ],
            source_geometry_sha256: Some(source),
            bbox_min: [0.0; 3],
            bbox_max: [1.0; 3],
        }
    }

    #[test]
    fn repeated_mixed_source_mesh_routes_once_per_virtual_placement() {
        let source = [7; 32];
        let route = build_route(&model(source), &archive(source)).unwrap();
        assert_eq!(route.virtual_placements.len(), 2);
        assert_eq!(route.virtual_placements[0].source_mesh_index, 0);
        assert_eq!(route.virtual_placements[0].placement_index, 0);
        assert!(route.virtual_placements[0].cast_shadow);
        assert_eq!(route.virtual_placements[1].placement_index, 1);
        assert!(!route.virtual_placements[1].cast_shadow);
        assert_eq!(route.compatibility_model_mesh_indices(), [1, 3]);
        assert!(route
            .compatibility_placements
            .iter()
            .all(|placement| matches!(
                placement.route,
                ModelVirtualCompatibilityRoute::Cooked(CompatibilityRecord {
                    reason: CompatibilityReason::AlphaBlend,
                    ..
                })
            )));

        let mut outer = IDENTITY_MAT4;
        outer[3][0] = 10.0;
        let mut previous_outer = IDENTITY_MAT4;
        previous_outer[3][0] = 8.0;
        let instances = route
            .virtual_instances(
                VirtualMeshId::FALLBACK,
                41,
                outer,
                previous_outer,
                [0.5, 0.75, 1.0, 1.0],
            )
            .unwrap();
        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].instance_id(), 41);
        assert_eq!(instances[1].instance_id(), 42);
        assert_eq!(instances[0].source_mesh_index(), Some(0));
        assert_eq!(instances[0].model()[3][0], 10.0);
        assert_eq!(instances[1].model()[3][0], 14.0);
        assert_eq!(instances[1].previous_model()[3][0], 12.0);
        assert_eq!(instances[1].model_tint(), [0.5, 0.75, 1.0, 1.0]);
    }

    #[test]
    fn incomplete_or_unrouted_model_fails_closed() {
        let source = [9; 32];
        let mut incomplete = model(source);
        incomplete.mesh_sources.pop();
        assert!(matches!(
            build_route(&incomplete, &archive(source)),
            Err(ModelVirtualGeometryRouteError::IncompletePlacementMetadata { .. })
        ));

        let mut unrouted = model(source);
        unrouted.mesh_sources[0].as_mut().unwrap().primitive_index = 99;
        assert_eq!(
            build_route(&unrouted, &archive(source)).unwrap_err(),
            ModelVirtualGeometryRouteError::UnroutedPrimitive {
                mesh_index: 0,
                primitive_index: 99,
            }
        );
    }

    #[test]
    fn alpha_masked_clusters_remain_ordinary_until_visibility_owns_cutoff() {
        let source = [11; 32];
        let mut archive = archive(source);
        archive.clusters[0].flags |= FLAG_ALPHA_MASKED;
        archive.clusters[0].material_index = Some(17);
        let route = build_route(&model(source), &archive).unwrap();
        assert!(route.virtual_placements.is_empty());
        assert_eq!(route.compatibility_model_mesh_indices(), [0, 1, 2, 3]);
        for placement in [
            &route.compatibility_placements[0],
            &route.compatibility_placements[2],
        ] {
            assert_eq!(
                placement.route,
                ModelVirtualCompatibilityRoute::AlphaMaskedVisibility {
                    material_index: Some(17),
                }
            );
        }
    }
}
