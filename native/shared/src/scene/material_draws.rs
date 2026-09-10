//! Retained material routing and compatibility-pass draws.

use super::*;

impl SceneGraph {
    pub fn has_transparent_nodes(&self) -> bool {
        self.has_transparent_nodes
    }

    pub(crate) fn has_layered_transparent_nodes(&self) -> bool {
        self.has_layered_transparent_nodes
    }

    pub fn has_refractive_nodes(&self) -> bool {
        self.has_refractive_nodes
    }

    /// Constant-time guard for renderer paths that only need to inspect
    /// physical-transmission metadata when at least one retained GI instance
    /// can contribute it.
    pub(crate) fn has_transmission_gi_nodes(&self) -> bool {
        self.has_transmission_gi_nodes
    }

    pub(crate) fn transparent_gi_instance_count(&self) -> u32 {
        if !self.has_transmission_gi_nodes {
            return 0;
        }
        self.nodes
            .iter()
            .filter(|(_handle, node)| {
                node.visible
                    && node.render_layer == 0
                    && node.card_first_slot.is_some()
                    && !node.indices().is_empty()
                    && node.material.has_gi_transmission()
            })
            .count()
            .min(u32::MAX as usize) as u32
    }

    /// O(n) only after the O(1) refractive gate succeeds. Shadow visibility is
    /// deliberately independent of the camera frustum: off-screen glass may
    /// project into a visible receiver.
    pub(crate) fn has_transmitted_shadow_casters(&self) -> bool {
        self.has_refractive_nodes
            && self.nodes.iter().any(|(_handle, node)| {
                node.visible
                    && !node.gi_only
                    && node.cast_shadow
                    && !node.indices().is_empty()
                    && node.material.transmission.is_active()
                    && node.gpu_refractive_material_bg.is_some()
            })
    }

    pub(crate) fn has_visible_opaque_iridescence(&self, imported_refraction_enabled: bool) -> bool {
        self.nodes.iter().any(|(_handle, node)| {
            node.visible
                && !node.gi_only
                && !node.indices().is_empty()
                && node.in_view_frustum
                && !node.occluded
                && node.material.alpha_mode != MaterialAlphaMode::Blend
                && !(imported_refraction_enabled && node.material.transmission.is_active())
                && node.material.layered_pbr.has_iridescence()
                && node.gpu_layered_material_bg.is_some()
        })
    }

    pub(crate) fn append_opaque_iridescence_draws<'a>(
        &'a self,
        out: &mut Vec<ImportedIridescenceDrawRef<'a>>,
    ) {
        for (_handle, node) in self.nodes.iter() {
            if !node.visible
                || node.gi_only
                || node.indices().is_empty()
                || !node.in_view_frustum
                || node.occluded
                || node.material.alpha_mode == MaterialAlphaMode::Blend
                || (self.imported_refraction_enabled && node.material.transmission.is_active())
                || !node.material.layered_pbr.has_iridescence()
            {
                continue;
            }
            let Some(uniforms) = node.gpu_uniform_bg.as_ref() else {
                continue;
            };
            let Some(material) = node.gpu_layered_material_bg.as_ref() else {
                continue;
            };
            let uses_uv1 = node.gpu_layered_uses_uv1;
            let (vertex, index, index_count, secondary_uv) = match node
                .lods
                .get(node.active_lod.max(0) as usize)
                .filter(|_| node.active_lod >= 0)
                .filter(|lod| !uses_uv1 || lod.gpu_secondary_uv_vb.is_some())
                .and_then(|lod| {
                    Some((
                        lod.gpu_vb.as_ref()?,
                        lod.gpu_ib.as_ref()?,
                        lod.gpu_index_count,
                        if uses_uv1 {
                            Some(lod.gpu_secondary_uv_vb.as_ref()?)
                        } else {
                            None
                        },
                    ))
                }) {
                Some(lod) => lod,
                None => {
                    let Some(vertex) = node.gpu_vb.as_ref() else {
                        continue;
                    };
                    let Some(index) = node.gpu_ib.as_ref() else {
                        continue;
                    };
                    let secondary_uv = if uses_uv1 {
                        let Some(buffer) = node.gpu_secondary_uv_vb.as_ref() else {
                            continue;
                        };
                        Some(buffer)
                    } else {
                        None
                    };
                    (vertex, index, node.gpu_index_count, secondary_uv)
                }
            };
            out.push(ImportedIridescenceDrawRef {
                uniforms,
                material,
                mesh: MeshDrawRef {
                    vertex,
                    index,
                    first_index: 0,
                    index_count,
                    base_vertex: 0,
                },
                secondary_uv,
                vertex_byte_offset: 0,
                index_byte_offset: 0,
            });
        }
    }

    pub(crate) fn visible_transparent_node_count(
        &self,
        imported_refraction_enabled: bool,
    ) -> usize {
        if !self.has_transparent_nodes {
            return 0;
        }
        self.nodes
            .iter()
            .filter(|(_handle, node)| {
                node.visible
                    && !node.gi_only
                    && !node.indices().is_empty()
                    && node.in_view_frustum
                    && !node.occluded
                    && node.material.alpha_mode == MaterialAlphaMode::Blend
                    && !(imported_refraction_enabled && node.material.transmission.is_active())
            })
            .count()
    }

    pub(crate) fn visible_refractive_node_count(&self) -> usize {
        if !self.has_refractive_nodes {
            return 0;
        }
        self.nodes
            .iter()
            .filter(|(_handle, node)| {
                node.visible
                    && !node.gi_only
                    && !node.indices().is_empty()
                    && node.in_view_frustum
                    && !node.occluded
                    && node.material.transmission.is_active()
            })
            .count()
    }

    pub(crate) fn append_refractive_draws<'a>(
        &'a self,
        out: &mut Vec<ImportedRefractiveDrawRef<'a>>,
        view_projection: &[[f32; 4]; 4],
        stable_id_base: usize,
    ) {
        for (node_index, (_handle, node)) in self.nodes.iter().enumerate() {
            if !node.visible
                || node.gi_only
                || node.indices().is_empty()
                || !node.in_view_frustum
                || node.occluded
                || !node.material.transmission.is_active()
            {
                continue;
            }
            let uses_uv1 = node.gpu_refractive_uses_uv1;
            let (vertex, index, index_count, secondary_uv) = match node
                .lods
                .get(node.active_lod.max(0) as usize)
                .filter(|_| node.active_lod >= 0)
                .filter(|lod| !uses_uv1 || lod.gpu_secondary_uv_vb.is_some())
                .and_then(|lod| {
                    Some((
                        lod.gpu_vb.as_ref()?,
                        lod.gpu_ib.as_ref()?,
                        lod.gpu_index_count,
                        if uses_uv1 {
                            Some(lod.gpu_secondary_uv_vb.as_ref()?)
                        } else {
                            None
                        },
                    ))
                }) {
                Some(lod) => lod,
                None => {
                    let Some(vertex) = &node.gpu_vb else {
                        continue;
                    };
                    let Some(index) = &node.gpu_ib else {
                        continue;
                    };
                    let secondary_uv = if uses_uv1 {
                        let Some(buffer) = node.gpu_secondary_uv_vb.as_ref() else {
                            continue;
                        };
                        Some(buffer)
                    } else {
                        None
                    };
                    (vertex, index, node.gpu_index_count, secondary_uv)
                }
            };
            let Some(uniforms) = &node.gpu_uniform_bg else {
                continue;
            };
            let Some(material) = &node.gpu_refractive_material_bg else {
                continue;
            };
            let center = if node.world_bounds_min[0] <= node.world_bounds_max[0] {
                [
                    (node.world_bounds_min[0] + node.world_bounds_max[0]) * 0.5,
                    (node.world_bounds_min[1] + node.world_bounds_max[1]) * 0.5,
                    (node.world_bounds_min[2] + node.world_bounds_max[2]) * 0.5,
                ]
            } else {
                let transform = node.world_transform();
                [transform[3][0], transform[3][1], transform[3][2]]
            };
            let pivot = crate::renderer::mat4_mul_vec4(
                view_projection,
                &[center[0], center[1], center[2], 1.0],
            );
            out.push(ImportedRefractiveDrawRef {
                view_depth: pivot[3],
                stable_id: stable_id_base + node_index,
                double_sided: node.material.double_sided,
                layered: node.gpu_refractive_layered,
                uniforms,
                material,
                mesh: MeshDrawRef {
                    vertex,
                    index,
                    first_index: 0,
                    index_count,
                    base_vertex: 0,
                },
                secondary_uv,
                vertex_byte_offset: 0,
                index_byte_offset: 0,
            });
        }
    }

    /// Render glTF BLEND nodes back-to-front into the forward translucent pass.
    /// Opaque depth remains read-only; these nodes never enter the depth prepass.
    pub fn render_transparent<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        single_sided_pipeline: &'a wgpu::RenderPipeline,
        double_sided_pipeline: &'a wgpu::RenderPipeline,
        view_projection: &[[f32; 4]; 4],
    ) {
        self.render_transparent_with_refraction(
            pass,
            single_sided_pipeline,
            double_sided_pipeline,
            view_projection,
            false,
        );
    }

    pub(crate) fn render_transparent_with_refraction<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        single_sided_pipeline: &'a wgpu::RenderPipeline,
        double_sided_pipeline: &'a wgpu::RenderPipeline,
        view_projection: &[[f32; 4]; 4],
        imported_refraction_enabled: bool,
    ) {
        let mut draws = Vec::new();
        self.append_transparent_draws(&mut draws, view_projection, 0, imported_refraction_enabled);
        draws.sort_by(|left, right| {
            right
                .view_depth
                .total_cmp(&left.view_depth)
                .then_with(|| left.stable_id.cmp(&right.stable_id))
        });
        let mut current_double_sided = None;
        for draw in draws {
            if current_double_sided != Some(draw.double_sided) {
                pass.set_pipeline(if draw.double_sided {
                    double_sided_pipeline
                } else {
                    single_sided_pipeline
                });
                current_double_sided = Some(draw.double_sided);
            }
            pass.set_bind_group(0, draw.uniforms, &[]);
            pass.set_bind_group(2, draw.material, &[]);
            pass.set_vertex_buffer(0, draw.mesh.vertex.slice(..));
            pass.set_index_buffer(draw.mesh.index.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(draw.mesh.index_range(), draw.mesh.base_vertex, 0..1);
        }
    }

    pub(crate) fn append_transparent_draws<'a>(
        &'a self,
        out: &mut Vec<ImportedTransparentDrawRef<'a>>,
        view_projection: &[[f32; 4]; 4],
        stable_id_base: usize,
        imported_refraction_enabled: bool,
    ) {
        for (node_index, (_handle, node)) in self.nodes.iter().enumerate() {
            if !node.visible
                || node.gi_only
                || node.indices().is_empty()
                || !node.in_view_frustum
                || node.occluded
                || node.material.alpha_mode != MaterialAlphaMode::Blend
                || (imported_refraction_enabled && node.material.transmission.is_active())
            {
                continue;
            }
            let Some(uniforms) = &node.gpu_uniform_bg else {
                continue;
            };
            let Some(base_material) = &node.gpu_material_bg else {
                continue;
            };
            let layered_material = node.gpu_layered_material_bg.as_ref();
            let uses_uv1 = layered_material.is_some() && node.gpu_layered_uses_uv1;
            let (vb, ib, index_count, secondary_uv) = match node
                .lods
                .get(node.active_lod.max(0) as usize)
                .filter(|_| node.active_lod >= 0)
                .filter(|lod| !uses_uv1 || lod.gpu_secondary_uv_vb.is_some())
                .and_then(|lod| {
                    Some((
                        lod.gpu_vb.as_ref()?,
                        lod.gpu_ib.as_ref()?,
                        lod.gpu_index_count,
                        if uses_uv1 {
                            Some(lod.gpu_secondary_uv_vb.as_ref()?)
                        } else {
                            None
                        },
                    ))
                }) {
                Some(lod) => lod,
                None => {
                    let Some(vb) = &node.gpu_vb else { continue };
                    let Some(ib) = &node.gpu_ib else { continue };
                    let secondary_uv = if uses_uv1 {
                        let Some(buffer) = node.gpu_secondary_uv_vb.as_ref() else {
                            continue;
                        };
                        Some(buffer)
                    } else {
                        None
                    };
                    (vb, ib, node.gpu_index_count, secondary_uv)
                }
            };
            let center = if node.world_bounds_min[0] <= node.world_bounds_max[0] {
                [
                    (node.world_bounds_min[0] + node.world_bounds_max[0]) * 0.5,
                    (node.world_bounds_min[1] + node.world_bounds_max[1]) * 0.5,
                    (node.world_bounds_min[2] + node.world_bounds_max[2]) * 0.5,
                ]
            } else {
                let transform = node.world_transform();
                [transform[3][0], transform[3][1], transform[3][2]]
            };
            let pivot = crate::renderer::mat4_mul_vec4(
                view_projection,
                &[center[0], center[1], center[2], 1.0],
            );
            out.push(ImportedTransparentDrawRef {
                view_depth: pivot[3],
                stable_id: stable_id_base + node_index,
                double_sided: node.material.double_sided,
                layered: layered_material.is_some(),
                uniforms,
                material: layered_material.unwrap_or(base_material),
                mesh: MeshDrawRef {
                    vertex: vb,
                    index: ib,
                    first_index: 0,
                    index_count,
                    base_vertex: 0,
                },
                secondary_uv,
                vertex_byte_offset: 0,
                index_byte_offset: 0,
            });
        }
    }

    /// Append retained-node records directly into the renderer's reused
    /// draw scratch. Off-frustum nodes remain in the list so the compute
    /// culler, rather than the CPU submission loop, owns visibility.
    pub(crate) fn append_gpu_driven_draws(&self, out: &mut Vec<GpuDrawRecord>) -> [u32; 3] {
        let mut compatibility = 0u32;
        let mut visible = 0u32;
        let mut culled = 0u32;
        if !self.gpu_driven_scene_active {
            compatibility = self
                .nodes
                .iter()
                .filter(|(_, node)| {
                    node.visible
                        && !node.gi_only
                        && node.render_layer == 0
                        && !node.indices().is_empty()
                        && node.in_view_frustum
                        && !node.occluded
                })
                .count() as u32;
            return [compatibility, visible, culled];
        }
        for (_handle, node) in self.nodes.iter() {
            if !node.visible
                || node.gi_only
                || node.render_layer != 0
                || node.indices().is_empty()
                || node.occluded
            {
                continue;
            }
            if !scene_node_gpu_driven_ready(node, self.imported_refraction_enabled) {
                compatibility += u32::from(node.in_view_frustum);
                continue;
            }
            let Some(slice) = node.gpu_geometry else {
                compatibility += u32::from(node.in_view_frustum);
                continue;
            };
            let Some(slot) = node.uniform_slot else {
                compatibility += u32::from(node.in_view_frustum);
                continue;
            };
            let offset = slot as usize * NODE_UNIFORM_STRIDE as usize;
            let end = offset + std::mem::size_of::<NodeUniforms>();
            let Some(bytes) = self.scratch.get(offset..end) else {
                compatibility += u32::from(node.in_view_frustum);
                continue;
            };
            let uniforms = bytemuck::pod_read_unaligned::<Uniforms3D>(bytes);
            out.push(GpuDrawRecord {
                uniforms,
                bounds_min: [
                    node.world_bounds_min[0],
                    node.world_bounds_min[1],
                    node.world_bounds_min[2],
                    f32::from_bits(gpu_driven::draw_flags(node.material.double_sided, true)),
                ],
                bounds_max: [
                    node.world_bounds_max[0],
                    node.world_bounds_max[1],
                    node.world_bounds_max[2],
                    0.0,
                ],
                draw: [
                    node.gpu_index_count,
                    slice.first_index,
                    slice.base_vertex as u32,
                    node.gpu_material_id.raw(),
                ],
            });
            if node.in_view_frustum {
                visible += 1;
            } else {
                culled += 1;
            }
        }
        [compatibility, visible, culled]
    }

    pub fn node_count(&self) -> usize {
        self.nodes.iter().count()
    }

    pub(crate) fn has_visible_opaque_nodes_in_layer(&self, render_layer: u32) -> bool {
        self.nodes.iter().any(|(_, node)| {
            node.visible
                && !node.gi_only
                && node.render_layer == render_layer
                && !node.indices().is_empty()
                && node.material.alpha_mode != MaterialAlphaMode::Blend
                && !(self.imported_refraction_enabled && node.material.transmission.is_active())
        })
    }

    /// Draw list for the planar-reflection probe: every visible node with
    /// uploaded geometry as (vb, ib, index_count, material_bg, transform).
    /// Frustum / occlusion flags are intentionally ignored — they were
    /// computed for the MAIN camera and the mirrored probe camera sees a
    /// different set. Base geometry only (no LOD swap): the probe is
    /// half-res and consumed through a perturbed water lookup, where a
    /// LOD pop would be more visible than the detail it saves.
    /// (Treats the composed public × imported transform as world.)
    pub fn reflect_draw_list(
        &self,
    ) -> Vec<(
        &wgpu::Buffer,
        &wgpu::Buffer,
        u32,
        &wgpu::BindGroup,
        [[f32; 4]; 4],
        [f32; 3],
        [f32; 3],
    )> {
        self.reflect_draw_list_with_refraction(false)
    }

    pub(crate) fn reflect_draw_list_with_refraction(
        &self,
        imported_refraction_enabled: bool,
    ) -> Vec<(
        &wgpu::Buffer,
        &wgpu::Buffer,
        u32,
        &wgpu::BindGroup,
        [[f32; 4]; 4],
        [f32; 3],
        [f32; 3],
    )> {
        let mut out = Vec::new();
        for (_handle, node) in self.nodes.iter() {
            if !node.visible
                || node.gi_only
                || node.material.alpha_mode == MaterialAlphaMode::Blend
                || (imported_refraction_enabled && node.material.transmission.is_active())
                || node.indices().is_empty()
            {
                continue;
            }
            let Some(vb) = &node.gpu_vb else { continue };
            let Some(ib) = &node.gpu_ib else { continue };
            let Some(mat_bg) = &node.gpu_material_bg else {
                continue;
            };
            // World bounds ride along so the probe pass can frustum-cull
            // against the MIRRORED camera (main-camera cull flags don't
            // apply there). Sentinel (min > max) = not yet computed →
            // never culled.
            out.push((
                vb,
                ib,
                node.gpu_index_count,
                mat_bg,
                node.world_transform(),
                node.world_bounds_min,
                node.world_bounds_max,
            ));
        }
        out
    }
}
