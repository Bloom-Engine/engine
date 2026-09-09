//! Shared sorted-translucency collection and dispatch.
//!
//! Imported BLEND geometry and custom translucent material commands used to
//! sort independently and then render as two whole lists. Each list was
//! internally stable, but a nearer draw from the first list could be blended
//! before a farther draw from the second. This module gives conventional
//! sorted translucency one depth/source/stable-ID order and one dispatcher.

use super::*;

pub(super) fn sorted_interleaving_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("BLOOM_SORTED_INTERLEAVING")
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "disabled"
                )
            })
            .unwrap_or(true)
    })
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SortedPipelineKey {
    Imported {
        double_sided: bool,
        layered: bool,
        secondary_uv: bool,
    },
    Custom(material_system::MaterialHandle),
}

#[derive(Copy, Clone, Debug)]
struct SortedTransparencyKey {
    view_depth: f32,
    /// Imported draws precede custom draws only at exactly equal depth. This
    /// is a deterministic tie policy, not a pass boundary.
    source_rank: u8,
    stable_id: usize,
}

fn compare_sorted_transparency(
    left: &SortedTransparencyKey,
    right: &SortedTransparencyKey,
) -> std::cmp::Ordering {
    right
        .view_depth
        .total_cmp(&left.view_depth)
        .then_with(|| left.source_rank.cmp(&right.source_rank))
        .then_with(|| left.stable_id.cmp(&right.stable_id))
}

impl material_system::MaterialSystem {
    pub(crate) fn has_temporal_reactive_commands(&self) -> bool {
        self.translucent_commands.iter().any(|command| {
            command
                .material
                .checked_sub(1)
                .and_then(|index| self.pipelines.get(index as usize))
                .and_then(|pipeline| pipeline.as_ref())
                .is_some_and(|pipeline| pipeline.writes_reactive)
        })
    }

    /// Compile attachment-compatible siblings only for custom materials that
    /// actually enter a mixed imported/custom TAA-reactive sorted frame.
    pub(crate) fn ensure_translucent_reactive_pipelines(&mut self, device: &wgpu::Device) {
        for command_index in 0..self.translucent_commands.len() {
            let material = self.translucent_commands[command_index].material;
            let Some(index) = material.checked_sub(1).map(|value| value as usize) else {
                continue;
            };
            if let Some(Some(pipeline)) = self.pipelines.get_mut(index) {
                if pipeline.ensure_reactive_pipeline(device) {
                    self.pipeline_creation_count = self.pipeline_creation_count.saturating_add(1);
                }
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn reactive_translucent_pipeline_count(&self) -> usize {
        self.pipelines
            .iter()
            .flatten()
            .filter(|pipeline| pipeline.reactive_pipeline.is_some())
            .count()
    }

    /// Draw one custom translucent command. `bind_material_state` is false
    /// only for an adjacent command using the same custom material; any
    /// imported draw between them resets the caller's state key.
    pub(crate) fn dispatch_translucent_command<'pass>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        command: &material_system::MaterialDrawCommand,
        mesh: MeshDrawRef<'pass>,
        reactive_compatible: bool,
        bind_material_state: bool,
    ) -> bool {
        let Some(index) = command.material.checked_sub(1).map(|value| value as usize) else {
            return false;
        };
        let Some(Some(material)) = self.pipelines.get(index) else {
            return false;
        };
        let pipeline = if reactive_compatible {
            let Some(pipeline) = material.reactive_pipeline.as_ref() else {
                return false;
            };
            pipeline
        } else {
            &material.pipeline
        };
        if bind_material_state {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.per_frame_bg, &[]);
            pass.set_bind_group(1, &self.per_view_bg, &[]);
            pass.set_bind_group(2, self.per_material_bg_for(command.material), &[]);
            if material.reads_scene && cfg!(not(fold_scene_inputs)) {
                if let Some(bind_group) = self.scene_inputs_bg.as_ref() {
                    pass.set_bind_group(4, bind_group, &[]);
                }
            }
        }
        pass.set_bind_group(3, &self.per_draw_bgs[command.draw_slot], &[]);
        pass.set_vertex_buffer(0, mesh.vertex.slice(..));
        pass.set_index_buffer(mesh.index.slice(..), wgpu::IndexFormat::Uint32);
        let instance_range = self.bind_instance_buffer(pass, &command.instance);
        if instance_range.end > instance_range.start {
            pass.draw_indexed(mesh.index_range(), mesh.base_vertex, instance_range);
        }
        true
    }

    pub(crate) fn dispatch_translucent_reactive<'pass, F>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        mut mesh_fetch: F,
    ) where
        F: FnMut(u64, usize) -> Option<MeshDrawRef<'pass>>,
    {
        let mut last_material = 0;
        for command in &self.translucent_commands {
            if let Some(mesh) = mesh_fetch(command.mesh_handle, command.mesh_idx) {
                let bind_material_state = command.material != last_material;
                if self.dispatch_translucent_command(pass, command, mesh, true, bind_material_state)
                {
                    last_material = command.material;
                }
            }
        }
    }
}

impl Renderer {
    fn collect_imported_transparency<'a>(
        &'a self,
        scene: &'a crate::scene::SceneGraph,
    ) -> Vec<ImportedTransparentDrawRef<'a>> {
        let camera_vp = mat4_multiply(
            self.current_proj_matrix_unjittered,
            self.current_view_matrix,
        );
        let camera_planes = crate::scene::extract_frustum_planes(&camera_vp);
        let mut draws = Vec::new();
        for (stable_id, command) in self.model_draw_commands.iter().enumerate() {
            let Some(Some(meshes)) = self.model_gpu_cache.get(&command.cache_handle) else {
                continue;
            };
            let Some(mesh) = meshes.get(command.mesh_idx) else {
                continue;
            };
            if mesh.alpha_mode != MaterialAlphaMode::Blend
                || (self.imported_refraction_enabled && mesh.transmission.is_active())
            {
                continue;
            }
            let (world_min, world_max) = command
                .bounds_override
                .unwrap_or_else(|| transform_aabb(&command.model, mesh.local_min, mesh.local_max));
            if world_min[0] <= world_max[0]
                && crate::scene::aabb_outside_frustum(&camera_planes, world_min, world_max)
            {
                continue;
            }
            let center = if world_min[0] <= world_max[0] {
                [
                    (world_min[0] + world_max[0]) * 0.5,
                    (world_min[1] + world_max[1]) * 0.5,
                    (world_min[2] + world_max[2]) * 0.5,
                ]
            } else {
                [
                    command.model[3][0],
                    command.model[3][1],
                    command.model[3][2],
                ]
            };
            let pivot = mat4_mul_vec4(
                &self.current_vp_matrix,
                &[center[0], center[1], center[2], 1.0],
            );
            let layered_material = mesh.layered_material_bg.as_ref();
            let secondary_uv = if layered_material.is_some() && mesh.layered_uses_uv1 {
                let Some(buffer) = mesh.layered_uv1_buffer.as_ref() else {
                    continue;
                };
                Some(buffer)
            } else {
                None
            };
            let (mesh_draw, vertex_byte_offset, index_byte_offset) = if secondary_uv.is_some() {
                self.gpu_driven
                    .mesh_draw_localized(&mesh.geometry, mesh.index_count)
            } else {
                (
                    self.gpu_driven.mesh_draw(&mesh.geometry, mesh.index_count),
                    0,
                    0,
                )
            };
            draws.push(ImportedTransparentDrawRef {
                view_depth: pivot[3],
                stable_id,
                double_sided: mesh.double_sided,
                layered: layered_material.is_some(),
                uniforms: &self.model_uniform_bind_groups[command.uniform_slot],
                material: layered_material.unwrap_or(&mesh.material_bg),
                mesh: mesh_draw,
                secondary_uv,
                vertex_byte_offset,
                index_byte_offset,
            });
        }
        scene.append_transparent_draws(
            &mut draws,
            &self.current_vp_matrix,
            self.model_draw_commands.len(),
            self.imported_refraction_enabled,
        );
        draws
    }

    /// Imported-only dispatcher retained for weighted OIT. Weighted
    /// accumulation is intentionally independent from custom material
    /// composition and therefore does not use the conventional sort list.
    pub(super) fn draw_imported_transparency<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        scene: &'a crate::scene::SceneGraph,
        weighted: bool,
        reactive: bool,
    ) {
        let mut draws = self.collect_imported_transparency(scene);
        draws.sort_by(|left, right| {
            right
                .view_depth
                .total_cmp(&left.view_depth)
                .then_with(|| left.stable_id.cmp(&right.stable_id))
        });
        let (single_sided, double_sided) = if weighted {
            (
                self.scene_weighted_transparent_pipeline
                    .as_ref()
                    .expect("weighted transparency initialized its pipeline"),
                self.scene_weighted_transparent_double_sided_pipeline
                    .as_ref()
                    .expect("weighted transparency initialized its double-sided pipeline"),
            )
        } else if reactive {
            (
                self.scene_transparent_reactive_pipeline
                    .as_ref()
                    .expect("reactive sorted transparency initialized its pipeline"),
                self.scene_transparent_reactive_double_sided_pipeline
                    .as_ref()
                    .expect("reactive sorted transparency initialized its double-sided pipeline"),
            )
        } else {
            (
                &self.scene_transparent_pipeline,
                &self.scene_transparent_double_sided_pipeline,
            )
        };
        pass.set_bind_group(1, &self.lighting_bind_group, &[]);
        pass.set_bind_group(3, &self.joint_bind_group, &[]);
        let mut current_pipeline_key = None;
        for draw in draws {
            let secondary_uv = draw.secondary_uv.is_some();
            let pipeline_key = (draw.layered, secondary_uv, draw.double_sided);
            if current_pipeline_key != Some(pipeline_key) {
                let pipeline = if draw.layered {
                    self.scene_layered_pbr_resources
                        .as_ref()
                        .expect("layered material initialized its pipelines")
                        .transparent_pipeline(secondary_uv, draw.double_sided, reactive, weighted)
                } else if draw.double_sided {
                    double_sided
                } else {
                    single_sided
                };
                pass.set_pipeline(pipeline);
                current_pipeline_key = Some(pipeline_key);
            }
            pass.set_bind_group(0, draw.uniforms, &[]);
            pass.set_bind_group(2, draw.material, &[]);
            pass.set_vertex_buffer(0, draw.mesh.vertex.slice(draw.vertex_byte_offset..));
            if let Some(secondary_uv) = draw.secondary_uv {
                pass.set_vertex_buffer(1, secondary_uv.slice(..));
            }
            pass.set_index_buffer(
                draw.mesh.index.slice(draw.index_byte_offset..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(draw.mesh.index_range(), draw.mesh.base_vertex, 0..1);
        }
    }

    /// Draw conventional imported and custom translucency in one global order.
    /// When `reactive` is true, imported pipelines write real R8 coverage and
    /// custom attachment-compatible siblings leave that target untouched.
    pub(super) fn draw_sorted_transparency<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        scene: &'a crate::scene::SceneGraph,
        reactive: bool,
    ) {
        let mut imported = self.collect_imported_transparency(scene);
        imported.sort_by(|left, right| {
            right
                .view_depth
                .total_cmp(&left.view_depth)
                .then_with(|| left.stable_id.cmp(&right.stable_id))
        });
        let mut imported = imported.into_iter().peekable();
        let mut custom = self
            .material_system
            .translucent_commands
            .iter()
            .enumerate()
            .peekable();

        let imported_pipelines = if reactive {
            (
                self.scene_transparent_reactive_pipeline
                    .as_ref()
                    .expect("reactive sorted transparency initialized its pipeline"),
                self.scene_transparent_reactive_double_sided_pipeline
                    .as_ref()
                    .expect("reactive sorted transparency initialized its double-sided pipeline"),
            )
        } else {
            (
                &self.scene_transparent_pipeline,
                &self.scene_transparent_double_sided_pipeline,
            )
        };
        let mut current_pipeline = None;
        while imported.peek().is_some() || custom.peek().is_some() {
            let take_imported = match (imported.peek(), custom.peek()) {
                (Some(imported), Some((stable_id, custom))) => compare_sorted_transparency(
                    &SortedTransparencyKey {
                        view_depth: imported.view_depth,
                        source_rank: 0,
                        stable_id: imported.stable_id,
                    },
                    &SortedTransparencyKey {
                        view_depth: custom.view_depth,
                        source_rank: 1,
                        stable_id: *stable_id,
                    },
                )
                .is_le(),
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };
            if take_imported {
                let draw = imported.next().expect("peeked imported draw");
                let key = SortedPipelineKey::Imported {
                    double_sided: draw.double_sided,
                    layered: draw.layered,
                    secondary_uv: draw.secondary_uv.is_some(),
                };
                if current_pipeline != Some(key) {
                    let pipeline = if draw.layered {
                        self.scene_layered_pbr_resources
                            .as_ref()
                            .expect("layered material initialized its pipelines")
                            .transparent_pipeline(
                                draw.secondary_uv.is_some(),
                                draw.double_sided,
                                reactive,
                                false,
                            )
                    } else if draw.double_sided {
                        imported_pipelines.1
                    } else {
                        imported_pipelines.0
                    };
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(1, &self.lighting_bind_group, &[]);
                    pass.set_bind_group(3, &self.joint_bind_group, &[]);
                    current_pipeline = Some(key);
                }
                pass.set_bind_group(0, draw.uniforms, &[]);
                pass.set_bind_group(2, draw.material, &[]);
                pass.set_vertex_buffer(0, draw.mesh.vertex.slice(draw.vertex_byte_offset..));
                if let Some(secondary_uv) = draw.secondary_uv {
                    pass.set_vertex_buffer(1, secondary_uv.slice(..));
                }
                pass.set_index_buffer(
                    draw.mesh.index.slice(draw.index_byte_offset..),
                    wgpu::IndexFormat::Uint32,
                );
                pass.draw_indexed(draw.mesh.index_range(), draw.mesh.base_vertex, 0..1);
            } else {
                let (_, command) = custom.next().expect("peeked custom draw");
                let Some(Some(meshes)) = self.model_gpu_cache.get(&command.mesh_handle) else {
                    continue;
                };
                let Some(mesh) = meshes.get(command.mesh_idx) else {
                    continue;
                };
                let key = SortedPipelineKey::Custom(command.material);
                let bind_material_state = current_pipeline != Some(key);
                let mesh = self.gpu_driven.mesh_draw(&mesh.geometry, mesh.index_count);
                if self.material_system.dispatch_translucent_command(
                    pass,
                    command,
                    mesh,
                    reactive,
                    bind_material_state,
                ) {
                    current_pipeline = Some(key);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(depth: f32, source_rank: u8, stable_id: usize) -> SortedTransparencyKey {
        SortedTransparencyKey {
            view_depth: depth,
            source_rank,
            stable_id,
        }
    }

    #[test]
    fn global_key_orders_depth_then_source_then_stable_id() {
        let mut draws = vec![
            key(5.0, 1, 2),
            key(20.0, 1, 0),
            key(10.0, 1, 1),
            key(10.0, 0, 8),
            key(10.0, 0, 3),
        ];
        draws.sort_by(compare_sorted_transparency);
        let order = draws
            .iter()
            .map(|draw| (draw.view_depth, draw.source_rank, draw.stable_id))
            .collect::<Vec<_>>();
        assert_eq!(
            order,
            vec![
                (20.0, 1, 0),
                (10.0, 0, 3),
                (10.0, 0, 8),
                (10.0, 1, 1),
                (5.0, 1, 2),
            ]
        );
    }
}
