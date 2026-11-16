//! Game-facing virtual geometry: a loaded model plus its cooked `.bgeo`
//! archive, drawn with one call per placement.
//!
//! `virtual_geometry_integration.rs` owns the producer chain and takes one
//! complete instance set per frame. This layer keeps what a game would
//! otherwise have to keep itself: which model has which archive, the split of
//! each model into virtual and compatibility primitives, and the frame's
//! instance list (with last frame's transforms for the velocity buffer).

use super::Renderer;
use crate::models::{ModelData, ModelVirtualGeometryRoute};
use crate::virtual_geometry::{
    GpuVirtualGeometryConfig, GpuVirtualInstance, GpuVirtualTraversalConfig, VirtualGeometryAsset,
    VirtualMeshId,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Distinguishes a model's compact compatibility cache from the complete
/// ordinary cache the same handle would own if it were also drawn normally.
const COMPATIBILITY_HANDLE_SALT: u64 = 0x5647_434f_4d50_4154;

/// Projected error, in pixels, below which a coarser cluster stands in.
const DEFAULT_TARGET_ERROR_PIXELS: f32 = 1.0;

struct VirtualModelBinding {
    mesh: VirtualMeshId,
    route: ModelVirtualGeometryRoute,
    compatibility_handle: u64,
    draws_compatibility: bool,
}

#[derive(Clone, Copy, PartialEq)]
struct PlacementKey {
    handle_bits: u64,
    model: [[f32; 4]; 4],
}

pub(crate) struct VirtualModelState {
    bindings: HashMap<u64, VirtualModelBinding>,
    instances: Vec<GpuVirtualInstance>,
    placements: Vec<PlacementKey>,
    previous_placements: Vec<PlacementKey>,
    target_error_pixels: f32,
    submitted_last_frame: bool,
    warned_submit: bool,
}

impl Default for VirtualModelState {
    fn default() -> Self {
        Self {
            bindings: HashMap::new(),
            instances: Vec::new(),
            placements: Vec::new(),
            previous_placements: Vec::new(),
            target_error_pixels: DEFAULT_TARGET_ERROR_PIXELS,
            submitted_last_frame: false,
            warned_submit: false,
        }
    }
}

impl Renderer {
    /// Ask, before the window exists, for the renderer to start in the
    /// visibility-shading mode virtual geometry is drawn through. Returns
    /// false where that mode does not exist (the web) or when the renderer
    /// already started without it.
    pub fn request_virtual_geometry() -> bool {
        super::visibility_buffer::request_shading()
    }

    /// Create the virtual producer chain with a physical page pool of
    /// `pool_megabytes` (clamped to what one storage buffer may hold on this
    /// device). The renderer must have started in visibility-shading mode:
    /// see `visibility_buffer::request_shading`.
    pub fn enable_virtual_models(&mut self, pool_megabytes: u32) -> Result<(), String> {
        if self.virtual_geometry.is_some() {
            return Ok(());
        }
        let limit = self.device.limits().max_storage_buffer_binding_size as u64;
        let page_stride_bytes = GpuVirtualGeometryConfig::default().page_stride_bytes;
        let wanted = u64::from(pool_megabytes.max(8)) * 1024 * 1024;
        let capacity_bytes =
            wanted.min(limit) / u64::from(page_stride_bytes) * u64::from(page_stride_bytes);
        self.enable_virtual_geometry(
            GpuVirtualGeometryConfig {
                capacity_bytes,
                page_stride_bytes,
                max_upload_bytes_per_frame: 16 * 1024 * 1024,
                max_upload_pages_per_frame: 256,
                max_evictions_per_frame: 256,
                // A cooked model's coarsest group is as large as its
                // hierarchy leaves it; 256 refused an 80,000-triangle house.
                max_clusters_per_group: 4_096,
                max_cluster_records: 1_048_576,
                ..GpuVirtualGeometryConfig::default()
            },
            GpuVirtualTraversalConfig::default(),
        )
        .map_err(|error| error.to_string())
    }

    pub fn virtual_models_enabled(&self) -> bool {
        self.virtual_geometry.is_some()
    }

    /// Projected error in pixels at which detail is refined (1 = a cluster
    /// is replaced once its error would cover a pixel). Larger is coarser.
    pub fn set_virtual_model_error_pixels(&mut self, pixels: f32) {
        self.virtual_models.target_error_pixels = pixels.clamp(0.1, 64.0);
    }

    /// Give a loaded model its cooked archive. The archive must have been
    /// cooked from the very file the model was loaded from; a mismatch is
    /// rejected. Idempotent for a model that already has one.
    pub fn attach_model_virtual_geometry(
        &mut self,
        handle_bits: u64,
        model: &ModelData,
        archive_bytes: Vec<u8>,
    ) -> Result<(), String> {
        if self.virtual_models.bindings.contains_key(&handle_bits) {
            return Ok(());
        }
        if self.virtual_geometry.is_none() {
            return Err("virtual geometry is not enabled".to_string());
        }
        let asset = Arc::new(
            VirtualGeometryAsset::from_bytes(archive_bytes).map_err(|error| error.to_string())?,
        );
        let route = model
            .route_virtual_geometry(&asset)
            .map_err(|error| error.to_string())?;
        if route.virtual_placements.is_empty() {
            return Err("the archive holds no virtual-eligible geometry for this model".to_string());
        }
        let queue = self.queue.clone();
        let mesh = self
            .virtual_geometry_pool_mut()
            .ok_or_else(|| "virtual geometry is not enabled".to_string())?
            .register_mesh(&queue, Arc::clone(&asset))
            .map_err(|error| error.to_string())?;
        if let Err(error) = self.bind_model_virtual_materials(mesh, model) {
            self.retire_virtual_mesh(mesh);
            return Err(error.to_string());
        }
        let compatibility_handle = handle_bits ^ COMPATIBILITY_HANDLE_SALT;
        let draws_compatibility = !route.compatibility_placements.is_empty();
        if draws_compatibility
            && !self.cache_model_virtual_compatibility(compatibility_handle, model, &route)
        {
            self.retire_virtual_mesh(mesh);
            return Err(
                "the model's non-virtual primitives could not be cached for the ordinary renderer"
                    .to_string(),
            );
        }
        self.virtual_models.bindings.insert(
            handle_bits,
            VirtualModelBinding {
                mesh,
                route,
                compatibility_handle,
                draws_compatibility,
            },
        );
        Ok(())
    }

    pub fn model_has_virtual_geometry(&self, handle_bits: u64) -> bool {
        self.virtual_models.bindings.contains_key(&handle_bits)
    }

    /// Release a model's archive, pages and compatibility cache.
    pub fn detach_model_virtual_geometry(&mut self, handle_bits: u64) -> bool {
        let Some(binding) = self.virtual_models.bindings.remove(&handle_bits) else {
            return false;
        };
        self.retire_virtual_mesh(binding.mesh);
        if binding.draws_compatibility {
            self.evict_model_cache(binding.compatibility_handle);
            self.model_virtual_compatibility_sources
                .remove(&binding.compatibility_handle);
        }
        true
    }

    fn retire_virtual_mesh(&mut self, mesh: VirtualMeshId) {
        let queue = self.queue.clone();
        if let Some(pool) = self.virtual_geometry_pool_mut() {
            let _ = pool.retire_mesh(&queue, mesh);
        }
        if let Some(state) = self.virtual_geometry.as_mut() {
            if let Some(materials) = state.take_owned_materials(mesh) {
                self.material_system
                    .indirection
                    .retire_materials(&self.queue, materials);
            }
        }
    }

    /// Draw one placement of a model through virtual geometry this frame.
    /// Its cut-out, translucent and skinned primitives go to the ordinary
    /// renderer in the same call. Returns false when the model has no archive
    /// attached, so the caller can fall back to an ordinary draw.
    pub fn draw_model_virtual(
        &mut self,
        handle_bits: u64,
        model_matrix: [[f32; 4]; 4],
        tint: [f32; 4],
    ) -> bool {
        let Some(binding) = self.virtual_models.bindings.get(&handle_bits) else {
            return false;
        };
        let mesh = binding.mesh;
        let draws_compatibility = binding.draws_compatibility;
        let compatibility_handle = binding.compatibility_handle;
        let route = binding.route.clone();

        // Same call order and same model as last frame: that placement's
        // previous transform. Anything else has no history and is still.
        let index = self.virtual_models.placements.len();
        let previous_matrix = self
            .virtual_models
            .previous_placements
            .get(index)
            .filter(|previous| previous.handle_bits == handle_bits)
            .map_or(model_matrix, |previous| previous.model);
        let Ok(first_instance_id) = u32::try_from(self.virtual_models.instances.len()) else {
            return false;
        };
        let Ok(instances) =
            route.virtual_instances(mesh, first_instance_id, model_matrix, previous_matrix, tint)
        else {
            return false;
        };
        self.virtual_models.instances.extend(instances);
        self.virtual_models.placements.push(PlacementKey {
            handle_bits,
            model: model_matrix,
        });
        if draws_compatibility {
            self.draw_model_cached_compatibility_transform(
                compatibility_handle,
                model_matrix,
                tint,
                &route,
            );
        }
        true
    }

    /// Hand the frame's placements to the producer chain. Called once by the
    /// engine before the frame is rendered.
    pub(crate) fn flush_virtual_model_draws(&mut self) {
        if self.virtual_geometry.is_none() {
            return;
        }
        let instances = std::mem::take(&mut self.virtual_models.instances);
        let had_draws = !instances.is_empty();
        // An empty submission after a frame that drew something clears the
        // retained batch; a renderer that never draws virtually submits none.
        if had_draws || self.virtual_models.submitted_last_frame {
            let target = self.virtual_models.target_error_pixels;
            if let Err(error) = self.submit_virtual_geometry_current_view(&instances, target) {
                if !self.virtual_models.warned_submit {
                    self.virtual_models.warned_submit = true;
                    eprintln!("bloom: virtual geometry submission failed: {error}");
                }
            }
        }
        self.virtual_models.submitted_last_frame = had_draws;
        let mut instances = instances;
        instances.clear();
        self.virtual_models.instances = instances;
        std::mem::swap(
            &mut self.virtual_models.previous_placements,
            &mut self.virtual_models.placements,
        );
        self.virtual_models.placements.clear();
    }

    /// Resident and capacity bytes of the page pool, pages resident, and the
    /// instances submitted this frame: for a game's debug read-out.
    pub fn virtual_model_stats(&self) -> [f64; 4] {
        let Some(state) = self.virtual_geometry.as_ref() else {
            return [0.0; 4];
        };
        let telemetry = state.pool_telemetry();
        [
            telemetry.resident_slot_bytes as f64,
            telemetry.total_gpu_bytes as f64,
            telemetry.resident_pages as f64,
            self.virtual_models.previous_placements.len() as f64,
        ]
    }
}
