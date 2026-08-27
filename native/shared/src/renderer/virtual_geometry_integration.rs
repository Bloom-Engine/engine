use super::material_indirection::MaterialId;
use super::{gpu_driven, specialized_scene_shader_source, Renderer};
use crate::virtual_geometry::{
    GpuVirtualDrawEmitter, GpuVirtualGeometryConfig, GpuVirtualGeometryPool,
    GpuVirtualHierarchySelector, GpuVirtualInstance, GpuVirtualPageStreamer,
    GpuVirtualStreamingConfig, GpuVirtualStreamingError, GpuVirtualTraversalConfig,
    GpuVirtualVisibilityFrame, GpuVirtualVisibilityRaster, GpuVirtualVisibilityShading,
    VirtualGeometryDrawEmissionError, VirtualGeometryGpuError, VirtualGeometryHiZFrame,
    VirtualGeometryTraversalError, VirtualGeometryView, VirtualGeometryVisibilityError,
    VirtualMaterialBinding, VirtualMeshId,
};
use bloom_geometry_format::FLAG_ALPHA_MASKED;
use std::collections::BTreeMap;
use std::fmt;

struct VirtualGeometryFrameSubmission {
    instances: Vec<GpuVirtualInstance>,
    view: VirtualGeometryView,
    visibility: GpuVirtualVisibilityFrame,
}

/// Explicit renderer-owned virtual-geometry producer chain. The renderer only
/// creates this state after `enable_virtual_geometry`; an ordinary renderer
/// therefore retains the established zero-allocation path.
pub(crate) struct RendererVirtualGeometry {
    pool: GpuVirtualGeometryPool,
    selector: GpuVirtualHierarchySelector,
    streamer: GpuVirtualPageStreamer,
    emitter: GpuVirtualDrawEmitter,
    raster: GpuVirtualVisibilityRaster,
    shading: Option<GpuVirtualVisibilityShading>,
    submission: Option<VirtualGeometryFrameSubmission>,
    prepared: bool,
    frame: u64,
    renderer_frame: u64,
    last_failure: Option<String>,
    owned_materials: BTreeMap<VirtualMeshId, Vec<MaterialId>>,
}

impl RendererVirtualGeometry {
    fn new(
        device: &wgpu::Device,
        pool_config: GpuVirtualGeometryConfig,
        traversal_config: GpuVirtualTraversalConfig,
        streaming_config: GpuVirtualStreamingConfig,
    ) -> Result<Self, RendererVirtualGeometryError> {
        let pool = GpuVirtualGeometryPool::new(device, pool_config)?;
        let selector = GpuVirtualHierarchySelector::new(device, &pool, traversal_config)?;
        let streamer = GpuVirtualPageStreamer::new(device, &selector, streaming_config)?;
        let emitter = GpuVirtualDrawEmitter::new(device, &selector)?;
        let raster = GpuVirtualVisibilityRaster::new(device, &pool, &selector, &emitter)?;
        Ok(Self {
            pool,
            selector,
            streamer,
            emitter,
            raster,
            shading: None,
            submission: None,
            prepared: false,
            frame: 0,
            renderer_frame: 0,
            last_failure: None,
            owned_materials: BTreeMap::new(),
        })
    }

    pub(crate) fn begin_frame(&mut self, device: &wgpu::Device) {
        self.streamer.poll(device);
        self.renderer_frame = self.renderer_frame.wrapping_add(1);
        self.submission = None;
        self.prepared = false;
        self.last_failure = None;
    }

    fn submit(
        &mut self,
        instances: &[GpuVirtualInstance],
        view: VirtualGeometryView,
        visibility: GpuVirtualVisibilityFrame,
    ) -> Result<(), RendererVirtualGeometryError> {
        if instances.len() > self.selector.config().max_instances as usize {
            return Err(RendererVirtualGeometryError::Traversal(
                VirtualGeometryTraversalError::TooManyInstances {
                    requested: instances.len(),
                    capacity: self.selector.config().max_instances,
                },
            ));
        }
        self.submission = Some(VirtualGeometryFrameSubmission {
            instances: instances.to_vec(),
            view,
            visibility,
        });
        self.prepared = false;
        self.last_failure = None;
        Ok(())
    }

    fn record_producers(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        hiz_frame: VirtualGeometryHiZFrame,
    ) -> Result<(), RendererVirtualGeometryError> {
        self.prepared = false;
        let Some(submission) = self.submission.as_mut() else {
            return Ok(());
        };
        let hiz_valid = self.selector.previous_hiz_history_valid(hiz_frame);
        for instance in &mut submission.instances {
            instance.set_previous_hiz_eligible(
                hiz_valid && self.selector.previous_hiz_contains(*instance),
            );
        }
        self.frame = self.frame.wrapping_add(1).max(1);
        self.pool.begin_frame(self.frame);
        self.streamer.service(&mut self.pool, queue);
        let dispatch = self.selector.record_with_previous_hiz(
            queue,
            encoder,
            &self.pool,
            &submission.instances,
            submission.view,
            hiz_frame,
        )?;
        if dispatch.instance_count != 0 {
            self.streamer.record(encoder, &self.selector);
        }
        self.emitter.record(queue, encoder, &self.selector)?;
        self.raster.prepare_frame(queue, submission.visibility)?;
        self.prepared = true;
        Ok(())
    }

    fn record_failure(&mut self, error: &RendererVirtualGeometryError) {
        let message = error.to_string();
        if self.last_failure.as_deref() != Some(message.as_str()) {
            log::error!("bloom: virtual-geometry frame suppressed as one batch: {message}");
            self.last_failure = Some(message);
        }
        self.prepared = false;
    }

    const fn frame_requested(&self) -> bool {
        self.submission.is_some()
    }

    const fn prepared(&self) -> bool {
        self.prepared
    }

    pub(crate) fn after_submit(&mut self) {
        self.streamer.after_submit();
        self.selector.after_submit_previous_hiz();
    }

    fn hiz_frame(
        &self,
        view_projection: [[f32; 4]; 4],
        view: [[f32; 4]; 4],
        render_extent: (u32, u32),
        camera_cut: bool,
    ) -> VirtualGeometryHiZFrame {
        VirtualGeometryHiZFrame {
            frame_index: self.renderer_frame,
            view_projection,
            view,
            render_extent,
            camera_cut,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_hiz_capture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        source_size: (u32, u32),
        view_projection: [[f32; 4]; 4],
        view: [[f32; 4]; 4],
        render_extent: (u32, u32),
    ) {
        let Some(submission) = self.submission.as_ref().filter(|_| self.prepared) else {
            return;
        };
        let frame = VirtualGeometryHiZFrame {
            frame_index: self.renderer_frame,
            view_projection,
            view,
            render_extent,
            camera_cut: false,
        };
        self.selector.record_previous_hiz_capture(
            device,
            queue,
            encoder,
            source,
            source_size,
            frame,
            &submission.instances,
        );
    }

    fn invalidate_hiz(&mut self, source_recreated: bool) {
        self.selector.invalidate_previous_hiz(source_recreated);
    }
}

/// Errors are explicit because the virtual path must fail closed: a rejected
/// producer never leaves partial IDs or half a virtual batch in the frame.
#[derive(Debug)]
pub enum RendererVirtualGeometryError {
    AlreadyEnabled,
    NotEnabled,
    VisibilityShadingRequired,
    MissingModelSourceClosure,
    ModelSourceClosureMismatch,
    MissingModelPrimitive {
        mesh_index: u32,
        primitive_index: u32,
    },
    InconsistentModelPrimitive {
        mesh_index: u32,
        primitive_index: u32,
    },
    UnsupportedModelMaterial {
        mesh_index: u32,
        primitive_index: u32,
    },
    ConflictingSourceMaterial {
        material_index: Option<u32>,
    },
    MaterialAllocationFailed {
        material_index: Option<u32>,
    },
    Pool(VirtualGeometryGpuError),
    Traversal(VirtualGeometryTraversalError),
    DrawEmission(VirtualGeometryDrawEmissionError),
    Visibility(VirtualGeometryVisibilityError),
    Streaming(GpuVirtualStreamingError),
}

impl From<VirtualGeometryGpuError> for RendererVirtualGeometryError {
    fn from(value: VirtualGeometryGpuError) -> Self {
        Self::Pool(value)
    }
}

impl From<VirtualGeometryTraversalError> for RendererVirtualGeometryError {
    fn from(value: VirtualGeometryTraversalError) -> Self {
        Self::Traversal(value)
    }
}

impl From<VirtualGeometryDrawEmissionError> for RendererVirtualGeometryError {
    fn from(value: VirtualGeometryDrawEmissionError) -> Self {
        Self::DrawEmission(value)
    }
}

impl From<VirtualGeometryVisibilityError> for RendererVirtualGeometryError {
    fn from(value: VirtualGeometryVisibilityError) -> Self {
        Self::Visibility(value)
    }
}

impl From<GpuVirtualStreamingError> for RendererVirtualGeometryError {
    fn from(value: GpuVirtualStreamingError) -> Self {
        Self::Streaming(value)
    }
}

impl fmt::Display for RendererVirtualGeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyEnabled => write!(formatter, "virtual geometry is already enabled"),
            Self::NotEnabled => write!(formatter, "virtual geometry is not enabled"),
            Self::VisibilityShadingRequired => write!(
                formatter,
                "virtual geometry requires the opt-in visibility shade path"
            ),
            Self::MissingModelSourceClosure => write!(
                formatter,
                "model loader did not resolve the complete virtual-geometry source closure"
            ),
            Self::ModelSourceClosureMismatch => write!(
                formatter,
                "model and registered virtual archive have different source closures"
            ),
            Self::MissingModelPrimitive {
                mesh_index,
                primitive_index,
            } => write!(
                formatter,
                "cooked glTF mesh {mesh_index} primitive {primitive_index} is absent from the runtime model"
            ),
            Self::InconsistentModelPrimitive {
                mesh_index,
                primitive_index,
            } => write!(
                formatter,
                "runtime placements disagree on glTF mesh {mesh_index} primitive {primitive_index}"
            ),
            Self::UnsupportedModelMaterial {
                mesh_index,
                primitive_index,
            } => write!(
                formatter,
                "virtual glTF mesh {mesh_index} primitive {primitive_index} uses transmission or layered PBR that remains compatibility-owned"
            ),
            Self::ConflictingSourceMaterial { material_index } => write!(
                formatter,
                "runtime primitives disagree on cooked material slot {material_index:?}"
            ),
            Self::MaterialAllocationFailed { material_index } => write!(
                formatter,
                "global material allocation failed for cooked material slot {material_index:?}"
            ),
            Self::Pool(error) => error.fmt(formatter),
            Self::Traversal(error) => error.fmt(formatter),
            Self::DrawEmission(error) => error.fmt(formatter),
            Self::Visibility(error) => error.fmt(formatter),
            Self::Streaming(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RendererVirtualGeometryError {}

impl Renderer {
    /// Cache only the ordinary-renderer half of an exact virtual-geometry
    /// route. The cache is compact: virtual-eligible vertex/index payloads are
    /// never uploaded to the ordinary static arena just to preserve source
    /// mesh indexing.
    pub fn cache_model_virtual_compatibility(
        &mut self,
        handle_bits: u64,
        model: &crate::models::ModelData,
        route: &crate::models::ModelVirtualGeometryRoute,
    ) -> bool {
        let source_indices = route.compatibility_model_mesh_indices();
        if !super::model_draw::cached_model_subset_is_canonical(
            model.meshes.len(),
            source_indices.iter().copied(),
        ) {
            return false;
        }
        if model.mesh_transforms.len() != model.meshes.len()
            || model.mesh_cast_shadows.len() != model.meshes.len()
            || model.mesh_sources.len() != model.meshes.len()
            || route.compatibility_placements.iter().any(|placement| {
                model.mesh_source(placement.model_mesh_index) != Some(placement.source)
            })
        {
            return false;
        }
        if let Some(cached_sources) = self.model_virtual_compatibility_sources.get(&handle_bits) {
            return cached_sources == &source_indices
                && self
                    .model_gpu_cache
                    .get(&handle_bits)
                    .is_some_and(Option::is_some);
        }
        if self.model_gpu_cache.contains_key(&handle_bits) {
            return false;
        }

        let meshes = source_indices
            .iter()
            .map(|&index| model.meshes[index].as_ref())
            .collect::<Vec<_>>();
        let transforms = source_indices
            .iter()
            .map(|&index| model.mesh_transform(index))
            .collect::<Vec<_>>();
        if !self.cache_model_if_static_with_transforms(handle_bits, &meshes, &transforms) {
            return false;
        }
        self.model_virtual_compatibility_sources
            .insert(handle_bits, source_indices);
        true
    }

    /// Create the complete fixed-budget virtual producer chain. This remains
    /// opt-in and requires `BLOOM_VISIBILITY_BUFFER=shade` at renderer startup;
    /// no virtual pool, transient buffers, pipeline, or target is allocated by
    /// the default renderer path.
    pub fn enable_virtual_geometry(
        &mut self,
        pool_config: GpuVirtualGeometryConfig,
        traversal_config: GpuVirtualTraversalConfig,
    ) -> Result<(), RendererVirtualGeometryError> {
        let streaming_config =
            GpuVirtualStreamingConfig::bounded_default(traversal_config.max_page_requests);
        self.enable_virtual_geometry_with_streaming(pool_config, traversal_config, streaming_config)
    }

    /// Advanced opt-in form with explicit asynchronous feedback budgets.
    pub fn enable_virtual_geometry_with_streaming(
        &mut self,
        pool_config: GpuVirtualGeometryConfig,
        traversal_config: GpuVirtualTraversalConfig,
        streaming_config: GpuVirtualStreamingConfig,
    ) -> Result<(), RendererVirtualGeometryError> {
        if self.virtual_geometry.is_some() {
            return Err(RendererVirtualGeometryError::AlreadyEnabled);
        }
        if !self.gpu_driven.visibility_shading_requested() {
            return Err(RendererVirtualGeometryError::VisibilityShadingRequired);
        }
        self.virtual_geometry = Some(RendererVirtualGeometry::new(
            &self.device,
            pool_config,
            traversal_config,
            streaming_config,
        )?);
        Ok(())
    }

    /// Drop every renderer-owned virtual resource and return to the ordinary
    /// visibility path. Existing non-virtual scene routing is untouched.
    pub fn disable_virtual_geometry(&mut self) -> bool {
        let Some(state) = self.virtual_geometry.take() else {
            return false;
        };
        self.material_system
            .indirection
            .retire_materials(&self.queue, state.owned_materials.into_values().flatten());
        true
    }

    /// Mutable residency/asset owner for explicit `.bgeo` registration and
    /// page streaming. It is unavailable until virtual geometry is enabled.
    pub fn virtual_geometry_pool_mut(&mut self) -> Option<&mut GpuVirtualGeometryPool> {
        self.virtual_geometry.as_mut().map(|state| &mut state.pool)
    }

    /// Allocate and bind the exact base-PBR material table for one registered
    /// cooked model. Source material indices are recovered from the archive;
    /// callers never manufacture renderer-global IDs. Masked clusters receive
    /// a valid (unused) binding because virtual raster discards them and the
    /// compatibility cache remains authoritative for their pixels.
    pub fn bind_model_virtual_materials(
        &mut self,
        virtual_mesh: VirtualMeshId,
        model: &crate::models::ModelData,
    ) -> Result<(), RendererVirtualGeometryError> {
        let asset = self
            .virtual_geometry
            .as_ref()
            .ok_or(RendererVirtualGeometryError::NotEnabled)?
            .pool
            .asset(virtual_mesh)?
            .clone();
        let source_hash = model
            .source_geometry_sha256
            .ok_or(RendererVirtualGeometryError::MissingModelSourceClosure)?;
        if source_hash != asset.archive().source_sha256 {
            return Err(RendererVirtualGeometryError::ModelSourceClosureMismatch);
        }

        let mut primitives = BTreeMap::<(u32, u32), &crate::models::MeshData>::new();
        for (model_mesh_index, mesh) in model.meshes.iter().enumerate() {
            let Some(source) = model.mesh_source(model_mesh_index) else {
                continue;
            };
            let key = (source.mesh_index, source.primitive_index);
            if let Some(previous) = primitives.get(&key).copied() {
                let previous_record = self.model_gpu_material_record(previous);
                let current_record = self.model_gpu_material_record(mesh);
                let previous_extended =
                    previous.transmission.is_active() || previous.layered_pbr.is_active();
                let current_extended =
                    mesh.transmission.is_active() || mesh.layered_pbr.is_active();
                if bytemuck::bytes_of(&previous_record) != bytemuck::bytes_of(&current_record)
                    || previous_extended != current_extended
                {
                    return Err(RendererVirtualGeometryError::InconsistentModelPrimitive {
                        mesh_index: source.mesh_index,
                        primitive_index: source.primitive_index,
                    });
                }
            } else {
                primitives.insert(key, mesh.as_ref());
            }
        }

        let mut records =
            BTreeMap::<Option<u32>, super::material_indirection::GpuMaterialRecord>::new();
        for cluster in &asset.archive().clusters {
            let key = (cluster.mesh_index, cluster.primitive_index);
            let mesh = primitives.get(&key).copied().ok_or(
                RendererVirtualGeometryError::MissingModelPrimitive {
                    mesh_index: cluster.mesh_index,
                    primitive_index: cluster.primitive_index,
                },
            )?;
            if cluster.flags & FLAG_ALPHA_MASKED == 0
                && ((self.imported_refraction_enabled && mesh.transmission.is_active())
                    || mesh.layered_pbr.is_active())
            {
                return Err(RendererVirtualGeometryError::UnsupportedModelMaterial {
                    mesh_index: cluster.mesh_index,
                    primitive_index: cluster.primitive_index,
                });
            }
            let record = self.model_gpu_material_record(mesh);
            if let Some(previous) = records.insert(cluster.material_index, record) {
                if bytemuck::bytes_of(&previous) != bytemuck::bytes_of(&record) {
                    return Err(RendererVirtualGeometryError::ConflictingSourceMaterial {
                        material_index: cluster.material_index,
                    });
                }
            }
        }

        let mut material_ids = Vec::with_capacity(records.len());
        let mut bindings = Vec::with_capacity(records.len());
        for (source_material_index, record) in records {
            let material_id = self
                .material_system
                .indirection
                .allocate_material(&self.device, record);
            if material_id == MaterialId::FALLBACK {
                self.material_system
                    .indirection
                    .retire_materials(&self.queue, material_ids);
                return Err(RendererVirtualGeometryError::MaterialAllocationFailed {
                    material_index: source_material_index,
                });
            }
            bindings.push(VirtualMaterialBinding {
                source_material_index,
                material_id: material_id.raw(),
            });
            material_ids.push(material_id);
        }

        let bind_result = self
            .virtual_geometry
            .as_mut()
            .expect("virtual state remained enabled while binding")
            .pool
            .bind_mesh_materials(&self.queue, virtual_mesh, &bindings);
        if let Err(error) = bind_result {
            self.material_system
                .indirection
                .retire_materials(&self.queue, material_ids);
            return Err(error.into());
        }
        let old = self
            .virtual_geometry
            .as_mut()
            .expect("virtual state remained enabled while recording ownership")
            .owned_materials
            .insert(virtual_mesh, material_ids);
        if let Some(old) = old {
            self.material_system
                .indirection
                .retire_materials(&self.queue, old);
        }
        Ok(())
    }

    /// Bounded feedback/readback/upload counters for debug UIs and captures.
    pub fn virtual_geometry_streaming_telemetry(
        &self,
    ) -> Option<crate::virtual_geometry::GpuVirtualStreamingTelemetry> {
        self.virtual_geometry
            .as_ref()
            .map(|state| state.streamer.telemetry())
    }

    pub fn virtual_geometry_hiz_telemetry(
        &self,
    ) -> Option<crate::virtual_geometry::GpuVirtualHiZTelemetry> {
        self.virtual_geometry
            .as_ref()
            .map(|state| state.selector.previous_hiz_telemetry())
    }

    /// Submit one complete virtual instance set for the current frame. A
    /// later submission replaces the earlier one, matching retained-scene
    /// semantics without accumulating hidden per-frame work.
    pub fn submit_virtual_geometry(
        &mut self,
        instances: &[GpuVirtualInstance],
        view: VirtualGeometryView,
        visibility: GpuVirtualVisibilityFrame,
    ) -> Result<(), RendererVirtualGeometryError> {
        self.virtual_geometry
            .as_mut()
            .ok_or(RendererVirtualGeometryError::NotEnabled)?
            .submit(instances, view, visibility)
    }

    /// Submit against the renderer's current camera with stable unjittered LOD
    /// selection and the exact jittered current/previous transforms used by
    /// Bloom's velocity buffer. This is the normal raylib/Unity-like entry
    /// point; the fully explicit form remains available for advanced tooling.
    pub fn submit_virtual_geometry_current_view(
        &mut self,
        instances: &[GpuVirtualInstance],
        target_error_pixels: f32,
    ) -> Result<(), RendererVirtualGeometryError> {
        let view_projection = super::mat4_multiply(
            self.current_proj_matrix_unjittered,
            self.current_view_matrix,
        );
        let (_, render_height) = self.render_extent();
        let view = VirtualGeometryView {
            frustum_planes: crate::scene::extract_frustum_planes(&view_projection),
            view_projection,
            camera_position: self.current_camera_pos,
            projection_scale: render_height as f32
                * 0.5
                * self.current_proj_matrix_unjittered[1][1].abs(),
            target_error_pixels,
        };
        let visibility =
            GpuVirtualVisibilityFrame::new(self.current_vp_matrix, self.velocity_ref_vp)?;
        self.submit_virtual_geometry(instances, view, visibility)
    }

    pub(crate) fn virtual_visibility_frame_requested(&self) -> bool {
        self.virtual_geometry
            .as_ref()
            .is_some_and(RendererVirtualGeometry::frame_requested)
    }

    pub(crate) fn virtual_geometry_report_json(&self) -> String {
        let Some(state) = self.virtual_geometry.as_ref() else {
            return concat!(
                "{\"enabled\":false,\"allocated\":false,",
                "\"frame_requested\":false,\"frame_prepared\":false,",
                "\"instances\":0,\"total_gpu_bytes\":0,",
                "\"streaming_pending_groups\":0,\"streaming_in_flight\":0,",
                "\"streaming_captures_completed\":0,\"streaming_uploaded_pages\":0,",
                "\"streaming_truncated_requests\":0,\"streaming_readback_bytes\":0,",
                "\"last_visible_groups\":0,\"last_frustum_culled_groups\":0,",
                "\"last_cone_culled_clusters\":0,\"last_refined_groups\":0,",
                "\"last_fallback_groups\":0,\"last_missing_current_pages\":0,",
                "\"last_selected_count\":0,\"last_selected_overflow\":0,",
                "\"last_request_overflow\":0,\"last_invalid_records\":0,",
                "\"last_depth_limit_fallbacks\":0,",
                "\"last_occlusion_culled_groups\":0,\"last_occlusion_uncertain_groups\":0,",
                "\"hiz_texture_bytes\":0,\"hiz_history_valid\":false,",
                "\"hiz_captures_submitted\":0,\"hiz_history_instances\":0}"
            )
            .to_string();
        };
        let telemetry = state.pool.telemetry();
        let streaming = state.streamer.telemetry();
        let hiz = state.selector.previous_hiz_telemetry();
        let submission_mode = match state.emitter.submission_mode() {
            crate::virtual_geometry::VirtualGeometrySubmissionMode::Counted => "counted",
            crate::virtual_geometry::VirtualGeometrySubmissionMode::BinnedFallback => {
                "binned-fallback"
            }
        };
        let instances = state
            .submission
            .as_ref()
            .map_or(0, |submission| submission.instances.len());
        format!(
            concat!(
                "{{\"enabled\":true,\"allocated\":true,",
                "\"frame_requested\":{},\"frame_prepared\":{},",
                "\"instances\":{},\"submission_mode\":\"{}\",",
                "\"pool_capacity_bytes\":{},\"total_gpu_bytes\":{},",
                "\"resident_pages\":{},\"active_meshes\":{},",
                "\"max_instances\":{},\"max_selected_clusters\":{},",
                "\"streaming_pending_groups\":{},\"streaming_in_flight\":{},",
                "\"streaming_captures_completed\":{},\"streaming_uploaded_pages\":{},",
                "\"streaming_truncated_requests\":{},\"streaming_readback_bytes\":{},",
                "\"last_visible_groups\":{},\"last_frustum_culled_groups\":{},",
                "\"last_cone_culled_clusters\":{},\"last_refined_groups\":{},",
                "\"last_fallback_groups\":{},\"last_missing_current_pages\":{},",
                "\"last_selected_count\":{},\"last_selected_overflow\":{},",
                "\"last_request_overflow\":{},\"last_invalid_records\":{},",
                "\"last_depth_limit_fallbacks\":{},",
                "\"last_occlusion_culled_groups\":{},\"last_occlusion_uncertain_groups\":{},",
                "\"hiz_texture_bytes\":{},\"hiz_history_valid\":{},",
                "\"hiz_captures_submitted\":{},\"hiz_history_instances\":{}}}"
            ),
            state.frame_requested(),
            state.prepared(),
            instances,
            submission_mode,
            telemetry.capacity_bytes,
            telemetry.total_gpu_bytes,
            telemetry.resident_pages,
            telemetry.active_meshes,
            state.selector.config().max_instances,
            state.selector.config().max_selected_clusters,
            streaming.pending_groups,
            streaming.in_flight_readbacks,
            streaming.captures_completed,
            streaming.uploaded_pages,
            streaming.truncated_requests,
            streaming.readback_bytes,
            streaming.last_visible_groups,
            streaming.last_frustum_culled_groups,
            streaming.last_cone_culled_clusters,
            streaming.last_refined_groups,
            streaming.last_fallback_groups,
            streaming.last_missing_current_pages,
            streaming.last_selected_count,
            streaming.last_selected_overflow,
            streaming.last_request_overflow,
            streaming.last_invalid_records,
            streaming.last_depth_limit_fallbacks,
            streaming.last_occlusion_culled_groups,
            streaming.last_occlusion_uncertain_groups,
            hiz.texture_bytes,
            hiz.history_valid,
            hiz.captures_submitted,
            hiz.history_instances,
        )
    }

    pub(crate) fn prepare_registered_virtual_visibility(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        visibility_target_recreated: bool,
    ) -> bool {
        if visibility_target_recreated {
            if let Some(state) = self.virtual_geometry.as_mut() {
                state.shading = None;
            }
        }
        let render_extent = self.render_extent();
        let current_vp = self.current_vp_matrix;
        let current_view = self.current_view_matrix;
        let camera_cut = self.temporal_camera_cut_active || self.temporal_camera_cut_pending;
        let producer_result = self.virtual_geometry.as_mut().map(|state| {
            let hiz_frame = state.hiz_frame(current_vp, current_view, render_extent, camera_cut);
            state.record_producers(&self.queue, encoder, hiz_frame)
        });
        if let Some(Err(error)) = producer_result {
            self.virtual_geometry
                .as_mut()
                .expect("virtual state existed while recording")
                .record_failure(&error);
            return false;
        }
        let needs_shading = self
            .virtual_geometry
            .as_ref()
            .is_some_and(|state| state.prepared() && state.shading.is_none());
        if needs_shading {
            let result = {
                let state = self
                    .virtual_geometry
                    .as_ref()
                    .expect("prepared virtual state exists");
                let Some(visibility) = self.gpu_driven.visibility_texture() else {
                    return false;
                };
                self.create_virtual_visibility_shading(
                    &state.pool,
                    &state.selector,
                    &state.raster,
                    visibility,
                )
                .map_err(RendererVirtualGeometryError::from)
            };
            match result {
                Ok(shading) => {
                    self.virtual_geometry
                        .as_mut()
                        .expect("prepared virtual state exists")
                        .shading = Some(shading);
                }
                Err(error) => {
                    self.virtual_geometry
                        .as_mut()
                        .expect("prepared virtual state exists")
                        .record_failure(&error);
                    return false;
                }
            }
        }
        self.virtual_geometry
            .as_ref()
            .is_some_and(RendererVirtualGeometry::prepared)
    }

    pub(crate) fn record_registered_virtual_hiz_capture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        source_size: (u32, u32),
    ) {
        let render_extent = self.render_extent();
        let view_projection = self.current_vp_matrix;
        let view = self.current_view_matrix;
        let source = &self.hiz_views[0];
        if let Some(state) = self.virtual_geometry.as_mut() {
            state.record_hiz_capture(
                &self.device,
                &self.queue,
                encoder,
                source,
                source_size,
                view_projection,
                view,
                render_extent,
            );
        }
    }

    pub(crate) fn invalidate_registered_virtual_hiz(&mut self, source_recreated: bool) {
        if let Some(state) = self.virtual_geometry.as_mut() {
            state.invalidate_hiz(source_recreated);
        }
    }

    pub(crate) fn draw_registered_virtual_visibility_raster<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
    ) -> bool {
        let Some(state) = self
            .virtual_geometry
            .as_ref()
            .filter(|state| state.prepared())
        else {
            return false;
        };
        match state.raster.draw(pass, &state.emitter) {
            Ok(()) => true,
            Err(error) => {
                log::error!("bloom: virtual-geometry raster suppressed: {error}");
                false
            }
        }
    }

    pub(crate) fn draw_registered_virtual_visibility_shading<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
    ) {
        let Some(state) = self
            .virtual_geometry
            .as_ref()
            .filter(|state| state.prepared())
        else {
            return;
        };
        let Some(shading) = state.shading.as_ref() else {
            return;
        };
        if let Err(error) = self.draw_virtual_visibility_shading(shading, pass, &state.selector) {
            log::error!("bloom: virtual-geometry shading suppressed: {error}");
        }
    }

    /// Build the explicit, unattached virtual-geometry PBR consumer against
    /// this renderer's exact lighting/material ABI. Ordinary frames do not
    /// call this and therefore allocate no virtual shading resources.
    pub fn create_virtual_visibility_shading(
        &self,
        pool: &GpuVirtualGeometryPool,
        selector: &GpuVirtualHierarchySelector,
        raster: &GpuVirtualVisibilityRaster,
        visibility: &wgpu::Texture,
    ) -> Result<GpuVirtualVisibilityShading, VirtualGeometryVisibilityError> {
        let Some(global_materials) = self.material_system.indirection.global_layout.as_ref() else {
            return Err(VirtualGeometryVisibilityError::PbrDeviceUnsupported);
        };
        let specialized = specialized_scene_shader_source(
            self.froxel.is_some(),
            self.shadow_map.virtual_map.requested(),
        );
        let gpu_source = gpu_driven::make_gpu_scene_shader(&specialized);
        GpuVirtualVisibilityShading::new(
            &self.device,
            pool,
            selector,
            raster,
            visibility,
            crate::virtual_geometry::shading::VirtualVisibilityPbrLayouts {
                draw: self.gpu_driven.draw_layout(),
                lighting: &self.lighting_layout,
                global_materials,
                joints: &self.joint_layout,
            },
            &gpu_source,
        )
    }

    /// Bind the renderer-owned scene globals and record one disjoint virtual
    /// fullscreen PBR pass into caller-provided four-MRT attachments.
    pub fn draw_virtual_visibility_shading<'a>(
        &'a self,
        shading: &'a GpuVirtualVisibilityShading,
        pass: &mut wgpu::RenderPass<'a>,
        selector: &GpuVirtualHierarchySelector,
    ) -> Result<(), VirtualGeometryVisibilityError> {
        let Some(global_materials) = self.material_system.indirection.global_bind_group.as_ref()
        else {
            return Err(VirtualGeometryVisibilityError::PbrDeviceUnsupported);
        };
        shading.draw(
            pass,
            selector,
            self.gpu_driven.draw_bind_group(),
            &self.lighting_bind_group,
            global_materials,
            &self.joint_bind_group,
        )
    }
}
