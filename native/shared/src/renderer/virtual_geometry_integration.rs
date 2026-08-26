use super::{gpu_driven, specialized_scene_shader_source, Renderer};
use crate::virtual_geometry::{
    GpuVirtualDrawEmitter, GpuVirtualGeometryConfig, GpuVirtualGeometryPool,
    GpuVirtualHierarchySelector, GpuVirtualInstance, GpuVirtualPageStreamer,
    GpuVirtualStreamingConfig, GpuVirtualStreamingError, GpuVirtualTraversalConfig,
    GpuVirtualVisibilityFrame, GpuVirtualVisibilityRaster, GpuVirtualVisibilityShading,
    VirtualGeometryDrawEmissionError, VirtualGeometryGpuError, VirtualGeometryTraversalError,
    VirtualGeometryView, VirtualGeometryVisibilityError,
};
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
    last_failure: Option<String>,
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
            last_failure: None,
        })
    }

    pub(crate) fn begin_frame(&mut self, device: &wgpu::Device) {
        self.streamer.poll(device);
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
    ) -> Result<(), RendererVirtualGeometryError> {
        self.prepared = false;
        let Some(submission) = self.submission.as_ref() else {
            return Ok(());
        };
        self.frame = self.frame.wrapping_add(1).max(1);
        self.pool.begin_frame(self.frame);
        self.streamer.service(&mut self.pool, queue);
        let dispatch = self.selector.record(
            queue,
            encoder,
            &self.pool,
            &submission.instances,
            submission.view,
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
    }
}

/// Errors are explicit because the virtual path must fail closed: a rejected
/// producer never leaves partial IDs or half a virtual batch in the frame.
#[derive(Debug)]
pub enum RendererVirtualGeometryError {
    AlreadyEnabled,
    NotEnabled,
    VisibilityShadingRequired,
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
        self.virtual_geometry.take().is_some()
    }

    /// Mutable residency/asset owner for explicit `.bgeo` registration and
    /// page streaming. It is unavailable until virtual geometry is enabled.
    pub fn virtual_geometry_pool_mut(&mut self) -> Option<&mut GpuVirtualGeometryPool> {
        self.virtual_geometry.as_mut().map(|state| &mut state.pool)
    }

    /// Bounded feedback/readback/upload counters for debug UIs and captures.
    pub fn virtual_geometry_streaming_telemetry(
        &self,
    ) -> Option<crate::virtual_geometry::GpuVirtualStreamingTelemetry> {
        self.virtual_geometry
            .as_ref()
            .map(|state| state.streamer.telemetry())
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
                "\"streaming_truncated_requests\":0,\"streaming_readback_bytes\":0}"
            )
            .to_string();
        };
        let telemetry = state.pool.telemetry();
        let streaming = state.streamer.telemetry();
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
                "\"streaming_truncated_requests\":{},\"streaming_readback_bytes\":{}}}"
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
        let producer_result = self
            .virtual_geometry
            .as_mut()
            .map(|state| state.record_producers(&self.queue, encoder));
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
