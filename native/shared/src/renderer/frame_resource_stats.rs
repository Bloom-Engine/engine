//! Allocation-free counters for GPU objects created while recording a frame.
//!
//! The renderer deliberately uses a fixed array instead of a map so measuring
//! steady-state object creation cannot itself introduce heap churn. Sites are
//! named and stable in quality telemetry, which lets qualification distinguish
//! a recurring allocation from a bounded rebuild after a resize or mode change.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(super) enum BindGroupCreationSite {
    SceneCompose,
    SsrTemporal,
    Upscale,
    Taa,
    TaaReactive,
    DepthOfField,
    MotionBlur,
    SubsurfaceScattering,
    ContrastAdaptiveSharpen,
    AutoExposure,
    FinalComposite,
    CustomPostPass,
}

impl BindGroupCreationSite {
    const COUNT: usize = 12;
    #[cfg(not(target_arch = "wasm32"))]
    const ALL: [Self; Self::COUNT] = [
        Self::SceneCompose,
        Self::SsrTemporal,
        Self::Upscale,
        Self::Taa,
        Self::TaaReactive,
        Self::DepthOfField,
        Self::MotionBlur,
        Self::SubsurfaceScattering,
        Self::ContrastAdaptiveSharpen,
        Self::AutoExposure,
        Self::FinalComposite,
        Self::CustomPostPass,
    ];

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::SceneCompose => "scene_compose",
            Self::SsrTemporal => "ssr_temporal",
            Self::Upscale => "upscale",
            Self::Taa => "taa",
            Self::TaaReactive => "taa_reactive",
            Self::DepthOfField => "depth_of_field",
            Self::MotionBlur => "motion_blur",
            Self::SubsurfaceScattering => "subsurface_scattering",
            Self::ContrastAdaptiveSharpen => "contrast_adaptive_sharpen",
            Self::AutoExposure => "auto_exposure",
            Self::FinalComposite => "final_composite",
            Self::CustomPostPass => "custom_post_pass",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct FrameResourceStats {
    bind_group_creations: [u32; BindGroupCreationSite::COUNT],
    pipeline_creation_count_at_begin: u64,
    pipeline_creations: u32,
    graph_compiles: u32,
    command_encoder_creations: u32,
    physical_texture_creations: u32,
    physical_buffer_creations: u32,
}

impl FrameResourceStats {
    pub(super) fn begin_frame(&mut self, pipeline_creation_count: u64) {
        self.bind_group_creations.fill(0);
        self.pipeline_creation_count_at_begin = pipeline_creation_count;
        self.pipeline_creations = 0;
        self.graph_compiles = 0;
        self.command_encoder_creations = 0;
        self.physical_texture_creations = 0;
        self.physical_buffer_creations = 0;
    }

    pub(super) fn created_bind_group(&mut self, site: BindGroupCreationSite) {
        let count = &mut self.bind_group_creations[site as usize];
        *count = count.saturating_add(1);
    }

    pub(super) fn created_graph_compiles(&mut self, count: u64) {
        self.graph_compiles = self
            .graph_compiles
            .saturating_add(count.min(u32::MAX as u64) as u32);
    }

    pub(super) fn finish_pipeline_creations(&mut self, pipeline_creation_count: u64) {
        let creations =
            pipeline_creation_count.saturating_sub(self.pipeline_creation_count_at_begin);
        self.pipeline_creations = creations.min(u32::MAX as u64) as u32;
    }

    pub(super) fn created_command_encoder(&mut self) {
        self.command_encoder_creations = self.command_encoder_creations.saturating_add(1);
    }

    pub(super) fn created_physical_textures(&mut self, count: u32) {
        self.physical_texture_creations = self.physical_texture_creations.saturating_add(count);
    }

    pub(super) fn created_physical_buffers(&mut self, count: u32) {
        self.physical_buffer_creations = self.physical_buffer_creations.saturating_add(count);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) const fn graph_compiles(&self) -> u32 {
        self.graph_compiles
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) const fn pipeline_creations(&self) -> u32 {
        self.pipeline_creations
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) const fn command_encoder_creations(&self) -> u32 {
        self.command_encoder_creations
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) const fn physical_texture_creations(&self) -> u32 {
        self.physical_texture_creations
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) const fn physical_buffer_creations(&self) -> u32 {
        self.physical_buffer_creations
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn total_bind_group_creations(&self) -> u32 {
        self.bind_group_creations
            .iter()
            .copied()
            .fold(0, u32::saturating_add)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn bind_group_creations(
        &self,
    ) -> impl Iterator<Item = (BindGroupCreationSite, u32)> + '_ {
        BindGroupCreationSite::ALL
            .into_iter()
            .map(|site| (site, self.bind_group_creations[site as usize]))
    }
}

impl super::Renderer {
    pub(super) fn created_pipelines(&mut self, count: u64) {
        self.pipeline_creation_count = self.pipeline_creation_count.saturating_add(count);
    }

    pub(super) fn total_pipeline_creation_count(&self) -> u64 {
        self.pipeline_creation_count
            .saturating_add(self.material_system.pipeline_creation_count)
    }

    pub(super) fn finish_frame_resource_stats(&mut self) {
        self.frame_resource_stats
            .finish_pipeline_creations(self.total_pipeline_creation_count());
        if cfg!(not(target_arch = "wasm32"))
            && (self.screenshot_requested || self.pending_quality_capture_dir.is_some())
        {
            return;
        }
        self.steady_state_frame_resource_stats = self.frame_resource_stats;
    }
}

#[cfg(test)]
mod tests {
    use super::{BindGroupCreationSite, FrameResourceStats};

    #[test]
    fn counters_are_named_bounded_and_reset_per_frame() {
        let mut stats = FrameResourceStats::default();
        stats.begin_frame(20);
        stats.created_bind_group(BindGroupCreationSite::Taa);
        stats.created_bind_group(BindGroupCreationSite::CustomPostPass);
        stats.created_bind_group(BindGroupCreationSite::CustomPostPass);
        stats.created_graph_compiles(1);
        stats.created_command_encoder();
        stats.created_physical_textures(2);
        stats.created_physical_buffers(3);
        stats.finish_pipeline_creations(22);

        assert_eq!(stats.total_bind_group_creations(), 3);
        assert_eq!(stats.pipeline_creations(), 2);
        assert_eq!(stats.graph_compiles(), 1);
        assert_eq!(stats.command_encoder_creations(), 1);
        assert_eq!(stats.physical_texture_creations(), 2);
        assert_eq!(stats.physical_buffer_creations(), 3);
        let custom = stats
            .bind_group_creations()
            .find(|(site, _)| *site == BindGroupCreationSite::CustomPostPass)
            .expect("custom post-pass site is reported");
        assert_eq!(custom.0.name(), "custom_post_pass");
        assert_eq!(custom.1, 2);

        stats.begin_frame(22);
        stats.finish_pipeline_creations(22);
        assert_eq!(stats.total_bind_group_creations(), 0);
        assert!(stats.bind_group_creations().all(|(_, count)| count == 0));
        assert_eq!(stats.graph_compiles(), 0);
        assert_eq!(stats.command_encoder_creations(), 0);
        assert_eq!(stats.physical_texture_creations(), 0);
        assert_eq!(stats.physical_buffer_creations(), 0);
        assert_eq!(stats.pipeline_creations(), 0);
    }
}
