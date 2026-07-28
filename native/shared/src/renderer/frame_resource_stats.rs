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
}

impl FrameResourceStats {
    pub(super) fn begin_frame(&mut self) {
        self.bind_group_creations.fill(0);
    }

    pub(super) fn created_bind_group(&mut self, site: BindGroupCreationSite) {
        let count = &mut self.bind_group_creations[site as usize];
        *count = count.saturating_add(1);
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

#[cfg(test)]
mod tests {
    use super::{BindGroupCreationSite, FrameResourceStats};

    #[test]
    fn counters_are_named_bounded_and_reset_per_frame() {
        let mut stats = FrameResourceStats::default();
        stats.created_bind_group(BindGroupCreationSite::Taa);
        stats.created_bind_group(BindGroupCreationSite::CustomPostPass);
        stats.created_bind_group(BindGroupCreationSite::CustomPostPass);

        assert_eq!(stats.total_bind_group_creations(), 3);
        let custom = stats
            .bind_group_creations()
            .find(|(site, _)| *site == BindGroupCreationSite::CustomPostPass)
            .expect("custom post-pass site is reported");
        assert_eq!(custom.0.name(), "custom_post_pass");
        assert_eq!(custom.1, 2);

        stats.begin_frame();
        assert_eq!(stats.total_bind_group_creations(), 0);
        assert!(stats.bind_group_creations().all(|(_, count)| count == 0));
    }
}
