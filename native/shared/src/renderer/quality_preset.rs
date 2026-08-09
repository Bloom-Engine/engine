//! Coherent resolution, reconstruction, and effect policy for quality presets.

use super::Renderer;

/// Balanced first-run default: 56% of native pixel shading instead of the
/// former 25%, while leaving a meaningful performance tier below it.
pub(super) const DEFAULT_RENDER_SCALE: f32 = 0.75;

#[derive(Clone, Copy, Debug, PartialEq)]
struct QualityPresetConfig {
    render_scale: f32,
    taa: bool,
    upscale_mode: u32,
    composite_sharpen: f32,
    cas_sharpen: f32,
    shadows: bool,
    ssao: bool,
    bloom: bool,
    ssr: bool,
    ssgi: bool,
    motion_blur: bool,
    sss: bool,
    chromatic_aberration: f32,
}

fn quality_preset_config(preset: u32) -> QualityPresetConfig {
    match preset {
        0 => QualityPresetConfig {
            render_scale: 0.50,
            taa: false,
            upscale_mode: 0,
            composite_sharpen: 0.0,
            cas_sharpen: 0.0,
            shadows: false,
            ssao: false,
            bloom: false,
            ssr: false,
            ssgi: false,
            motion_blur: false,
            sss: false,
            chromatic_aberration: 0.0,
        },
        1 => QualityPresetConfig {
            render_scale: 0.67,
            taa: false,
            upscale_mode: 1,
            composite_sharpen: 0.25,
            cas_sharpen: 0.0,
            shadows: false,
            ssao: false,
            bloom: true,
            ssr: false,
            ssgi: false,
            motion_blur: false,
            sss: false,
            chromatic_aberration: 0.0,
        },
        2 => QualityPresetConfig {
            render_scale: DEFAULT_RENDER_SCALE,
            taa: true,
            upscale_mode: 1,
            composite_sharpen: 0.40,
            cas_sharpen: 0.0,
            shadows: true,
            ssao: true,
            bloom: true,
            ssr: false,
            ssgi: false,
            motion_blur: false,
            sss: false,
            chromatic_aberration: 0.0,
        },
        3 => QualityPresetConfig {
            render_scale: 0.85,
            taa: true,
            upscale_mode: 1,
            composite_sharpen: 0.45,
            cas_sharpen: 0.0,
            shadows: true,
            ssao: true,
            bloom: true,
            ssr: true,
            ssgi: true,
            motion_blur: false,
            sss: false,
            chromatic_aberration: 0.002,
        },
        _ => QualityPresetConfig {
            render_scale: 1.0,
            taa: true,
            upscale_mode: 1,
            composite_sharpen: 0.50,
            cas_sharpen: 0.0,
            shadows: true,
            ssao: true,
            bloom: true,
            ssr: true,
            ssgi: true,
            motion_blur: true,
            // The current screen-space SSS pass has no per-pixel material
            // classification, so enabling it here diffuses every opaque
            // surface (wood, stone, metal, and glass included). Keep the
            // explicit SSS API available for authored experiments, but do not
            // turn a material-specific effect into a full-frame Ultra blur.
            sss: false,
            chromatic_aberration: 0.003,
        },
    }
}

impl Renderer {
    /// Apply one coherent quality tier. Resolution, temporal reconstruction,
    /// upscale filtering, sharpening, and effects change together. Individual
    /// setters remain overrides and should be called after this method.
    pub fn apply_quality_preset(&mut self, preset: u32) {
        let config = quality_preset_config(preset);
        self.set_render_scale(config.render_scale);
        self.set_taa_enabled(config.taa);
        self.set_upscale_mode(config.upscale_mode);
        self.set_sharpen_strength(config.composite_sharpen);
        self.set_cas_strength(config.cas_sharpen);
        self.set_shadows_enabled(config.shadows);
        self.set_ssao_enabled(config.ssao);
        self.set_bloom_enabled(config.bloom);
        self.set_ssr_enabled(config.ssr);
        self.set_ssgi_enabled(config.ssgi);
        self.set_motion_blur_enabled(config.motion_blur);
        self.set_sss_enabled(config.sss);
        self.set_chromatic_aberration(config.chromatic_aberration);
    }
}

#[cfg(test)]
mod tests {
    use super::{quality_preset_config, DEFAULT_RENDER_SCALE};

    #[test]
    fn tiers_raise_resolution_monotonically_and_ultra_is_native() {
        let configs = (0..=4).map(quality_preset_config).collect::<Vec<_>>();
        assert_eq!(DEFAULT_RENDER_SCALE, 0.75);
        assert_eq!(configs[4].render_scale, 1.0);
        for pair in configs.windows(2) {
            assert!(pair[0].render_scale < pair[1].render_scale);
        }
    }

    #[test]
    fn reconstruction_and_sharpening_are_explicit_per_tier() {
        let off = quality_preset_config(0);
        let low = quality_preset_config(1);
        let medium = quality_preset_config(2);
        let high = quality_preset_config(3);
        let ultra = quality_preset_config(4);

        assert!(!off.taa && !low.taa);
        assert!(medium.taa && high.taa && ultra.taa);
        assert_eq!(off.upscale_mode, 0);
        assert_eq!(low.upscale_mode, 1);
        assert_eq!(off.composite_sharpen, 0.0);
        assert!(low.composite_sharpen < medium.composite_sharpen);
        assert!(medium.composite_sharpen < high.composite_sharpen);
        assert!(high.composite_sharpen < ultra.composite_sharpen);
        assert!([off, low, medium, high, ultra]
            .iter()
            .all(|config| config.cas_sharpen == 0.0));
        assert!([off, low, medium, high, ultra]
            .iter()
            .all(|config| !config.sss));
    }

    #[test]
    fn out_of_range_presets_clamp_to_ultra_policy() {
        assert_eq!(quality_preset_config(5), quality_preset_config(4));
        assert_eq!(quality_preset_config(u32::MAX), quality_preset_config(4));
    }
}
