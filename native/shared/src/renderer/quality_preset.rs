//! Coherent resolution, reconstruction, and effect policy for quality presets.

use super::{formats, Renderer};

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
            composite_sharpen: 0.85,
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
    /// Spacing of the screen-space GI probes, in half-resolution pixels
    /// (default 8, range 4..=32). Probe count, and with it the trace,
    /// temporal and resolve cost, falls with the square of the spacing.
    /// Indirect light is low-frequency, so a high-density display can take
    /// 12 or 16 with little visible change: at 3200x1800 on an M1 Max,
    /// 16 took 3.1 ms off a 19.1 ms frame. Rebuilds the probe targets.
    pub fn set_ssgi_probe_spacing(&mut self, pixels: u32) {
        let tile = pixels.clamp(4, 32);
        if tile != formats::probe_tile_size() {
            formats::store_probe_tile_size(tile);
            self.rebuild_render_targets();
        }
    }

    /// Divisor from the render resolution to the ambient-occlusion targets
    /// (default 2 = half resolution, range 1..=4). The Hi-Z pyramid it
    /// samples stays at half resolution. At 3200x1800 on an M1 Max, 3 took
    /// 2.1 ms and 4 took 2.3 ms off a 19.2 ms frame. Coarser targets also
    /// weaken the occlusion: at 3 a dim interior read about 5 of 255 levels
    /// brighter. Rebuilds the ambient-occlusion targets.
    pub fn set_ssao_downscale(&mut self, divisor: u32) {
        let divisor = divisor.clamp(1, 4);
        if divisor != formats::ao_divisor() {
            formats::store_ao_divisor(divisor);
            self.rebuild_render_targets();
        }
    }

    fn rebuild_render_targets(&mut self) {
        let (w, h) = (self.native_width, self.native_height);
        let (lw, lh) = (self.logical_width, self.logical_height);
        if w > 0 && h > 0 {
            self.resize(w, h, lw, lh);
        }
    }

    /// Whether TAA also upscales when the render scale is below 1 (a temporal
    /// upscaler: render-resolution input, history and output at the surface's
    /// size). Off, TAA resolves at the render resolution and the upscale pass
    /// scales its result, the order mobile renderers use (Unity URP): the
    /// temporal pass then costs render_scale² as much, which on a phone's 2-3x
    /// density display is most of the frame. Default off on Android, on
    /// elsewhere. Triggers a resize, since TAA's history changes size.
    pub fn set_temporal_upscale(&mut self, enabled: bool) {
        if enabled != self.temporal_upscale {
            self.temporal_upscale = enabled;
            self.taa_current_idx = 0;
            self.taa_frame_index = 0;
            self.taa_history_valid = false;
            self.taa_history_written = false;
            let (w, h) = (self.native_width, self.native_height);
            self.resize(w, h, self.logical_width, self.logical_height);
        }
    }

    /// The fast TAA: the current pixel as rendered, history clamped to its
    /// 5-pixel cross, one bilinear history fetch (`fs_main_fast`), instead of
    /// the full reconstruction with its provenance and detail-lock
    /// heuristics. The low tier of temporal AA, for TAA resolving at the
    /// render resolution (`set_temporal_upscale(false)`). Default on Android
    /// (Pixel 4a: 16 -> 4 ms at 0.5 render scale).
    pub fn set_taa_fast(&mut self, on: bool) {
        if on != self.taa_fast {
            self.taa_fast = on;
            self.taa_history_valid = false;
        }
    }

    /// The sun shadow's filter: `true` takes 4 hardware-PCF taps (a
    /// 3x3-texel tent: the filter mobile renderers default to, a quarter of
    /// the cost), `false` the 16-tap Poisson disc. Default 4-tap on Android.
    /// Applies from the current frame (dir_light_count.w).
    pub fn set_shadow_filter_4tap(&mut self, on: bool) {
        self.shadow_map.filter_4tap = on;
        self.lighting_uniforms.dir_light_count[3] = if on { 1.0 } else { 0.0 };
    }

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
