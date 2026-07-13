//! Lighting, shadows, post-FX, environment, and render-quality settings.
//!
//! Section of [`define_core_ffi!`](crate::define_core_ffi) — see
//! `ffi_core/mod.rs` for the architecture and the invoking-crate contract.
//! Internal: platform crates must invoke `define_core_ffi!()`, never the
//! section macros directly.

#[doc(hidden)]
#[macro_export]
macro_rules! __bloom_ffi_visual {
    () => {

        // bloom_set_env_clear_from_hdr  [source: curated; gated: image-extras]
        #[cfg(feature = "image-extras")]
        #[no_mangle]
        pub extern "C" fn bloom_set_env_clear_from_hdr(path_ptr: *const u8) {
            $crate::ffi::guard("bloom_set_env_clear_from_hdr", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                engine().renderer.set_env_clear_from_hdr_file(path);
        })
        }
        #[cfg(not(feature = "image-extras"))]
        #[no_mangle]
        pub extern "C" fn bloom_set_env_clear_from_hdr(_path_ptr: *const u8) {
            $crate::ffi::feature_off_warn_once("bloom_set_env_clear_from_hdr", "image-extras");
        }

        // bloom_set_target_fps  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_target_fps(fps: f64) {
            $crate::ffi::guard("bloom_set_target_fps", move || {
                engine().target_fps = fps;
        })
        }

        // bloom_set_direct_2d_mode  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_direct_2d_mode(on: f64) {
            $crate::ffi::guard("bloom_set_direct_2d_mode", move || {
                engine().direct_2d_mode = on > 0.5;
        })
        }

        // bloom_set_sound_volume  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_sound_volume(handle: f64, volume: f64) {
            $crate::ffi::guard("bloom_set_sound_volume", move || {
                engine().audio.set_sound_volume(handle, volume as f32);
        })
        }

        // bloom_set_master_volume  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_master_volume(volume: f64) {
            $crate::ffi::guard("bloom_set_master_volume", move || {
                engine().audio.set_master_volume(volume as f32);
        })
        }

        // bloom_set_listener_position  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_listener_position(x: f64, y: f64, z: f64, fx: f64, fy: f64, fz: f64) {
            $crate::ffi::guard("bloom_set_listener_position", move || {
                engine().audio.set_listener_position(x as f32, y as f32, z as f32, fx as f32, fy as f32, fz as f32);
        })
        }

        // bloom_set_texture_filter  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_texture_filter(handle: f64, mode: f64) {
            $crate::ffi::guard("bloom_set_texture_filter", move || {
                let eng = engine();
                if let Some(tex) = eng.textures.get(handle) {
                    let bind_group_idx = tex.bind_group_idx;
                    eng.renderer.set_texture_filter(bind_group_idx, mode > 0.5);
                }
        })
        }

        // bloom_set_material_reflection_probe  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_reflection_probe(
            material: f64, probe: f64,
        ) {
            $crate::ffi::guard("bloom_set_material_reflection_probe", move || {
                engine().renderer.set_material_reflection_probe(material as u32, probe as u32);
        })
        }

        // bloom_set_material_texture_array  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_texture_array(
            material: f64, slot: f64, array: f64,
        ) {
            $crate::ffi::guard("bloom_set_material_texture_array", move || {
                engine().renderer.set_material_texture_array(
                    material as u32, slot as u32, array as u32,
                );
        })
        }

        // bloom_set_material_shading_model  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_shading_model(
            material: f64, model: f64,
        ) {
            $crate::ffi::guard("bloom_set_material_shading_model", move || {
                engine().renderer.set_material_shading_model(material as u32, model as u32);
        })
        }

        // bloom_set_material_probe_visible  [source: shared]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_probe_visible(
            material: f64, visible: f64,
        ) {
            $crate::ffi::guard("bloom_set_material_probe_visible", move || {
                engine().renderer.set_material_probe_visible(material as u32, visible != 0.0);
        })
        }

        // bloom_set_material_foliage  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_foliage(
            material: f64,
            trans_r: f64, trans_g: f64, trans_b: f64,
            trans_amount: f64, wrap_factor: f64,
        ) {
            $crate::ffi::guard("bloom_set_material_foliage", move || {
                engine().renderer.set_material_foliage(
                    material as u32,
                    [trans_r as f32, trans_g as f32, trans_b as f32],
                    trans_amount as f32, wrap_factor as f32,
                );
        })
        }

        // bloom_set_post_pass  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_post_pass(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_set_post_pass", move || {
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.set_post_pass(source) {
                    Ok(()) => 1.0,
                    Err(e) => { eprintln!("[post_pass] compile failed: {:?}", e); 0.0 }
                }
        })
        }

        // bloom_set_joint_test  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_joint_test(joint_index: f64, angle: f64) {
            $crate::ffi::guard("bloom_set_joint_test", move || {
                engine().renderer.set_joint_test(joint_index as usize, angle as f32);
        })
        }

        // bloom_set_ambient_light  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ambient_light(r: f64, g: f64, b: f64, intensity: f64) {
            $crate::ffi::guard("bloom_set_ambient_light", move || {
                engine().renderer.set_ambient_light(r, g, b, intensity);
        })
        }

        // bloom_set_directional_light  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_directional_light(dx: f64, dy: f64, dz: f64, r: f64, g: f64, b: f64, intensity: f64) {
            $crate::ffi::guard("bloom_set_directional_light", move || {
                engine().renderer.set_directional_light(dx, dy, dz, r, g, b, intensity);
        })
        }

        // bloom_set_procedural_sky  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_procedural_sky(enabled: f64, rayleigh_density: f64, mie_density: f64, ground_albedo: f64) {
            $crate::ffi::guard("bloom_set_procedural_sky", move || {
                engine().renderer.set_procedural_sky(
                    enabled != 0.0,
                    rayleigh_density as f32,
                    mie_density as f32,
                    ground_albedo as f32,
                );
        })
        }

        // bloom_set_sun_direction  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_sun_direction(dx: f64, dy: f64, dz: f64, intensity: f64) {
            $crate::ffi::guard("bloom_set_sun_direction", move || {
                engine().renderer.set_sun_direction(dx as f32, dy as f32, dz as f32, intensity as f32);
        })
        }

        // bloom_set_fog  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_fog(r: f64, g: f64, b: f64, density: f64, height_ref: f64, height_falloff: f64) {
            $crate::ffi::guard("bloom_set_fog", move || {
                let r_ = engine();
                r_.renderer.set_fog_color(r as f32, g as f32, b as f32);
                r_.renderer.set_fog_density(density as f32);
                r_.renderer.set_fog_height_falloff(height_ref as f32, height_falloff as f32);
        })
        }

        // bloom_set_chromatic_aberration  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_chromatic_aberration(strength: f64) {
            $crate::ffi::guard("bloom_set_chromatic_aberration", move || {
                engine().renderer.set_chromatic_aberration(strength as f32);
        })
        }

        // bloom_set_vignette  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_vignette(strength: f64, softness: f64) {
            $crate::ffi::guard("bloom_set_vignette", move || {
                engine().renderer.set_vignette(strength as f32, softness as f32);
        })
        }

        // bloom_set_film_grain  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_film_grain(strength: f64) {
            $crate::ffi::guard("bloom_set_film_grain", move || {
                engine().renderer.set_film_grain(strength as f32);
        })
        }

        // bloom_set_sharpen_strength  [round-2 audit F8]
        #[no_mangle]
        pub extern "C" fn bloom_set_sharpen_strength(strength: f64) {
            $crate::ffi::guard("bloom_set_sharpen_strength", move || {
                engine().renderer.set_sharpen_strength(strength as f32);
        })
        }

        // bloom_set_present_mode  [round-2 audit F6]
        #[no_mangle]
        pub extern "C" fn bloom_set_present_mode(mode: f64) {
            $crate::ffi::guard("bloom_set_present_mode", move || {
                engine().renderer.set_present_mode(mode as u32);
        })
        }

        // bloom_set_sun_shafts  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_sun_shafts(strength: f64, decay: f64, r: f64, g: f64, b: f64) {
            $crate::ffi::guard("bloom_set_sun_shafts", move || {
                let eng = engine();
                eng.renderer.set_sun_shaft_strength(strength as f32);
                eng.renderer.set_sun_shaft_decay(decay as f32);
                eng.renderer.set_sun_shaft_color(r as f32, g as f32, b as f32);
        })
        }

        // bloom_set_auto_exposure  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_auto_exposure(on: f64) {
            $crate::ffi::guard("bloom_set_auto_exposure", move || {
                engine().renderer.set_auto_exposure(on != 0.0);
        })
        }

        #[no_mangle]
        pub extern "C" fn bloom_set_occlusion_culling(on: f64) {
            $crate::ffi::guard("bloom_set_occlusion_culling", move || {
                engine().renderer.occlusion.enabled = on != 0.0;
            })
        }

        // bloom_set_taa_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_taa_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_taa_enabled", move || {
                engine().renderer.set_taa_enabled(on != 0.0);
        })
        }

        // bloom_set_render_scale  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_render_scale(scale: f64) {
            $crate::ffi::guard("bloom_set_render_scale", move || {
                engine().renderer.set_render_scale(scale as f32);
        })
        }

        // bloom_set_upscale_mode  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_upscale_mode(mode: f64) {
            $crate::ffi::guard("bloom_set_upscale_mode", move || {
                engine().renderer.set_upscale_mode(mode as u32);
        })
        }

        // bloom_set_cas_strength  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_cas_strength(strength: f64) {
            $crate::ffi::guard("bloom_set_cas_strength", move || {
                engine().renderer.set_cas_strength(strength as f32);
        })
        }

        // bloom_set_auto_resolution  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_auto_resolution(target_hz: f64, enabled: f64) {
            $crate::ffi::guard("bloom_set_auto_resolution", move || {
                let eng = engine();
                if enabled != 0.0 {
                    let current = eng.renderer.render_scale();
                    eng.drs.enable(target_hz as f32, current);
                } else {
                    eng.drs.disable();
                }
        })
        }

        // bloom_set_manual_exposure  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_manual_exposure(value: f64) {
            $crate::ffi::guard("bloom_set_manual_exposure", move || {
                engine().renderer.set_manual_exposure(value as f32);
        })
        }

        // bloom_set_env_intensity  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_env_intensity(intensity: f64) {
            $crate::ffi::guard("bloom_set_env_intensity", move || {
                engine().renderer.set_env_intensity(intensity as f32);
        })
        }

        // bloom_set_ssgi_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ssgi_enabled(enabled: f64) {
            $crate::ffi::guard("bloom_set_ssgi_enabled", move || {
                engine().renderer.set_ssgi_enabled(enabled != 0.0);
        })
        }

        // bloom_set_path_tracing — 0 off / 1 progressive / 2 realtime
        // (docs/pt/pt-roadmap.md). Needs hardware ray query; without it
        // the request is stored but nothing engages — check
        // bloom_path_tracing_supported to know which world you are in.
        #[no_mangle]
        pub extern "C" fn bloom_set_path_tracing(mode: f64) {
            $crate::ffi::guard("bloom_set_path_tracing", move || {
                engine().renderer.set_path_tracing(mode as u32);
        })
        }

        // bloom_path_tracing_supported — 1.0 when the device can trace
        // (same ray-query requirement as Lumen's HW backend), else 0.0.
        #[no_mangle]
        pub extern "C" fn bloom_path_tracing_supported() -> f64 {
            $crate::ffi::guard("bloom_path_tracing_supported", move || {
                if engine().renderer.pt_supported() { 1.0 } else { 0.0 }
        })
        }

        // bloom_set_ssgi_intensity  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ssgi_intensity(intensity: f64) {
            $crate::ffi::guard("bloom_set_ssgi_intensity", move || {
                engine().renderer.set_ssgi_intensity(intensity as f32);
        })
        }

        // bloom_set_ssgi_radius  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ssgi_radius(radius: f64) {
            $crate::ffi::guard("bloom_set_ssgi_radius", move || {
                engine().renderer.set_ssgi_radius(radius as f32);
        })
        }

        // bloom_set_dof  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_dof(enabled: f64, focus_distance: f64, aperture: f64) {
            $crate::ffi::guard("bloom_set_dof", move || {
                let r = &mut engine().renderer;
                r.set_dof_enabled(enabled != 0.0);
                r.set_dof_focus_distance(focus_distance as f32);
                r.set_dof_aperture(aperture as f32);
        })
        }

        // bloom_set_quality_preset  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_quality_preset(preset: f64) {
            $crate::ffi::guard("bloom_set_quality_preset", move || {
                engine().renderer.apply_quality_preset(preset as u32);
        })
        }

        // bloom_set_shadows_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_shadows_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_shadows_enabled", move || {
                engine().renderer.set_shadows_enabled(on != 0.0);
        })
        }

        // bloom_set_shadows_always_fresh  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_shadows_always_fresh(on: f64) {
            $crate::ffi::guard("bloom_set_shadows_always_fresh", move || {
                engine().renderer.set_shadows_always_fresh(on != 0.0);
        })
        }

        // bloom_set_bloom_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_bloom_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_bloom_enabled", move || {
                engine().renderer.set_bloom_enabled(on != 0.0);
        })
        }

        // bloom_set_bloom_intensity  [source: art-direction]
        // Scales the bloom contribution added to the HDR scene before tonemap
        // (0 = none, ~0.04 subtle default, higher = stronger glow).
        #[no_mangle]
        pub extern "C" fn bloom_set_bloom_intensity(value: f64) {
            $crate::ffi::guard("bloom_set_bloom_intensity", move || {
                engine().renderer.set_bloom_intensity(value as f32);
        })
        }

        // bloom_set_tonemap  [source: art-direction]
        // Selects the tonemap operator: 0 = ACES (default), 1 = AgX (more
        // filmic, better highlight desaturation + a punchier look).
        #[no_mangle]
        pub extern "C" fn bloom_set_tonemap(kind: f64) {
            $crate::ffi::guard("bloom_set_tonemap", move || {
                engine().renderer.set_tonemap_kind(kind as u32);
        })
        }

        // bloom_set_auto_exposure_key  [source: art-direction]
        // Target scene-average luma for auto-exposure. Lower = the auto-
        // exposure aims for a darker, more saturated midpoint (less wash-out);
        // higher = brighter.
        #[no_mangle]
        pub extern "C" fn bloom_set_auto_exposure_key(key: f64) {
            $crate::ffi::guard("bloom_set_auto_exposure_key", move || {
                engine().renderer.set_auto_exposure_key(key as f32);
        })
        }

        // bloom_set_auto_exposure_rate  [source: art-direction]
        // Per-frame adaptation rate for auto-exposure (0 = frozen, ~0.05 = a
        // smooth eye-adaptation feel, 1 = instant).
        #[no_mangle]
        pub extern "C" fn bloom_set_auto_exposure_rate(rate: f64) {
            $crate::ffi::guard("bloom_set_auto_exposure_rate", move || {
                engine().renderer.set_auto_exposure_rate(rate as f32);
        })
        }

        // bloom_set_ssao_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ssao_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_ssao_enabled", move || {
                engine().renderer.set_ssao_enabled(on != 0.0);
        })
        }

        // bloom_set_ssao_intensity  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ssao_intensity(value: f64) {
            $crate::ffi::guard("bloom_set_ssao_intensity", move || {
                engine().renderer.set_ssao_strength(value as f32);
        })
        }

        // bloom_set_ssao_radius  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ssao_radius(world_radius: f64) {
            $crate::ffi::guard("bloom_set_ssao_radius", move || {
                engine().renderer.set_ssao_radius(world_radius as f32);
        })
        }

        // bloom_set_wind  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_wind(dir_x: f64, dir_z: f64, amplitude: f64, frequency: f64) {
            $crate::ffi::guard("bloom_set_wind", move || {
                engine().renderer.set_wind(dir_x as f32, dir_z as f32, amplitude as f32, frequency as f32);
        })
        }

        // bloom_set_model_foliage_wind  [EN-041]
        //
        // Mark a cached model as a plant so the wind bends it. amount ~1.0 for a
        // tree. The engine used to sway alpha-cut materials only, so leaf cards
        // fluttered and every trunk stood rigid.
        #[no_mangle]
        pub extern "C" fn bloom_set_model_foliage_wind(model: f64, amount: f64) {
            $crate::ffi::guard("bloom_set_model_foliage_wind", move || {
                engine().renderer.set_model_foliage_wind(model.to_bits(), amount as f32);
        })
        }

        // bloom_set_foliage_shadow_motion  [EN-041]
        //
        // Let foliage sway in the shadow pass too, so the canopy dapple moves.
        // NOT free: a moving caster cannot reuse the cached static shadow depth.
        #[no_mangle]
        pub extern "C" fn bloom_set_foliage_shadow_motion(on: f64) {
            $crate::ffi::guard("bloom_set_foliage_shadow_motion", move || {
                engine().renderer.set_foliage_shadow_motion(on > 0.5);
        })
        }

        // bloom_set_output_scale  [EN-046]
        //
        // Shrink the SWAPCHAIN, not the G-buffer. This is the only knob that touches
        // the fixed cost of the TSR upscale + final composite, which is what actually
        // dominates a 4K frame — render_scale does not.
        #[no_mangle]
        pub extern "C" fn bloom_set_output_scale(scale: f64) {
            $crate::ffi::guard("bloom_set_output_scale", move || {
                engine().renderer.set_output_scale(scale as f32);
        })
        }

        #[no_mangle]
        pub extern "C" fn bloom_get_output_scale() -> f64 {
            $crate::ffi::guard("bloom_get_output_scale", move || {
                engine().renderer.output_scale() as f64
        })
        }

        // bloom_set_cloud_shadows  [EN-040]
        //
        // Opt the world into the deck the sky is already drawing. strength 0
        // (the default) = sky-only clouds, which is what every game that never
        // calls this keeps.
        #[no_mangle]
        pub extern "C" fn bloom_set_cloud_shadows(
            strength: f64, deck_height: f64, feature_scale: f64, drift_speed: f64,
        ) {
            $crate::ffi::guard("bloom_set_cloud_shadows", move || {
                engine().renderer.set_cloud_shadows(
                    strength as f32, deck_height as f32,
                    feature_scale as f32, drift_speed as f32);
        })
        }

        // bloom_set_ssr_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_ssr_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_ssr_enabled", move || {
                engine().renderer.set_ssr_enabled(on != 0.0);
        })
        }

        // bloom_set_motion_blur_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_motion_blur_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_motion_blur_enabled", move || {
                engine().renderer.set_motion_blur_enabled(on != 0.0);
        })
        }

        // bloom_set_sss_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_sss_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_sss_enabled", move || {
                engine().renderer.set_sss_enabled(on != 0.0);
        })
        }

        // bloom_set_profiler_enabled  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_profiler_enabled(on: f64) {
            $crate::ffi::guard("bloom_set_profiler_enabled", move || {
                engine().profiler.set_enabled(on != 0.0);
        })
        }

        // bloom_set_music_volume  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_music_volume(handle: f64, volume: f64) {
            $crate::ffi::guard("bloom_set_music_volume", move || {
                engine().audio.set_music_volume(handle, volume as f32);
        })
        }

        // bloom_add_directional_light  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_add_directional_light(
            dx: f64, dy: f64, dz: f64,
            r: f64, g: f64, b: f64,
            intensity: f64,
        ) {
            $crate::ffi::guard("bloom_add_directional_light", move || {
                engine().renderer.add_directional_light(
                    dx as f32, dy as f32, dz as f32,
                    r as f32, g as f32, b as f32,
                    intensity as f32,
                );
        })
        }

        // bloom_add_point_light  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_add_point_light(
            x: f64, y: f64, z: f64, range: f64,
            r: f64, g: f64, b: f64,
            intensity: f64,
        ) {
            $crate::ffi::guard("bloom_add_point_light", move || {
                engine().renderer.add_point_light(
                    x as f32, y as f32, z as f32, range as f32,
                    r as f32, g as f32, b as f32,
                    intensity as f32,
                );
        })
        }

        // bloom_set_cursor_shape  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_set_cursor_shape(shape: f64) {
            $crate::ffi::guard("bloom_set_cursor_shape", move || {
                engine().input.cursor_shape = shape as u32;
        })
        }

        // bloom_enable_shadows  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_enable_shadows() {
            $crate::ffi::guard("bloom_enable_shadows", move || {
                engine().renderer.shadow_map.enable();
        })
        }

        // bloom_disable_shadows  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_disable_shadows() {
            $crate::ffi::guard("bloom_disable_shadows", move || {
                engine().renderer.shadow_map.disable();
        })
        }

        // bloom_dump_shadow_map  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_dump_shadow_map(path_ptr: *const u8) {
            $crate::ffi::guard("bloom_dump_shadow_map", move || {
                let path = $crate::string_header::str_from_header(path_ptr).to_string();
                engine().renderer.dump_shadow_map(&path);
        })
        }

        // bloom_enable_postfx  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_enable_postfx() {
            $crate::ffi::guard("bloom_enable_postfx", move || {
                let eng = engine();
                let w = eng.renderer.width();
                let h = eng.renderer.height();
                let fmt = eng.renderer.surface_format();
                eng.postfx = Some($crate::postfx::PostFxPipeline::new(
                    &eng.renderer.device, w, h, fmt,
                ));
        })
        }

        // bloom_disable_postfx  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_disable_postfx() {
            $crate::ffi::guard("bloom_disable_postfx", move || {
                engine().postfx = None;
        })
        }

        // bloom_splat_impulse  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_splat_impulse(x: f64, z: f64, radius: f64, strength: f64) {
            $crate::ffi::guard("bloom_splat_impulse", move || {
                engine().renderer.impulse_field.submit_splat(
                    x as f32, z as f32, radius as f32, strength as f32,
                );
        })
        }

    };
}
