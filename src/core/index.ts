import { Color, Camera2D, Camera3D } from './types';

export type { Color, Vec2, Vec3, Vec4, Rect, Camera2D, Camera3D, Texture, Font, Sound, Music, Quat, Ray, BoundingBox, Model, Mat4, RayHit, FrustumPlanes } from './types';
export { Color, ColorConstants, Colors } from './colors';
export { Key, MouseButton } from './keys';

// FFI declarations
declare function bloom_init_window(width: number, height: number, title: number, fullscreen: number): void;
declare function bloom_attach_native(handle: number, width: number, height: number): number;
declare function bloom_close_window(): void;
declare function bloom_attach_hwnd(hwnd: number, width: number, height: number): void;
declare function bloom_resize(physW: number, physH: number, logW: number, logH: number): void;
declare function bloom_window_should_close(): number;
declare function bloom_begin_drawing(): void;
declare function bloom_end_drawing(): void;
declare function bloom_take_screenshot(path: number): void;
declare function bloom_clear_background(r: number, g: number, b: number, a: number): void;
declare function bloom_set_env_clear_from_hdr(path: number): void;
declare function bloom_set_fog(r: number, g: number, b: number, density: number, height_ref: number, height_falloff: number): void;
declare function bloom_set_chromatic_aberration(strength: number): void;
declare function bloom_set_vignette(strength: number, softness: number): void;
declare function bloom_set_film_grain(strength: number): void;
declare function bloom_set_sharpen_strength(strength: number): void;
declare function bloom_set_sun_shafts(strength: number, decay: number, r: number, g: number, b: number): void;
declare function bloom_set_auto_exposure(on: number): void;
declare function bloom_set_taa_enabled(on: number): void;
declare function bloom_set_occlusion_culling(on: number): void;
declare function bloom_set_render_scale(scale: number): void;
declare function bloom_get_render_scale(): number;
declare function bloom_set_upscale_mode(mode: number): void;
declare function bloom_set_cas_strength(strength: number): void;
declare function bloom_get_physical_width(): number;
declare function bloom_get_physical_height(): number;
declare function bloom_set_auto_resolution(targetHz: number, enabled: number): void;
declare function bloom_set_manual_exposure(value: number): void;
declare function bloom_set_env_intensity(intensity: number): void;
declare function bloom_set_ssgi_enabled(on: number): void;
declare function bloom_set_ssgi_intensity(intensity: number): void;
declare function bloom_set_ssgi_radius(radius: number): void;
declare function bloom_set_dof(enabled: number, focusDistance: number, aperture: number): void;
declare function bloom_set_quality_preset(preset: number): void;
declare function bloom_set_shadows_enabled(on: number): void;
declare function bloom_set_shadows_always_fresh(on: number): void;
declare function bloom_set_bloom_enabled(on: number): void;
declare function bloom_set_bloom_intensity(value: number): void;
declare function bloom_set_tonemap(kind: number): void;
declare function bloom_set_auto_exposure_key(key: number): void;
declare function bloom_set_auto_exposure_rate(rate: number): void;
declare function bloom_set_ssao_enabled(on: number): void;
declare function bloom_set_post_pass(source: number): number;
declare function bloom_clear_post_pass(): void;
declare function bloom_add_post_pass(source: number): number;
declare function bloom_clear_all_post_passes(): void;
declare function bloom_set_ssao_intensity(intensity: number): void;
declare function bloom_set_ssao_radius(worldRadius: number): void;
declare function bloom_set_wind(dirX: number, dirZ: number, amplitude: number, frequency: number): void;
declare function bloom_set_ssr_enabled(on: number): void;
declare function bloom_set_motion_blur_enabled(on: number): void;
declare function bloom_set_sss_enabled(on: number): void;
declare function bloom_set_profiler_enabled(on: number): void;
declare function bloom_get_profiler_frame_cpu_us(): number;
declare function bloom_get_profiler_frame_gpu_us(): number;
declare function bloom_print_profiler_summary(): void;
declare function bloom_profiler_overlay_text(): string;
declare function bloom_profiler_frame_history(): string;
declare function bloom_splat_impulse(x: number, z: number, radius: number, strength: number): void;
declare function bloom_set_material_params(handle: number, paramsPtr: any, paramCount: number): void;
declare function bloom_set_material_params_scratch(handle: number, paramCount: number): void;
declare function bloom_mesh_scratch_reset(): void;
declare function bloom_mesh_scratch_push_f32(v: number): void;
declare function bloom_set_target_fps(fps: number): void;
declare function bloom_set_direct_2d_mode(on: number): void;
declare function bloom_get_delta_time(): number;
declare function bloom_get_fps(): number;
declare function bloom_get_screen_width(): number;
declare function bloom_get_screen_height(): number;
declare function bloom_is_key_pressed(key: number): number;
declare function bloom_is_key_down(key: number): number;
declare function bloom_is_key_released(key: number): number;
declare function bloom_get_mouse_x(): number;
declare function bloom_get_mouse_y(): number;
declare function bloom_is_mouse_button_pressed(btn: number): number;
declare function bloom_is_mouse_button_down(btn: number): number;
declare function bloom_is_mouse_button_released(btn: number): number;

// Camera FFI
declare function bloom_begin_mode_2d(ox: number, oy: number, tx: number, ty: number, rot: number, zoom: number): void;
declare function bloom_end_mode_2d(): void;
declare function bloom_begin_mode_3d(px: number, py: number, pz: number, tx: number, ty: number, tz: number, ux: number, uy: number, uz: number, fovy: number, proj: number): void;
declare function bloom_end_mode_3d(): void;

// Gamepad FFI
declare function bloom_is_gamepad_available(): number;
declare function bloom_get_gamepad_axis(axis: number): number;
declare function bloom_is_gamepad_button_pressed(btn: number): number;
declare function bloom_is_gamepad_button_down(btn: number): number;
declare function bloom_is_gamepad_button_released(btn: number): number;
declare function bloom_get_gamepad_axis_count(): number;

// Touch FFI
declare function bloom_get_touch_x(index: number): number;
declare function bloom_get_touch_y(index: number): number;
declare function bloom_get_touch_count(): number;

// Input injection FFI
declare function bloom_inject_key_down(key: number): void;
declare function bloom_inject_key_up(key: number): void;
declare function bloom_inject_gamepad_axis(axis: number, value: number): void;
declare function bloom_inject_gamepad_button_down(button: number): void;
declare function bloom_inject_gamepad_button_up(button: number): void;
declare function bloom_get_platform(): number;
declare function bloom_get_language(): number;
declare function bloom_is_any_input_pressed(): number;
declare function bloom_get_crown_rotation(): number;

// Utility FFI
declare function bloom_toggle_fullscreen(): void;
declare function bloom_set_window_title(title: number): void;
declare function bloom_get_time(): number;
declare function bloom_set_window_icon(path: number): void;
declare function bloom_disable_cursor(): void;
declare function bloom_enable_cursor(): void;
declare function bloom_get_mouse_delta_x(): number;
declare function bloom_get_mouse_delta_y(): number;
declare function bloom_get_mouse_wheel(): number;
declare function bloom_get_char_pressed(): number;
declare function bloom_set_cursor_shape(shape: number): void;
declare function bloom_set_clipboard_text(text: number): void;
declare function bloom_get_clipboard_text(): number;
declare function bloom_open_file_dialog(filter: number, title: number): number;
declare function bloom_save_file_dialog(defaultName: number, title: number): number;
declare function bloom_write_file(path: number, data: number): number;
declare function bloom_file_exists(path: number): number;
declare function bloom_read_file(path: number): number;
declare function bloom_run_game(callback: number): void;

// Window management

export function initWindow(width: number, height: number, title: string, fullscreen: boolean = false): void {
  bloom_init_window(width, height, title as any, fullscreen ? 1.0 : 0.0);
}

/**
 * Attach the engine to a host-owned native render surface instead of
 * creating its own window (PerryTS/perry#5519). `handle` is the
 * platform's native view / window / surface pointer — e.g. the `NSView*`
 * / `UIView*` / `GtkWidget*` / `ANativeWindow*` / `HWND` returned by
 * Perry UI's `bloomViewGetNativeHandle`. `width`/`height` are the host
 * view's size in logical points. On success the host owns the run loop
 * and drives frames with `beginDrawing()` / `endDrawing()` as usual.
 *
 * Returns `true` if the engine attached and built its surface, `false`
 * on a null/invalid handle or if surface bring-up failed. Idempotent: a
 * second call once attached is a no-op that returns `true`.
 *
 * The platform-named aliases below forward to this same entry point; use
 * whichever reads clearest for the target you're building.
 */
export function attachToNativeView(handle: number, width: number, height: number): boolean {
  return bloom_attach_native(handle, width, height) !== 0;
}

/** macOS — attach to a host `NSView*`. See {@link attachToNativeView}. */
export function attachToNSView(view: number, width: number, height: number): boolean {
  return bloom_attach_native(view, width, height) !== 0;
}

/** iOS / tvOS / visionOS — attach to a host `UIView*`. See {@link attachToNativeView}. */
export function attachToUIView(view: number, width: number, height: number): boolean {
  return bloom_attach_native(view, width, height) !== 0;
}

/**
 * Linux/GTK4 (`GtkWidget*`), Android (`ANativeWindow*`), Windows (`HWND`)
 * — attach to a host surface handle. See {@link attachToNativeView}.
 */
export function attachToSurface(handle: number, width: number, height: number): boolean {
  return bloom_attach_native(handle, width, height) !== 0;
}

export function closeWindow(): void {
  bloom_close_window();
}

/**
 * Embed Bloom inside a host-provided native window — e.g. a Perry UI
 * `BloomView` widget. Pass the window handle from `bloomViewGetHwnd(view)`
 * and the logical viewport size. Bloom builds its render surface on that
 * window and subclasses it for resize/input; the host owns the message loop,
 * so drive frames yourself with `beginDrawing()` / `update` / `endDrawing()`
 * (do NOT call `runGame`, which blocks). Call once, after the host window is
 * shown and laid out (e.g. on the first `onFrame` tick).
 */
export function attachToHwnd(hwnd: number, width: number, height: number): void {
  bloom_attach_hwnd(hwnd, width, height);
}

/** Resize the embedded surface explicitly (physical + logical pixels). */
export function resize(physW: number, physH: number, logW: number, logH: number): void {
  bloom_resize(physW, physH, logW, logH);
}

export function windowShouldClose(): boolean {
  return bloom_window_should_close() !== 0;
}

// Drawing lifecycle

export function beginDrawing(): void {
  bloom_begin_drawing();
}

export function endDrawing(): void {
  bloom_end_drawing();
}

/**
 * Capture the next rendered frame and write it as a PNG to `path`.
 * The actual capture happens during the next `endDrawing()` call —
 * call this immediately before that endDrawing(), and the file will
 * be on disk afterwards.
 *
 * Used by `bloom-diff` and CI image regression workflows.
 */
export function takeScreenshot(path: string): void {
  bloom_take_screenshot(path as any);
}

export function clearBackground(color: Color): void {
  bloom_clear_background(color.r, color.g, color.b, color.a);
}

/**
 * Set the clear color from the average luminance-weighted color of
 * an HDR environment map (.hdr / Radiance format). A stand-in for
 * proper equirect-sky-pass rendering until that lands — lets us
 * immediately close most of the background-color gap between Bloom's
 * realtime output and the path-traced reference.
 */
export function setEnvClearFromHdr(path: string): void {
  bloom_set_env_clear_from_hdr(path as any);
}

// ---- Post-FX knobs ----
// All default to off. Calling these turns the corresponding
// composite-pass / TAA-pass effect on for the rest of the run
// (or until called again with 0 / disabled values).

/** Height-based exponential fog. Density 0 = off. */
export function setFog(r: number, g: number, b: number, density: number, heightRef: number, heightFalloff: number): void {
  bloom_set_fog(r, g, b, density, heightRef, heightFalloff);
}

/** Radial RGB-channel split at the screen edges. 0 = off. */
export function setChromaticAberration(strength: number): void {
  bloom_set_chromatic_aberration(strength);
}

/** Smooth radial darkening of the corners. strength 0..1, softness 0..1. */
export function setVignette(strength: number, softness: number): void {
  bloom_set_vignette(strength, softness);
}

/** Animated film grain post-tonemap. 0 = off. */
export function setFilmGrain(strength: number): void {
  bloom_set_film_grain(strength);
}

/**
 * Composite unsharp-mask strength. Engine default 0.8; 0 disables the
 * sharpen taps entirely. At high output resolutions the default visibly
 * halos high-contrast silhouettes — tune per game.
 */
export function setSharpenStrength(strength: number): void {
  bloom_set_sharpen_strength(strength);
}

/** Screen-space sun shafts (god rays). strength 0 = off. */
export function setSunShafts(strength: number, decay: number, r: number, g: number, b: number): void {
  bloom_set_sun_shafts(strength, decay, r, g, b);
}

/** Toggle physically-based auto-exposure. 18% gray target, log-average metered. */
export function setAutoExposure(on: boolean): void {
  bloom_set_auto_exposure(on ? 1 : 0);
}

/** Toggle temporal anti-aliasing (sub-pixel jitter + reprojected history blend). */
/**
 * Hi-Z occlusion culling: scene nodes provably hidden behind other
 * geometry (per last frame's depth) are skipped in the main camera
 * pass. Conservative with one frame of latency; on by default. This is
 * the kill switch for debugging or for scenes that pathologically
 * defeat it (e.g. every object visible every frame).
 */
export function setOcclusionCulling(on: boolean): void {
  bloom_set_occlusion_culling(on ? 1 : 0);
}

export function setTaaEnabled(on: boolean): void {
  bloom_set_taa_enabled(on ? 1 : 0);
}

/**
 * Render-resolution multiplier. 0.5 = quarter-pixel shading (cheap, soft);
 * 1.0 = native (sharp, expensive). Clamped to [0.5, 1.0]. Once called
 * explicitly, the choice sticks across `setTaaEnabled` toggles instead of
 * being overridden by the legacy 0.5↔1.0 coupling.
 *
 * On a 4K display: 0.75 hits a quality/perf sweet spot for 3D scenes.
 * Catmull-Rom is the default upscale filter (see `setUpscaleMode`).
 */
export function setRenderScale(scale: number): void {
  bloom_set_render_scale(Math.min(1.0, Math.max(0.5, scale)));
}
export function getRenderScale(): number { return bloom_get_render_scale(); }

/** Upscale filter when render_scale < 1 and TAA is off. "bilinear" = cheap/soft, "catmull-rom" = sharper (default). */
export type UpscaleMode = "bilinear" | "catmull-rom";
export function setUpscaleMode(mode: UpscaleMode): void {
  bloom_set_upscale_mode(mode === "catmull-rom" ? 1 : 0);
}

/**
 * Contrast-adaptive sharpen strength. 0 = off (default, pass skipped);
 * 0.3 subtle; 0.6 punchy; 1.0 max. Useful at any render_scale — pairs
 * particularly well with TAA-softened native or Catmull-Rom upscale.
 */
export function setCasStrength(strength: number): void {
  bloom_set_cas_strength(strength);
}

/** Physical-pixel size of the GPU surface (HiDPI-aware on macOS today). */
export function getPhysicalWidth(): number { return bloom_get_physical_width(); }
export function getPhysicalHeight(): number { return bloom_get_physical_height(); }

/**
 * Dynamic resolution scaling. When enabled, the engine self-tunes
 * `render_scale` toward the given target framerate using a 6-rung
 * ladder (0.50–1.00) with EMA-smoothed frame time, asymmetric
 * hysteresis, and a 30-frame cooldown between steps.
 *
 * Call with `enabled = false` to disarm. Manual `setRenderScale`
 * still works while DRS is on (DRS will simply step away from the
 * value on its next eligible frame).
 */
export function setAutoResolution(targetHz: number, enabled: boolean = true): void {
  bloom_set_auto_resolution(targetHz, enabled ? 1 : 0);
}

/** Manual exposure multiplier (ignored when auto-exposure is on). 1.0 = default. */
export function setManualExposure(value: number): void {
  bloom_set_manual_exposure(value);
}

/** Env-map intensity multiplier for IBL + sky pass. 1.0 = reference, 0.2–0.5 typical for bright outdoor HDRs. */
export function setEnvIntensity(intensity: number): void {
  bloom_set_env_intensity(intensity);
}

/** Toggle screen-space global illumination (single-bounce indirect diffuse). Default on. */
export function setSsgiEnabled(on: boolean): void {
  bloom_set_ssgi_enabled(on ? 1 : 0);
}

/** SSGI intensity multiplier. 0 = off, 0.5 = default, 1+ = strong. */
export function setSsgiIntensity(intensity: number): void {
  bloom_set_ssgi_intensity(intensity);
}

/** SSGI max view-space march distance in meters. Default 20. Tune to scene scale. */
export function setSsgiRadius(radius: number): void {
  bloom_set_ssgi_radius(radius);
}

/** Depth of field. focusDistance = view-space distance in world units. aperture = blur strength (0 = off, 0.03 = subtle, 0.1 = heavy). */
export function setDepthOfField(focusDistance: number, aperture: number): void {
  bloom_set_dof(aperture > 0 ? 1 : 0, focusDistance, aperture);
}

// ============================================================
// Render quality — let games pick a preset for the target device
// or toggle individual effects. Presets apply a known-good set of
// flags; individual setters can override afterward.
// ============================================================

export enum QualityPreset {
  /** Bare minimum — no shadows, no SSAO, no bloom, no TAA, no SSR/SSGI/DoF/MB/SSS. */
  Off = 0,
  /** Base pipeline only: HDR tonemap + bloom. No shadows/SSAO/TAA. */
  Low = 1,
  /** Balanced default: shadows + SSAO + bloom + TAA. No SSR/SSGI/cinematic FX. */
  Medium = 2,
  /** + SSR, SSGI, subtle chromatic aberration. */
  High = 3,
  /** Everything on (plus DoF if aperture > 0). */
  Ultra = 4,
}

/** Apply a quality preset in one call. Call individual setters after for fine-tuning. */
export function setQualityPreset(preset: QualityPreset): void {
  bloom_set_quality_preset(preset);
}

/** Toggle cascaded shadow maps. Default on. Disable on low-end GPUs — biggest single win. */
export function setShadowsEnabled(on: boolean): void {
  bloom_set_shadows_enabled(on ? 1 : 0);
}

/**
 * Force cascaded shadow maps to re-render every frame, bypassing the
 * static-caster cache (ticket 004). Default off. Turn on for games
 * with continuously-changing light state (day/night cycles set from
 * native code, heavily-deformable casters) where the cache hit rate
 * would be ~zero anyway, so skipping the check saves a few µs.
 */
export function setShadowsAlwaysFresh(on: boolean): void {
  bloom_set_shadows_always_fresh(on ? 1 : 0);
}

/** Toggle the bloom down/upsample chain (~10 passes). Default on. */
export function setBloomEnabled(on: boolean): void {
  bloom_set_bloom_enabled(on ? 1 : 0);
}

/**
 * Bloom contribution strength added to the HDR scene before tonemap.
 * 0 = none, ~0.04 subtle default, higher = stronger glow around bright pixels.
 */
export function setBloomIntensity(intensity: number): void {
  bloom_set_bloom_intensity(intensity);
}

/** Tonemap operator selection. */
export enum Tonemap {
  /** Filmic ACES (default). */
  ACES = 0,
  /** AgX — more filmic highlight roll-off + a punchier, better-saturated look. */
  AgX = 1,
}

/** Choose the tonemap operator applied in the composite pass. */
export function setTonemap(kind: Tonemap): void {
  bloom_set_tonemap(kind);
}

/**
 * Auto-exposure target (scene-average luma key). Lower aims for a darker,
 * more saturated midpoint (counteracts wash-out from very bright skies);
 * higher aims brighter. Only affects frames where auto-exposure is on.
 */
export function setAutoExposureKey(key: number): void {
  bloom_set_auto_exposure_key(key);
}

/** Auto-exposure adaptation rate per frame (0 = frozen, ~0.05 smooth, 1 = instant). */
export function setAutoExposureRate(rate: number): void {
  bloom_set_auto_exposure_rate(rate);
}

/** Toggle screen-space ambient occlusion + its bilateral blur. Default on. */
export function setSsaoEnabled(on: boolean): void {
  bloom_set_ssao_enabled(on ? 1 : 0);
}

/** SSAO strength. 0 disables corner darkening, 1 is default, 2 is heavy. */
export function setSsaoIntensity(intensity: number): void {
  bloom_set_ssao_intensity(intensity);
}

/** SSAO sampling radius in world units. 0.1..2.0 m is the sane range. */
export function setSsaoRadius(worldRadius: number): void {
  bloom_set_ssao_radius(worldRadius);
}

/// EN-017 — install a game-supplied fullscreen WGSL post-pass.
/// Runs after composite + tonemapping, before the 2D overlay, so
/// the HUD stays crisp. The fragment shader sees scene_color_tex
/// (LDR, post-tonemap) and scene_depth_tex at @group(0).
///
/// Example — underwater tint:
///   setPostPass(`
///     @fragment fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
///       let scene = textureSample(scene_color_tex, scene_color_samp, uv);
///       return vec4<f32>(scene.rgb * vec3<f32>(0.4, 0.7, 0.9), 1.0);
///     }
///   `);
///
/// Returns true on successful compile, false on shader error.
/// In V2 this is shorthand for `clearAllPostPasses()` followed by
/// `addPostPass(wgsl)` — the existing stack is wiped before the new
/// pass is installed, matching V1 single-slot semantics.
export function setPostPass(wgslSource: string): boolean {
  return bloom_set_post_pass(wgslSource as any) > 0;
}

/// EN-017 — uninstall the active post-pass. The composite output
/// goes directly to the swapchain again (zero post-pass cost).
/// V2 alias for `clearAllPostPasses()`.
export function clearPostPass(): void {
  bloom_clear_post_pass();
}

/// EN-017 V2 — append a fullscreen WGSL post-pass to the stack.
/// Each pass samples the previous pass's output (or scene_color_tex
/// for the first pass) and writes either to the next intermediate
/// (if more passes follow) or to the swapchain (if last).
///
/// Stack order matters: addPostPass(A); addPostPass(B); means A
/// runs first, then B sees A's output. Compose effects (e.g.
/// underwater tint, then damage flash, then scope vignette) in the
/// order they should apply.
///
/// Returns the 0-based index of the newly added pass on success,
/// or -1 if the shader failed to compile (existing stack untouched).
export function addPostPass(wgslSource: string): number {
  const r = bloom_add_post_pass(wgslSource as any);
  return r > 0 ? (r - 1) : -1;
}

/// EN-017 V2 — wipe the entire post-pass stack. The composite
/// output goes directly to the swapchain again (zero post-pass cost).
export function clearAllPostPasses(): void {
  bloom_clear_all_post_passes();
}

/// Set the global wind field used by foliage materials.
/// dirX/dirZ define the wind direction in the XZ plane (need not be
/// normalised; magnitude scales effective amplitude).
/// amplitude is the displacement scale (~0.1 m typical for grass).
/// frequency is in Hz (~1.0 typical).
export function setWind(dirX: number, dirZ: number, amplitude: number, frequency: number): void {
  bloom_set_wind(dirX, dirZ, amplitude, frequency);
}

/** Toggle screen-space reflections. Default on. */
export function setSsrEnabled(on: boolean): void {
  bloom_set_ssr_enabled(on ? 1 : 0);
}

/** Toggle per-object motion blur. Default off. */
export function setMotionBlurEnabled(on: boolean): void {
  bloom_set_motion_blur_enabled(on ? 1 : 0);
}

/** Toggle subsurface scattering (for skin/wax materials). Default off. */
export function setSssEnabled(on: boolean): void {
  bloom_set_sss_enabled(on ? 1 : 0);
}

// ============================================================
// Profiler — measure CPU phase timings and (on supported GPUs)
// per-pass GPU times. Off by default; enable at runtime.
// ============================================================

/** Enable/disable the frame profiler. When off, it has zero per-frame cost. */
export function setProfilerEnabled(on: boolean): void {
  bloom_set_profiler_enabled(on ? 1 : 0);
}

/** Average total CPU frame time (sum of all phases) over the rolling window, in microseconds. */
export function getProfilerFrameCpuUs(): number {
  return bloom_get_profiler_frame_cpu_us();
}

/** Average total GPU frame time over the rolling window, in microseconds. 0 if GPU timing unavailable. */
export function getProfilerFrameGpuUs(): number {
  return bloom_get_profiler_frame_gpu_us();
}

/** Print a per-phase CPU/GPU timing table to stdout. Useful for quick diagnostics. */
export function printProfilerSummary(): void {
  bloom_print_profiler_summary();
}

/**
 * Phase 7 — submit a world-space impulse splat. Per-frame compute
 * accumulates + decays into a 256×256 top-down field covering a 128m
 * centred square. Refractive/translucent materials sampling
 * `impulse_tex` (group 4 binding 4) see the result. Up to 16 splats
 * per frame; excess is dropped. Typical uses:
 *   - Footsteps in mud / wet pavement
 *   - Splashes when the player enters water
 *   - Explosion rings, impact ripples
 */
export function splatImpulse(x: number, z: number, radius: number, strength: number): void {
  bloom_splat_impulse(x, z, radius, strength);
}

/**
 * Phase 5 — set per-material `user_params` (ABI §1.4). Bytes are
 * uploaded to `@group(2) @binding(11)` for the next dispatch of the
 * given material. The shader casts them to whatever struct it
 * declared. Up to 64 floats (256-byte ABI cap).
 *
 * Pass an empty array to revert to the default zero-initialised UBO.
 *
 * Example: a water material with a dynamic tint + wave amplitude
 *   `setMaterialParams(matWater, [0.10, 0.30, 0.40, 1.0,  0.20])`
 * lets game code change colour per-zone without recompiling WGSL.
 */
export function setMaterialParams(handle: number, params: number[]): void {
  // Perry 0.5.x rejects JS arrays passed into pointer params, so the floats
  // go through the all-f64 mesh scratch (same fix as createMesh /
  // createInstanceBuffer). ≤ 64 floats per the 256-byte UBO cap, so the
  // per-float FFI cost is negligible.
  bloom_mesh_scratch_reset();
  for (let i = 0; i < params.length; i++) bloom_mesh_scratch_push_f32(params[i]);
  bloom_set_material_params_scratch(handle, params.length);
}

/**
 * Per-pass timings, sorted descending by CPU time. Each row:
 *   `{ label, cpuUs, gpuUs }` (gpuUs = -1 when the device has no
 *   TIMESTAMP_QUERY feature).
 * Intended for an in-game overlay — games call it at draw time and
 * render one `drawText` per entry.
 */
export function getProfilerOverlay(): { label: string, cpuUs: number, gpuUs: number }[] {
  const raw = bloom_profiler_overlay_text();
  if (!raw || raw.length === 0) return [];
  const out: { label: string, cpuUs: number, gpuUs: number }[] = [];
  const lines = raw.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.length === 0) continue;
    const parts = line.split('|');
    if (parts.length < 3) continue;
    out.push({
      label: parts[0],
      cpuUs: parseFloat(parts[1]),
      gpuUs: parseFloat(parts[2]),
    });
  }
  return out;
}

/**
 * Phase 8 — last ~120 frames' CPU + GPU totals, oldest first.
 * Useful for an overlay bar-chart of frame-time variance. GPU time
 * is 0 when the device lacks TIMESTAMP_QUERY.
 */
export function getProfilerFrameHistory(): { cpuUs: number, gpuUs: number }[] {
  const raw = bloom_profiler_frame_history();
  if (!raw || raw.length === 0) return [];
  const out: { cpuUs: number, gpuUs: number }[] = [];
  const lines = raw.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.length === 0) continue;
    const parts = line.split('|');
    if (parts.length < 2) continue;
    out.push({ cpuUs: parseFloat(parts[0]), gpuUs: parseFloat(parts[1]) });
  }
  return out;
}

// Timing

export function setTargetFPS(fps: number): void {
  bloom_set_target_fps(fps);
}

/**
 * Enable direct-to-swapchain 2D mode. Skips scene prep, shadow maps,
 * HDR/tonemap, SSAO, bloom, SDF/WSRC bakes and mesh-card passes —
 * rendering goes straight through the batched 2D pipeline. Intended
 * for pure-2D games that never populate the scene graph; on mobile
 * GPUs this is the difference between ~15 fps and 60 fps.
 *
 * Call once after initWindow(). Off by default.
 */
export function setDirect2DMode(on: boolean): void {
  bloom_set_direct_2d_mode(on ? 1.0 : 0.0);
}

export function getDeltaTime(): number {
  return bloom_get_delta_time();
}

export function getFPS(): number {
  return bloom_get_fps();
}

export function getTime(): number {
  return bloom_get_time();
}

// Screen

export function getScreenWidth(): number {
  return bloom_get_screen_width();
}

export function getScreenHeight(): number {
  return bloom_get_screen_height();
}

// Keyboard input

export function isKeyPressed(key: number): boolean {
  return bloom_is_key_pressed(key) !== 0;
}

export function isKeyDown(key: number): boolean {
  return bloom_is_key_down(key) !== 0;
}

export function isKeyReleased(key: number): boolean {
  return bloom_is_key_released(key) !== 0;
}

// Mouse input

export function getMouseX(): number {
  return bloom_get_mouse_x();
}

export function getMouseY(): number {
  return bloom_get_mouse_y();
}

export function isMouseButtonPressed(button: number): boolean {
  return bloom_is_mouse_button_pressed(button) !== 0;
}

export function isMouseButtonDown(button: number): boolean {
  return bloom_is_mouse_button_down(button) !== 0;
}

export function isMouseButtonReleased(button: number): boolean {
  return bloom_is_mouse_button_released(button) !== 0;
}

// Convenience wrappers

export function getMousePosition(): { x: number; y: number } {
  return { x: bloom_get_mouse_x(), y: bloom_get_mouse_y() };
}

export function getTouchPosition(index: number): { x: number; y: number } {
  return { x: bloom_get_touch_x(index), y: bloom_get_touch_y(index) };
}

// Camera 2D

export function beginMode2D(camera: Camera2D): void {
  bloom_begin_mode_2d(camera.offset.x, camera.offset.y, camera.target.x, camera.target.y, camera.rotation, camera.zoom);
}

// Raw variant: takes primitives directly. Workaround for aarch64 Android
// Perry miscompilation where obj.field reads feeding f64 FFI args arrive as NaN.
export function beginMode2DRaw(offsetX: number, offsetY: number, targetX: number, targetY: number, rotation: number, zoom: number): void {
  bloom_begin_mode_2d(offsetX, offsetY, targetX, targetY, rotation, zoom);
}

export function endMode2D(): void {
  bloom_end_mode_2d();
}

// Camera 3D

export function beginMode3D(camera: Camera3D): void {
  const proj = camera.projection === "orthographic" ? 1 : 0;
  bloom_begin_mode_3d(
    camera.position.x, camera.position.y, camera.position.z,
    camera.target.x, camera.target.y, camera.target.z,
    camera.up.x, camera.up.y, camera.up.z,
    camera.fovy, proj,
  );
}

export function endMode3D(): void {
  bloom_end_mode_3d();
}

// Gamepad — spec-compliant signatures with gamepad ID

export function isGamepadAvailable(id?: number): boolean {
  return bloom_is_gamepad_available() !== 0;
}

export function getGamepadAxisValue(id: number, axis: number): number {
  return bloom_get_gamepad_axis(axis);
}

export function getGamepadAxis(axis: number): number {
  return bloom_get_gamepad_axis(axis);
}

export function isGamepadButtonPressed(button: number): boolean {
  return bloom_is_gamepad_button_pressed(button) !== 0;
}

export function isGamepadButtonDown(button: number): boolean {
  return bloom_is_gamepad_button_down(button) !== 0;
}

export function isGamepadButtonReleased(button: number): boolean {
  return bloom_is_gamepad_button_released(button) !== 0;
}

export function getGamepadAxisCount(): number {
  return bloom_get_gamepad_axis_count();
}

// Touch

export function getTouchX(index: number): number {
  return bloom_get_touch_x(index);
}

export function getTouchY(index: number): number {
  return bloom_get_touch_y(index);
}

export function getTouchCount(): number {
  return bloom_get_touch_count();
}

export function getTouchPointCount(): number {
  return bloom_get_touch_count();
}

// Utility

export function toggleFullscreen(): void {
  bloom_toggle_fullscreen();
}

export function setWindowTitle(title: string): void {
  bloom_set_window_title(title as any);
}

export function setWindowIcon(path: string): void {
  bloom_set_window_icon(path as any);
}

export function disableCursor(): void {
  bloom_disable_cursor();
}

export function enableCursor(): void {
  bloom_enable_cursor();
}

export function getMouseDeltaX(): number {
  return bloom_get_mouse_delta_x();
}

export function getMouseDeltaY(): number {
  return bloom_get_mouse_delta_y();
}

/**
 * Accumulated vertical scroll-wheel delta since the last call to this
 * function. Positive values mean scrolling up (away from user on macOS);
 * use this for camera zoom and scrollable UI panels. Reading consumes
 * the value, so call it exactly once per frame.
 */
export function getMouseWheel(): number {
  return bloom_get_mouse_wheel();
}

/**
 * Dequeue the next typed character as a Unicode codepoint. Returns 0 when
 * the queue is empty. Call in a loop each frame to consume all pending
 * characters:
 *
 *   let c = getCharPressed();
 *   while (c !== 0) {
 *     // handle character c
 *     c = getCharPressed();
 *   }
 *
 * Printable characters (codepoint >= 32) plus backspace (8), return (13),
 * and tab (9) are enqueued. Platform-specific text input methods (NSEvent
 * characters on macOS, WM_CHAR on Windows, etc.) feed this queue.
 */
export function getCharPressed(): number {
  return bloom_get_char_pressed();
}

/**
 * Set the mouse cursor shape. Values:
 *   0 = default (arrow), 1 = hand, 2 = move, 3 = text (I-beam),
 *   4 = resize horizontal, 5 = resize vertical, 6 = crosshair.
 * Applied per-frame by the platform event loop.
 */
export const CursorShape = { Default: 0, Hand: 1, Move: 2, Text: 3, ResizeH: 4, ResizeV: 5, Crosshair: 6 } as const;

export function setCursorShape(shape: number): void {
  bloom_set_cursor_shape(shape);
}

/**
 * Copy text to the system clipboard.
 */
export function setClipboardText(text: string): void {
  bloom_set_clipboard_text(text as any);
}

/**
 * Read text from the system clipboard. Returns empty string on failure.
 */
export function getClipboardText(): string {
  return bloom_get_clipboard_text() as any;
}

/**
 * Open a native file-open dialog. Returns the selected file path, or
 * empty string if the user cancelled. `filter` is a file extension
 * (e.g. "world.json") or empty for all files.
 */
export function openFileDialog(filter: string, title: string): string {
  return bloom_open_file_dialog(filter as any, title as any) as any;
}

/**
 * Open a native file-save dialog. Returns the chosen save path, or
 * empty string if cancelled.
 */
export function saveFileDialog(defaultName: string, title: string): string {
  return bloom_save_file_dialog(defaultName as any, title as any) as any;
}

// File I/O

export function writeFile(path: string, data: string): boolean {
  return bloom_write_file(path as any, data as any) !== 0.0;
}

export function fileExists(path: string): boolean {
  return bloom_file_exists(path as any) !== 0.0;
}

export function readFile(path: string): string {
  return bloom_read_file(path as any) as any;
}

// Input injection

export function injectKeyDown(key: number): void { bloom_inject_key_down(key); }
export function injectKeyUp(key: number): void { bloom_inject_key_up(key); }
export function injectGamepadAxis(axis: number, value: number): void { bloom_inject_gamepad_axis(axis, value); }
export function injectGamepadButtonDown(button: number): void { bloom_inject_gamepad_button_down(button); }
export function injectGamepadButtonUp(button: number): void { bloom_inject_gamepad_button_up(button); }

// Platform detection

export const Platform = { UNKNOWN: 0, MACOS: 1, IOS: 2, WINDOWS: 3, LINUX: 4, ANDROID: 5, TVOS: 6, WEB: 7, WATCHOS: 8, VISIONOS: 9 } as const;

export function getPlatform(): number { return bloom_get_platform(); }

/// User's preferred OS language as a packed 2-letter code (`c0 * 256 + c1`,
/// ASCII of the lowercased ISO-639 primary subtag, e.g. "en" = 101*256+110).
/// Script subtags are dropped (zh-Hans -> "zh"); callers map to their variant.
export function getLanguage(): number { return bloom_get_language(); }

export function isMobile(): boolean {
  const p = bloom_get_platform();
  return p === 2 || p === 5;
}

export function isTV(): boolean {
  return bloom_get_platform() === 6;
}

export function isWatch(): boolean {
  return bloom_get_platform() === 8;
}

/**
 * Digital Crown rotation accumulated since the last call, in radians.
 * Positive values = clockwise (scrolling away from the user).
 * Returns 0 on platforms without a crown. Reading consumes the accumulator.
 */
export function getCrownRotation(): number {
  return bloom_get_crown_rotation();
}

export function isAnyInputPressed(): boolean {
  return bloom_is_any_input_pressed() !== 0;
}

/**
 * Cross-platform game loop entry point (Emscripten-style).
 *
 * On native: blocks in a while loop calling beginDrawing/update/endDrawing each frame.
 * On web: passes the callback to the JS runtime which drives it via requestAnimationFrame.
 *
 * Usage:
 *   initWindow(800, 600, "My Game");
 *   runGame((dt) => {
 *     clearBackground({ r: 0, g: 0, b: 0, a: 255 });
 *     // game logic + draw calls
 *   });
 */
export function runGame(update: (dt: number) => void): void {
  const platform = bloom_get_platform();
  if (platform === 7) {
    // Web: delegate to JS glue layer via FFI.
    // bloom_glue.js intercepts this call and sets up requestAnimationFrame.
    bloom_run_game(update as any);
  } else {
    // Native: blocking game loop
    while (!windowShouldClose()) {
      beginDrawing();
      update(getDeltaTime());
      endDrawing();
    }
  }
}

// Pure TS camera helpers

export function getScreenToWorld2D(position: { x: number; y: number }, camera: Camera2D): { x: number; y: number } {
  const cos = Math.cos(camera.rotation * Math.PI / 180);
  const sin = Math.sin(camera.rotation * Math.PI / 180);
  const dx = (position.x - camera.offset.x) / camera.zoom;
  const dy = (position.y - camera.offset.y) / camera.zoom;
  return {
    x: cos * dx + sin * dy + camera.target.x,
    y: -sin * dx + cos * dy + camera.target.y,
  };
}

export function getWorldToScreen2D(position: { x: number; y: number }, camera: Camera2D): { x: number; y: number } {
  const cos = Math.cos(camera.rotation * Math.PI / 180);
  const sin = Math.sin(camera.rotation * Math.PI / 180);
  const dx = position.x - camera.target.x;
  const dy = position.y - camera.target.y;
  return {
    x: (cos * dx - sin * dy) * camera.zoom + camera.offset.x,
    y: (sin * dx + cos * dy) * camera.zoom + camera.offset.y,
  };
}
