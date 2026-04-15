import { Color, Camera2D, Camera3D } from './types';

export type { Color, Vec2, Vec3, Vec4, Rect, Camera2D, Camera3D, Texture, Font, Sound, Music, Quat, Ray, BoundingBox, Model, Mat4, RayHit, FrustumPlanes } from './types';
export { ColorConstants, Colors } from './colors';
export { ColorConstants as Color } from './colors';
export { Key, MouseButton } from './keys';

// FFI declarations
declare function bloom_init_window(width: number, height: number, title: number, fullscreen: number): void;
declare function bloom_close_window(): void;
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
declare function bloom_set_sun_shafts(strength: number, decay: number, r: number, g: number, b: number): void;
declare function bloom_set_auto_exposure(on: number): void;
declare function bloom_set_manual_exposure(value: number): void;
declare function bloom_set_env_intensity(intensity: number): void;
declare function bloom_set_target_fps(fps: number): void;
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
declare function bloom_is_any_input_pressed(): number;

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

export function closeWindow(): void {
  bloom_close_window();
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

/** Screen-space sun shafts (god rays). strength 0 = off. */
export function setSunShafts(strength: number, decay: number, r: number, g: number, b: number): void {
  bloom_set_sun_shafts(strength, decay, r, g, b);
}

/** Toggle physically-based auto-exposure. 18% gray target, log-average metered. */
export function setAutoExposure(on: boolean): void {
  bloom_set_auto_exposure(on ? 1 : 0);
}

/** Manual exposure multiplier (ignored when auto-exposure is on). 1.0 = default. */
export function setManualExposure(value: number): void {
  bloom_set_manual_exposure(value);
}

/** Env-map intensity multiplier for IBL + sky pass. 1.0 = reference, 0.2–0.5 typical for bright outdoor HDRs. */
export function setEnvIntensity(intensity: number): void {
  bloom_set_env_intensity(intensity);
}

// Timing

export function setTargetFPS(fps: number): void {
  bloom_set_target_fps(fps);
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

export const Platform = { UNKNOWN: 0, MACOS: 1, IOS: 2, WINDOWS: 3, LINUX: 4, ANDROID: 5, TVOS: 6, WEB: 7 } as const;

export function getPlatform(): number { return bloom_get_platform(); }

export function isMobile(): boolean {
  const p = bloom_get_platform();
  return p === 2 || p === 5;
}

export function isTV(): boolean {
  return bloom_get_platform() === 6;
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
