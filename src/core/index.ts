import { Color, Camera2D, Camera3D } from './types';

export type { Color, Vec2, Vec3, Vec4, Rect, Camera2D, Camera3D, Texture, Font, Sound, Music, Quat, Ray, BoundingBox, Model, Mat4, RayHit, FrustumPlanes } from './types';
export { ColorConstants, Colors } from './colors';
export { ColorConstants as Color } from './colors';
export { Key, MouseButton } from './keys';

// FFI declarations
declare function bloom_init_window(width: number, height: number, title: number): void;
declare function bloom_close_window(): void;
declare function bloom_window_should_close(): number;
declare function bloom_begin_drawing(): void;
declare function bloom_end_drawing(): void;
declare function bloom_clear_background(r: number, g: number, b: number, a: number): void;
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
declare function bloom_write_file(path: number, data: number): number;
declare function bloom_file_exists(path: number): number;
declare function bloom_read_file(path: number): number;

// Window management

export function initWindow(width: number, height: number, title: string): void {
  bloom_init_window(width, height, title as any);
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

export function clearBackground(color: Color): void {
  bloom_clear_background(color.r, color.g, color.b, color.a);
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

// File I/O

export function writeFile(path: string, data: string): boolean {
  return bloom_write_file(path as any, data as any) !== 0;
}

export function fileExists(path: string): boolean {
  return bloom_file_exists(path as any) !== 0;
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

export const Platform = { UNKNOWN: 0, MACOS: 1, IOS: 2, WINDOWS: 3, LINUX: 4, ANDROID: 5, TVOS: 6 } as const;

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
