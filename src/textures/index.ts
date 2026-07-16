import { spawn, parallelMap } from 'perry/thread';
import { Color, Rect, Texture } from '../core/types';

// FFI declarations
declare function bloom_load_texture(path: number): number;
declare function bloom_unload_texture(handle: number): void;
declare function bloom_draw_texture(handle: number, x: number, y: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_texture_rec(handle: number, sx: number, sy: number, sw: number, sh: number, dx: number, dy: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_texture_pro(handle: number, sx: number, sy: number, sw: number, sh: number, dx: number, dy: number, dw: number, dh: number, ox: number, oy: number, rot: number, r: number, g: number, b: number, a: number): void;
declare function bloom_get_texture_width(handle: number): number;
declare function bloom_get_texture_height(handle: number): number;
declare function bloom_load_image(path: number): number;
declare function bloom_image_resize(handle: number, w: number, h: number): void;
declare function bloom_image_crop(handle: number, x: number, y: number, w: number, h: number): void;
declare function bloom_image_flip_h(handle: number): void;
declare function bloom_image_flip_v(handle: number): void;
declare function bloom_load_texture_from_image(handle: number): number;
declare function bloom_gen_texture_mipmaps(handle: number): void;
declare function bloom_set_texture_filter(handle: number, mode: number): void;

export const FILTER_LINEAR = 0;
export const FILTER_NEAREST = 1;

export function setTextureFilter(texture: Texture, mode: number): void {
  bloom_set_texture_filter(texture.handle, mode);
}

/**
 * Load a texture from a PNG/JPEG/BMP/TGA file.
 *
 * On failure (missing file, undecodable data) returns a Texture with
 * `handle === 0` and zero dimensions — drawing it is a safe no-op and the
 * engine logs the failure once. Check `tex.handle !== 0` when you need to
 * react to missing assets.
 */
export function loadTexture(path: string): Texture {
  const id = bloom_load_texture(path as any);
  const width = bloom_get_texture_width(id);
  const height = bloom_get_texture_height(id);
  return { handle: id, width, height };
}

export function unloadTexture(texture: Texture): void {
  bloom_unload_texture(texture.handle);
}

export function drawTexture(texture: Texture, x: number, y: number, tint: Color): void {
  bloom_draw_texture(texture.handle, x, y, tint.r, tint.g, tint.b, tint.a);
}

export function drawTextureRec(texture: Texture, source: Rect, position: { x: number; y: number }, tint: Color): void {
  bloom_draw_texture_rec(texture.handle, source.x, source.y, source.width, source.height, position.x, position.y, tint.r, tint.g, tint.b, tint.a);
}

export function drawTexturePro(
  texture: Texture, source: Rect, dest: Rect,
  origin: { x: number; y: number }, rotation: number, tint: Color,
): void {
  bloom_draw_texture_pro(
    texture.handle,
    source.x, source.y, source.width, source.height,
    dest.x, dest.y, dest.width, dest.height,
    origin.x, origin.y, rotation,
    tint.r, tint.g, tint.b, tint.a,
  );
}

// Raw variant: takes primitives directly. Workaround for aarch64 Android
// Perry miscompilation where obj.field reads feeding f64 FFI args arrive as NaN.
export function drawTextureProRaw(
  textureId: number,
  sx: number, sy: number, sw: number, sh: number,
  dx: number, dy: number, dw: number, dh: number,
  ox: number, oy: number, rotation: number,
  r: number, g: number, b: number, a: number,
): void {
  bloom_draw_texture_pro(textureId, sx, sy, sw, sh, dx, dy, dw, dh, ox, oy, rotation, r, g, b, a);
}

export function getTextureWidth(texture: Texture): number {
  return texture.width;
}

export function getTextureHeight(texture: Texture): number {
  return texture.height;
}

export function loadImage(path: string): number {
  return bloom_load_image(path as any);
}

export function imageResize(imageHandle: number, width: number, height: number): void {
  bloom_image_resize(imageHandle, width, height);
}

export function imageCrop(imageHandle: number, x: number, y: number, width: number, height: number): void {
  bloom_image_crop(imageHandle, x, y, width, height);
}

export function imageFlipH(imageHandle: number): void {
  bloom_image_flip_h(imageHandle);
}

export function imageFlipV(imageHandle: number): void {
  bloom_image_flip_v(imageHandle);
}

export function loadTextureFromImage(imageHandle: number): Texture {
  const id = bloom_load_texture_from_image(imageHandle);
  const width = bloom_get_texture_width(id);
  const height = bloom_get_texture_height(id);
  return { handle: id, width, height };
}

export function genTextureMipmaps(texture: Texture): void {
  bloom_gen_texture_mipmaps(texture.handle);
}

// Async / threaded loading

declare function bloom_stage_texture(path: number): number;
declare function bloom_commit_texture(handle: number): number;
declare function bloom_load_render_texture(width: number, height: number): number;
declare function bloom_unload_render_texture(handle: number): void;
declare function bloom_begin_texture_mode(handle: number): void;
declare function bloom_end_texture_mode(): void;
declare function bloom_get_render_texture_texture(handle: number): number;

export async function loadTextureAsync(path: string): Promise<Texture> {
  const stagingHandle = await spawn(() => bloom_stage_texture(path as any));
  const id = bloom_commit_texture(stagingHandle);
  const width = bloom_get_texture_width(id);
  const height = bloom_get_texture_height(id);
  return { handle: id, width, height };
}

export function stageTextures(paths: string[]): number[] {
  return parallelMap(paths, (path: string) => bloom_stage_texture(path as any));
}

export function commitTexture(stagingHandle: number): Texture {
  const id = bloom_commit_texture(stagingHandle);
  const width = bloom_get_texture_width(id);
  const height = bloom_get_texture_height(id);
  return { handle: id, width, height };
}

// ============================================================
// Q1: Offscreen Render Targets
// ============================================================

/**
 * Create an offscreen render texture. Rendering commands between
 * beginTextureMode and endTextureMode draw into this texture instead
 * of the screen. Use getRenderTextureTexture to get a Texture for
 * drawing the result via drawTexture.
 *
 * NOTE: GPU implementation is a stub in this version. The FFI surface
 * is stable; the actual render-to-texture plumbing will land in a
 * focused GPU session.
 */
export function loadRenderTexture(width: number, height: number): number {
  return bloom_load_render_texture(width, height);
}

export function unloadRenderTexture(handle: number): void {
  bloom_unload_render_texture(handle);
}

export function beginTextureMode(handle: number): void {
  bloom_begin_texture_mode(handle);
}

export function endTextureMode(): void {
  bloom_end_texture_mode();
}

export function getRenderTextureTexture(handle: number): Texture {
  // `handle`, not `id` — Texture renamed the field in v0.5 and this function
  // was never migrated. Every draw call reads texture.handle, so the old
  // `{ id }` shape fed `undefined` into a native f64 param the first time
  // anyone drew a render texture (nobody had, until editor thumbnails).
  // Width/height are 0 because the FFI only returns the texture id; callers
  // pass explicit source rects.
  const id = bloom_get_render_texture_texture(handle);
  return { handle: id, width: 0, height: 0 };
}
