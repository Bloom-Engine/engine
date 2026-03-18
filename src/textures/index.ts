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

export function loadTexture(path: string): Texture {
  const id = bloom_load_texture(path as any);
  const width = bloom_get_texture_width(id);
  const height = bloom_get_texture_height(id);
  return { id, width, height };
}

export function unloadTexture(texture: Texture): void {
  bloom_unload_texture(texture.id);
}

export function drawTexture(texture: Texture, x: number, y: number, tint: Color): void {
  bloom_draw_texture(texture.id, x, y, tint.r, tint.g, tint.b, tint.a);
}

export function drawTextureRec(texture: Texture, source: Rect, position: { x: number; y: number }, tint: Color): void {
  bloom_draw_texture_rec(texture.id, source.x, source.y, source.width, source.height, position.x, position.y, tint.r, tint.g, tint.b, tint.a);
}

export function drawTexturePro(
  texture: Texture, source: Rect, dest: Rect,
  origin: { x: number; y: number }, rotation: number, tint: Color,
): void {
  bloom_draw_texture_pro(
    texture.id,
    source.x, source.y, source.width, source.height,
    dest.x, dest.y, dest.width, dest.height,
    origin.x, origin.y, rotation,
    tint.r, tint.g, tint.b, tint.a,
  );
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
  return { id, width, height };
}

export function genTextureMipmaps(texture: Texture): void {
  bloom_gen_texture_mipmaps(texture.id);
}
