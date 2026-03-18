import { Color, Font, Vec2 } from '../core/types';

// FFI declarations
declare function bloom_draw_text(text: number, x: number, y: number, size: number, r: number, g: number, b: number, a: number): void;
declare function bloom_measure_text(text: number, size: number): number;
declare function bloom_load_font(path: number, size: number): number;
declare function bloom_draw_text_ex(font: number, text: number, x: number, y: number, size: number, spacing: number, r: number, g: number, b: number, a: number): void;
declare function bloom_measure_text_ex(font: number, text: number, size: number, spacing: number): number;
declare function bloom_unload_font(font: number): void;

export function drawText(text: string, x: number, y: number, size: number, color: Color): void {
  bloom_draw_text(text as any, x, y, size, color.r, color.g, color.b, color.a);
}

export function measureText(text: string, size: number): number {
  return bloom_measure_text(text as any, size);
}

export function loadFont(path: string, size: number): Font {
  const handle = bloom_load_font(path as any, size);
  return { handle, size };
}

export function loadFontEx(path: string, size: number): Font {
  const handle = bloom_load_font(path as any, size);
  return { handle, size };
}

export function unloadFont(font: Font): void {
  bloom_unload_font(font.handle);
}

export function drawTextEx(font: Font, text: string, pos: Vec2, size: number, spacing: number, color: Color): void {
  bloom_draw_text_ex(font.handle, text as any, pos.x, pos.y, size, spacing, color.r, color.g, color.b, color.a);
}

export function measureTextEx(font: Font, text: string, size: number, spacing: number): Vec2 {
  const width = bloom_measure_text_ex(font.handle, text as any, size, spacing);
  return { x: width, y: size };
}
