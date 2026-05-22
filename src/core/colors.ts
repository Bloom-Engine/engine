import { Color as ColorType } from './types';

// Canonical color palette. Re-exported as `Color` from `bloom/core` and
// declared as a real top-level binding (not an alias re-export) so Perry
// emits a `_perry_fn_src_core_colors_ts__Color` symbol that examples
// importing `Color` from `bloom/core` can link against.
export const Color: Record<string, ColorType> = {
  Snow:       { r: 245, g: 245, b: 245, a: 255 },
  White:      { r: 255, g: 255, b: 255, a: 255 },
  Black:      { r: 0,   g: 0,   b: 0,   a: 255 },
  Red:        { r: 230, g: 41,  b: 55,  a: 255 },
  Green:      { r: 0,   g: 228, b: 48,  a: 255 },
  Blue:       { r: 0,   g: 121, b: 241, a: 255 },
  Yellow:     { r: 253, g: 249, b: 0,   a: 255 },
  Orange:     { r: 255, g: 161, b: 0,   a: 255 },
  Pink:       { r: 255, g: 109, b: 194, a: 255 },
  Purple:     { r: 200, g: 122, b: 255, a: 255 },
  DarkGray:   { r: 80,  g: 80,  b: 80,  a: 255 },
  LightGray:  { r: 200, g: 200, b: 200, a: 255 },
  Gray:       { r: 130, g: 130, b: 130, a: 255 },
  DarkBlue:   { r: 0,   g: 82,  b: 172, a: 255 },
  SkyBlue:    { r: 102, g: 191, b: 255, a: 255 },
  Lime:       { r: 0,   g: 158, b: 47,  a: 255 },
  DarkGreen:  { r: 0,   g: 117, b: 44,  a: 255 },
  Gold:       { r: 255, g: 203, b: 0,   a: 255 },
  Maroon:     { r: 190, g: 33,  b: 55,  a: 255 },
  Brown:      { r: 127, g: 106, b: 79,  a: 255 },
  Beige:      { r: 211, g: 176, b: 131, a: 255 },
  Magenta:    { r: 255, g: 0,   b: 255, a: 255 },
  Violet:     { r: 135, g: 60,  b: 190, a: 255 },
  Blank:      { r: 0,   g: 0,   b: 0,   a: 0   },
};

// Backward-compatible alias — same object, kept for older imports.
export const ColorConstants = Color;

// Backward-compatible alias with SCREAMING_SNAKE keys
export const Colors: Record<string, ColorType> = {
  WHITE:      Color.White,
  BLACK:      Color.Black,
  RED:        Color.Red,
  GREEN:      Color.Green,
  BLUE:       Color.Blue,
  YELLOW:     Color.Yellow,
  ORANGE:     Color.Orange,
  PINK:       Color.Pink,
  PURPLE:     Color.Purple,
  DARKGRAY:   Color.DarkGray,
  LIGHTGRAY:  Color.LightGray,
  GRAY:       Color.Gray,
  DARKBLUE:   Color.DarkBlue,
  SKYBLUE:    Color.SkyBlue,
  LIME:       Color.Lime,
  DARKGREEN:  Color.DarkGreen,
  GOLD:       Color.Gold,
  MAROON:     Color.Maroon,
  BROWN:      Color.Brown,
  BEIGE:      Color.Beige,
  MAGENTA:    Color.Magenta,
  VIOLET:     Color.Violet,
  SNOW:       Color.Snow,
  BLANK:      Color.Blank,
};
