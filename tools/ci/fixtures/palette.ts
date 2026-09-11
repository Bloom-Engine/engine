import { Colors, ColorConstants, Color, Key, MouseButton, CursorShape, Platform, getPlatform } from 'bloom/core';
import {
  Colors as RootColors, ColorConstants as RootColorConstants, Key as RootKey,
  MouseButton as RootMouseButton, CursorShape as RootCursorShape, Platform as RootPlatform,
  FILTER_LINEAR, FILTER_NEAREST, TEXTURE_KIND_COLOR, TEXTURE_KIND_NORMAL, TEXTURE_KIND_DATA,
  MATERIAL_TEXTURE_SLOT_BASE_COLOR, MATERIAL_TEXTURE_SLOT_NORMAL, MATERIAL_TEXTURE_SLOT_METALLIC_ROUGHNESS,
  MATERIAL_TEXTURE_SLOT_EMISSIVE, MATERIAL_TEXTURE_SLOT_OCCLUSION,
} from 'bloom';

function row(name: string, upper: Color, mixed: Color, rootUpper: Color, rootMixed: Color): string {
  return '["' + name + '",' + upper.r + ',' + upper.g + ',' + upper.b + ',' + upper.a
    + ',' + mixed.r + ',' + mixed.g + ',' + mixed.b + ',' + mixed.a
    + ',' + rootUpper.r + ',' + rootUpper.g + ',' + rootUpper.b + ',' + rootUpper.a
    + ',' + rootMixed.r + ',' + rootMixed.g + ',' + rootMixed.b + ',' + rootMixed.a
    + ',' + (upper === mixed) + ',' + (upper === rootUpper) + ',' + (upper === rootMixed) + ']';
}

// Read the public barrel, not the implementation module. Both naming styles
// must retain their values and refer to the same mutable color objects.
let result = '{"colors":[';
result = result + row('SNOW', Colors.SNOW, ColorConstants.Snow, RootColors.SNOW, RootColorConstants.Snow);
result = result + ',' + row('WHITE', Colors.WHITE, ColorConstants.White, RootColors.WHITE, RootColorConstants.White);
result = result + ',' + row('BLACK', Colors.BLACK, ColorConstants.Black, RootColors.BLACK, RootColorConstants.Black);
result = result + ',' + row('RED', Colors.RED, ColorConstants.Red, RootColors.RED, RootColorConstants.Red);
result = result + ',' + row('GREEN', Colors.GREEN, ColorConstants.Green, RootColors.GREEN, RootColorConstants.Green);
result = result + ',' + row('BLUE', Colors.BLUE, ColorConstants.Blue, RootColors.BLUE, RootColorConstants.Blue);
result = result + ',' + row('YELLOW', Colors.YELLOW, ColorConstants.Yellow, RootColors.YELLOW, RootColorConstants.Yellow);
result = result + ',' + row('ORANGE', Colors.ORANGE, ColorConstants.Orange, RootColors.ORANGE, RootColorConstants.Orange);
result = result + ',' + row('PINK', Colors.PINK, ColorConstants.Pink, RootColors.PINK, RootColorConstants.Pink);
result = result + ',' + row('PURPLE', Colors.PURPLE, ColorConstants.Purple, RootColors.PURPLE, RootColorConstants.Purple);
result = result + ',' + row('DARKGRAY', Colors.DARKGRAY, ColorConstants.DarkGray, RootColors.DARKGRAY, RootColorConstants.DarkGray);
result = result + ',' + row('LIGHTGRAY', Colors.LIGHTGRAY, ColorConstants.LightGray, RootColors.LIGHTGRAY, RootColorConstants.LightGray);
result = result + ',' + row('GRAY', Colors.GRAY, ColorConstants.Gray, RootColors.GRAY, RootColorConstants.Gray);
result = result + ',' + row('DARKBLUE', Colors.DARKBLUE, ColorConstants.DarkBlue, RootColors.DARKBLUE, RootColorConstants.DarkBlue);
result = result + ',' + row('SKYBLUE', Colors.SKYBLUE, ColorConstants.SkyBlue, RootColors.SKYBLUE, RootColorConstants.SkyBlue);
result = result + ',' + row('LIME', Colors.LIME, ColorConstants.Lime, RootColors.LIME, RootColorConstants.Lime);
result = result + ',' + row('DARKGREEN', Colors.DARKGREEN, ColorConstants.DarkGreen, RootColors.DARKGREEN, RootColorConstants.DarkGreen);
result = result + ',' + row('GOLD', Colors.GOLD, ColorConstants.Gold, RootColors.GOLD, RootColorConstants.Gold);
result = result + ',' + row('MAROON', Colors.MAROON, ColorConstants.Maroon, RootColors.MAROON, RootColorConstants.Maroon);
result = result + ',' + row('BROWN', Colors.BROWN, ColorConstants.Brown, RootColors.BROWN, RootColorConstants.Brown);
result = result + ',' + row('BEIGE', Colors.BEIGE, ColorConstants.Beige, RootColors.BEIGE, RootColorConstants.Beige);
result = result + ',' + row('MAGENTA', Colors.MAGENTA, ColorConstants.Magenta, RootColors.MAGENTA, RootColorConstants.Magenta);
result = result + ',' + row('VIOLET', Colors.VIOLET, ColorConstants.Violet, RootColors.VIOLET, RootColorConstants.Violet);
result = result + ',' + row('BLANK', Colors.BLANK, ColorConstants.Blank, RootColors.BLANK, RootColorConstants.Blank);
function numbers(values: number[]): string {
  let text = '[';
  for (let i = 0; i < values.length; i++) {
    if (i > 0) text = text + ',';
    text = text + values[i];
  }
  return text + ']';
}
const inputs = [
  Key.A, Key.B, Key.C, Key.D, Key.E, Key.F, Key.G, Key.H,
  Key.I, Key.J, Key.K, Key.L, Key.M, Key.N, Key.O, Key.P,
  Key.Q, Key.R, Key.S, Key.T, Key.U, Key.V, Key.W, Key.X,
  Key.Y, Key.Z, Key.ZERO, Key.ONE, Key.TWO, Key.THREE, Key.FOUR, Key.FIVE,
  Key.SIX, Key.SEVEN, Key.EIGHT, Key.NINE, Key.F1, Key.F2, Key.F3, Key.F4,
  Key.F5, Key.F6, Key.F7, Key.F8, Key.F9, Key.F10, Key.F11, Key.F12,
  Key.UP, Key.DOWN, Key.LEFT, Key.RIGHT, Key.SPACE, Key.ENTER, Key.ESCAPE, Key.TAB,
  Key.BACKSPACE, Key.DELETE, Key.INSERT, Key.HOME, Key.END, Key.PAGE_UP, Key.PAGE_DOWN, Key.LEFT_SHIFT,
  Key.RIGHT_SHIFT, Key.LEFT_CONTROL, Key.RIGHT_CONTROL, Key.LEFT_ALT, Key.RIGHT_ALT, Key.LEFT_SUPER, Key.RIGHT_SUPER, Key.APOSTROPHE,
  Key.COMMA, Key.MINUS, Key.PERIOD, Key.SLASH, Key.SEMICOLON, Key.EQUAL, Key.LEFT_BRACKET, Key.BACKSLASH,
  Key.RIGHT_BRACKET, Key.GRAVE, MouseButton.LEFT, MouseButton.RIGHT, MouseButton.MIDDLE, CursorShape.Default, CursorShape.Hand, CursorShape.Move,
  CursorShape.Text, CursorShape.ResizeH, CursorShape.ResizeV, CursorShape.Crosshair, Platform.UNKNOWN, Platform.MACOS, Platform.IOS, Platform.WINDOWS,
  Platform.LINUX, Platform.ANDROID, Platform.TVOS, Platform.WEB, Platform.WATCHOS, Platform.VISIONOS,
];
const rootInputs = [
  RootKey.A, RootKey.B, RootKey.C, RootKey.D, RootKey.E, RootKey.F, RootKey.G, RootKey.H,
  RootKey.I, RootKey.J, RootKey.K, RootKey.L, RootKey.M, RootKey.N, RootKey.O, RootKey.P,
  RootKey.Q, RootKey.R, RootKey.S, RootKey.T, RootKey.U, RootKey.V, RootKey.W, RootKey.X,
  RootKey.Y, RootKey.Z, RootKey.ZERO, RootKey.ONE, RootKey.TWO, RootKey.THREE, RootKey.FOUR, RootKey.FIVE,
  RootKey.SIX, RootKey.SEVEN, RootKey.EIGHT, RootKey.NINE, RootKey.F1, RootKey.F2, RootKey.F3, RootKey.F4,
  RootKey.F5, RootKey.F6, RootKey.F7, RootKey.F8, RootKey.F9, RootKey.F10, RootKey.F11, RootKey.F12,
  RootKey.UP, RootKey.DOWN, RootKey.LEFT, RootKey.RIGHT, RootKey.SPACE, RootKey.ENTER, RootKey.ESCAPE, RootKey.TAB,
  RootKey.BACKSPACE, RootKey.DELETE, RootKey.INSERT, RootKey.HOME, RootKey.END, RootKey.PAGE_UP, RootKey.PAGE_DOWN, RootKey.LEFT_SHIFT,
  RootKey.RIGHT_SHIFT, RootKey.LEFT_CONTROL, RootKey.RIGHT_CONTROL, RootKey.LEFT_ALT, RootKey.RIGHT_ALT, RootKey.LEFT_SUPER, RootKey.RIGHT_SUPER, RootKey.APOSTROPHE,
  RootKey.COMMA, RootKey.MINUS, RootKey.PERIOD, RootKey.SLASH, RootKey.SEMICOLON, RootKey.EQUAL, RootKey.LEFT_BRACKET, RootKey.BACKSLASH,
  RootKey.RIGHT_BRACKET, RootKey.GRAVE, RootMouseButton.LEFT, RootMouseButton.RIGHT, RootMouseButton.MIDDLE, RootCursorShape.Default, RootCursorShape.Hand, RootCursorShape.Move,
  RootCursorShape.Text, RootCursorShape.ResizeH, RootCursorShape.ResizeV, RootCursorShape.Crosshair, RootPlatform.UNKNOWN, RootPlatform.MACOS, RootPlatform.IOS, RootPlatform.WINDOWS,
  RootPlatform.LINUX, RootPlatform.ANDROID, RootPlatform.TVOS, RootPlatform.WEB, RootPlatform.WATCHOS, RootPlatform.VISIONOS,
];
const sharedMaps = Colors === RootColors && ColorConstants === RootColorConstants && Key === RootKey
  && MouseButton === RootMouseButton && CursorShape === RootCursorShape && Platform === RootPlatform;
const rootConstants = [FILTER_LINEAR, FILTER_NEAREST, TEXTURE_KIND_COLOR, TEXTURE_KIND_NORMAL, TEXTURE_KIND_DATA,
  MATERIAL_TEXTURE_SLOT_BASE_COLOR, MATERIAL_TEXTURE_SLOT_NORMAL, MATERIAL_TEXTURE_SLOT_METALLIC_ROUGHNESS,
  MATERIAL_TEXTURE_SLOT_EMISSIVE, MATERIAL_TEXTURE_SLOT_OCCLUSION];
const white = Colors.WHITE;
white.r = 17;
const aliasMutation = ColorConstants.White.r === 17;
white.r = 255;
const restored = Colors.WHITE.r === 255 && ColorConstants.White.r === 255;
result = result + '],"aliasMutation":' + aliasMutation + ',"restored":' + restored
  + ',"inputs":' + numbers(inputs) + ',"rootInputs":' + numbers(rootInputs)
  + ',"sharedMaps":' + sharedMaps + ',"rootConstants":' + numbers(rootConstants) + '}';
console.log('BLOOM_PALETTE_RESULT:' + result);
// The smoke runner appends getPlatform() only to its WASM guard-failure control.
