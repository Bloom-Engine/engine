// Platform-agnostic key code constants.
// Platform crates map native keycodes to these Bloom key values.
export const Key = {
  // Letters
  A: 65, B: 66, C: 67, D: 68, E: 69, F: 70, G: 71, H: 72,
  I: 73, J: 74, K: 75, L: 76, M: 77, N: 78, O: 79, P: 80,
  Q: 81, R: 82, S: 83, T: 84, U: 85, V: 86, W: 87, X: 88,
  Y: 89, Z: 90,

  // Numbers
  ZERO: 48, ONE: 49, TWO: 50, THREE: 51, FOUR: 52,
  FIVE: 53, SIX: 54, SEVEN: 55, EIGHT: 56, NINE: 57,

  // Function keys
  F1: 112, F2: 113, F3: 114, F4: 115, F5: 116, F6: 117,
  F7: 118, F8: 119, F9: 120, F10: 121, F11: 122, F12: 123,

  // Arrow keys
  UP: 256, DOWN: 257, LEFT: 258, RIGHT: 259,

  // Special keys
  SPACE: 32,
  ENTER: 265,
  ESCAPE: 27,
  TAB: 9,
  BACKSPACE: 8,
  DELETE: 127,
  INSERT: 260,
  HOME: 261,
  END: 262,
  PAGE_UP: 263,
  PAGE_DOWN: 264,

  // Modifiers
  LEFT_SHIFT: 280,
  RIGHT_SHIFT: 281,
  LEFT_CONTROL: 282,
  RIGHT_CONTROL: 283,
  LEFT_ALT: 284,
  RIGHT_ALT: 285,
  LEFT_SUPER: 286,
  RIGHT_SUPER: 287,

  // Punctuation
  APOSTROPHE: 39,
  COMMA: 44,
  MINUS: 45,
  PERIOD: 46,
  SLASH: 47,
  SEMICOLON: 59,
  EQUAL: 61,
  LEFT_BRACKET: 91,
  BACKSLASH: 92,
  RIGHT_BRACKET: 93,
  GRAVE: 96,
};

// Mouse button constants
export const MouseButton = {
  LEFT: 0,
  RIGHT: 1,
  MIDDLE: 2,
};
