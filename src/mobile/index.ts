import { Color } from '../core/types';
import {
  injectKeyDown, injectKeyUp,
  injectGamepadAxis, injectGamepadButtonDown, injectGamepadButtonUp,
  getTouchX, getTouchY, getTouchCount,
  getScreenWidth, getScreenHeight,
  isKeyDown,
  getGamepadAxis,
} from '../core/index';
import { drawCircle, drawCircleLines } from '../shapes/index';
import { drawText, measureText } from '../text/index';
import { Key } from '../core/keys';

// ============================================================
// Types
// ============================================================

export interface VirtualJoystick {
  zone: 'left' | 'right';
  radius: number;
  deadzone: number;
  axisX: number;
  axisY: number;
  // Runtime state
  active: boolean;
  touchIndex: number;
  originX: number;
  originY: number;
  handleX: number;
  handleY: number;
  valueX: number;
  valueY: number;
}

export interface VirtualButton {
  x: number;
  y: number;
  radius: number;
  label: string;
  key: number | null;
  gamepadButton: number | null;
  // Runtime state
  active: boolean;
  touchIndex: number;
}

// ============================================================
// Touch claim system
// ============================================================

const claimedTouches = new Set<number>();

export function resetTouchClaims(): void {
  claimedTouches.clear();
}

// ============================================================
// Virtual Joystick
// ============================================================

export function createVirtualJoystick(opts?: {
  zone?: 'left' | 'right';
  radius?: number;
  deadzone?: number;
  axisX?: number;
  axisY?: number;
}): VirtualJoystick {
  return {
    zone: opts?.zone ?? 'left',
    radius: opts?.radius ?? 60,
    deadzone: opts?.deadzone ?? 0.15,
    axisX: opts?.axisX ?? 0,
    axisY: opts?.axisY ?? 1,
    active: false,
    touchIndex: -1,
    originX: 0,
    originY: 0,
    handleX: 0,
    handleY: 0,
    valueX: 0,
    valueY: 0,
  };
}

export function updateVirtualJoystick(js: VirtualJoystick): void {
  const screenW = getScreenWidth();
  const touchCount = getTouchCount();

  // If already active, check if our finger is still down
  if (js.active) {
    let stillActive = false;
    for (let i = 0; i < touchCount; i++) {
      const tx = getTouchX(i);
      const ty = getTouchY(i);
      if (i === js.touchIndex && tx !== 0 && ty !== 0) {
        stillActive = true;
        break;
      }
    }
    // Also check that the touch index is still within count
    if (js.touchIndex >= touchCount) {
      stillActive = false;
    }
    if (!stillActive) {
      js.active = false;
      js.touchIndex = -1;
      js.valueX = 0;
      js.valueY = 0;
      injectGamepadAxis(js.axisX, 0);
      injectGamepadAxis(js.axisY, 0);
      return;
    }
  }

  // Look for a new touch if not active
  if (!js.active) {
    for (let i = 0; i < touchCount; i++) {
      if (claimedTouches.has(i)) continue;
      const tx = getTouchX(i);
      const ty = getTouchY(i);
      if (tx === 0 && ty === 0) continue;

      // Check if touch is in the correct zone
      const inZone = js.zone === 'left' ? tx < screenW / 2 : tx >= screenW / 2;
      if (!inZone) continue;

      // PUBG-style: joystick origin appears at touch position
      js.active = true;
      js.touchIndex = i;
      js.originX = tx;
      js.originY = ty;
      js.handleX = tx;
      js.handleY = ty;
      claimedTouches.add(i);
      break;
    }
  }

  if (!js.active) return;

  claimedTouches.add(js.touchIndex);

  // Read current touch position
  const tx = getTouchX(js.touchIndex);
  const ty = getTouchY(js.touchIndex);

  const dx = tx - js.originX;
  const dy = ty - js.originY;
  const dist = Math.sqrt(dx * dx + dy * dy);

  if (dist < js.deadzone * js.radius) {
    js.handleX = tx;
    js.handleY = ty;
    js.valueX = 0;
    js.valueY = 0;
  } else {
    const clamped = Math.min(dist, js.radius);
    const norm = clamped / dist;
    js.handleX = js.originX + dx * norm;
    js.handleY = js.originY + dy * norm;
    js.valueX = (dx * norm) / js.radius;
    js.valueY = (dy * norm) / js.radius;
  }

  injectGamepadAxis(js.axisX, js.valueX);
  injectGamepadAxis(js.axisY, js.valueY);
}

export function drawVirtualJoystick(js: VirtualJoystick): void {
  if (!js.active) return;

  const bgColor: Color = { r: 255, g: 255, b: 255, a: 60 };
  const handleColor: Color = { r: 255, g: 255, b: 255, a: 140 };

  drawCircleLines(js.originX, js.originY, js.radius, bgColor);
  drawCircle(js.handleX, js.handleY, js.radius * 0.4, handleColor);
}

// ============================================================
// Virtual Button
// ============================================================

export function createVirtualButton(x: number, y: number, opts?: {
  radius?: number;
  label?: string;
  key?: number | null;
  gamepadButton?: number | null;
}): VirtualButton {
  return {
    x,
    y,
    radius: opts?.radius ?? 30,
    label: opts?.label ?? '',
    key: opts?.key ?? null,
    gamepadButton: opts?.gamepadButton ?? null,
    active: false,
    touchIndex: -1,
  };
}

export function updateVirtualButton(btn: VirtualButton): void {
  const touchCount = getTouchCount();
  const wasActive = btn.active;

  // Check if any unclaimed touch is inside the button
  let nowActive = false;
  let activeIndex = -1;

  for (let i = 0; i < touchCount; i++) {
    if (claimedTouches.has(i)) continue;
    const tx = getTouchX(i);
    const ty = getTouchY(i);
    if (tx === 0 && ty === 0) continue;

    const dx = tx - btn.x;
    const dy = ty - btn.y;
    if (dx * dx + dy * dy <= btn.radius * btn.radius) {
      nowActive = true;
      activeIndex = i;
      break;
    }
  }

  if (nowActive) {
    claimedTouches.add(activeIndex);
    btn.touchIndex = activeIndex;
  }

  if (nowActive && !wasActive) {
    // Just pressed
    btn.active = true;
    if (btn.key !== null) injectKeyDown(btn.key);
    if (btn.gamepadButton !== null) injectGamepadButtonDown(btn.gamepadButton);
  } else if (!nowActive && wasActive) {
    // Just released
    btn.active = false;
    btn.touchIndex = -1;
    if (btn.key !== null) injectKeyUp(btn.key);
    if (btn.gamepadButton !== null) injectGamepadButtonUp(btn.gamepadButton);
  }
}

export function drawVirtualButton(btn: VirtualButton): void {
  const bgColor: Color = btn.active
    ? { r: 255, g: 255, b: 255, a: 120 }
    : { r: 255, g: 255, b: 255, a: 60 };

  drawCircle(btn.x, btn.y, btn.radius, bgColor);

  if (btn.label) {
    const fontSize = btn.radius * 0.8;
    const textWidth = measureText(btn.label, fontSize);
    const labelColor: Color = { r: 255, g: 255, b: 255, a: 200 };
    drawText(btn.label, btn.x - textWidth / 2, btn.y - fontSize / 2, fontSize, labelColor);
  }
}

// ============================================================
// Convenience: unified movement input
// ============================================================

export function getMovementInput(): { x: number; y: number } {
  let x = 0;
  let y = 0;

  // WASD
  if (isKeyDown(Key.D)) x += 1;
  if (isKeyDown(Key.A)) x -= 1;
  if (isKeyDown(Key.S)) y += 1;
  if (isKeyDown(Key.W)) y -= 1;

  // Arrow keys
  if (isKeyDown(Key.RIGHT)) x += 1;
  if (isKeyDown(Key.LEFT)) x -= 1;
  if (isKeyDown(Key.DOWN)) y += 1;
  if (isKeyDown(Key.UP)) y -= 1;

  // Gamepad axes (overrides if non-zero)
  const gx = getGamepadAxis(0);
  const gy = getGamepadAxis(1);
  if (Math.abs(gx) > 0.1 || Math.abs(gy) > 0.1) {
    x = gx;
    y = gy;
  }

  // Clamp
  const len = Math.sqrt(x * x + y * y);
  if (len > 1) {
    x /= len;
    y /= len;
  }

  return { x, y };
}
