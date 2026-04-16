// ============================================================
// Bloom Sponza Showcase
// ============================================================
// The Khronos Sponza atrium — industry-standard PBR benchmark.
// 260K triangles, 25 materials, architectural columns + arches.
// Tests: shadows through columns, AO in arches, IBL on marble,
// auto-exposure in bright courtyard vs shadowed corridors.

import {
  initWindow, windowShouldClose, beginDrawing, endDrawing, takeScreenshot,
  setEnvClearFromHdr, setTargetFPS, getDeltaTime,
  isKeyDown, isKeyPressed,
  getMouseDeltaX, getMouseDeltaY,
  disableCursor, enableCursor,
  beginMode3D, endMode3D,
  setFog, setVignette, setChromaticAberration,
  setAutoExposure, setEnvIntensity, setManualExposure,
} from "bloom/core";
import { Key } from "bloom/core";
import { drawText } from "bloom/text";
import {
  setAmbientLight, setDirectionalLight, loadModel, drawModel,
} from "bloom/models";
import {
  enableShadows, addDirectionalLight,
} from "bloom/scene";
import { clamp } from "bloom/math";

const SCREEN_W = 1280;
const SCREEN_H = 720;
const MOUSE_SENS = 0.003;
const MOVE_SPEED = 5.0;
const SPRINT_MULT = 2.5;

// Auto-capture args
declare const process: { argv: string[] };
const argv: string[] = process.argv;
let captureFrames = 0;
let capturePath = "";
let frameCount = 0;
for (let i = 2; i < argv.length; i = i + 1) {
  if (argv[i] === "--capture" && i + 2 < argv.length) {
    captureFrames = Math.floor(parseFloat(argv[i + 1]));
    capturePath = argv[i + 2];
  }
}

// ---- Init ----
initWindow(SCREEN_W, SCREEN_H, "Bloom Sponza", 0);
setTargetFPS(60);
setEnvClearFromHdr("assets/outdoor.hdr");
enableShadows();

// Sponza ceilings face down = dark IBL. High env_intensity
// compensates for lack of GI bounce.
setEnvIntensity(1.5);
setAutoExposure(false);
setManualExposure(2.5);
// Fog disabled — was causing brightness variation in corridors
// setFog(0.7, 0.75, 0.82, 0.008, 0.0, 0.1);
setVignette(0.25, 0.25);
setChromaticAberration(0.001);

// ---- Load Sponza ----
const sponza = loadModel("assets/Sponza.glb");

// ---- Camera ----
// Sponza courtyard center, looking down the main axis
let camX = 0.0;
let camY = 2.0;
let camZ = 0.0;
let camYaw = 0.0;
let camPitch = 0.0;
let cursorLocked = false;

// ---- Main loop ----
while (!windowShouldClose()) {
  const dt = getDeltaTime();

  // Camera controls
  if (cursorLocked) {
    camYaw = camYaw - getMouseDeltaX() * MOUSE_SENS;
    camPitch = camPitch - getMouseDeltaY() * MOUSE_SENS;
    camPitch = clamp(camPitch, -1.4, 1.4);
  }

  const speed = isKeyDown(Key.LEFT_SHIFT) ? MOVE_SPEED * SPRINT_MULT : MOVE_SPEED;
  const fwdX = -Math.sin(camYaw);
  const fwdZ = -Math.cos(camYaw);
  const rightX = Math.cos(camYaw);
  const rightZ = -Math.sin(camYaw);

  if (isKeyDown(Key.W) || isKeyDown(Key.UP))    { camX = camX + fwdX * speed * dt; camZ = camZ + fwdZ * speed * dt; }
  if (isKeyDown(Key.S) || isKeyDown(Key.DOWN))   { camX = camX - fwdX * speed * dt; camZ = camZ - fwdZ * speed * dt; }
  if (isKeyDown(Key.A) || isKeyDown(Key.LEFT))   { camX = camX - rightX * speed * dt; camZ = camZ - rightZ * speed * dt; }
  if (isKeyDown(Key.D) || isKeyDown(Key.RIGHT))  { camX = camX + rightX * speed * dt; camZ = camZ + rightZ * speed * dt; }
  if (isKeyDown(Key.SPACE))        { camY = camY + speed * dt; }
  if (isKeyDown(Key.C))            { camY = camY - speed * dt; }

  if (isKeyPressed(Key.TAB)) {
    cursorLocked = !cursorLocked;
    if (cursorLocked) { disableCursor(); } else { enableCursor(); }
  }

  const lookX = camX + Math.cos(camPitch) * fwdX * 100;
  const lookY = camY + Math.sin(camPitch) * 100;
  const lookZ = camZ + Math.cos(camPitch) * fwdZ * 100;

  // ---- Rendering ----
  beginDrawing();

  // Lighting
  setAmbientLight({ r: 160, g: 165, b: 180, a: 255 }, 0.3);
  // Primary sun
  setDirectionalLight(
    { x: 0.6, y: 0.8, z: 0.3 },
    { r: 255, g: 245, b: 230, a: 255 },
    1.5,
  );
  // Fill light from below — bounced light that would come from
  // the lit floor illuminating the vaulted ceilings. Without GI
  // this is the only way to keep ceilings visible.
  // Gentle fill — just enough to keep ceilings from going black.
  // Kept weak so back corridors stay naturally darker than the
  // sunlit courtyard (matching real light falloff).
  addDirectionalLight(0.0, -1.0, 0.0, 0.5, 0.55, 0.65, 2.0);

  beginMode3D({
    position: { x: camX, y: camY, z: camZ },
    target: { x: lookX, y: lookY, z: lookZ },
    up: { x: 0, y: 1, z: 0 },
    fovy: 60,
    projection: "perspective",
  });

  if (sponza.handle !== 0) {
    drawModel(sponza, { x: 0, y: 0, z: 0 }, 1.0, { r: 255, g: 255, b: 255, a: 255 });
  }

  endMode3D();

  // HUD
  drawText("Bloom Sponza", 10, 10, 20, { r: 255, g: 255, b: 255, a: 255 });
  drawText("WASD move / Mouse look / Tab cursor", 10, SCREEN_H - 30, 14, { r: 180, g: 180, b: 180, a: 255 });

  // Auto-capture for automated testing
  if (captureFrames > 0) {
    frameCount = frameCount + 1;
    if (frameCount === captureFrames) { takeScreenshot(capturePath); }
    if (frameCount > captureFrames) { endDrawing(); break; }
  }

  endDrawing();
}
