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
  setAutoExposure, setEnvIntensity, setManualExposure, setTaaEnabled,
} from "bloom/core";
import { Key } from "bloom/core";
import { drawText } from "bloom/text";
import {
  setAmbientLight, setDirectionalLight, loadModel,
} from "bloom/models";
import {
  enableShadows, addDirectionalLight,
  createSceneNode, attachModelToNode, setSceneNodeTransform,
  setSceneNodeCastShadow,
} from "bloom/scene";
import { clamp, mat4Identity } from "bloom/math";

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
let initYaw = 0.0;
let taaOverride = -1; // -1 = default, 0 = force off, 1 = force on
for (let i = 2; i < argv.length; i = i + 1) {
  if (argv[i] === "--capture" && i + 2 < argv.length) {
    captureFrames = Math.floor(parseFloat(argv[i + 1]));
    capturePath = argv[i + 2];
  }
  if (argv[i] === "--yaw" && i + 1 < argv.length) {
    initYaw = parseFloat(argv[i + 1]);
  }
  if (argv[i] === "--taa" && i + 1 < argv.length) {
    taaOverride = parseInt(argv[i + 1]);
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
setAutoExposure(true);
if (taaOverride === 0) { setTaaEnabled(false); }
if (taaOverride === 1) { setTaaEnabled(true); }
// Fog disabled — was causing brightness variation in corridors
// setFog(0.7, 0.75, 0.82, 0.008, 0.0, 0.1);
setVignette(0.25, 0.25);
setChromaticAberration(0.001);

// ---- Load Sponza into scene graph ----
const sponza = loadModel("assets/Sponza.glb");
const identity = mat4Identity();
for (let i = 0; i < sponza.meshCount; i = i + 1) {
  const node = createSceneNode();
  attachModelToNode(node, sponza.handle, i);
  setSceneNodeTransform(node, identity);
}

// ---- Camera ----
// Sponza courtyard center, looking down the main axis
let camX = 0.0;
let camY = 2.0;
let camZ = 0.0;
let camYaw = initYaw;
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

  setAmbientLight({ r: 160, g: 165, b: 180, a: 255 }, 0.3);
  setDirectionalLight(
    { x: 0.6, y: 0.8, z: 0.3 },
    { r: 255, g: 245, b: 230, a: 255 },
    1.5,
  );
  // Gentle fill from below — safety net for ceilings that SSGI
  // bounce light might not fully reach. Kept very low (0.5) since
  // SSGI now provides natural indirect diffuse bounce from the
  // sunlit floor.
  addDirectionalLight(0.0, -1.0, 0.0, 0.5, 0.55, 0.65, 0.5);

  beginMode3D({
    position: { x: camX, y: camY, z: camZ },
    target: { x: lookX, y: lookY, z: lookZ },
    up: { x: 0, y: 1, z: 0 },
    fovy: 60,
    projection: "perspective",
  });

  // Scene graph handles all rendering (shadows + PBR). No
  // drawModel needed — it would double-render without shadows.

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
