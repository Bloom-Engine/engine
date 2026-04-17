// ============================================================
// Bloom Bistro Validation Scene
// ============================================================
// Amazon Lumberyard Bistro — Parisian street corner scene. Different
// profile from Sponza: outdoor lighting dominated by a single sun,
// varied materials (stone, brick, painted wood, glass, fabric awnings,
// metal fixtures, foliage), and long sight lines. A good cross-check
// that the rendering pipeline doesn't over-fit to Sponza's atrium
// geometry and IBL.
//
// Assets aren't shipped with the repo — they total ~1.2 GB. To set
// up this scene, clone zeux/niagara_bistro (MIT-licensed glTF
// conversion of NVIDIA's Bistro) into `assets/`:
//
//   cd examples/bistro
//   git clone https://github.com/zeux/niagara_bistro.git assets
//
// The scene loads `assets/bistro.gltf` (exterior). An interior
// variant `assets/bistrox.gltf` also exists — swap the filename
// below to open that one instead.

import {
  initWindow, windowShouldClose, beginDrawing, endDrawing, takeScreenshot,
  setEnvClearFromHdr, setTargetFPS, getDeltaTime, getFPS,
  isKeyDown, isKeyPressed,
  getMouseDeltaX, getMouseDeltaY,
  disableCursor, enableCursor,
  beginMode3D, endMode3D,
  setFog, setSunShafts, setVignette, setChromaticAberration,
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
} from "bloom/scene";
import { clamp, mat4Identity } from "bloom/math";

const SCREEN_W = 800;
const SCREEN_H = 450;
const MOUSE_SENS = 0.003;
const MOVE_SPEED = 5.0;
const SPRINT_MULT = 2.5;

// Auto-capture args (matches the sponza examples' CLI)
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
initWindow(SCREEN_W, SCREEN_H, "Bloom Bistro", 0);
setTargetFPS(60);
setEnvClearFromHdr("assets/outdoor.hdr");
enableShadows();

// Open-air street scene: the sky is the dominant IBL source. 1.2×
// env intensity gives colourful ambient reflection without washing
// out direct sunlight.
setEnvIntensity(1.2);
setAutoExposure(true);
if (taaOverride === 0) { setTaaEnabled(false); }
if (taaOverride === 1) { setTaaEnabled(true); }

// Warm Parisian haze — cream-white with a slight yellow shift.
// Lower density than the first attempt so distant buildings still
// register texture and colour rather than fading into flat blue fog.
setFog(0.92, 0.90, 0.84, 0.006, 0.0, 0.05);
// Subtle shafts — exterior scene so the sun is usually off-frame or
// clipped by buildings. Lower strength than Sponza's atrium.
setSunShafts(0.25, 0.96, 1.0, 0.94, 0.82);
setVignette(0.20, 0.25);
setChromaticAberration(0.001);

// ---- Load Bistro into scene graph ----
// `bistro.gltf` = exterior street corner. Swap to `bistrox.gltf`
// for the interior wine-bar variant.
const bistro = loadModel("assets/bistro.gltf");
const identity = mat4Identity();
for (let i = 0; i < bistro.meshCount; i = i + 1) {
  const node = createSceneNode();
  attachModelToNode(node, bistro.handle, i);
  setSceneNodeTransform(node, identity);
}

// ---- Camera ----
// Matches the preset glTF camera in zeux/niagara_bistro (translation
// -26.43, 3.16, 11.17 aimed toward the bistro façade near the world
// origin). Gives a clean opening frame showing the signature corner
// with the lantern, awning, and cobble street.
let camX = -26.43;
let camY = 3.16;
let camZ = 11.17;
let camYaw = initYaw !== 0.0 ? initYaw : -1.17; // ≈ 67° left of -Z
let camPitch = 0.0;
let cursorLocked = false;

// ---- Main loop ----
while (!windowShouldClose()) {
  const dt = getDeltaTime();

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

  beginDrawing();

  setAmbientLight({ r: 150, g: 160, b: 180, a: 255 }, 0.3);
  // Parisian afternoon sun — warm, angled slightly from the side.
  setDirectionalLight(
    { x: -0.5, y: 0.75, z: 0.4 },
    { r: 255, g: 240, b: 220, a: 255 },
    1.8,
  );
  // Tiny fill from below — same trick as Sponza, keeps overhangs
  // and awnings from bottoming out when SSGI misses them.
  addDirectionalLight(0.0, -1.0, 0.0, 0.5, 0.55, 0.7, 0.4);

  beginMode3D({
    position: { x: camX, y: camY, z: camZ },
    target: { x: lookX, y: lookY, z: lookZ },
    up: { x: 0, y: 1, z: 0 },
    fovy: 60,
    projection: "perspective",
  });

  endMode3D();

  // HUD
  drawText("Bloom Bistro", 10, 10, 20, { r: 255, g: 255, b: 255, a: 255 });
  const fps = getFPS();
  const ms = fps > 0.0 ? 1000.0 / fps : 0.0;
  const fpsColor = fps >= 55.0
    ? { r: 120, g: 230, b: 120, a: 255 }
    : fps >= 30.0
      ? { r: 230, g: 220, b: 120, a: 255 }
      : { r: 230, g: 120, b: 120, a: 255 };
  const fpsText = `FPS ${Math.round(fps)}  (${ms.toFixed(1)} ms)`;
  drawText(fpsText, 10, 35, 16, fpsColor);
  drawText("WASD move / Mouse look / Tab cursor", 10, SCREEN_H - 30, 14, { r: 180, g: 180, b: 180, a: 255 });

  if (captureFrames > 0) {
    frameCount = frameCount + 1;
    if (frameCount === captureFrames) { takeScreenshot(capturePath); }
    if (frameCount > captureFrames) { endDrawing(); break; }
  }

  endDrawing();
}
