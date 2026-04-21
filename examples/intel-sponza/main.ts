// ============================================================
// Bloom Intel Sponza Stress Test
// ============================================================
// Intel's 2022 photogrammetry rework of Sponza — ~10M+ triangles,
// 4K PBR textures, 68 materials. Stress-tests:
//   - Draw-call batching (68 meshes vs Khronos's 25)
//   - Texture memory pressure (4K PBR sets)
//   - Shadow pass perf under heavy vertex load
//   - Auto-fit shadow bounds on a non-Khronos scene
//   - TAA + GTAO stability under high-frequency detail

import {
  initWindow, windowShouldClose, beginDrawing, endDrawing, takeScreenshot,
  setEnvClearFromHdr, setTargetFPS, getDeltaTime, getFPS,
  isKeyDown, isKeyPressed,
  getMouseDeltaX, getMouseDeltaY,
  disableCursor, enableCursor,
  beginMode3D, endMode3D,
  setFog, setSunShafts, setVignette, setChromaticAberration,
  setAutoExposure, setEnvIntensity, setManualExposure, setTaaEnabled,
  setProfilerEnabled, printProfilerSummary,
  getProfilerFrameCpuUs, getProfilerFrameGpuUs,
  setQualityPreset, QualityPreset,
  setShadowsEnabled, setSsaoEnabled, setSsrEnabled, setSsgiEnabled,
} from "bloom/core";
import { Key } from "bloom/core";
import { drawText } from "bloom/text";
import {
  setAmbientLight, setDirectionalLight, loadModel,
} from "bloom/models";
import {
  enableShadows, addDirectionalLight,
  createSceneNode, attachModelToNode, setSceneNodeTransform,
  setSceneNodeCastShadow, dumpShadowMap,
} from "bloom/scene";
import { clamp, mat4Identity } from "bloom/math";

const SCREEN_W = 800;
const SCREEN_H = 450;
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
let dumpShadowFrames = 0;
let dumpShadowPath = "";
let profileFrames = 0;
let fpsOnlyFrames = 0;
let qualityPreset = -1;
let forceShadows = -1;
let forceSsao = -1;
let forceSsr = -1;
let forceSsgi = -1;
let initCamX: number | null = null;
let initCamY: number | null = null;
let initCamZ: number | null = null;
let initPitch = 0.0;
let dumpPoseEveryFrame = false;
// Shadow-drag repro: start at `initYaw`, let history accumulate for
// `turnAt` frames, then snap to `turnYaw`. If a temporal pass is
// reprojecting history wrong, the frames captured after the snap
// will show a dark 'ghost' sliding across the frame.
let turnYaw = Number.NaN;
let turnAt = 0;
// `--no-pan` freezes the auto-camera during `--profile` / `--fps-only`.
// Needed to measure the ticket-004 shadow cache — stationary camera is
// the cache-hit path; the default 0.005 rad/frame pan invalidates the
// cascade VPs every frame and forces a re-render.
let noPan = false;
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
  if (argv[i] === "--dump-shadow" && i + 2 < argv.length) {
    dumpShadowFrames = Math.floor(parseFloat(argv[i + 1]));
    dumpShadowPath = argv[i + 2];
  }
  if (argv[i] === "--profile" && i + 1 < argv.length) {
    profileFrames = Math.floor(parseFloat(argv[i + 1]));
  }
  // --fps-only runs the same auto-camera loop as --profile but WITHOUT
  // turning the profiler on, so there's no readback stall polluting
  // the FPS number.
  if (argv[i] === "--fps-only" && i + 1 < argv.length) {
    fpsOnlyFrames = Math.floor(parseFloat(argv[i + 1]));
  }
  // --quality N applies a preset (0=Off, 1=Low, 2=Medium, 3=High, 4=Ultra).
  if (argv[i] === "--quality" && i + 1 < argv.length) {
    qualityPreset = Math.floor(parseFloat(argv[i + 1]));
  }
  // Individual effect overrides — applied AFTER --quality so they win.
  if (argv[i] === "--shadows" && i + 1 < argv.length) {
    forceShadows = parseInt(argv[i + 1]);
  }
  if (argv[i] === "--ssao" && i + 1 < argv.length) {
    forceSsao = parseInt(argv[i + 1]);
  }
  if (argv[i] === "--ssr" && i + 1 < argv.length) {
    forceSsr = parseInt(argv[i + 1]);
  }
  if (argv[i] === "--ssgi" && i + 1 < argv.length) {
    forceSsgi = parseInt(argv[i + 1]);
  }
  if (argv[i] === "--taa" && i + 1 < argv.length) {
    taaOverride = parseInt(argv[i + 1]);
  }
  // Camera pose overrides — lets the headless --capture path hit any
  // view the interactive player can walk to. Pair with --yaw.
  if (argv[i] === "--cam-x" && i + 1 < argv.length) {
    initCamX = parseFloat(argv[i + 1]);
  }
  if (argv[i] === "--cam-y" && i + 1 < argv.length) {
    initCamY = parseFloat(argv[i + 1]);
  }
  if (argv[i] === "--cam-z" && i + 1 < argv.length) {
    initCamZ = parseFloat(argv[i + 1]);
  }
  if (argv[i] === "--pitch" && i + 1 < argv.length) {
    initPitch = parseFloat(argv[i + 1]);
  }
  // Prints current cam pose every frame — use to capture your
  // exact interactive view then replay via --cam-x/y/z --yaw --pitch.
  if (argv[i] === "--dump-pose") {
    dumpPoseEveryFrame = true;
  }
  // Shadow-drag repro: start at --yaw, let N frames accumulate,
  // then snap to --turn-yaw. Capture with --capture frame where
  // frame > turn-at to see the post-snap ghost.
  if (argv[i] === "--turn-yaw" && i + 1 < argv.length) {
    turnYaw = parseFloat(argv[i + 1]);
  }
  if (argv[i] === "--turn-at" && i + 1 < argv.length) {
    turnAt = Math.floor(parseFloat(argv[i + 1]));
  }
  if (argv[i] === "--no-pan") {
    noPan = true;
  }
}

// ---- Init ----
initWindow(SCREEN_W, SCREEN_H, "Bloom Intel Sponza Stress", 0);
setTargetFPS(60);
setEnvClearFromHdr("assets/outdoor.hdr");
enableShadows();

// Sponza ceilings face down = dark IBL. High env_intensity
// compensates for lack of GI bounce.
// Env 1.5 was overdriving the marble column's IBL specular response,
// producing a bright vertical stripe Cycles doesn't show. 1.2 matches
// the Bistro preset and keeps indirect-diffuse reasonable.
setEnvIntensity(1.2);
// Manual exposure for apples-to-apples with Unity/Cycles comparisons;
// auto-exposure was hiding a blowout on the column at this pose by
// pulling the whole image up, which then clipped the sunlit stone to
// pure white.
setAutoExposure(false);
setManualExposure(1.0);
if (taaOverride === 0) { setTaaEnabled(false); }
if (taaOverride === 1) { setTaaEnabled(true); }
// Subtle warm atmospheric haze — dust in the air catches the sun
// coming through the atrium. Density low enough to preserve scene
// contrast; height falloff thins it above head-height so the
// upper walls stay clean.
// Fog density reduced from 0.010 — at the atrium pose it was washing
// out the back wall and flattening mid-ground contrast vs the Cycles
// reference. 0.003 keeps the sun-shaft volumetric feel without muting
// textures.
setFog(0.86, 0.82, 0.72, 0.003, 0.0, 0.12);
// Sun shafts from the open atrium. Warm tint matches the outdoor HDR
// environment; high decay gives long streaks that fade gently with
// distance from the sun's projected screen position.
// Sun shafts were producing a bright vertical stripe through the column
// (the sun's screen-space radial-blur streak didn't respect depth).
// Turned off for now; can re-enable with a depth-aware variant later.
setSunShafts(0.0, 0.97, 1.0, 0.92, 0.78);
setVignette(0.25, 0.25);
// CA was tinting every bright specular speck pink (R/B channels
// split on narrow highlights). Off for interior scenes — lens-
// abberation feel is a bigger negative than a positive here.
setChromaticAberration(0.0);

// Profiler — enable once we've built up some rolling averages. Turned
// on before the main loop so the first sampled frame doesn't include
// one-shot setup costs.
if (profileFrames > 0) {
  setProfilerEnabled(true);
}

// Apply quality preset if requested on the CLI (overrides the per-FX
// calls above since preset.apply() runs after them via FFI).
if (qualityPreset >= 0) {
  setQualityPreset(qualityPreset as QualityPreset);
}
// Individual overrides — these apply AFTER the preset so you can, e.g.,
// `--quality 1 --shadows 1` to test "Low + only shadows".
if (forceShadows >= 0) { setShadowsEnabled(forceShadows !== 0); }
if (forceSsao >= 0) { setSsaoEnabled(forceSsao !== 0); }
if (forceSsr >= 0) { setSsrEnabled(forceSsr !== 0); }
if (forceSsgi >= 0) { setSsgiEnabled(forceSsgi !== 0); }
if (taaOverride === 0) { setTaaEnabled(false); }
if (taaOverride === 1) { setTaaEnabled(true); }

// ---- Load Sponza into scene graph ----
// Intel Sponza ships as loose glTF + .bin + 68 textures. The
// filename in Intel's bundle is typically `NewSponza_Main_glTF_003.gltf`
// or similar — adjust after extracting to match whatever it turns out to be.
const sponza = loadModel("assets/NewSponza_Main_glTF_003.gltf");
const identity = mat4Identity();
for (let i = 0; i < sponza.meshCount; i = i + 1) {
  const node = createSceneNode();
  attachModelToNode(node, sponza.handle, i);
  setSceneNodeTransform(node, identity);
}

// ---- Camera ----
// Sponza courtyard center, looking down the main axis. CLI overrides
// (--cam-x/y/z --yaw --pitch) replace the defaults so a headless
// --capture can reproduce any player-visible pose.
let camX = initCamX !== null ? initCamX : 0.0;
let camY = initCamY !== null ? initCamY : 2.0;
let camZ = initCamZ !== null ? initCamZ : 0.0;
let camYaw = initYaw;
let camPitch = initPitch;
let cursorLocked = false;

// ---- Main loop ----
while (!windowShouldClose()) {
  const dt = getDeltaTime();

  // Shadow-drag repro: after `turnAt` frames at the initial yaw,
  // start rotating continuously toward `turnYaw` at mouse-like
  // speed. Simulates a real drag (continuous small velocities)
  // rather than a single snap — important because temporal
  // reprojection can behave differently at 1 UV/frame vs
  // 0.01 UV/frame per pixel.
  if (!Number.isNaN(turnYaw) && frameCount >= turnAt) {
    const perFrameRate = 0.025; // radians/frame ≈ fast mouse drag
    const delta = turnYaw - camYaw;
    if (Math.abs(delta) > perFrameRate) {
      camYaw = camYaw + Math.sign(delta) * perFrameRate;
    } else {
      camYaw = turnYaw;
    }
  }

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

  // Pose dump: press P to print the current camera pose — or run
  // with --dump-pose to print every frame. Paste into a replay:
  //   ./main --cam-x X --cam-y Y --cam-z Z --yaw YAW --pitch PITCH
  if (isKeyPressed(Key.P) || dumpPoseEveryFrame) {
    console.log(
      "pose: --cam-x " + camX.toFixed(3) +
      " --cam-y " + camY.toFixed(3) +
      " --cam-z " + camZ.toFixed(3) +
      " --yaw " + camYaw.toFixed(3) +
      " --pitch " + camPitch.toFixed(3),
    );
  }

  const lookX = camX + Math.cos(camPitch) * fwdX * 100;
  const lookY = camY + Math.sin(camPitch) * 100;
  const lookZ = camZ + Math.cos(camPitch) * fwdZ * 100;

  // ---- Rendering ----
  beginDrawing();

  // Ambient swung from cool blue-grey (160,165,180) toward warm
  // stone-bounce (175,160,140). The cool tint was an artifact of the
  // HDR sky alone; in reality the atrium floor of sunlit stone bounces
  // amber-yellow back up into the shaded interior.
  setAmbientLight({ r: 175, g: 160, b: 140, a: 255 }, 0.3);
  setDirectionalLight(
    { x: 0.6, y: 0.8, z: 0.3 },
    { r: 255, g: 245, b: 230, a: 255 },
    1.0,
  );
  // Fill from above aimed at downward-facing surfaces (undersides of
  // arches, lantern caps). Previously cool green-blue; flipped to
  // amber since what actually hits a ceiling's underside is bounce
  // light from the sun-lit stone floor, not sky. Matches Cycles'
  // warmer undersoffit tone.
  addDirectionalLight(0.0, -1.0, 0.0, 0.6, 0.5, 0.4, 0.5);

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
  drawText("Bloom Intel Sponza Stress", 10, 10, 20, { r: 255, g: 255, b: 255, a: 255 });
  const fps = getFPS();
  const ms = fps > 0.0 ? 1000.0 / fps : 0.0;
  // Color the FPS line based on perf bucket so glances give
  // instant feedback during stress tests.
  const fpsColor = fps >= 55.0
    ? { r: 120, g: 230, b: 120, a: 255 }
    : fps >= 30.0
      ? { r: 230, g: 220, b: 120, a: 255 }
      : { r: 230, g: 120, b: 120, a: 255 };
  const fpsText = `FPS ${Math.round(fps)}  (${ms.toFixed(1)} ms)`;
  drawText(fpsText, 10, 35, 16, fpsColor);
  drawText("WASD move / Mouse look / Tab cursor", 10, SCREEN_H - 30, 14, { r: 180, g: 180, b: 180, a: 255 });

  // Auto-capture for automated testing
  if (captureFrames > 0) {
    frameCount = frameCount + 1;
    if (frameCount === captureFrames) { takeScreenshot(capturePath); }
    if (frameCount > captureFrames) { endDrawing(); break; }
  }

  // Shadow-map dump for diagnostics — dump cascade 0 after N frames then exit.
  if (dumpShadowFrames > 0) {
    frameCount = frameCount + 1;
    if (frameCount === dumpShadowFrames) {
      endDrawing();
      dumpShadowMap(dumpShadowPath);
      break;
    }
  }

  // Profiler auto-run: run N frames, print summary, exit. Camera
  // pans slightly so each frame exercises shadow recomputation.
  if (profileFrames > 0) {
    frameCount = frameCount + 1;
    if (!noPan) { camYaw = camYaw + 0.005; }
    if (frameCount >= profileFrames) {
      endDrawing();
      const cpuUs = getProfilerFrameCpuUs();
      const gpuUs = getProfilerFrameGpuUs();
      const measuredFps = getFPS();
      const measuredMs = measuredFps > 0 ? 1000 / measuredFps : 0;
      console.log(`\n=== PROFILE (${frameCount} frames) ===`);
      console.log(`FPS:       ${measuredFps.toFixed(1)} (${measuredMs.toFixed(2)} ms/frame)`);
      console.log(`Total CPU: ${(cpuUs / 1000).toFixed(2)} ms`);
      console.log(`Total GPU: ${(gpuUs / 1000).toFixed(2)} ms`);
      printProfilerSummary();
      break;
    }
  }

  // FPS-only run: same camera pan, no profiler → pure FPS signal.
  if (fpsOnlyFrames > 0) {
    frameCount = frameCount + 1;
    if (!noPan) { camYaw = camYaw + 0.005; }
    if (frameCount >= fpsOnlyFrames) {
      endDrawing();
      const measuredFps = getFPS();
      const measuredMs = measuredFps > 0 ? 1000 / measuredFps : 0;
      console.log(`\n=== FPS-ONLY (${frameCount} frames, no profiler) ===`);
      console.log(`FPS: ${measuredFps.toFixed(1)} (${measuredMs.toFixed(2)} ms/frame)`);
      break;
    }
  }

  endDrawing();
}
