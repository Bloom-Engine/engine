// ============================================================
// Bloom Renderer Test Scene
// ============================================================
// Walkable 3D showcase exercising every current rendering feature.
// Visual regression target for bloom-renderer-spec-v2.md.
//
// As renderer features land (GI, SSR, bloom, volumetrics, TAA,
// VSM, etc.), this scene is designed to reveal their impact
// without code changes — the geometry and materials are chosen
// to make quality differences obvious.
//
// Controls:
//   WASD / Arrows  Move (horizontal)
//   Mouse          Look
//   Space          Up
//   C              Down
//   Shift          Sprint
//   Tab            Toggle cursor lock
//   1-6            Teleport to zone
//
// Zones:
//   1  PBR Material Gallery   (origin)
//   2  Multi-Light Arena      (+X)
//   3  Shadow Quality         (-X)
//   4  Water Surface          (+Z)
//   5  Geometry Density       (-Z, +X)
//   6  Thin Geometry / AA     (-Z, -X)
// ============================================================

import {
  initWindow, closeWindow, windowShouldClose, beginDrawing, endDrawing, takeScreenshot,
  clearBackground, setEnvClearFromHdr, setTargetFPS, getDeltaTime, getFPS, getTime,
  isKeyDown, isKeyPressed,
  getMouseDeltaX, getMouseDeltaY,
  disableCursor, enableCursor,
  beginMode3D, endMode3D,
  setFog, setChromaticAberration, setVignette, setFilmGrain, setSunShafts,
  setAutoExposure, setEnvIntensity,
} from "bloom/core";
import { Key } from "bloom/core";
import { drawText } from "bloom/text";

// HUD colors (0-255 range, matching bloom Color struct)
const WHITE  = { r: 255, g: 255, b: 255, a: 255 };
const LGRAY  = { r: 200, g: 200, b: 200, a: 255 };
const GRAY   = { r: 130, g: 130, b: 130, a: 255 };
import {
  setAmbientLight, setDirectionalLight,
  drawGrid, genMeshCube, createMesh, loadModel, drawModel, drawCube,
} from "bloom/models";
import {
  createSceneNode, setSceneNodeTransform,
  updateSceneNodeGeometry,
  setSceneNodeColor, setSceneNodePbr,
  setSceneNodeCastShadow, setSceneNodeReceiveShadow,
  enableShadows,
  addDirectionalLight, addPointLight,
  attachModelToNode,
  setSceneNodeWaterMaterial,
} from "bloom/scene";
import {
  mat4Identity, mat4Translate, mat4Scale, mat4RotateY,
  clamp,
} from "bloom/math";

// ---- Constants ----

const SCREEN_W = 1280;
const SCREEN_H = 720;
const MOUSE_SENS = 0.003;
const MOVE_SPEED = 8.0;
const SPRINT_MULT = 2.5;
const PI = 3.14159265;
const TWO_PI = 6.28318530;

// ---- Headless / spec-driven mode parsing ----
// When launched with `--spec FILE --out FILE`, we skip the interactive
// loop: read the camera from the spec, render N warmup frames (so
// lighting uniforms & framebuffer are populated), capture a screenshot
// to the given path, then exit. This matches how bloom-reference
// renders the same spec, so the two outputs are directly comparable
// via bloom-diff. Without these flags, the app runs as the normal
// interactive walkthrough with F12 screenshots.

let headlessSpecPath = "";
let headlessOutPath = "";
let headlessMode = false;
let headlessCamX = 0.0;
let headlessCamY = 0.0;
let headlessCamZ = 0.0;
let headlessTargetX = 0.0;
let headlessTargetY = 0.0;
let headlessTargetZ = 0.0;
let headlessFov = 45.0;
let headlessResW = 0;
let headlessResH = 0;
// Auto-capture in interactive mode: render the full scene (all zones,
// HUD, the actual interactive path) for N frames, then screenshot and
// exit. Used to programmatically hunt the TAA+bloom corruption that
// only appears with surface presentation.
let interactiveCaptureFrames = 0;
let interactiveCapturePath = "";

declare const process: { argv: string[] };
const argv: string[] = process.argv;
for (let i = 2; i < argv.length; i = i + 1) {
  if (argv[i] === "--spec" && i + 1 < argv.length) {
    headlessSpecPath = argv[i + 1];
    headlessMode = true;
  } else if (argv[i] === "--out" && i + 1 < argv.length) {
    headlessOutPath = argv[i + 1];
  } else if (argv[i] === "--camera" && i + 9 < argv.length) {
    // --camera px py pz tx ty tz fov
    // Primary path for headless mode — avoids JSON array access
    // which has known Perry LLVM backend issues (see Phase 5 notes).
    headlessCamX = parseFloat(argv[i + 1]);
    headlessCamY = parseFloat(argv[i + 2]);
    headlessCamZ = parseFloat(argv[i + 3]);
    headlessTargetX = parseFloat(argv[i + 4]);
    headlessTargetY = parseFloat(argv[i + 5]);
    headlessTargetZ = parseFloat(argv[i + 6]);
    headlessFov = parseFloat(argv[i + 7]);
    headlessMode = true;
  } else if (argv[i] === "--res" && i + 2 < argv.length) {
    // --res W H — overrides the window size for headless captures
    // so validate.sh can match the reference's resolution exactly.
    // Use parseFloat (parseInt has shown odd behavior under Perry's
    // current backend); cast back to int via Math.floor.
    headlessResW = Math.floor(parseFloat(argv[i + 1]));
    headlessResH = Math.floor(parseFloat(argv[i + 2]));
    headlessMode = true;
  } else if (argv[i] === "--interactive-capture" && i + 2 < argv.length) {
    interactiveCaptureFrames = Math.floor(parseFloat(argv[i + 1]));
    interactiveCapturePath = argv[i + 2];
  }
}


// ---- Mesh generation ----

function makeSphereVertices(radius: number, segs: number, rings: number): number[] {
  const v: number[] = [];
  for (let r = 0; r <= rings; r = r + 1) {
    const phi = PI * r / rings;
    const sp = Math.sin(phi);
    const cp = Math.cos(phi);
    for (let s = 0; s <= segs; s = s + 1) {
      const theta = TWO_PI * s / segs;
      const st = Math.sin(theta);
      const ct = Math.cos(theta);
      const x = sp * ct;
      const y = cp;
      const z = sp * st;
      v.push(x * radius, y * radius, z * radius); // position
      v.push(x, y, z);                             // normal
      v.push(1, 1, 1, 1);                          // color (white)
      v.push(s / segs, r / rings);                  // uv
    }
  }
  return v;
}

function makeSphereIndices(segs: number, rings: number): number[] {
  const idx: number[] = [];
  for (let r = 0; r < rings; r = r + 1) {
    for (let s = 0; s < segs; s = s + 1) {
      const a = r * (segs + 1) + s;
      const b = a + segs + 1;
      idx.push(a, b, a + 1);
      idx.push(b, b + 1, a + 1);
    }
  }
  return idx;
}

function makePlaneVertices(w: number, d: number): number[] {
  const hw = w / 2;
  const hd = d / 2;
  return [
    -hw, 0, -hd,  0, 1, 0,  1, 1, 1, 1,  0, 0,
     hw, 0, -hd,  0, 1, 0,  1, 1, 1, 1,  1, 0,
     hw, 0,  hd,  0, 1, 0,  1, 1, 1, 1,  1, 1,
    -hw, 0,  hd,  0, 1, 0,  1, 1, 1, 1,  0, 1,
  ];
}

function makePlaneIndices(): number[] {
  return [0, 2, 1, 0, 3, 2];
}

// ---- Shared mesh handles (initialized after initWindow) ----

let sphereHandle = 0;
let cubeHandle = 0;

function initSharedMeshes(): void {
  const sv = makeSphereVertices(0.5, 24, 16);
  const si = makeSphereIndices(24, 16);
  sphereHandle = createMesh(sv, si).handle;
  cubeHandle = genMeshCube(1, 1, 1).handle;
}

// ---- Helpers ----

function placeNode(
  modelHandle: number, meshIdx: number,
  px: number, py: number, pz: number,
  sx: number, sy: number, sz: number,
): number {
  const node = createSceneNode();
  attachModelToNode(node, modelHandle, meshIdx);
  let m = mat4Identity();
  m = mat4Translate(m, { x: px, y: py, z: pz });
  m = mat4Scale(m, { x: sx, y: sy, z: sz });
  setSceneNodeTransform(node, m);
  setSceneNodeCastShadow(node, true);
  setSceneNodeReceiveShadow(node, true);
  return node;
}

function placeSphere(
  px: number, py: number, pz: number,
  scale: number,
  cr: number, cg: number, cb: number,
  roughness: number, metalness: number,
): number {
  const node = placeNode(sphereHandle, 0, px, py, pz, scale, scale, scale);
  setSceneNodeColor(node, cr, cg, cb);
  setSceneNodePbr(node, roughness, metalness);
  return node;
}

function placeCube(
  px: number, py: number, pz: number,
  sx: number, sy: number, sz: number,
  cr: number, cg: number, cb: number,
  roughness: number, metalness: number,
): number {
  const node = placeNode(cubeHandle, 0, px, py, pz, sx, sy, sz);
  setSceneNodeColor(node, cr, cg, cb);
  setSceneNodePbr(node, roughness, metalness);
  return node;
}

// ============================================================
// Zone 1: PBR Material Gallery (centered at origin)
// ============================================================
// 7 columns = roughness steps, 5 rows = different materials.
// This is the single most important zone for renderer quality
// verification. When SSR, GI, or better BRDF lands, the
// difference will be immediately visible on these spheres.

function setupMaterialGallery(): void {
  const roughSteps = [0.05, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0];

  // Row colors [r, g, b] and metalness
  // Metals (metalness=1): reflections, Fresnel, specular color = albedo
  // Dielectrics (metalness=0): diffuse dominant, white specular
  const rows: number[][] = [
    //  R     G     B    metal
    [1.00, 0.76, 0.33, 1.0],  // Gold
    [0.95, 0.64, 0.54, 1.0],  // Copper
    [0.91, 0.92, 0.92, 1.0],  // Silver / Chrome
    [0.80, 0.05, 0.05, 0.0],  // Red plastic
    [0.90, 0.88, 0.82, 0.0],  // White ceramic
  ];

  // Pedestal under the gallery
  placeCube(0, -0.15, 0, 14, 0.3, 10, 0.15, 0.15, 0.17, 0.9, 0.0);

  for (let row = 0; row < 5; row = row + 1) {
    const mat = rows[row];
    for (let col = 0; col < 7; col = col + 1) {
      const x = (col - 3) * 1.8;
      const z = (row - 2) * 1.8;
      placeSphere(x, 0.6, z, 1.0, mat[0], mat[1], mat[2], roughSteps[col], mat[3]);
    }
  }
}

// ============================================================
// Zone 2: Multi-Light Arena (centered at x=28)
// ============================================================
// White/gray objects lit by multiple colored point lights.
// Tests light accumulation, specular from multiple sources,
// shadow interaction. Animated lights orbit the center.

function setupLightArena(): void {
  const cx = 28.0;

  // Central pillar
  placeCube(cx, 2.5, 0, 1.5, 5, 1.5, 0.85, 0.85, 0.85, 0.3, 0.0);

  // Surrounding spheres (white, varying roughness)
  const count = 8;
  for (let i = 0; i < count; i = i + 1) {
    const angle = TWO_PI * i / count;
    const x = cx + Math.cos(angle) * 6;
    const z = Math.sin(angle) * 6;
    const roughness = 0.05 + 0.95 * i / (count - 1);
    placeSphere(x, 0.8, z, 1.4, 0.9, 0.9, 0.9, roughness, 0.0);
  }

  // Floor disc (dark, metallic — to catch reflections when SSR lands)
  placeCube(cx, -0.05, 0, 16, 0.1, 16, 0.05, 0.05, 0.07, 0.1, 1.0);

  // Smaller metallic accents
  placeSphere(cx - 3, 0.5, -3, 0.8, 0.95, 0.93, 0.88, 0.05, 1.0);
  placeSphere(cx + 3, 0.5, 3, 0.8, 1.0, 0.76, 0.33, 0.1, 1.0);
  placeSphere(cx, 0.5, -5, 0.8, 0.95, 0.64, 0.54, 0.15, 1.0);
}

// ============================================================
// Zone 3: Shadow Quality (centered at x=-28)
// ============================================================
// Pillars at different heights, small objects casting fine
// shadows, floating platforms. Tests cascade transitions,
// shadow resolution, PCF quality, contact shadows.

function setupShadowTest(): void {
  // Single pillar at origin on a white floor. Simplest possible
  // shadow test.
  const floor = placeCube(0, -0.05, 0, 40, 0.1, 40, 1.0, 1.0, 1.0, 0.8, 0.0);
  setSceneNodeCastShadow(floor, false);

  placeCube(0, 4, 0, 1.5, 8, 1.5, 0.5, 0.5, 0.5, 0.5, 0.0);
}

// ============================================================
// Zone 4: Water Surface (centered at z=25)
// ============================================================
// Large water plane with objects above and partially submerged.
// Tests water material, and will test reflections, refraction,
// caustics, foam, and volumetric fog when those land.

function setupWater(): void {
  const cz = 25.0;

  // Water plane
  const waterNode = createSceneNode();
  const wv = makePlaneVertices(30, 20);
  const wi = makePlaneIndices();
  updateSceneNodeGeometry(waterNode, wv, wi);
  const wm = mat4Translate(mat4Identity(), { x: 0, y: 0.2, z: cz });
  setSceneNodeTransform(waterNode, wm);
  setSceneNodeWaterMaterial(waterNode, 0.15, 1.5, 0.1, 0.3, 0.5, 0.6);
  setSceneNodeReceiveShadow(waterNode, true);

  // Rocks / objects sticking out of water
  placeSphere(3, 0.8, cz - 3, 1.5, 0.45, 0.42, 0.4, 0.8, 0.0);
  placeSphere(-4, 0.5, cz + 2, 1.2, 0.5, 0.45, 0.4, 0.9, 0.0);
  placeSphere(6, 1.2, cz + 4, 2.0, 0.4, 0.38, 0.35, 0.7, 0.0);

  // Pillar rising from water
  placeCube(0, 2, cz, 1, 4, 1, 0.55, 0.5, 0.48, 0.4, 0.0);

  // Metallic sphere floating above (for reflection testing)
  placeSphere(0, 4.5, cz, 1.0, 0.95, 0.93, 0.88, 0.05, 1.0);

  // Shore / ground under water extending outward
  placeCube(0, -0.5, cz - 12, 40, 0.1, 8, 0.6, 0.55, 0.45, 0.8, 0.0);
}

// ============================================================
// Zone 5: Geometry Density (centered at x=25, z=-25)
// ============================================================
// 10x10 grid of small objects to stress-test draw calls,
// culling, batching, and future LOD / virtualized geometry.

function setupGeometryDensity(): void {
  const cx = 25.0;
  const cz = -25.0;

  // Ground
  placeCube(cx, -0.05, cz, 22, 0.1, 22, 0.2, 0.22, 0.25, 0.6, 0.0);

  for (let row = 0; row < 10; row = row + 1) {
    for (let col = 0; col < 10; col = col + 1) {
      const x = cx - 9 + col * 2;
      const z = cz - 9 + row * 2;
      const height = 0.5 + Math.sin(row * 1.3 + col * 0.7) * 0.3;

      // Alternate cubes and spheres
      if ((row + col) % 2 === 0) {
        const r = 0.3 + col * 0.07;
        const g = 0.3 + row * 0.07;
        const b = 0.5;
        placeCube(x, height / 2, z, 0.8, height, 0.8, r, g, b, 0.5, 0.0);
      } else {
        const r = 0.5;
        const g = 0.3 + col * 0.07;
        const b = 0.3 + row * 0.07;
        placeSphere(x, height / 2 + 0.25, z, 0.6, r, g, b, 0.4, 0.2);
      }
    }
  }
}

// ============================================================
// Zone 6: Thin Geometry / Anti-Aliasing (centered at x=-25, z=-25)
// ============================================================
// Thin bars at various angles to test aliasing. When TAA,
// MSAA, or TSR lands, improvements will be immediately visible
// on this zone — shimmering and crawling on thin geometry is
// the hardest temporal stability test.

function setupThinGeometry(): void {
  const cx = -25.0;
  const cz = -25.0;

  // Ground
  placeCube(cx, -0.05, cz, 20, 0.1, 16, 0.3, 0.3, 0.32, 0.5, 0.0);

  // Vertical fence posts
  for (let i = 0; i < 20; i = i + 1) {
    const x = cx - 9.5 + i * 1.0;
    placeCube(x, 1.0, cz - 5, 0.06, 2.0, 0.06, 0.35, 0.35, 0.38, 0.4, 1.0);
  }

  // Horizontal fence rail
  placeCube(cx, 1.8, cz - 5, 20, 0.08, 0.08, 0.35, 0.35, 0.38, 0.4, 1.0);
  placeCube(cx, 0.6, cz - 5, 20, 0.08, 0.08, 0.35, 0.35, 0.38, 0.4, 1.0);

  // Diagonal bars (worst case for aliasing)
  for (let i = 0; i < 12; i = i + 1) {
    const x = cx - 5.5 + i * 1.0;
    const node = createSceneNode();
    attachModelToNode(node, cubeHandle, 0);
    setSceneNodeColor(node, 0.6, 0.6, 0.62);
    setSceneNodePbr(node, 0.3, 1.0);
    setSceneNodeCastShadow(node, true);
    setSceneNodeReceiveShadow(node, true);
    let m = mat4Identity();
    m = mat4Translate(m, { x: x, y: 1.5, z: cz + 2 });
    m = mat4RotateY(m, 0.3 + i * 0.15);
    m = mat4Scale(m, { x: 0.05, y: 3.0, z: 0.05 });
    setSceneNodeTransform(node, m);
  }

  // Grid of very thin wires (subpixel test)
  for (let i = 0; i < 30; i = i + 1) {
    const x = cx - 7 + i * 0.5;
    placeCube(x, 1.0, cz + 6, 0.02, 2.0, 0.02, 0.7, 0.7, 0.72, 0.2, 1.0);
  }
}

// ============================================================
// Ground plane & sky reference objects
// ============================================================

function setupGround(): void {
  // Ground plane (flat box covering the whole scene)
  placeCube(0, -0.3, 0, 120, 0.1, 120, 0.25, 0.27, 0.22, 0.85, 0.0);

  // Bright sphere high up — tests tone mapping / bloom when those land
  placeSphere(0, 20, -10, 3.0, 5.0, 4.8, 4.0, 0.1, 0.0);

  // Dark reference sphere (test shadow/AO in crevices)
  placeSphere(10, 0.5, 10, 1.0, 0.02, 0.02, 0.02, 0.9, 0.0);

  // Mid-gray reference (linear 0.18 — 18% gray card for exposure)
  placeCube(12, 0.75, 10, 1.5, 1.5, 0.1, 0.18, 0.18, 0.18, 0.9, 0.0);
}

// ============================================================
// Zone 7: glTF Reality Check (DamagedHelmet)
// ============================================================
// Canonical Khronos PBR test model. If renderer upgrades look good
// here, they'll look good on real game art. This is where procedural
// cubes/spheres stop being a meaningful test.
//
// Uses immediate-mode drawModel() inside the render loop, matching
// how david/garden render glTF — the scene graph attach path has
// issues we haven't debugged yet.

let gltfModel = { handle: 0, meshCount: 0, materialCount: 0, transform: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] };

function setupGltfModel(): void {
  gltfModel = loadModel("assets/DamagedHelmet.glb");
  // Pedestal (still rendered via scene graph)
  placeCube(0, 0.1, -30, 6, 0.2, 6, 0.15, 0.15, 0.17, 0.9, 0.0);
}

// ============================================================
// Camera state
// ============================================================

// Spawn high above zone 3 looking down at the floor. From this
// angle the shadows cast by the pillars appear clearly as
// separate dark streaks on the white floor, rather than joining
// visually onto each pillar's base.
let camX = 0.0;
let camY = 4.0;
let camZ = 16.0;
let camYaw = 0.0;
let camPitch = -0.2;
let cursorLocked = true;

// Zone teleport positions [x, y, z, yaw]
const zones: number[][] = [
  [0, 4, 16, 0],              // 1: Material gallery
  [28, 4, 16, 0],             // 2: Light arena
  [-28, 4, 16, 0],            // 3: Shadow test
  [0, 6, 40, 0],              // 4: Water
  [25, 6, -14, PI],           // 5: Geometry density
  [-25, 4, -14, PI],          // 6: Thin geometry
  [0, 3.5, -24, PI],          // 7: glTF reality check
];

const zoneNames: string[] = [
  "PBR Materials",
  "Multi-Light",
  "Shadows",
  "Water",
  "Geometry Density",
  "Thin Geo / AA",
  "glTF Reality Check",
];

// ============================================================
// Init
// ============================================================

// Use --res override when provided (validation/CI), otherwise the
// interactive walkthrough size.
const winW = headlessResW > 0 ? headlessResW : SCREEN_W;
const winH = headlessResH > 0 ? headlessResH : SCREEN_H;
initWindow(winW, winH, "Bloom Renderer Test", 0);
setTargetFPS(60);
if (!headlessMode) {
  disableCursor();
}


// Initialize shared meshes (requires engine)
initSharedMeshes();

// In headless mode, seed the clear color from the HDR env map so
// the background color matches the path-traced reference. Proper
// sky rendering (equirect sample per background pixel) comes in a
// follow-up — this first-pass "solid color from env average" already
// closes most of the background-gap RMSE.
// Always load the HDR env map — needed for IBL and for the sky pass
// to render anything other than the default clear color.
setEnvClearFromHdr("assets/outdoor.hdr");

// Build the scene. In headless mode we draw ONLY the glTF model at
// origin so the view matches what bloom-reference renders from the
// same spec. Interactive mode builds all seven zones for walking
// around and eyeballing.
if (headlessMode) {
  setupGltfModel();
} else {
  setupGround();
  setupMaterialGallery();
  setupLightArena();
  setupShadowTest();
  setupWater();
  setupGeometryDensity();
  setupThinGeometry();
  setupGltfModel();
}

if (!headlessMode) {
  enableShadows();
}

if (!headlessMode) {
  setEnvIntensity(0.3);
  setAutoExposure(true);
  setFog(0.65, 0.72, 0.80, 0.02, 0.0, 0.18);
  setVignette(0.35, 0.30);
  setFilmGrain(0.025);
  setChromaticAberration(0.0025);
  setSunShafts(0.6, 0.97, 1.0, 0.92, 0.78);
}

// ============================================================
// Main loop
// ============================================================

let headlessFrame = 0;
const HEADLESS_WARMUP_FRAMES = 30;

while (!windowShouldClose()) {
  const dt = getDeltaTime();
  const t = getTime();

  // ---- Camera controls ----

  if (headlessMode) {
    // No input — the spec's camera pose is applied directly below.
  } else if (cursorLocked) {
    camYaw = camYaw - getMouseDeltaX() * MOUSE_SENS;
    camPitch = camPitch - getMouseDeltaY() * MOUSE_SENS;
    camPitch = clamp(camPitch, -1.4, 1.4);
  }

  // Movement
  const speed = isKeyDown(Key.LEFT_SHIFT) ? MOVE_SPEED * SPRINT_MULT : MOVE_SPEED;
  const fwdX = -Math.sin(camYaw);
  const fwdZ = -Math.cos(camYaw);
  const rightX = Math.cos(camYaw);
  const rightZ = -Math.sin(camYaw);

  if (isKeyDown(Key.W) || isKeyDown(Key.UP)) {
    camX = camX + fwdX * speed * dt;
    camZ = camZ + fwdZ * speed * dt;
  }
  if (isKeyDown(Key.S) || isKeyDown(Key.DOWN)) {
    camX = camX - fwdX * speed * dt;
    camZ = camZ - fwdZ * speed * dt;
  }
  if (isKeyDown(Key.A) || isKeyDown(Key.LEFT)) {
    camX = camX - rightX * speed * dt;
    camZ = camZ - rightZ * speed * dt;
  }
  if (isKeyDown(Key.D) || isKeyDown(Key.RIGHT)) {
    camX = camX + rightX * speed * dt;
    camZ = camZ + rightZ * speed * dt;
  }
  if (isKeyDown(Key.SPACE)) {
    camY = camY + speed * dt;
  }
  if (isKeyDown(Key.C)) {
    camY = camY - speed * dt;
  }

  // Cursor toggle
  if (isKeyPressed(Key.TAB)) {
    cursorLocked = !cursorLocked;
    if (cursorLocked) {
      disableCursor();
    } else {
      enableCursor();
    }
  }

  // F12 → screenshot for bloom-diff comparison against bloom-reference
  if (isKeyPressed(Key.F12)) {
    takeScreenshot("renderer-test-screenshot.png");
  }

  // Zone teleport (keys 1-7)
  for (let zi = 0; zi < 7; zi = zi + 1) {
    // Key.ONE = 49, Key.TWO = 50, ...
    if (isKeyPressed(49 + zi)) {
      const z = zones[zi];
      camX = z[0];
      camY = z[1];
      camZ = z[2];
      camYaw = z[3];
      camPitch = -0.2;
    }
  }

  // Look target
  const lookX = camX + Math.cos(camPitch) * (-Math.sin(camYaw)) * 100;
  const lookY = camY + Math.sin(camPitch) * 100;
  const lookZ = camZ + Math.cos(camPitch) * (-Math.cos(camYaw)) * 100;

  // ---- Rendering ----

  beginDrawing();
  // Interactive mode: explicit dark clear color matching the spec
  // scenes. Headless mode uses the HDR env average seeded at init
  // time, so we skip clearBackground to preserve that color.
  if (!headlessMode) {
    clearBackground({ r: 12, g: 14, b: 22, a: 255 });
  }

  // Global lighting — set only outside headless mode. The shared
  // helmet spec (`specs/helmet.json`) has `sun: null`, so the
  // bloom-reference path tracer renders env-only. Adding a sun +
  // ambient here would skew the realtime brighter and warmer than
  // the reference for no good reason during validation.
  if (!headlessMode) {
    setAmbientLight({ r: 70, g: 80, b: 100, a: 255 }, 0.25);
    // ~45° afternoon sun — shadows roughly pillar-length.
    setDirectionalLight(
      { x: -0.5, y: 0.7, z: 0.3 },
      { r: 255, g: 248, b: 235, a: 255 },
      2.0,
    );
  }

  // Zone 2 animated point lights — irrelevant in headless mode
  // (camera is on the helmet, point lights are 28+ units away with
  // range 18 so they can't reach), and adding them risks unstable
  // diff numbers when light state leaks into the shader's counters.
  if (!headlessMode) {
    const lightRadius = 7.0;
    addPointLight(
      28 + Math.cos(t * 0.8) * lightRadius,
      3.0,
      Math.sin(t * 0.8) * lightRadius,
      18, 1.0, 0.25, 0.08, 4.0
    );
    addPointLight(
      28 + Math.cos(t * 0.8 + 2.09) * lightRadius,
      3.0,
      Math.sin(t * 0.8 + 2.09) * lightRadius,
      18, 0.08, 0.4, 1.0, 4.0
    );
    addPointLight(
      28 + Math.cos(t * 0.8 + 4.19) * lightRadius,
      3.0,
      Math.sin(t * 0.8 + 4.19) * lightRadius,
      18, 0.15, 1.0, 0.25, 4.0
    );

    // Additional warm fill light near shadow zone
    addPointLight(-28, 6, 5, 20, 1.0, 0.9, 0.7, 1.5);
  }

  // 3D rendering. Headless mode uses the spec's camera/fov verbatim
  // and draws the helmet at origin so both renderers see the same
  // transform. Interactive mode uses the FPS camera and zone 7's
  // world-space position.
  if (headlessMode) {
    beginMode3D({
      position: { x: headlessCamX, y: headlessCamY, z: headlessCamZ },
      target: { x: headlessTargetX, y: headlessTargetY, z: headlessTargetZ },
      up: { x: 0, y: 1, z: 0 },
      fovy: headlessFov,
      projection: "perspective",
    });
    // TEMP: visual sanity check — a bright cube at origin. If this
    // shows but the glTF helmet doesn't, the issue is in the glTF
    // model load, not the render pass.
    drawCube({ x: 0, y: 0, z: 0 }, 0.8, 0.8, 0.8, { r: 255, g: 100, b: 100, a: 255 });
    if (gltfModel.handle !== 0) {
      drawModel(gltfModel, { x: 0, y: 0, z: 0 }, 1.0, { r: 255, g: 255, b: 255, a: 255 });
    }
  } else {
    beginMode3D({
      position: { x: camX, y: camY, z: camZ },
      target: { x: lookX, y: lookY, z: lookZ },
      up: { x: 0, y: 1, z: 0 },
      fovy: 60,
      projection: "perspective",
    });
    drawGrid(60, 2.0);
    // Zone 7: glTF helmet drawn in immediate mode
    if (gltfModel.handle !== 0) {
      drawModel(gltfModel, { x: 0, y: 2.5, z: -30 }, 2.0, { r: 255, g: 255, b: 255, a: 255 });
    }
  }
  endMode3D();

  // ---- HUD ---- (skipped in headless — reference never shows text)

  if (!headlessMode) {
    drawText("Bloom Renderer Test", 10, 10, 22, WHITE);
    drawText("FPS: " + getFPS().toString(), 10, 38, 16, LGRAY);

    drawText("WASD move / Mouse look / Shift sprint / Tab cursor", 10, SCREEN_H - 50, 14, GRAY);
    drawText("Press 1-6 to teleport to zones", 10, SCREEN_H - 30, 14, GRAY);

    // Zone legend
    const legendX = SCREEN_W - 220;
    drawText("Zones:", legendX, 10, 16, LGRAY);
    for (let i = 0; i < 7; i = i + 1) {
      const label = (i + 1).toString() + "  " + zoneNames[i];
      drawText(label, legendX, 32 + i * 20, 14, GRAY);
    }
  }

  // Headless: after warmup frames, capture the frame to --out and
  // exit. We request the screenshot BEFORE endDrawing() so the
  // renderer's pending-screenshot path picks it up during present.
  if (headlessMode) {
    headlessFrame = headlessFrame + 1;
    if (headlessFrame === HEADLESS_WARMUP_FRAMES && headlessOutPath.length > 0) {
      takeScreenshot(headlessOutPath);
    }
    if (headlessFrame > HEADLESS_WARMUP_FRAMES) {
      endDrawing();
      break;
    }
  } else if (interactiveCaptureFrames > 0) {
    headlessFrame = headlessFrame + 1;
    if (headlessFrame === interactiveCaptureFrames) {
      takeScreenshot(interactiveCapturePath);
    }
    if (headlessFrame > interactiveCaptureFrames) {
      endDrawing();
      break;
    }
  }

  endDrawing();
}

// Clean shutdown — headless mode relies on this so the PNG flushes
// to disk before the process exits.
closeWindow();
