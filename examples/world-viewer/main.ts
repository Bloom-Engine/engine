// world-viewer — the reference consumer of the shared world format.
//
// Loads any `*.world.json` and shows it with a fly camera, going through the
// GENERIC path end to end: loadWorld → instantiateWorld (terrain, entities,
// prefabs, water, rivers) → applyWorldEnvironment every frame. If a world
// looks right here, it will look right in any game that uses the same calls —
// this example exists to keep that path honest (it had zero consumers before;
// every game hand-rolled its own spawn code).
//
//   perry compile main.ts -o world-viewer
//   cd <your-game>   # modelRef paths resolve relative to CWD
//   <path>/world-viewer --world assets/worlds/level.world.json [--prefabs assets/prefabs]
//
// Controls: WASD move, Q/E down/up, hold right mouse to look, Shift = fast.

import {
  initWindow, windowShouldClose, closeWindow,
  beginDrawing, endDrawing, clearBackground,
  beginMode3D, endMode3D, setTargetFPS, getDeltaTime,
  isKeyDown, Key, drawText,
  isMouseButtonDown, MouseButton, getMouseDeltaX, getMouseDeltaY,
  loadModel, Camera3D,
} from 'bloom';
import { readdirSync } from 'fs';
import {
  loadWorld, instantiateWorld, applyWorldEnvironment,
  InstantiateContext, WorldData,
  createPrefabRegistry, registerPrefab, loadPrefab, PrefabRegistry,
} from 'bloom/world';

// ---- args --------------------------------------------------------------------

let worldPath = '';
let prefabsDir = '';
for (let i = 0; i < process.argv.length; i++) {
  if (process.argv[i] === '--world' && i + 1 < process.argv.length) {
    worldPath = process.argv[i + 1];
  }
  if (process.argv[i] === '--prefabs' && i + 1 < process.argv.length) {
    prefabsDir = process.argv[i + 1];
  }
}
if (worldPath.length === 0) {
  console.error('usage: world-viewer --world <path/to/x.world.json> [--prefabs <dir>]');
  process.exit(2);
}

// ---- load ----------------------------------------------------------------------

const world: WorldData = loadWorld(worldPath);

initWindow(1280, 800, 'world-viewer — ' + world.name);
setTargetFPS(60);

// Model cache: modelRef strings are paths relative to the game root, so run
// this viewer from the game's root directory.
const modelHandles = new Map<string, number>();
function getModelHandle(modelRef: string): number {
  const cached = modelHandles.get(modelRef);
  if (cached !== undefined) return cached;
  const model = loadModel(modelRef);
  modelHandles.set(modelRef, model.handle);
  return model.handle;
}

// Prefab registry from --prefabs (optional).
let registry: PrefabRegistry | null = null;
if (prefabsDir.length > 0) {
  registry = createPrefabRegistry();
  let files: string[] = [];
  try {
    files = readdirSync(prefabsDir) as string[];
  } catch (e) {
    console.error('world-viewer: cannot read --prefabs dir ' + prefabsDir);
  }
  for (let i = 0; i < files.length; i++) {
    if (!files[i].endsWith('.prefab.json')) continue;
    try {
      registerPrefab(registry, loadPrefab(prefabsDir + '/' + files[i]));
    } catch (e) {
      console.error('world-viewer: skipping prefab ' + files[i] + ': ' + (e as Error).message);
    }
  }
}

const ctx: InstantiateContext = {
  getModelHandle: getModelHandle,
  prefabRegistry: registry,
};

const result = instantiateWorld(world, ctx);
for (let i = 0; i < result.warnings.length; i++) {
  console.error('world-viewer: ' + result.warnings[i]);
}

// ---- fly camera -----------------------------------------------------------------

// Start behind the world bounds' center, looking at it.
const bcx = (world.bounds.min[0] + world.bounds.max[0]) / 2;
const bcy = (world.bounds.min[1] + world.bounds.max[1]) / 2;
const bcz = (world.bounds.min[2] + world.bounds.max[2]) / 2;
const spanX = world.bounds.max[0] - world.bounds.min[0];
const spanZ = world.bounds.max[2] - world.bounds.min[2];
let span = spanX > spanZ ? spanX : spanZ;
if (span < 10) span = 10;

let camX = bcx;
let camY = bcy + span * 0.35;
let camZ = bcz + span * 0.7;
let yaw = Math.PI;          // Facing -Z, toward the center.
let pitch = -0.35;

while (!windowShouldClose()) {
  const dt = getDeltaTime();

  // Look (hold RMB).
  if (isMouseButtonDown(MouseButton.RIGHT)) {
    yaw -= getMouseDeltaX() * 0.003;
    pitch -= getMouseDeltaY() * 0.003;
    if (pitch > 1.5) pitch = 1.5;
    if (pitch < -1.5) pitch = -1.5;
  }

  const fwdX = Math.sin(yaw) * Math.cos(pitch);
  const fwdY = Math.sin(pitch);
  const fwdZ = Math.cos(yaw) * Math.cos(pitch);
  const rightX = Math.sin(yaw - Math.PI / 2);
  const rightZ = Math.cos(yaw - Math.PI / 2);

  let speed = span * 0.15 * dt;
  if (isKeyDown(Key.LEFT_SHIFT)) speed *= 4;

  if (isKeyDown(Key.W)) { camX += fwdX * speed; camY += fwdY * speed; camZ += fwdZ * speed; }
  if (isKeyDown(Key.S)) { camX -= fwdX * speed; camY -= fwdY * speed; camZ -= fwdZ * speed; }
  if (isKeyDown(Key.A)) { camX -= rightX * speed; camZ -= rightZ * speed; }
  if (isKeyDown(Key.D)) { camX += rightX * speed; camZ += rightZ * speed; }
  if (isKeyDown(Key.Q)) { camY -= speed; }
  if (isKeyDown(Key.E)) { camY += speed; }

  const cam: Camera3D = {
    position: { x: camX, y: camY, z: camZ },
    target: { x: camX + fwdX, y: camY + fwdY, z: camZ + fwdZ },
    up: { x: 0, y: 1, z: 0 },
    fovy: 60,
    projection: 'perspective',
  };

  beginDrawing();
  clearBackground({
    r: Math.floor(world.environment.skyColor[0] * 255),
    g: Math.floor(world.environment.skyColor[1] * 255),
    b: Math.floor(world.environment.skyColor[2] * 255),
    a: 255,
  });

  // The whole point: the renderer clears lighting per frame, so the world's
  // environment + point lights re-apply per frame through the SHARED helper.
  applyWorldEnvironment(world);

  beginMode3D(cam);
  // Scene nodes spawned by instantiateWorld draw themselves (retained mode).
  endMode3D();

  drawText(world.name + '  —  ' + world.entities.length + ' entities, ' +
    world.lights.length + ' lights, ' + world.water.length + ' water, ' +
    world.rivers.length + ' rivers' +
    (result.warnings.length > 0 ? '  (' + result.warnings.length + ' warnings, see console)' : ''),
    12, 12, 18, { r: 255, g: 255, b: 255, a: 220 });
  drawText('WASD move · Q/E down/up · hold RMB look · Shift fast',
    12, 34, 14, { r: 255, g: 255, b: 255, a: 140 });

  endDrawing();
}

closeWindow();
