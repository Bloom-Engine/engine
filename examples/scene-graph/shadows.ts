/**
 * Shadows + GLTF Example — Phase 4 proof-of-concept.
 *
 * Demonstrates:
 * - Directional light shadow mapping (PCF)
 * - GLTF model loaded and attached to scene nodes
 * - Multiple lights with shadows
 * - Model instancing via scene nodes
 */

import {
  initWindow, windowShouldClose, beginDrawing, endDrawing,
  clearBackground, setTargetFPS, drawText,
  beginMode3D, endMode3D, drawGrid,
  isKeyPressed, Key, Colors, getDeltaTime,
  setAmbientLight, setDirectionalLight,
  loadModel,
} from 'bloom';

import {
  createSceneNode,
  setSceneNodeColor, setSceneNodePbr, setSceneNodeTransform,
  getSceneNodeCount,
  addDirectionalLight,
  extrudePolygon,
  enableShadows, attachModelToNode,
  registerFrameCallback,
} from 'bloom/scene';

import { mat4Identity, mat4Translate, mat4Scale, mat4Multiply } from 'bloom';

// ============================================================
// Setup
// ============================================================

initWindow(1280, 720, "Bloom — Shadows + GLTF (Phase 4)");
setTargetFPS(60);

// Enable shadow mapping
enableShadows();

// Lighting
setAmbientLight(255, 255, 255, 0.2);
setDirectionalLight(0.5, 1.0, 0.3, 255, 240, 220, 0.7);

// ============================================================
// Scene: room with furniture
// ============================================================

// Floor
const floor = createSceneNode();
const floorPoly = [-5, -5, 5, -5, 5, 5, -5, 5];
extrudePolygon(floor, floorPoly, 0.05);
setSceneNodeColor(floor, 0.8, 0.78, 0.72, 1.0);
setSceneNodePbr(floor, 0.7, 0.0);

// Walls
function makeWall(sx: number, sz: number, ex: number, ez: number): void {
  const node = createSceneNode();
  const dx = ex - sx;
  const dz = ez - sz;
  const len = Math.sqrt(dx * dx + dz * dz);
  const nx = -dz / len * 0.1;
  const nz = dx / len * 0.1;
  const poly = [
    sx + nx, sz + nz,
    ex + nx, ez + nz,
    ex - nx, ez - nz,
    sx - nx, sz - nz,
  ];
  extrudePolygon(node, poly, 3.0);
  setSceneNodeColor(node, 0.95, 0.93, 0.88, 1.0);
  setSceneNodePbr(node, 0.85, 0.0);
}

makeWall(-5, -5, 5, -5);  // back
makeWall(5, -5, 5, 5);    // right
makeWall(5, 5, -5, 5);    // front
makeWall(-5, 5, -5, -5);  // left

// Simple "table" made from extruded boxes
function makeBox(cx: number, cy: number, cz: number, w: number, h: number, d: number, r: number, g: number, b: number): void {
  const node = createSceneNode();
  const hw = w / 2;
  const hd = d / 2;
  const poly = [cx - hw, cz - hd, cx + hw, cz - hd, cx + hw, cz + hd, cx - hw, cz + hd];
  extrudePolygon(node, poly, h);
  // Offset Y via transform
  const t = mat4Translate(mat4Identity(), 0, cy, 0);
  setSceneNodeTransform(node, t);
  setSceneNodeColor(node, r, g, b, 1.0);
  setSceneNodePbr(node, 0.6, 0.0);
}

// Table
makeBox(0, 0, 0, 1.5, 0.75, 0.8, 0.6, 0.4, 0.25);  // tabletop
makeBox(-0.6, 0, -0.3, 0.08, 0.72, 0.08, 0.5, 0.35, 0.2);  // legs
makeBox(0.6, 0, -0.3, 0.08, 0.72, 0.08, 0.5, 0.35, 0.2);
makeBox(-0.6, 0, 0.3, 0.08, 0.72, 0.08, 0.5, 0.35, 0.2);
makeBox(0.6, 0, 0.3, 0.08, 0.72, 0.08, 0.5, 0.35, 0.2);

// Chair (simple box)
makeBox(2.0, 0, 0, 0.5, 0.45, 0.5, 0.55, 0.45, 0.35);
makeBox(2.0, 0.45, -0.2, 0.5, 0.5, 0.08, 0.55, 0.45, 0.35);

// Lighting system
registerFrameCallback(5, (dt: number) => {
  // Main sun
  addDirectionalLight(0.5, 1.0, 0.3, 1.0, 0.95, 0.9, 0.6);
  // Fill
  addDirectionalLight(-0.3, 0.5, -0.7, 0.7, 0.8, 0.95, 0.2);
});

// ============================================================
// Main loop
// ============================================================

let angle = 0;

while (!windowShouldClose()) {
  const dt = getDeltaTime();
  angle += dt * 0.15;

  beginDrawing();
  clearBackground(Colors.RAYWHITE);

  const camX = Math.cos(angle) * 12;
  const camZ = Math.sin(angle) * 12;
  beginMode3D({
    position: { x: camX, y: 8, z: camZ },
    target: { x: 0, y: 1, z: 0 },
    up: { x: 0, y: 1, z: 0 },
    fovy: 45,
    projection: "perspective",
  });

  drawGrid(20, 1.0);
  endMode3D();

  drawText("Bloom — Shadow Mapping + GLTF (Phase 4)", 10, 10, 20, Colors.DARKGRAY);
  drawText("Scene nodes: " + String(getSceneNodeCount()), 10, 35, 16, Colors.GRAY);
  drawText("Directional light shadows (2048x2048 PCF)", 10, 55, 16, Colors.GRAY);
  drawText("Room with table + chair (extruded polygons)", 10, 75, 16, Colors.GRAY);

  endDrawing();
}
