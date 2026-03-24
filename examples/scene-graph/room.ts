/**
 * Room Scene Example — Phase 2 proof-of-concept.
 *
 * Demonstrates the Pascal Editor's architecture compiled natively:
 * - Zustand-like store (flat node dictionary + dirty tracking)
 * - sceneRegistry (Map<nodeId, BloomHandle>)
 * - System pattern (frame callbacks with priority ordering)
 * - Polygon extrusion (walls and slabs via Bloom's earcut-based extrusion)
 * - CSG box subtraction (door cutout in wall)
 * - Multiple lights (3 directional + point lights)
 */

import {
  initWindow, windowShouldClose, beginDrawing, endDrawing,
  clearBackground, setTargetFPS, drawText,
  beginMode3D, endMode3D, drawGrid,
  isKeyPressed, Key, Colors, getDeltaTime,
  setAmbientLight, setDirectionalLight,
} from 'bloom';

import {
  createSceneNode, destroySceneNode,
  setSceneNodeTransform, updateSceneNodeGeometry,
  setSceneNodeColor, setSceneNodePbr, setSceneNodeVisible,
  getSceneNodeCount,
  registerFrameCallback, addDirectionalLight, addPointLight,
  extrudePolygon, subtractBox,
} from 'bloom/scene';

import { mat4Identity } from 'bloom';

// ============================================================
// Zustand-like store (simplified useScene)
// ============================================================

interface WallNode {
  id: string;
  type: 'wall';
  start: [number, number];
  end: [number, number];
  thickness: number;
  height: number;
  children: string[]; // door IDs
}

interface SlabNode {
  id: string;
  type: 'slab';
  polygon: [number, number][];
  elevation: number;
}

interface DoorNode {
  id: string;
  type: 'door';
  parentId: string;
  width: number;
  height: number;
  position: number; // 0-1 along wall
}

type AnyNode = WallNode | SlabNode | DoorNode;

const nodes: Map<string, AnyNode> = new Map();
const dirtyNodes: Set<string> = new Set();

function createNode(node: AnyNode): void {
  nodes.set(node.id, node);
  dirtyNodes.add(node.id);
}

function markDirty(id: string): void {
  dirtyNodes.add(id);
}

function clearDirty(id: string): void {
  dirtyNodes.delete(id);
}

// ============================================================
// Scene Registry (Map<nodeId, BloomHandle>)
// ============================================================

const sceneRegistry: Map<string, number> = new Map();

function ensureSceneNode(id: string): number {
  let handle = sceneRegistry.get(id);
  if (handle === undefined) {
    handle = createSceneNode();
    sceneRegistry.set(id, handle);
  }
  return handle;
}

// ============================================================
// Wall geometry generator (pure TypeScript math)
// ============================================================

function generateWallVertices(
  wall: WallNode,
): number[] {
  const [sx, sz] = wall.start;
  const [ex, ez] = wall.end;
  const t = wall.thickness;
  const h = wall.height;

  const dx = ex - sx;
  const dz = ez - sz;
  const len = Math.sqrt(dx * dx + dz * dz);
  if (len < 0.001) return [];

  // Normal perpendicular to wall
  const nx = -dz / len;
  const nz = dx / len;
  const hx = nx * t * 0.5;
  const hz = nz * t * 0.5;

  // 2D polygon footprint for extrusion
  // 4 corners of wall cross-section in XZ plane
  const polygon: number[] = [
    sx + hx, sz + hz,
    ex + hx, ez + hz,
    ex - hx, ez - hz,
    sx - hx, sz - hz,
  ];

  return polygon;
}

// ============================================================
// Systems (frame callbacks, matching R3F useFrame pattern)
// ============================================================

// SlabSystem (priority 1) — generates slab geometry for dirty slabs
function slabSystem(dt: number): void {
  for (const id of dirtyNodes) {
    const node = nodes.get(id);
    if (!node || node.type !== 'slab') continue;

    const handle = ensureSceneNode(id);
    const slab = node as SlabNode;

    // Flatten polygon for extrusion
    const flat: number[] = [];
    for (const [x, z] of slab.polygon) {
      flat.push(x);
      flat.push(z);
    }

    // Extrude polygon to create slab
    extrudePolygon(handle, flat, slab.elevation);

    // Gray floor material
    setSceneNodeColor(handle, 0.75, 0.75, 0.70, 1.0);
    setSceneNodePbr(handle, 0.6, 0.0);

    clearDirty(id);
  }
}

// WallSystem (priority 4) — generates wall geometry with door cutouts
function wallSystem(dt: number): void {
  for (const id of dirtyNodes) {
    const node = nodes.get(id);
    if (!node || node.type !== 'wall') continue;

    const handle = ensureSceneNode(id);
    const wall = node as WallNode;

    // Generate wall polygon
    const polygon = generateWallVertices(wall);
    if (polygon.length === 0) continue;

    // Extrude wall polygon
    extrudePolygon(handle, polygon, wall.height);

    // Apply door cutouts via CSG box subtraction
    for (const childId of wall.children) {
      const child = nodes.get(childId);
      if (!child || child.type !== 'door') continue;
      const door = child as DoorNode;

      // Compute door position along wall
      const [sx, sz] = wall.start;
      const [ex, ez] = wall.end;
      const dx = ex - sx;
      const dz = ez - sz;
      const len = Math.sqrt(dx * dx + dz * dz);
      const nx = -dz / len;
      const nz = dx / len;

      // Door center along wall
      const cx = sx + dx * door.position;
      const cz = sz + dz * door.position;

      // Cutout box (slightly wider than wall thickness for clean cut)
      const halfW = door.width * 0.5;
      const wt = wall.thickness * 1.5;
      subtractBox(handle,
        cx - halfW * (dx / len) - nx * wt, 0.0, cz - halfW * (dz / len) - nz * wt,
        cx + halfW * (dx / len) + nx * wt, door.height, cz + halfW * (dz / len) + nz * wt,
      );
    }

    // White wall material
    setSceneNodeColor(handle, 0.95, 0.95, 0.92, 1.0);
    setSceneNodePbr(handle, 0.8, 0.0);

    clearDirty(id);
  }
}

// LightSystem (priority 5) — sets up lighting each frame
function lightSystem(dt: number): void {
  // Main directional light (sun-like)
  addDirectionalLight(0.5, 1.0, 0.3, 1.0, 0.95, 0.9, 0.7);
  // Fill light
  addDirectionalLight(-0.3, 0.5, -0.7, 0.8, 0.85, 0.95, 0.3);
  // Rim light
  addDirectionalLight(0.0, -0.2, 1.0, 0.9, 0.9, 1.0, 0.15);
}

// ============================================================
// Scene setup
// ============================================================

initWindow(1280, 720, "Bloom — Room Scene (Phase 2)");
setTargetFPS(60);

// Set ambient light
setAmbientLight(255, 255, 255, 0.25);
setDirectionalLight(0.5, 1.0, 0.3, 255, 240, 230, 0.6);

// Create scene nodes (like Pascal Editor's createNode)
const ROOM_W = 6;
const ROOM_D = 5;
const WALL_H = 3;
const WALL_T = 0.2;
const SLAB_T = 0.15;

// Floor slab
createNode({
  id: 'slab_1',
  type: 'slab',
  polygon: [[0, 0], [ROOM_W, 0], [ROOM_W, ROOM_D], [0, ROOM_D]],
  elevation: SLAB_T,
});

// Back wall
createNode({
  id: 'wall_1',
  type: 'wall',
  start: [0, 0],
  end: [ROOM_W, 0],
  thickness: WALL_T,
  height: WALL_H,
  children: [],
});

// Right wall (with door)
createNode({
  id: 'door_1',
  type: 'door',
  parentId: 'wall_2',
  width: 0.9,
  height: 2.1,
  position: 0.5,
});

createNode({
  id: 'wall_2',
  type: 'wall',
  start: [ROOM_W, 0],
  end: [ROOM_W, ROOM_D],
  thickness: WALL_T,
  height: WALL_H,
  children: ['door_1'],
});

// Front wall
createNode({
  id: 'wall_3',
  type: 'wall',
  start: [ROOM_W, ROOM_D],
  end: [0, ROOM_D],
  thickness: WALL_T,
  height: WALL_H,
  children: [],
});

// Left wall
createNode({
  id: 'wall_4',
  type: 'wall',
  start: [0, ROOM_D],
  end: [0, 0],
  thickness: WALL_T,
  height: WALL_H,
  children: [],
});

// Register systems as frame callbacks (priority-ordered, like useFrame)
registerFrameCallback(1, slabSystem);
registerFrameCallback(4, wallSystem);
registerFrameCallback(5, lightSystem);

// ============================================================
// Main loop
// ============================================================

let angle = 0;

while (!windowShouldClose()) {
  const dt = getDeltaTime();
  angle += dt * 0.2;

  // Toggle wall visibility with 1-4 keys
  if (isKeyPressed(Key.ONE)) {
    const node = nodes.get('wall_1') as WallNode;
    if (node) {
      const handle = sceneRegistry.get('wall_1');
      if (handle !== undefined) {
        setSceneNodeVisible(handle, false);
      }
    }
  }

  beginDrawing();
  clearBackground(Colors.RAYWHITE);

  const camX = 3 + Math.cos(angle) * 12;
  const camZ = 2.5 + Math.sin(angle) * 12;
  beginMode3D({
    position: { x: camX, y: 5, z: camZ },
    target: { x: 3, y: 1.5, z: 2.5 },
    up: { x: 0, y: 1, z: 0 },
    fovy: 45,
    projection: "perspective",
  });

  // Grid (immediate mode, drawn alongside scene graph nodes)
  drawGrid(20, 1.0);

  endMode3D();

  // HUD
  drawText("Bloom — Multi-Object Room (Phase 2)", 10, 10, 20, Colors.DARKGRAY);
  drawText("Scene nodes: " + String(getSceneNodeCount()), 10, 35, 16, Colors.GRAY);
  drawText("4 walls + 1 slab + 1 door cutout", 10, 55, 16, Colors.GRAY);
  drawText("3 directional lights (sun + fill + rim)", 10, 75, 16, Colors.GRAY);
  drawText("Frame callbacks: slab@1, wall@4, light@5", 10, 95, 16, Colors.GRAY);

  endDrawing();
}
