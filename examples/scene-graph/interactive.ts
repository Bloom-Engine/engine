/**
 * Interactive Wall Editor — Phase 3 proof-of-concept.
 *
 * Demonstrates the Pascal Editor's interaction model compiled natively:
 * - Scene picking (click to select walls)
 * - Wall drawing tool (click to place endpoints)
 * - Camera orbit controls (right-click drag)
 * - Event pipeline: mouse input → raycast → domain event → state update
 */

import {
  initWindow, windowShouldClose, beginDrawing, endDrawing,
  clearBackground, setTargetFPS, drawText,
  beginMode3D, endMode3D, drawGrid, drawRay,
  isKeyPressed, isMouseButtonPressed, isMouseButtonDown,
  getMouseX, getMouseY, getMouseDeltaX, getMouseDeltaY,
  Key, MouseButton, Colors, getDeltaTime,
  setAmbientLight, setDirectionalLight,
} from 'bloom';

import {
  createSceneNode,
  setSceneNodeColor, setSceneNodePbr,
  getSceneNodeCount,
  addDirectionalLight,
  extrudePolygon,
  pickScene,
  registerFrameCallback,
} from 'bloom/scene';

import type { SceneNodeHandle, PickHit } from 'bloom/scene';

// ============================================================
// Store (Zustand-like: flat Map + dirty tracking)
// ============================================================

interface WallData {
  id: string;
  start: [number, number];
  end: [number, number];
  thickness: number;
  height: number;
  handle: SceneNodeHandle;
}

const walls: Map<string, WallData> = new Map();
const dirtyWalls: Set<string> = new Set();
let nextWallId = 1;
let selectedWallId: string | null = null;

// Floor node
const floorHandle = createSceneNode();
const floorPolygon = [-10, -10, 10, -10, 10, 10, -10, 10];
extrudePolygon(floorHandle, floorPolygon, 0.02);
setSceneNodeColor(floorHandle, 0.85, 0.85, 0.82, 1.0);
setSceneNodePbr(floorHandle, 0.7, 0.0);

// Handle → wall ID lookup (for picking)
const handleToWallId: Map<number, string> = new Map();

function createWall(sx: number, sz: number, ex: number, ez: number): string {
  const id = "wall_" + String(nextWallId);
  nextWallId += 1;

  const handle = createSceneNode();
  const wall: WallData = {
    id,
    start: [sx, sz],
    end: [ex, ez],
    thickness: 0.2,
    height: 3.0,
    handle,
  };
  walls.set(id, wall);
  handleToWallId.set(handle, id);
  dirtyWalls.add(id);
  return id;
}

// ============================================================
// Wall System (frame callback, priority 4)
// ============================================================

function wallSystem(dt: number): void {
  for (const id of dirtyWalls) {
    const wall = walls.get(id);
    if (!wall) continue;

    const [sx, sz] = wall.start;
    const [ex, ez] = wall.end;
    const t = wall.thickness;

    const dx = ex - sx;
    const dz = ez - sz;
    const len = Math.sqrt(dx * dx + dz * dz);
    if (len < 0.001) continue;

    const nx = -dz / len;
    const nz = dx / len;
    const hx = nx * t * 0.5;
    const hz = nz * t * 0.5;

    // Wall footprint polygon
    const polygon = [
      sx + hx, sz + hz,
      ex + hx, ez + hz,
      ex - hx, ez - hz,
      sx - hx, sz - hz,
    ];

    extrudePolygon(wall.handle, polygon, wall.height);

    // Color based on selection
    if (wall.id === selectedWallId) {
      setSceneNodeColor(wall.handle, 0.3, 0.6, 1.0, 1.0);
    } else {
      setSceneNodeColor(wall.handle, 0.95, 0.95, 0.92, 1.0);
    }
    setSceneNodePbr(wall.handle, 0.8, 0.0);

    dirtyWalls.delete(id);
  }
}

// Light system (priority 5)
function lightSystem(dt: number): void {
  addDirectionalLight(0.5, 1.0, 0.3, 1.0, 0.95, 0.9, 0.6);
  addDirectionalLight(-0.3, 0.5, -0.7, 0.8, 0.85, 0.95, 0.25);
}

// ============================================================
// Camera orbit controls
// ============================================================

let camAngle = 0.5;
let camPitch = 0.4;
let camDist = 15.0;
let camTargetX = 0.0;
let camTargetZ = 0.0;

function updateCamera(dt: number): void {
  // Right-click drag to orbit
  if (isMouseButtonDown(MouseButton.RIGHT)) {
    const dx = getMouseDeltaX();
    const dy = getMouseDeltaY();
    camAngle -= dx * 0.005;
    camPitch -= dy * 0.005;
    camPitch = Math.max(0.1, Math.min(1.4, camPitch));
  }
}

function getCameraPosition(): { x: number; y: number; z: number } {
  return {
    x: camTargetX + Math.cos(camAngle) * Math.cos(camPitch) * camDist,
    y: Math.sin(camPitch) * camDist,
    z: camTargetZ + Math.sin(camAngle) * Math.cos(camPitch) * camDist,
  };
}

// ============================================================
// Tool state (wall drawing)
// ============================================================

type ToolMode = 'select' | 'draw';
let toolMode: ToolMode = 'draw';
let drawStart: [number, number] | null = null;
let previewHandle: SceneNodeHandle | null = null;

// ============================================================
// Main
// ============================================================

initWindow(1280, 720, "Bloom — Interactive Wall Editor (Phase 3)");
setTargetFPS(60);
setAmbientLight(255, 255, 255, 0.3);
setDirectionalLight(0.5, 1.0, 0.3, 255, 240, 230, 0.5);

// Register systems
registerFrameCallback(4, wallSystem);
registerFrameCallback(5, lightSystem);

// Create some initial walls
createWall(0, 0, 5, 0);
createWall(5, 0, 5, 4);
createWall(5, 4, 0, 4);
createWall(0, 4, 0, 0);

while (!windowShouldClose()) {
  const dt = getDeltaTime();
  updateCamera(dt);

  // Toggle tool mode with Tab
  if (isKeyPressed(Key.TAB)) {
    toolMode = toolMode === 'select' ? 'draw' : 'select';
    drawStart = null;
  }

  const cam = getCameraPosition();

  beginDrawing();
  clearBackground(Colors.RAYWHITE);

  beginMode3D({
    position: cam,
    target: { x: camTargetX, y: 1.5, z: camTargetZ },
    up: { x: 0, y: 1, z: 0 },
    fovy: 45,
    projection: "perspective",
  });

  // Handle mouse click (left button)
  if (isMouseButtonPressed(MouseButton.LEFT)) {
    const mx = getMouseX();
    const my = getMouseY();

    if (toolMode === 'select') {
      // Pick scene — raycast against all scene nodes
      const hit = pickScene(mx, my);
      if (hit.hit) {
        const wallId = handleToWallId.get(hit.handle);
        if (wallId) {
          // Deselect old wall
          if (selectedWallId) dirtyWalls.add(selectedWallId);
          // Select new wall
          selectedWallId = wallId;
          dirtyWalls.add(wallId);
        }
      } else {
        // Clicked empty space — deselect
        if (selectedWallId) {
          dirtyWalls.add(selectedWallId);
          selectedWallId = null;
        }
      }
    } else {
      // Draw mode: pick ground plane for wall placement
      const hit = pickScene(mx, my);
      if (hit.hit && hit.handle === floorHandle) {
        const wx = Math.round(hit.point.x * 2) / 2; // snap to 0.5 grid
        const wz = Math.round(hit.point.z * 2) / 2;

        if (drawStart === null) {
          drawStart = [wx, wz];
        } else {
          // Create wall from start to clicked point
          createWall(drawStart[0], drawStart[1], wx, wz);
          drawStart = null;
        }
      }
    }
  }

  // Escape to cancel draw
  if (isKeyPressed(Key.ESCAPE)) {
    drawStart = null;
  }

  // Delete selected wall
  if (isKeyPressed(Key.BACKSPACE) && selectedWallId) {
    const wall = walls.get(selectedWallId);
    if (wall) {
      // Note: in a full implementation we'd call destroySceneNode(wall.handle)
      setSceneNodeColor(wall.handle, 0, 0, 0, 0);
      handleToWallId.delete(wall.handle);
      walls.delete(selectedWallId);
      selectedWallId = null;
    }
  }

  drawGrid(20, 0.5);
  endMode3D();

  // HUD
  drawText("Interactive Wall Editor", 10, 10, 20, Colors.DARKGRAY);
  drawText("Mode: " + toolMode + " (Tab to toggle)", 10, 35, 16, Colors.GRAY);
  drawText("Scene nodes: " + String(getSceneNodeCount()), 10, 55, 16, Colors.GRAY);
  drawText("Walls: " + String(walls.size), 10, 75, 16, Colors.GRAY);

  if (toolMode === 'select') {
    drawText("LEFT CLICK: select wall | BACKSPACE: delete", 10, 100, 14, Colors.BLUE);
    if (selectedWallId) {
      drawText("Selected: " + selectedWallId, 10, 120, 14, Colors.BLUE);
    }
  } else {
    drawText("LEFT CLICK: place wall endpoint | ESC: cancel", 10, 100, 14, Colors.GREEN);
    if (drawStart) {
      drawText("Start: " + String(drawStart[0]) + ", " + String(drawStart[1]) + " — click to place end", 10, 120, 14, Colors.GREEN);
    }
  }

  drawText("RIGHT DRAG: orbit camera", 10, 145, 14, Colors.GRAY);

  endDrawing();
}
