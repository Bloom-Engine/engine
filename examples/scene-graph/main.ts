/**
 * Scene Graph Example — Demonstrates retained-mode 3D rendering.
 *
 * This is a proof-of-concept for the Pascal Editor native compilation path:
 * - Creates persistent scene nodes (like React Three Fiber's <mesh>)
 * - Updates geometry dynamically (like WallSystem's useFrame callback)
 * - Sets per-node transforms and materials
 *
 * The scene graph nodes persist across frames — unlike immediate-mode drawCube(),
 * they don't need to be re-submitted each frame.
 */

import {
  initWindow, windowShouldClose, beginDrawing, endDrawing,
  clearBackground, setTargetFPS, drawText,
  beginMode3D, endMode3D, drawGrid,
  isKeyPressed, Key,
  Colors, getDeltaTime,
} from 'bloom';

import {
  createSceneNode, destroySceneNode,
  setSceneNodeTransform, updateSceneNodeGeometry,
  setSceneNodeColor, setSceneNodePbr,
  setSceneNodeVisible, getSceneNodeCount,
} from 'bloom/scene';

import {
  mat4Identity, mat4Translate, mat4RotateY, mat4Multiply,
} from 'bloom';

// ============================================================
// Wall geometry generator (simplified version of Pascal Editor's
// generateExtrudedWall — pure TypeScript math, no Three.js)
// ============================================================

function generateWallVertices(
  startX: number, startZ: number,
  endX: number, endZ: number,
  height: number,
  thickness: number,
): { vertices: number[]; indices: number[] } {
  // Wall direction vector
  const dx = endX - startX;
  const dz = endZ - startZ;
  const len = Math.sqrt(dx * dx + dz * dz);
  if (len < 0.001) return { vertices: [], indices: [] };

  // Normal perpendicular to wall direction (in XZ plane)
  const nx = -dz / len;
  const nz = dx / len;

  // Half thickness offset
  const hx = nx * thickness * 0.5;
  const hz = nz * thickness * 0.5;

  // 8 corner vertices of the wall box
  //   0--1  (top, front)
  //   |  |
  //   3--2  (top, back)
  //
  //   4--5  (bottom, front)
  //   |  |
  //   7--6  (bottom, back)
  const corners = [
    // Front face (start side)
    [startX + hx, height, startZ + hz],  // 0: top-front-start
    [endX + hx, height, endZ + hz],      // 1: top-front-end
    [endX - hx, height, endZ - hz],      // 2: top-back-end
    [startX - hx, height, startZ - hz],  // 3: top-back-start
    [startX + hx, 0, startZ + hz],       // 4: bottom-front-start
    [endX + hx, 0, endZ + hz],           // 5: bottom-front-end
    [endX - hx, 0, endZ - hz],           // 6: bottom-back-end
    [startX - hx, 0, startZ - hz],       // 7: bottom-back-start
  ];

  const vertices: number[] = [];
  const indices: number[] = [];

  function addFace(
    v0: number[], v1: number[], v2: number[], v3: number[],
    fnx: number, fny: number, fnz: number,
  ) {
    const baseIdx = vertices.length / 12;
    // 4 vertices per face, each with 12 floats: xyz, nxnynz, rgba, uv
    for (const [v, u, uv_u, uv_v] of [[v0, 0, 0, 0], [v1, 1, 1, 0], [v2, 2, 1, 1], [v3, 3, 0, 1]] as any) {
      vertices.push(
        v[0], v[1], v[2],       // position
        fnx, fny, fnz,          // normal
        1.0, 1.0, 1.0, 1.0,    // color (white)
        uv_u, uv_v,            // UV
      );
    }
    // Two triangles per face
    indices.push(baseIdx, baseIdx + 1, baseIdx + 2);
    indices.push(baseIdx, baseIdx + 2, baseIdx + 3);
  }

  // Front face (+normal direction)
  addFace(corners[0], corners[1], corners[5], corners[4], nx, 0, nz);
  // Back face (-normal direction)
  addFace(corners[2], corners[3], corners[7], corners[6], -nx, 0, -nz);
  // Top face
  addFace(corners[0], corners[3], corners[2], corners[1], 0, 1, 0);
  // Bottom face
  addFace(corners[4], corners[5], corners[6], corners[7], 0, -1, 0);
  // Start cap
  addFace(corners[3], corners[0], corners[4], corners[7], -dx / len, 0, -dz / len);
  // End cap
  addFace(corners[1], corners[2], corners[6], corners[5], dx / len, 0, dz / len);

  return { vertices, indices };
}

// ============================================================
// Main
// ============================================================

initWindow(1280, 720, "Bloom — Scene Graph Demo");
setTargetFPS(60);

// Create scene graph nodes (persistent, like R3F <mesh> elements)
const wall1 = createSceneNode();
const wall2 = createSceneNode();
const wall3 = createSceneNode();
const floor = createSceneNode();

// Generate wall geometry (like WallSystem's generateExtrudedWall)
const wall1Geo = generateWallVertices(0, 0, 5, 0, 3, 0.2);
const wall2Geo = generateWallVertices(5, 0, 5, 4, 3, 0.2);
const wall3Geo = generateWallVertices(0, 0, 0, 4, 3, 0.2);

// Upload geometry to GPU (like WallSystem assigning mesh.geometry)
updateSceneNodeGeometry(wall1, wall1Geo.vertices, wall1Geo.indices);
updateSceneNodeGeometry(wall2, wall2Geo.vertices, wall2Geo.indices);
updateSceneNodeGeometry(wall3, wall3Geo.vertices, wall3Geo.indices);

// Floor: a flat rectangle
const floorGeo = generateWallVertices(0, 0, 5, 0, 0, 4);
// Floor needs different geometry — let's make a simple quad
const floorVerts: number[] = [
  // x, y, z, nx, ny, nz, r, g, b, a, u, v
  0, 0, 0,   0, 1, 0,   0.8, 0.8, 0.8, 1.0,   0, 0,
  5, 0, 0,   0, 1, 0,   0.8, 0.8, 0.8, 1.0,   1, 0,
  5, 0, 4,   0, 1, 0,   0.8, 0.8, 0.8, 1.0,   1, 1,
  0, 0, 4,   0, 1, 0,   0.8, 0.8, 0.8, 1.0,   0, 1,
];
const floorIdx: number[] = [0, 1, 2, 0, 2, 3];
updateSceneNodeGeometry(floor, floorVerts, floorIdx);

// Set materials
setSceneNodeColor(wall1, 0.95, 0.95, 0.92, 1.0);
setSceneNodeColor(wall2, 0.92, 0.92, 0.88, 1.0);
setSceneNodeColor(wall3, 0.90, 0.90, 0.86, 1.0);
setSceneNodeColor(floor, 0.7, 0.7, 0.65, 1.0);

// Set PBR properties
setSceneNodePbr(wall1, 0.8, 0.0);
setSceneNodePbr(wall2, 0.8, 0.0);
setSceneNodePbr(wall3, 0.8, 0.0);
setSceneNodePbr(floor, 0.6, 0.0);

let angle = 0;
let wall3Visible = true;

while (!windowShouldClose()) {
  const dt = getDeltaTime();

  // Toggle wall visibility with Space
  if (isKeyPressed(Key.SPACE)) {
    wall3Visible = !wall3Visible;
    setSceneNodeVisible(wall3, wall3Visible);
  }

  // Slowly rotate camera angle
  angle += dt * 0.3;

  beginDrawing();
  clearBackground(Colors.RAYWHITE);

  // Camera orbits around the room
  const camX = 2.5 + Math.cos(angle) * 10;
  const camZ = 2.0 + Math.sin(angle) * 10;
  beginMode3D({
    position: { x: camX, y: 6, z: camZ },
    target: { x: 2.5, y: 1.5, z: 2 },
    up: { x: 0, y: 1, z: 0 },
    fovy: 45,
    projection: "perspective",
  });

  // Scene graph nodes render automatically via end_frame_with_scene
  // We only need to draw the grid manually (immediate mode)
  drawGrid(20, 1.0);

  endMode3D();

  drawText("Bloom Scene Graph Demo", 10, 10, 20, Colors.DARKGRAY);
  drawText("Scene nodes: " + String(getSceneNodeCount()), 10, 35, 16, Colors.GRAY);
  drawText("Press SPACE to toggle wall 3", 10, 55, 16, Colors.GRAY);
  drawText("Walls render automatically (retained mode)", 10, 75, 16, Colors.GRAY);

  endDrawing();
}
