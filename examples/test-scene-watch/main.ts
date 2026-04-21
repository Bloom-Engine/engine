// Minimal retained-mode scene-graph test for watchOS.
// Creates two cubes via bloom_scene_* FFI, animates rotation, adds a
// directional light, renders through the BloomSceneView SceneKit bridge.

import {
  initWindow, windowShouldClose, beginDrawing, endDrawing,
  clearBackground, setTargetFPS, getDeltaTime,
  beginMode3D, endMode3D,
} from 'bloom/core';

import {
  createSceneNode,
  setSceneNodeTransform, updateSceneNodeGeometry,
  setSceneNodeColor, setSceneNodePbr,
} from 'bloom/scene';

declare function bloom_add_directional_light(
  dx: number, dy: number, dz: number,
  r: number, g: number, b: number,
  intensity: number,
): void;

// 12 floats per vertex: xyz, nx ny nz, rgba, uv.
function cubeVerts(s: number): number[] {
  const verts: number[] = [];
  // Six faces, each a quad → two triangles. Normal points out.
  const faces: [number[], number[]][] = [
    [[ 0, 0, 1], [ 1, 1, 1, 1]], [[ 0, 0,-1], [1, 1, 1, 1]],
    [[ 1, 0, 0], [ 1, 1, 1, 1]], [[-1, 0, 0], [1, 1, 1, 1]],
    [[ 0, 1, 0], [ 1, 1, 1, 1]], [[ 0,-1, 0], [1, 1, 1, 1]],
  ];
  const quadUVs = [[0,0],[1,0],[1,1],[0,1]];
  const cornerFor = (n: number[], i: number): number[] => {
    // Produce 4 corners of each face (CCW as seen from outside).
    if (n[2] ===  1) return [[-s,-s, s],[ s,-s, s],[ s, s, s],[-s, s, s]][i];
    if (n[2] === -1) return [[ s,-s,-s],[-s,-s,-s],[-s, s,-s],[ s, s,-s]][i];
    if (n[0] ===  1) return [[ s,-s, s],[ s,-s,-s],[ s, s,-s],[ s, s, s]][i];
    if (n[0] === -1) return [[-s,-s,-s],[-s,-s, s],[-s, s, s],[-s, s,-s]][i];
    if (n[1] ===  1) return [[-s, s, s],[ s, s, s],[ s, s,-s],[-s, s,-s]][i];
    /*  n[1] = -1 */ return [[-s,-s,-s],[ s,-s,-s],[ s,-s, s],[-s,-s, s]][i];
  };
  for (const [n, c] of faces) {
    for (let i = 0; i < 4; i = i + 1) {
      const p = cornerFor(n, i);
      verts.push(p[0], p[1], p[2], n[0], n[1], n[2], c[0], c[1], c[2], c[3], quadUVs[i][0], quadUVs[i][1]);
    }
  }
  return verts;
}

function cubeIndices(): number[] {
  const idx: number[] = [];
  for (let f = 0; f < 6; f = f + 1) {
    const base = f * 4;
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return idx;
}

initWindow(800, 600, "Bloom Scene Watch");
setTargetFPS(30);

bloom_add_directional_light(-0.5, -1, -0.3, 1, 1, 1, 0.9);

const v = cubeVerts(1.0);
const i = cubeIndices();

const a = createSceneNode();
updateSceneNodeGeometry(a, v, i);
setSceneNodeColor(a, 220, 60, 60, 255);
setSceneNodePbr(a, 0.4, 0.0);

const b = createSceneNode();
updateSceneNodeGeometry(b, v, i);
setSceneNodeColor(b, 60, 160, 220, 255);
setSceneNodePbr(b, 0.2, 0.9);

let t = 0.0;

while (!windowShouldClose()) {
  beginDrawing();
  clearBackground({ r: 40, g: 50, b: 70, a: 255 });

  t = t + getDeltaTime();
  const c = Math.cos(t);
  const s = Math.sin(t);

  // Rotate cube A around Y, position at (-2, 0, 0)
  setSceneNodeTransform(a, [
     c, 0, s, 0,
     0, 1, 0, 0,
    -s, 0, c, 0,
    -2, 0, 0, 1,
  ]);
  // Rotate cube B the other way, position (2, 0, 0)
  setSceneNodeTransform(b, [
     c, 0,-s, 0,
     0, 1, 0, 0,
     s, 0, c, 0,
     2, 0, 0, 1,
  ]);

  beginMode3D({
    position: { x: 0, y: 3, z: 8 },
    target: { x: 0, y: 0, z: 0 },
    up: { x: 0, y: 1, z: 0 },
    fovy: 45,
    projection: "perspective",
  });
  endMode3D();

  endDrawing();
}
