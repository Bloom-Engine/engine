// glTF loader test for watchOS — exercises the multi-primitive + scene-
// hierarchy path via Buggy.glb (205 nodes, 34 multi-primitive meshes).

import {
  initWindow, windowShouldClose, beginDrawing, endDrawing,
  clearBackground, setTargetFPS, getDeltaTime,
  beginMode3D, endMode3D,
} from 'bloom/core';

import {
  createSceneNode, setSceneNodeTransform,
} from 'bloom/scene';

import { loadModel } from 'bloom/models';

declare function bloom_add_directional_light(
  dx: number, dy: number, dz: number,
  r: number, g: number, b: number,
  intensity: number,
): void;
declare function bloom_scene_attach_model(node: number, model: number, meshIdx: number): void;
declare function bloom_set_vignette(strength: number, softness: number): void;
declare function bloom_enable_postfx(): void;

initWindow(800, 600, "Bloom glTF Watch");
setTargetFPS(30);

bloom_add_directional_light(-0.5, -0.8, -0.3, 1, 1, 1, 1.2);
bloom_enable_postfx();
bloom_set_vignette(0.5, 0.3);

const fox = loadModel("assets/Fox.glb");

const root = createSceneNode();
bloom_scene_attach_model(root, fox.handle, 0);

let t = 0.0;

while (!windowShouldClose()) {
  beginDrawing();
  clearBackground({ r: 20, g: 24, b: 32, a: 255 });

  t = t + getDeltaTime();
  const c = Math.cos(t * 0.4);
  const s = Math.sin(t * 0.4);

  // Fox mesh: ~25×78×150 in original units. Scale down 100×, center.
  const scl = 0.012;
  setSceneNodeTransform(root, [
     c * scl, 0,        s * scl, 0,
     0,       scl,      0,       0,
    -s * scl, 0,        c * scl, 0,
     0,       0,        0,       1,
  ]);

  beginMode3D({
    position: { x: 0, y: 0.8, z: 2.5 },
    target: { x: 0, y: 0.4, z: 0 },
    up: { x: 0, y: 1, z: 0 },
    fovy: 50,
    projection: "perspective",
  });
  endMode3D();

  endDrawing();
}
