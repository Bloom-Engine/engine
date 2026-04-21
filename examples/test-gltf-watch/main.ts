// glTF loader test for watchOS — loads DamagedHelmet.glb from the app
// bundle, attaches it to a scene node, rotates it, and renders through
// the SceneKit bridge.

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
bloom_set_vignette(0.7, 0.25);  // strength 0.7, 25% soft center

const helmet = loadModel("assets/DamagedHelmet.glb");

const node = createSceneNode();
bloom_scene_attach_model(node, helmet.handle, 0);

let t = 0.0;

while (!windowShouldClose()) {
  beginDrawing();
  clearBackground({ r: 20, g: 22, b: 28, a: 255 });

  t = t + getDeltaTime();
  const c = Math.cos(t * 0.6);
  const s = Math.sin(t * 0.6);

  // Rotate helmet around Y.
  setSceneNodeTransform(node, [
     c, 0, s, 0,
     0, 1, 0, 0,
    -s, 0, c, 0,
     0, 0, 0, 1,
  ]);

  beginMode3D({
    position: { x: 0, y: 0, z: 3 },
    target: { x: 0, y: 0, z: 0 },
    up: { x: 0, y: 1, z: 0 },
    fovy: 50,
    projection: "perspective",
  });
  endMode3D();

  endDrawing();
}
