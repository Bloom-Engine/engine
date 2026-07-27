// Deterministic skinned-motion + alpha-test qualification scene.
// The Fox supplies skeletal deformation/motion vectors; a single Sponza
// curtain primitive supplies a textured MASK surface and cutout shadow.

import {
  initWindow, closeWindow, windowShouldClose,
  beginDrawing, endDrawing, beginMode3D, endMode3D,
  setEnvClearFromHdr, setTargetFPS, setAutoExposure, setManualExposure,
  setTaaEnabled, setEnvIntensity, getCommandLineArgs, resize,
} from "bloom/core";
import { parseQualityRun, QualityRun } from "bloom/quality";
import {
  loadModel, loadModelAnimation, updateModelAnimation, drawModel,
  setAmbientLight, setDirectionalLight,
} from "bloom/models";
import {
  enableShadows, createSceneNode, attachModelToNode,
  setSceneNodeTransform, setSceneNodeCastShadow, setSceneNodeReceiveShadow,
} from "bloom/scene";
import { mat4Identity, mat4Translate, mat4Scale } from "bloom/math";

const argv: string[] = getCommandLineArgs();
const config = parseQualityRun(argv);

initWindow(800, 450, "Bloom Quality: Skinned + Alpha Motion", 0);
if (config !== null) resize(800, 450, 800, 450);
setTargetFPS(60);
setEnvClearFromHdr("../renderer-test/assets/outdoor.hdr");
setEnvIntensity(0.8);
setAutoExposure(false);
setManualExposure(1.0);
setTaaEnabled(true);
enableShadows();

const quality: QualityRun | null = config !== null ? new QualityRun(config) : null;
const foxModel = loadModel("../test-gltf-watch/assets/Fox.glb");
const foxAnimation = loadModelAnimation("../test-gltf-watch/assets/Fox.glb");

// Khronos Sponza primitive 0 is MASK foliage. The loader already bakes
// the glTF scene transforms (including its authored unit conversion), so the
// scene node must remain identity-scaled; applying another 0.01 made the
// backdrop disappear from the qualification camera.
const sponza = loadModel("../sponza/assets/Sponza.glb");
const curtain = createSceneNode();
attachModelToNode(curtain, sponza.handle, 0);
// Sponza's flattened primitive is centred at (3.962, 1.185, 1.588).
// Compose T(target) * S * T(-centre) so enlarging the authored foliage does
// not scale its baked world-space offset and move it out of frame.
let curtainTransform = mat4Identity();
curtainTransform = mat4Translate(
  curtainTransform,
  { x: 4.86, y: 1.20, z: -2.00 },
);
curtainTransform = mat4Scale(
  curtainTransform,
  { x: 4.0, y: 4.0, z: 1.0 },
);
curtainTransform = mat4Translate(
  curtainTransform,
  { x: -3.962, y: -1.185, z: -1.588 },
);
setSceneNodeTransform(curtain, curtainTransform);
setSceneNodeCastShadow(curtain, true);
setSceneNodeReceiveShadow(curtain, true);

let simulationTime = 0.0;
while (!windowShouldClose()) {
  const capture = quality !== null ? quality.beginFrame() : false;
  const dt = quality !== null ? quality.deltaTime() : 1 / 60;
  simulationTime = simulationTime + dt;

  // Fox clip 1 ("Walk") has continuous limb motion and a stable loop.
  updateModelAnimation(foxAnimation, 1, simulationTime, 0.02, 4.86, 0.15, -1.25, 0.0);

  beginDrawing();
  setAmbientLight({ r: 120, g: 130, b: 145, a: 255 }, 0.25);
  setDirectionalLight(
    { x: 0.45, y: 0.85, z: 0.25 },
    { r: 255, g: 244, b: 226, a: 255 },
    1.8,
  );
  beginMode3D({
    position: { x: 4.86, y: 1.45, z: 2.2 },
    target: { x: 4.86, y: 1.2, z: -1.6 },
    up: { x: 0, y: 1, z: 0 },
    fovy: 48,
    projection: "perspective",
  });
  drawModel(
    foxModel,
    { x: 4.86, y: 0.15, z: -1.25 },
    0.02,
    { r: 255, g: 255, b: 255, a: 255 },
  );
  endMode3D();
  if (capture && quality !== null) quality.requestCapture();
  endDrawing();

  if (quality !== null && quality.endFrame()) break;
}

closeWindow();
