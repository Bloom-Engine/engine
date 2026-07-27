// Deterministic 10k-draw + many-light qualification scene.
// Startup/node construction is intentionally outside the measured window.

import {
  initWindow, closeWindow, windowShouldClose,
  beginDrawing, endDrawing, beginMode3D, endMode3D,
  setEnvClearFromHdr, setTargetFPS, setAutoExposure, setManualExposure,
  setTaaEnabled, setEnvIntensity, getCommandLineArgs, resize,
} from "bloom/core";
import { parseQualityRun, QualityRun } from "bloom/quality";
import {
  genMeshCube, setAmbientLight, setDirectionalLight,
} from "bloom/models";
import {
  enableShadows, addPointLight, createSceneNode, attachModelToNode,
  setSceneNodeTransform, setSceneNodeColor, setSceneNodePbr,
  setSceneNodeCastShadow, setSceneNodeReceiveShadow,
} from "bloom/scene";
import { mat4Identity, mat4Translate, mat4Scale } from "bloom/math";

const argv: string[] = getCommandLineArgs();
const config = parseQualityRun(argv);

const GRID_X = 128;
const GRID_Z = 80;
const DRAW_COUNT = GRID_X * GRID_Z; // 10,240 independently submitted nodes.
const LIGHT_COUNT = 192;

initWindow(1280, 720, "Bloom Quality: 10k Draws + Many Lights", 0);
if (config !== null) resize(1280, 720, 1280, 720);
setTargetFPS(60);
setEnvClearFromHdr("../renderer-test/assets/outdoor.hdr");
setEnvIntensity(0.35);
setAutoExposure(false);
setManualExposure(1.0);
setTaaEnabled(true);
enableShadows();
const quality: QualityRun | null = config !== null ? new QualityRun(config) : null;

const cube = genMeshCube(1, 1, 1);
for (let i = 0; i < DRAW_COUNT; i = i + 1) {
  const xIndex = i % GRID_X;
  const zIndex = Math.floor(i / GRID_X);
  const x = (xIndex - GRID_X / 2) * 0.72;
  const z = (zIndex - GRID_Z / 2) * 0.72;
  const y = 0.35 + 0.22 * Math.sin(xIndex * 0.37) * Math.cos(zIndex * 0.29);
  const node = createSceneNode();
  attachModelToNode(node, cube.handle, 0);
  let transform = mat4Identity();
  transform = mat4Translate(transform, { x: x, y: y, z: z });
  transform = mat4Scale(transform, { x: 0.26, y: 0.55, z: 0.26 });
  setSceneNodeTransform(node, transform);
  setSceneNodeColor(
    node,
    70 + (xIndex % 7) * 22,
    90 + (zIndex % 5) * 25,
    130 + ((xIndex + zIndex) % 4) * 24,
  );
  setSceneNodePbr(node, 0.22 + (xIndex % 6) * 0.12, (zIndex % 9 === 0) ? 0.8 : 0.05);
  // Keep the stress focused on main-view draw submission. A 10k-caster
  // shadow map would test a different budget and conceal clustered-light cost.
  setSceneNodeCastShadow(node, false);
  setSceneNodeReceiveShadow(node, true);
}

while (!windowShouldClose()) {
  const capture = quality !== null ? quality.beginFrame() : false;
  beginDrawing();
  setAmbientLight({ r: 105, g: 115, b: 135, a: 255 }, 0.25);
  setDirectionalLight(
    { x: -0.4, y: 0.85, z: 0.3 },
    { r: 255, g: 244, b: 228, a: 255 },
    1.4,
  );
  for (let i = 0; i < LIGHT_COUNT; i = i + 1) {
    const col = i % 24;
    const row = Math.floor(i / 24);
    const hue = i % 3;
    addPointLight(
      (col - 11.5) * 3.3,
      2.0 + (i % 4) * 0.6,
      (row - 3.5) * 6.0,
      5.5,
      hue === 0 ? 1.0 : 0.2,
      hue === 1 ? 1.0 : 0.25,
      hue === 2 ? 1.0 : 0.2,
      3.0,
    );
  }
  beginMode3D({
    position: { x: 0, y: 38, z: 52 },
    target: { x: 0, y: 0, z: 0 },
    up: { x: 0, y: 1, z: 0 },
    fovy: 58,
    projection: "perspective",
  });
  endMode3D();
  if (capture && quality !== null) quality.requestCapture();
  endDrawing();
  if (quality !== null && quality.endFrame()) break;
}

closeWindow();
