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
  enableShadows, addPointLight, addShadowedPointLight, createSceneNode, attachModelToNode,
  setSceneNodeTransform, setSceneNodeColor, setSceneNodePbr,
  setSceneNodeCastShadow, setSceneNodeReceiveShadow,
} from "bloom/scene";
import { mat4Identity, mat4Translate, mat4Scale } from "bloom/math";

const argv: string[] = getCommandLineArgs();
const config = parseQualityRun(argv);
let vsmGpuCasterFixture = false;
let vsmLocalLightsFixture = false;
let vsmLocalReferenceFixture = false;
for (let i = 0; i < argv.length; i = i + 1) {
  if (argv[i] === "--vsm-gpu-casters") vsmGpuCasterFixture = true;
  if (argv[i] === "--vsm-local-lights") vsmLocalLightsFixture = true;
  if (argv[i] === "--vsm-local-reference") vsmLocalReferenceFixture = true;
}

const localLightLayout = vsmLocalLightsFixture || vsmLocalReferenceFixture;
const GRID_X = localLightLayout ? 16 : (vsmGpuCasterFixture ? 32 : 128);
const GRID_Z = localLightLayout ? 16 : (vsmGpuCasterFixture ? 16 : 80);
const DRAW_COUNT = GRID_X * GRID_Z;
const LIGHT_COUNT = localLightLayout ? 128 : 192;

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
if (localLightLayout) {
  const floor = createSceneNode();
  attachModelToNode(floor, cube.handle, 0);
  let floorTransform = mat4Identity();
  floorTransform = mat4Translate(floorTransform, { x: 0, y: -0.15, z: 0 });
  floorTransform = mat4Scale(floorTransform, { x: 14, y: 0.2, z: 10 });
  setSceneNodeTransform(floor, floorTransform);
  setSceneNodeColor(floor, 125, 130, 138);
  setSceneNodePbr(floor, 0.72, 0.0);
  setSceneNodeCastShadow(floor, false);
  setSceneNodeReceiveShadow(floor, true);
}
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
  // The ordinary 10k case stays focused on main-view submission. The opt-in
  // 512-node VSM oracle stays below the per-page compatibility cap.
  setSceneNodeCastShadow(node, vsmGpuCasterFixture || localLightLayout);
  setSceneNodeReceiveShadow(node, true);
}

let fixtureFrame = 0;
while (!windowShouldClose()) {
  const capture = quality !== null ? quality.beginFrame() : false;
  fixtureFrame = fixtureFrame + 1;
  beginDrawing();
  setAmbientLight(
    { r: 105, g: 115, b: 135, a: 255 },
    localLightLayout ? 0.08 : 0.25,
  );
  setDirectionalLight(
    vsmGpuCasterFixture && Math.floor(fixtureFrame / 30) % 2 !== 0
      ? { x: -0.28, y: 0.90, z: 0.34 }
      : { x: -0.4, y: 0.85, z: 0.3 },
    { r: 255, g: 244, b: 228, a: 255 },
    localLightLayout ? 0.15 : 1.4,
  );
  for (let i = 0; i < LIGHT_COUNT; i = i + 1) {
    const columns = localLightLayout ? 16 : 24;
    const col = i % columns;
    const row = Math.floor(i / columns);
    const hue = i % 3;
    const priorityLocalLight = localLightLayout && i < 5;
    const args = [
      priorityLocalLight ? 0.0
        : (localLightLayout ? (col - 7.5) * 1.0 : (col - 11.5) * 3.3),
      priorityLocalLight ? 6.0 : 2.0 + (i % 4) * 0.6,
      priorityLocalLight ? 4.0
        : (localLightLayout ? (row - 3.5) * 1.8 : (row - 3.5) * 6.0),
      priorityLocalLight ? 15.0 : (localLightLayout ? 4.5 : 5.5),
      hue === 0 ? 1.0 : 0.2,
      hue === 1 ? 1.0 : 0.25,
      hue === 2 ? 1.0 : 0.2,
      priorityLocalLight ? 2.0 : (localLightLayout ? 8.0 : 3.0),
    ] as const;
    if (vsmLocalLightsFixture) {
      if (!addShadowedPointLight(...args) && fixtureFrame === 1) {
        throw new Error(`shadowed point-light request ${i} was rejected`);
      }
    } else if (
      !vsmLocalReferenceFixture
      || i < 5
    ) {
      addPointLight(...args);
    }
  }
  beginMode3D({
    position: localLightLayout
      ? { x: 0, y: 9, z: 17 }
      : { x: 0, y: 38, z: 52 },
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
