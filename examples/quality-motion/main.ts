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
  loadModel, loadModelAnimation, updateModelAnimation, drawModel, genMeshCube,
  setAmbientLight, setDirectionalLight,
} from "bloom/models";
import {
  enableShadows, createSceneNode, attachModelToNode,
  setSceneNodeTransform, setSceneNodeCastShadow, setSceneNodeReceiveShadow,
  setSceneNodeVisible, setSceneNodeColor, setSceneNodePbr,
} from "bloom/scene";
import { mat4Identity, mat4Translate, mat4Scale } from "bloom/math";

const argv: string[] = getCommandLineArgs();
const config = parseQualityRun(argv);
let vsmDynamicFixture = false;
let vsmScrollFixture = false;
let vsmLightMotionFixture = false;
let vsmGpuReceiverFixture = false;
let vsmContactDetailFixture = false;
for (let i = 0; i < argv.length; i = i + 1) {
  if (argv[i] === "--vsm-dynamic") vsmDynamicFixture = true;
  if (argv[i] === "--vsm-scroll") vsmScrollFixture = true;
  if (argv[i] === "--vsm-light-motion") vsmLightMotionFixture = true;
  if (argv[i] === "--vsm-gpu-receivers") vsmGpuReceiverFixture = true;
  if (argv[i] === "--vsm-contact-detail") vsmContactDetailFixture = true;
}

const captureWidth = vsmContactDetailFixture ? 1280 : 800;
const captureHeight = vsmContactDetailFixture ? 720 : 450;
initWindow(captureWidth, captureHeight, "Bloom Quality: Skinned + Alpha Motion", 0);
if (config !== null) resize(captureWidth, captureHeight, captureWidth, captureHeight);
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
if (vsmContactDetailFixture) setSceneNodeVisible(curtain, false);

// Opt-in VSM oracle: the large static receiver creates a production-sized
// demand set while the animated Fox exercises current-frame skinned overlays.
// The ordinary quality-motion corpus remains byte-for-byte unchanged.
if (vsmDynamicFixture || vsmContactDetailFixture) {
  const groundModel = genMeshCube(28.0, 0.08, 24.0);
  const ground = createSceneNode();
  attachModelToNode(ground, groundModel.handle, 0);
  let groundTransform = mat4Identity();
  groundTransform = mat4Translate(
    groundTransform,
    { x: 4.86, y: 0.02, z: -5.0 },
  );
  setSceneNodeTransform(ground, groundTransform);
  setSceneNodeCastShadow(ground, false);
  setSceneNodeReceiveShadow(ground, true);
  if (vsmContactDetailFixture) {
    setSceneNodeColor(ground, 178, 184, 194);
    setSceneNodePbr(ground, 0.82, 0.0);
  }
}

// Opt-in directional contact-detail oracle. Repeated sub-pixel-width posts
// and their ground contacts expose the resolution retained by page-local VSM
// depth without changing the established motion fixture.
if (vsmContactDetailFixture) {
  const postModel = genMeshCube(0.055, 0.72, 0.055);
  for (let row = 0; row < 14; row = row + 1) {
    for (let column = 0; column < 19; column = column + 1) {
      const post = createSceneNode();
      attachModelToNode(post, postModel.handle, 0);
      let postTransform = mat4Identity();
      postTransform = mat4Translate(
        postTransform,
        {
          x: 4.86 + (column - 9) * 0.24,
          y: 0.42,
          z: -0.45 - row * 0.42,
        },
      );
      setSceneNodeTransform(post, postTransform);
      setSceneNodeCastShadow(post, true);
      setSceneNodeReceiveShadow(post, true);
      setSceneNodeColor(
        post,
        142 + (column % 3) * 18,
        132 + (row % 3) * 14,
        118,
      );
      setSceneNodePbr(post, 0.7, 0.0);
    }
  }
}

// Opt-in receiver-compaction stress oracle. These small receive-only nodes
// remain fully below the large ground, so they cannot alter the capture, but
// their camera-visible bounds exercise the >=1,024 receiver path. Continuous
// sub-page motion requires asynchronous results to make forward progress
// without a same-frame CPU readback.
const gpuReceiverNodes: number[] = [];
if (vsmGpuReceiverFixture) {
  const receiverModel = genMeshCube(0.12, 0.02, 0.12);
  for (let index = 0; index < 18 * 64; index = index + 1) {
    const receiver = createSceneNode();
    attachModelToNode(receiver, receiverModel.handle, 0);
    setSceneNodeCastShadow(receiver, false);
    setSceneNodeReceiveShadow(receiver, true);
    gpuReceiverNodes.push(receiver);
  }
}

let simulationTime = 0.0;
let fixtureFrame = 0;
while (!windowShouldClose()) {
  const capture = quality !== null ? quality.beginFrame() : false;
  const dt = quality !== null ? quality.deltaTime() : 1 / 60;
  simulationTime = simulationTime + dt;
  fixtureFrame = fixtureFrame + 1;

  if (vsmGpuReceiverFixture) {
    const receiverMotion = Math.sin(simulationTime * 2.0) * 0.02;
    for (let index = 0; index < gpuReceiverNodes.length; index = index + 1) {
      const column = index % 18;
      const row = Math.floor(index / 18);
      let transform = mat4Identity();
      transform = mat4Translate(
        transform,
        {
          x: 4.86 + (column - 8.5) * 0.38 + receiverMotion,
          y: 0.0,
          z: -1.2 - row * 0.08,
        },
      );
      setSceneNodeTransform(gpuReceiverNodes[index], transform);
    }
  }

  // Fox clip 1 ("Walk") has continuous limb motion and a stable loop.
  if (!vsmContactDetailFixture) {
    updateModelAnimation(foxAnimation, 1, simulationTime, 0.02, 4.86, 0.15, -1.25, 0.0);
  }

  beginDrawing();
  setAmbientLight({ r: 120, g: 130, b: 145, a: 255 }, 0.25);
  // Opt-in light-basis transition oracle. Frame 240 returns to the ordinary
  // fixture direction, so it can be compared directly with a settled capture
  // while telemetry observes the conservative one-frame invalidation.
  const alternateLight = vsmLightMotionFixture
    ? Math.floor(fixtureFrame / 30) % 2
    : 0;
  const lightDirection = vsmContactDetailFixture
    ? { x: 0.62, y: 0.58, z: 0.36 }
    : alternateLight !== 0
      ? { x: 0.28, y: 0.90, z: 0.34 }
      : { x: 0.45, y: 0.85, z: 0.25 };
  setDirectionalLight(
    lightDirection,
    { r: 255, g: 244, b: 226, a: 255 },
    1.8,
  );
  // Opt-in snapped-origin transition oracle. The 30-frame hold lets new
  // boundary pages settle, while frame 240 deliberately lands one frame after
  // a rebase so the capture and telemetry expose its safe fallback behavior.
  const cameraScroll = vsmScrollFixture
    ? (Math.floor(fixtureFrame / 30) % 2) * 2.5
    : 0.0;
  // Exact unnormalized right vector for the qualification sun, so this move
  // crosses planar page origins without also changing the depth projection.
  const cameraScrollX = cameraScroll * 0.25;
  const cameraScrollZ = cameraScroll * -0.45;
  beginMode3D({
    position: vsmContactDetailFixture
      ? { x: 4.86, y: 2.5, z: 3.4 }
      : { x: 4.86 + cameraScrollX, y: 1.45, z: 2.2 + cameraScrollZ },
    target: vsmContactDetailFixture
      ? { x: 4.86, y: 0.18, z: -3.25 }
      : { x: 4.86 + cameraScrollX, y: 1.2, z: -1.6 + cameraScrollZ },
    up: { x: 0, y: 1, z: 0 },
    fovy: 48,
    projection: "perspective",
  });
  if (!vsmContactDetailFixture) {
    drawModel(
      foxModel,
      { x: 4.86, y: 0.15, z: -1.25 },
      0.02,
      { r: 255, g: 255, b: 255, a: 255 },
    );
  }
  endMode3D();
  if (capture && quality !== null) quality.requestCapture();
  endDrawing();

  if (quality !== null && quality.endFrame()) break;
}

closeWindow();
