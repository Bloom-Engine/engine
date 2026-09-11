import {
  initWindow, runGame, clearBackground, closeWindow,
  captureFrameToPng, isFrameCaptureReady,
  setDirect2DMode, writeFile,
} from "@bloomengine/engine/core";
import { drawRect } from "@bloomengine/engine/shapes";
import {
  createWorld, destroyWorld, sphereShape, releaseShape, createBody,
  destroyBody, bodyCount, isBodyValid, stepVariable, getBodyPosition,
} from "@bloomengine/engine/physics";

// Native installed-package acceptance: a real Jolt body must fall before the
// renderer produces the white square checked by the host. No random inputs.
initWindow(128, 128, "Bloom installed native startup");
const BLOOM_SMOKE_DIRECT_2D = false;
setDirect2DMode(BLOOM_SMOKE_DIRECT_2D);
const world = createWorld({ gravity: { x: 0, y: -9.81, z: 0 }, maxBodies: 64, numThreads: 1 });
const shape = sphereShape(0.5);
const body = createBody(world, shape, { motionType: 2, position: { x: 0, y: 4, z: 0 } });
let frames = 0;
let cleanups = 0;
runGame((_dt) => {
  stepVariable(world, 1 / 60, 1);
  const y = getBodyPosition(body).y;
  const physicsReady = isBodyValid(body) && bodyCount(world) === 1 && y > 0 && y < 4;
  clearBackground({ r: 0, g: 0, b: 0, a: 255 });
  drawRect(32, 32, 64, 64, {
    r: 255, g: physicsReady ? 255 : 0, b: physicsReady ? 255 : 0, a: 255,
  });
  frames = frames + 1;
  if (frames === 8) captureFrameToPng("native-startup.png");
  if (frames > 8 && isFrameCaptureReady()) closeWindow();
  // A failed capture must terminate and fail the host's image check.
  if (frames >= 120) closeWindow();
}, () => {
  cleanups = cleanups + 1;
  destroyBody(body);
  releaseShape(shape);
  destroyWorld(world);
  writeFile("native-cleanup.txt", cleanups.toString());
});
