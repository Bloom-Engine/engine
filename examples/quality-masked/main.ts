// Deterministic masked-alpha qualification scene: coverage-preserving cards
// span multiple projected mip sizes while moving and casting real silhouettes.

import {
  initWindow, closeWindow, windowShouldClose,
  beginDrawing, endDrawing, beginMode3D, endMode3D,
  setTargetFPS, setAutoExposure, setManualExposure, setTaaEnabled,
  setShadowsEnabled, getCommandLineArgs, resize,
} from "bloom/core";
import { parseQualityRun, QualityRun } from "bloom/quality";
import {
  loadModel, drawModelTransform, drawCube,
  setAmbientLight, setDirectionalLight,
} from "bloom/models";
import {
  mat4Identity, mat4Translate, mat4Scale, mat4RotateY, mat4RotateZ,
} from "bloom/math";

const argv: string[] = getCommandLineArgs();
const config = parseQualityRun(argv);
const WIDTH = 960;
const HEIGHT = 540;
const DEPTH_ROWS = 6;
const COLUMNS = 8;

initWindow(WIDTH, HEIGHT, "Bloom Quality: Masked Alpha Coverage", 0);
if (config !== null) resize(WIDTH, HEIGHT, WIDTH, HEIGHT);
setTargetFPS(60);
setAutoExposure(false);
setManualExposure(1.0);
setTaaEnabled(true);
setShadowsEnabled(true);

const quality: QualityRun | null = config !== null ? new QualityRun(config) : null;
const card = loadModel("assets/masked-card.gltf");
let simulationTime = 0.0;

while (!windowShouldClose()) {
  const capture = quality !== null ? quality.beginFrame() : false;
  const dt = quality !== null ? quality.deltaTime() : 1 / 60;
  simulationTime = simulationTime + dt;

  beginDrawing();
  setAmbientLight({ r: 155, g: 170, b: 190, a: 255 }, 0.55);
  setDirectionalLight(
    { x: -0.38, y: 0.82, z: 0.42 },
    { r: 255, g: 245, b: 225, a: 255 },
    1.2,
  );
  beginMode3D({
    position: { x: 0, y: 5.0, z: 10.0 },
    target: { x: 0, y: -0.3, z: -4.0 },
    up: { x: 0, y: 1, z: 0 },
    fovy: 52,
    projection: "perspective",
  });

  drawCube(
    { x: 0, y: -1.25, z: -4.5 },
    22.0, 0.16, 30.0,
    { r: 42, g: 48, b: 58, a: 255 },
  );

  for (let row = 0; row < DEPTH_ROWS; row = row + 1) {
    const z = 3.0 - row * 3.0;
    const spread = 8.5 + row * 2.0;
    const projectedScale = Math.pow(0.72, row);
    for (let column = 0; column < COLUMNS; column = column + 1) {
      const phase = row * 0.71 + column * 0.37;
      const x = (column / (COLUMNS - 1) - 0.5) * spread;
      let transform = mat4Identity();
      transform = mat4Translate(transform, {
        x,
        y: -0.15 + row * 0.35 + (column % 2) * 0.06,
        z,
      });
      transform = mat4RotateY(
        transform,
        Math.sin(simulationTime * 0.42 + phase) * 0.12,
      );
      transform = mat4RotateZ(
        transform,
        Math.sin(simulationTime * 0.63 + phase) * 0.035,
      );
      transform = mat4Scale(transform, {
        x: 0.55 * projectedScale,
        y: 0.82 * projectedScale,
        z: 1.0,
      });
      drawModelTransform(
        card,
        transform,
        {
          r: 150 + (column * 13) % 80,
          g: 205 + (row * 7) % 45,
          b: 145 + (row * 11 + column * 5) % 75,
          a: 255,
        },
      );
    }
  }

  endMode3D();
  if (capture && quality !== null) quality.requestCapture();
  endDrawing();
  if (quality !== null && quality.endFrame()) break;
}

closeWindow();
