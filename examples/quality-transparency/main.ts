// Deterministic weighted-transparency qualification scene: 96 imported glTF
// BLEND quads arranged as 12 intersecting eight-layer cells.

import {
  initWindow, closeWindow, windowShouldClose,
  beginDrawing, endDrawing, beginMode3D, endMode3D,
  setTargetFPS, setAutoExposure, setManualExposure, setTaaEnabled,
  getCommandLineArgs, resize, setTransparencyCompositionMode,
} from "bloom/core";
import { parseQualityRun, QualityRun } from "bloom/quality";
import {
  loadModel, drawModelTransform, drawCube,
  setAmbientLight, setDirectionalLight, createPlanarReflection,
  compileTransparentMaterial, drawMeshWithMaterial,
} from "bloom/models";
import {
  createSceneNode, attachModelToNode, setSceneNodeTransform,
  setSceneNodeCastShadow,
} from "bloom/scene";
import {
  mat4Identity, mat4Translate, mat4Scale, mat4RotateX,
  mat4RotateY, mat4RotateZ,
} from "bloom/math";

const argv: string[] = getCommandLineArgs();
const config = parseQualityRun(argv);
const sortedInterleavingRoute = argv.includes("--sorted-interleaving");
const sortedRoute = argv.includes("--sorted") || sortedInterleavingRoute;
// Focused unversioned glass-reflection oracle. One smooth pane reflects
// opaque objects on the camera side; unlike the 96-layer transmission stress
// route, this gives the screen-space reflection tier deliberate valid hits.
const reflectionHierarchyRoute = argv.includes("--reflection-hierarchy");
const refractiveRoute = argv.includes("--refractive") || reflectionHierarchyRoute;
// Unversioned focused route for physical-transmission GI performance A/B.
// Unlike drawModelTransform(), retained nodes enter Mesh-Cards and the TLAS.
const transparentGiRoute = argv.includes("--transparent-gi");

const WIDTH = 960;
const HEIGHT = 540;
const CELLS_X = 4;
const CELLS_Y = 3;
const LAYERS = 8;

initWindow(WIDTH, HEIGHT, "Bloom Quality: Weighted Transparency", 0);
if (config !== null) resize(WIDTH, HEIGHT, WIDTH, HEIGHT);
setTargetFPS(60);
setAutoExposure(false);
setManualExposure(1.0);
setTaaEnabled(true);
setTransparencyCompositionMode(sortedRoute ? "sorted" : "weighted");

const quality: QualityRun | null = config !== null ? new QualityRun(config) : null;
const quad = loadModel(
  refractiveRoute || transparentGiRoute
    ? "assets/transmission-quad.gltf"
    : "assets/transparent-quad.gltf",
);
const customTransparentMaterial = sortedInterleavingRoute
  ? compileTransparentMaterial(`
#include "material_abi.wgsl"

struct VsOut {
  @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VsOut {
  var out: VsOut;
  out.clip_position = draw.mvp * vec4<f32>(in.position, 1.0);
  return out;
}

@fragment
fn fs_main(_in: VsOut) -> TranslucentOut {
  var out: TranslucentOut;
  out.hdr = draw.model_tint;
  return out;
}
`)
  : 0;
let reflectedModel = quad;
if (reflectionHierarchyRoute) {
  createPlanarReflection(0.01, 0.0, 1.0, 0.0, 512);
  reflectedModel = loadModel("../renderer-test/assets/DamagedHelmet.glb");
}
const retainedQuads: number[] = [];
if (transparentGiRoute) {
  for (let index = 0; index < CELLS_X * CELLS_Y * LAYERS; index = index + 1) {
    const node = createSceneNode();
    attachModelToNode(node, quad.handle, 0);
    // Isolate transparent-GI timings from the independent transmitted-shadow
    // route while preserving camera-facing physical refraction.
    setSceneNodeCastShadow(node, false);
    retainedQuads.push(node);
  }
}
let simulationTime = 0.0;

while (!windowShouldClose()) {
  const capture = quality !== null ? quality.beginFrame() : false;
  const dt = quality !== null ? quality.deltaTime() : 1 / 60;
  simulationTime = simulationTime + dt;

  beginDrawing();
  setAmbientLight({ r: 190, g: 200, b: 220, a: 255 }, 0.7);
  setDirectionalLight(
    { x: -0.35, y: 0.8, z: 0.45 },
    { r: 255, g: 246, b: 232, a: 255 },
    1.0,
  );
  beginMode3D({
    position: reflectionHierarchyRoute
      ? { x: 0, y: 3.4, z: 8.0 }
      : { x: 0, y: 0, z: 9.0 },
    target: { x: 0, y: 0, z: 0 },
    up: { x: 0, y: 1, z: 0 },
    fovy: 48,
    projection: "perspective",
  });

  drawCube(
    { x: 0, y: 0, z: -1.8 },
    9.5, 6.5, 0.2,
    { r: 24, g: 31, b: 48, a: 255 },
  );

  if (reflectionHierarchyRoute) {
    // Opaque camera-side geometry supplies reflection hits. The tall side
    // columns remain directly visible while the smaller center objects sit
    // outside the pane's direct view and appear through its reflected rays.
    drawCube(
      { x: -1.65, y: 1.15, z: 0.1 },
      0.9, 2.3, 0.9,
      { r: 245, g: 58, b: 44, a: 255 },
    );
    drawCube(
      { x: 0.0, y: 1.0, z: -1.45 },
      1.2, 1.2, 1.2,
      { r: 36, g: 220, b: 105, a: 255 },
    );
    drawCube(
      { x: 1.65, y: 0.9, z: 0.35 },
      0.85, 1.8, 0.85,
      { r: 250, g: 185, b: 36, a: 255 },
    );
    drawCube(
      { x: 0.0, y: 2.35, z: -0.7 },
      1.4, 0.55, 0.55,
      { r: 48, g: 105, b: 250, a: 255 },
    );
    let reflectedTransform = mat4Identity();
    reflectedTransform = mat4Translate(
      reflectedTransform,
      { x: 0.0, y: 1.15, z: -0.6 },
    );
    reflectedTransform = mat4RotateY(reflectedTransform, simulationTime * 0.35);
    reflectedTransform = mat4Scale(
      reflectedTransform,
      { x: 1.35, y: 1.35, z: 1.35 },
    );
    drawModelTransform(
      reflectedModel,
      reflectedTransform,
      { r: 255, g: 255, b: 255, a: 255 },
    );

    let pane = mat4Identity();
    pane = mat4Translate(pane, { x: 0.0, y: 0.01, z: 0.4 });
    pane = mat4RotateX(pane, -1.57079632679);
    pane = mat4Scale(pane, { x: 2.8, y: 2.2, z: 1.0 });
    drawModelTransform(
      quad,
      pane,
      { r: 225, g: 238, b: 255, a: 255 },
    );
  } else for (let cell = 0; cell < CELLS_X * CELLS_Y; cell = cell + 1) {
    const cx = (cell % CELLS_X - (CELLS_X - 1) * 0.5) * 2.05;
    const cy = (Math.floor(cell / CELLS_X) - (CELLS_Y - 1) * 0.5) * 1.75;
    for (let layer = 0; layer < LAYERS; layer = layer + 1) {
      const phase = layer * 0.73 + cell * 0.19;
      let transform = mat4Identity();
      transform = mat4Translate(transform, {
        x: cx,
        y: cy,
        z: -0.22 + layer * 0.055,
      });
      transform = mat4RotateY(
        transform,
        -0.55 + layer * 0.16 + Math.sin(simulationTime * 0.7 + phase) * 0.09,
      );
      transform = mat4RotateX(
        transform,
        -0.24 + (layer % 4) * 0.16,
      );
      transform = mat4RotateZ(
        transform,
        Math.sin(simulationTime * 0.45 + phase) * 0.12,
      );
      transform = mat4Scale(transform, { x: 0.92, y: 0.74, z: 1.0 });
      if (transparentGiRoute) {
        setSceneNodeTransform(retainedQuads[cell * LAYERS + layer], transform);
      } else {
        drawModelTransform(
          quad,
          transform,
          {
            r: 70 + (layer * 47 + cell * 11) % 185,
            g: 65 + (layer * 29 + cell * 23) % 190,
            b: 85 + (layer * 61 + cell * 17) % 170,
            a: 255,
          },
        );
        if (sortedInterleavingRoute) {
          drawMeshWithMaterial(
            customTransparentMaterial,
            quad,
            {
              x: cx + 0.13,
              y: cy - 0.09,
              z: -0.2475 + layer * 0.055,
            },
            0.46,
            {
              r: 45 + (layer * 31 + cell * 17) % 170,
              g: 90 + (layer * 43 + cell * 13) % 150,
              b: 55 + (layer * 23 + cell * 29) % 180,
              a: 76,
            },
          );
        }
      }
    }
  }

  endMode3D();
  if (capture && quality !== null) quality.requestCapture();
  endDrawing();
  if (quality !== null && quality.endFrame()) break;
}

closeWindow();
