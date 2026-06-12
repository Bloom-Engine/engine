// ============================================================
// Bloom PBR Material Grid (validation scene)
// ============================================================
// 5×5 grid of spheres sharing a base color (gold-ish), one row
// per metallic value (top = full metal, bottom = full dielectric)
// and one column per roughness value (left = smooth, right = rough).
// All lit purely by an HDR environment map — no directional,
// ambient, point, or sun lights — so any divergence from the path-
// traced reference comes from the material system, not from lighting
// setup mismatch.
//
// Run modes:
//   - Interactive: orbit-cam (no inputs yet, just a fixed angle).
//   - Headless:    --camera px py pz tx ty tz fov --out path.png
//                  (matches renderer-test's headless interface).

import {
  initWindow, beginDrawing, endDrawing, takeScreenshot,
  setEnvClearFromHdr,
  setTargetFPS,
  windowShouldClose,
  beginMode3D, endMode3D,
} from "bloom/core";
import { genMeshCube, createMeshExplicit } from "bloom/models";
import {
  createSceneNode, setSceneNodeTransform,
  setSceneNodeColor, setSceneNodePbr,
  setSceneNodeCastShadow, setSceneNodeReceiveShadow,
  attachModelToNode,
} from "bloom/scene";
import { mat4Identity, mat4Translate, mat4Scale } from "bloom/math";

// ---- Scene parameters ----
// Picked so the grid spans roughly ±3 units in X/Y, fitting a
// 45° fov from ~10 units away.
const GRID_N = 5;          // 5×5 spheres
const SPHERE_R = 0.42;     // unit sphere scaled
const SPACING = 1.1;       // center-to-center distance
const BASE_R = 1.00;       // base color
const BASE_G = 0.71;       // gold-ish — makes metallic vs dielectric
const BASE_B = 0.29;       // visually unambiguous (metals tint, dielectrics stay yellow-ish white)
// ---- Headless mode args (mirrors renderer-test) ----
let headlessOutPath = "";
let headlessMode = false;
let headlessCamX = 0.0;
let headlessCamY = 0.0;
let headlessCamZ = 6.0;
let headlessTargetX = 0.0;
let headlessTargetY = 0.0;
let headlessTargetZ = 0.0;
let headlessFov = 45.0;
let headlessResW = 512;
let headlessResH = 512;

declare const process: { argv: string[] };
const argv: string[] = process.argv;
for (let i = 2; i < argv.length; i = i + 1) {
  if (argv[i] === "--out" && i + 1 < argv.length) {
    headlessOutPath = argv[i + 1];
    headlessMode = true;
  } else if (argv[i] === "--camera" && i + 7 < argv.length) {
    headlessCamX = parseFloat(argv[i + 1]);
    headlessCamY = parseFloat(argv[i + 2]);
    headlessCamZ = parseFloat(argv[i + 3]);
    headlessTargetX = parseFloat(argv[i + 4]);
    headlessTargetY = parseFloat(argv[i + 5]);
    headlessTargetZ = parseFloat(argv[i + 6]);
    headlessFov = parseFloat(argv[i + 7]);
    headlessMode = true;
  } else if (argv[i] === "--res" && i + 2 < argv.length) {
    headlessResW = Math.floor(parseFloat(argv[i + 1]));
    headlessResH = Math.floor(parseFloat(argv[i + 2]));
    headlessMode = true;
  }
}

// ---- Init ----
initWindow(headlessResW, headlessResH, "Bloom PBR Spheres", 0);
setTargetFPS(60);
setEnvClearFromHdr("assets/outdoor.hdr");

// Mirror renderer-test's pattern exactly: declare let-binding for
// the handle, populate inside a function (function-local scope
// behaves differently from top-level under Perry), use the bare
// number handle thereafter.
const PI = 3.14159265;
const TWO_PI = 6.28318530;

function makeSphereVertices(radius: number, segs: number, rings: number): number[] {
  const v: number[] = [];
  for (let r = 0; r <= rings; r = r + 1) {
    const phi = PI * r / rings;
    const sp = Math.sin(phi);
    const cp = Math.cos(phi);
    for (let s = 0; s <= segs; s = s + 1) {
      const theta = TWO_PI * s / segs;
      const st = Math.sin(theta);
      const ct = Math.cos(theta);
      const x = sp * ct;
      const y = cp;
      const z = sp * st;
      v.push(x * radius, y * radius, z * radius);
      v.push(x, y, z);
      v.push(1, 1, 1, 1);
      v.push(s / segs, r / rings);
    }
  }
  return v;
}

function makeSphereIndices(segs: number, rings: number): number[] {
  const idx: number[] = [];
  for (let r = 0; r < rings; r = r + 1) {
    for (let s = 0; s < segs; s = s + 1) {
      const a = r * (segs + 1) + s;
      const b = a + segs + 1;
      idx.push(a, b, a + 1);
      idx.push(b, b + 1, a + 1);
    }
  }
  return idx;
}

let sphereHandle = 0;
let cubeHandle = 0;
// Procedural sphere mesh generator. Pre-sizes the arrays via
// `new Array(N)` then writes each slot by index. Avoids `.push()`
// entirely because Perry's current backend has bugs around .push:
// `.length` reports the literal-init size (not post-push), and the
// pushed data sometimes doesn't land where the FFI reads it.
// `new Array(N)` + index assignment is documented as working
// (per the perry-llvm fix history).
function makeSphere(segs: number, rings: number): {
  vertices: number[]; vertexCount: number;
  indices: number[];  indexCount: number;
} {
  const vCount = (rings + 1) * (segs + 1);
  const iCount = rings * segs * 6;
  const verts: number[] = new Array(vCount * 12);
  const inds:  number[] = new Array(iCount);
  let vi = 0;
  for (let r = 0; r <= rings; r = r + 1) {
    const phi = PI * r / rings;
    const sp = Math.sin(phi);
    const cp = Math.cos(phi);
    for (let s = 0; s <= segs; s = s + 1) {
      const theta = TWO_PI * s / segs;
      const x = sp * Math.cos(theta);
      const y = cp;
      const z = sp * Math.sin(theta);
      verts[vi]      = x; verts[vi + 1]  = y; verts[vi + 2]  = z;       // pos
      verts[vi + 3]  = x; verts[vi + 4]  = y; verts[vi + 5]  = z;       // normal
      verts[vi + 6]  = 1; verts[vi + 7]  = 1; verts[vi + 8]  = 1;
      verts[vi + 9]  = 1;                                                // color rgba
      verts[vi + 10] = s / segs; verts[vi + 11] = r / rings;             // uv
      vi = vi + 12;
    }
  }
  let ii = 0;
  for (let r = 0; r < rings; r = r + 1) {
    for (let s = 0; s < segs; s = s + 1) {
      const a = r * (segs + 1) + s;
      const b = a + segs + 1;
      inds[ii]     = a; inds[ii + 1] = b; inds[ii + 2] = a + 1;
      inds[ii + 3] = b; inds[ii + 4] = b + 1; inds[ii + 5] = a + 1;
      ii = ii + 6;
    }
  }
  return { vertices: verts, vertexCount: vCount, indices: inds, indexCount: iCount };
}

function initSharedMeshes(): void {
  const sphere = makeSphere(24, 16);
  sphereHandle = createMeshExplicit(
    sphere.vertices, sphere.vertexCount,
    sphere.indices,  sphere.indexCount,
  ).handle;
  cubeHandle = genMeshCube(1.0, 1.0, 1.0).handle;
}
initSharedMeshes();

// Build the grid. Roughness clamped away from exact 0 / 1 because
// the GGX BRDF is undefined at zero roughness and the prefilter
// shader's mip lookup behaves oddly at exactly 1. The 0.05/0.95
// endpoints match the standard reference scenes (Khronos
// MetalRoughSpheres uses similar bounds).
const ROUGH_MIN = 0.05;
const ROUGH_MAX = 0.95;

for (let row = 0; row < GRID_N; row = row + 1) {
  // metallic varies along Y (top row = full metal)
  const metallic = (GRID_N - 1 - row) / (GRID_N - 1);
  for (let col = 0; col < GRID_N; col = col + 1) {
    const roughness = ROUGH_MIN + (ROUGH_MAX - ROUGH_MIN) * (col / (GRID_N - 1));

    const node = createSceneNode();
    attachModelToNode(node, sphereHandle, 0);
    setSceneNodeColor(node, BASE_R * 255, BASE_G * 255, BASE_B * 255);
    setSceneNodePbr(node, roughness, metallic);
    setSceneNodeCastShadow(node, false);
    setSceneNodeReceiveShadow(node, false);

    const x = (col - (GRID_N - 1) / 2) * SPACING;
    const y = (row - (GRID_N - 1) / 2) * SPACING;
    let m = mat4Identity();
    m = mat4Translate(m, { x: x, y: y, z: 0.0 });
    m = mat4Scale(m, { x: SPHERE_R, y: SPHERE_R, z: SPHERE_R });
    setSceneNodeTransform(node, m);
  }
}

// ---- Loop ----
let headlessFrame = 0;
const HEADLESS_WARMUP_FRAMES = 30;

while (!windowShouldClose()) {
  beginDrawing();

  beginMode3D({
    position: { x: headlessCamX, y: headlessCamY, z: headlessCamZ },
    target: { x: headlessTargetX, y: headlessTargetY, z: headlessTargetZ },
    up: { x: 0, y: 1, z: 0 },
    fovy: headlessFov,
    projection: "perspective",
  });
  endMode3D();

  endDrawing();

  if (headlessMode) {
    headlessFrame = headlessFrame + 1;
    if (headlessFrame === HEADLESS_WARMUP_FRAMES) {
      takeScreenshot(headlessOutPath);
    } else if (headlessFrame > HEADLESS_WARMUP_FRAMES) {
      break;
    }
  }
}
