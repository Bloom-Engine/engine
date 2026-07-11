// Shared spawn helpers for the parts of a world that are not entities: water
// volumes and rivers.
//
// Both the runtime loader (`instantiateWorld`, used by games) and the world
// editor call these, so a river looks the same in the editor as it does in the
// game. Anything that renders world data belongs here rather than in either
// consumer — the editor previously drew its own translucent debug cubes, which
// is exactly how the two drift apart.
//
// Colour convention: the world schema stores RGBA as 0-1 floats, while the
// scene API takes 0-255. The conversion lives here, once.

import {
  createSceneNode, setSceneNodeVisible, setSceneNodeWaterMaterial,
  attachModelToNode, setSceneNodeColor,
  SceneNodeHandle,
} from '../scene/index';
import { genMeshCube, genMeshSplineRibbon } from '../models/index';
import { setSceneNodeTransform } from '../scene/index';
import { WaterVolume, RiverSpline } from './types';

// A water volume renders as a box whose *top face* sits at `surfaceHeight`
// (the schema's `center.y` positions the body of water; the surface is what the
// player sees and what the wave shader animates).
export function spawnWaterVolume(volume: WaterVolume): SceneNodeHandle {
  const node = createSceneNode();
  const cube = genMeshCube(1, 1, 1);
  attachModelToNode(node, cube.handle, 0);

  const sx = volume.size[0];
  const sy = volume.size[1];
  const sz = volume.size[2];

  // Column-major TRS: scale on the diagonal, translation in the last column.
  // Y is placed so the top face lands on surfaceHeight.
  const cy = volume.surfaceHeight - sy / 2;
  setSceneNodeTransform(node, [
    sx, 0, 0, 0,
    0, sy, 0, 0,
    0, 0, sz, 0,
    volume.center[0], cy, volume.center[2], 1,
  ]);

  const c = volume.color;
  setSceneNodeWaterMaterial(
    node,
    volume.waveAmplitude, volume.waveSpeed,
    c[0] * 255, c[1] * 255, c[2] * 255, c[3] * 255,
  );
  setSceneNodeVisible(node, true);
  return node;
}

// A river renders as a ribbon mesh swept along its control points, dropped by
// `depth` so it sits in its channel rather than on top of the terrain. Widths
// are per control point; a river with fewer widths than points repeats the last.
export function spawnRiver(river: RiverSpline): SceneNodeHandle {
  const pointCount = river.controlPoints.length;
  if (pointCount < 2) return 0;

  const points: number[] = [];
  for (let i = 0; i < pointCount; i++) {
    const p = river.controlPoints[i];
    points.push(p[0]);
    points.push(p[1] - river.depth);
    points.push(p[2]);
  }

  const widths: number[] = [];
  for (let i = 0; i < pointCount; i++) {
    const w = i < river.widths.length
      ? river.widths[i]
      : (river.widths.length > 0 ? river.widths[river.widths.length - 1] : 1);
    widths.push(w);
  }

  const ribbon = genMeshSplineRibbon(points, widths);
  if (ribbon.handle === 0) return 0;

  const node = createSceneNode();
  attachModelToNode(node, ribbon.handle, 0);

  const c = river.color;
  // Flow speed drives the same wave animation as a water volume; a river with
  // no flow still ripples gently rather than reading as a flat plastic strip.
  setSceneNodeWaterMaterial(
    node,
    0.05, river.flowSpeed,
    c[0] * 255, c[1] * 255, c[2] * 255, c[3] * 255,
  );
  setSceneNodeVisible(node, true);
  return node;
}

// Editor-only: tint a water/river node to show selection. Games never call this.
export function setWaterHighlight(handle: SceneNodeHandle, selected: boolean): void {
  if (selected) {
    setSceneNodeColor(handle, 255, 220, 120, 255);
  }
}
