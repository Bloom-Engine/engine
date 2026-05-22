/**
 * R3F Bridge Example — Phase 6 proof-of-concept.
 *
 * This demonstrates what Pascal Editor renderer code looks like when compiled
 * via perry-react-three-fiber. The JSX below shows R3F patterns that Perry
 * compiles to Bloom FFI calls through the bridge.
 *
 * In practice, Perry's JSX compiler transforms:
 *   <mesh castShadow><boxGeometry args={[1,2,3]}/><meshStandardMaterial color="white"/></mesh>
 * into:
 *   buildR3FIntrinsic("mesh", {castShadow: true}, [
 *     buildR3FIntrinsic("boxGeometry", {args: [1,2,3]}, null),
 *     buildR3FIntrinsic("meshStandardMaterial", {color: "white"}, null),
 *   ])
 * which calls:
 *   bloom_scene_create_node()
 *   bloom_scene_set_cast_shadow(handle, 1)
 *   bloom_scene_update_geometry(handle, boxVertices, ...)
 *   bloom_scene_set_material_color(handle, 1, 1, 1, 1)
 *   bloom_scene_set_material_pbr(handle, 0.8, 0.0)
 */

import {
  initWindow, windowShouldClose, beginDrawing, endDrawing,
  clearBackground, setTargetFPS, drawText,
  beginMode3D, endMode3D, drawGrid,
  Colors, getDeltaTime,
  setAmbientLight, setDirectionalLight,
} from 'bloom';

import {
  buildR3FIntrinsic, useFrame,
} from 'perry-react-three-fiber';

import {
  getSceneNodeCount, enableShadows,
  setPostFxSelected, pickScene, projectToScreen,
  addDirectionalLight,
  registerFrameCallback,
} from 'bloom/scene';

import { isMouseButtonPressed, getMouseX, getMouseY, MouseButton } from 'bloom';

// ============================================================
// Simulate Pascal Editor's WallRenderer (compiled from R3F JSX)
// ============================================================

// This is what Perry generates from:
//   <mesh castShadow receiveShadow>
//     <boxGeometry args={[5, 3, 0.2]} />
//     <meshStandardMaterial color="white" roughness={0.8} />
//   </mesh>
const wall1 = buildR3FIntrinsic("mesh", { castShadow: true, receiveShadow: true }, null);
buildR3FIntrinsic("boxGeometry", { args: [5, 3, 0.2] }, null);
buildR3FIntrinsic("meshStandardMaterial", { color: 0xf2f2ee, roughness: 0.8 }, null);

// Second wall
const wall2 = buildR3FIntrinsic("mesh", { castShadow: true, receiveShadow: true }, null);
buildR3FIntrinsic("boxGeometry", { args: [0.2, 3, 4] }, null);
buildR3FIntrinsic("meshStandardMaterial", { color: 0xeeeee8, roughness: 0.8 }, null);

// Floor
const floor = buildR3FIntrinsic("mesh", { receiveShadow: true }, null);
buildR3FIntrinsic("planeGeometry", { args: [10, 10] }, null);
buildR3FIntrinsic("meshStandardMaterial", { color: 0xccccbb, roughness: 0.6 }, null);

// Simulate: <directionalLight position={[5, 10, 3]} intensity={0.7} castShadow />
buildR3FIntrinsic("directionalLight", {
  position: [5, 10, 3],
  intensity: 0.7,
  color: 0xfffff0,
  castShadow: true,
}, null);

// ============================================================
// Simulate useFrame (R3F render loop hooks)
// ============================================================

// This is what Perry generates from:
//   useFrame((state, delta) => {
//     addDirectionalLight(0.5, 1, 0.3, 1, 0.95, 0.9, 0.6);
//   }, 5);
useFrame((state: any, delta: number) => {
  addDirectionalLight(0.5, 1.0, 0.3, 1.0, 0.95, 0.9, 0.6);
  addDirectionalLight(-0.3, 0.5, -0.7, 0.8, 0.85, 0.95, 0.2);
}, 5);

// ============================================================
// Main loop
// ============================================================

initWindow(1280, 720, "Bloom — R3F Bridge Demo (Phase 6)");
setTargetFPS(60);
setAmbientLight(255, 255, 255, 0.25);
enableShadows();

let angle = 0;
let selectedHandle: number = 0;

while (!windowShouldClose()) {
  const dt = getDeltaTime();
  angle += dt * 0.2;

  beginDrawing();
  clearBackground(Colors.SNOW);

  const camX = Math.cos(angle) * 10;
  const camZ = Math.sin(angle) * 10;
  beginMode3D({
    position: { x: camX, y: 6, z: camZ },
    target: { x: 0, y: 1.5, z: 0 },
    up: { x: 0, y: 1, z: 0 },
    fovy: 45,
    projection: "perspective",
  });

  // Click to select (like Pascal Editor's SelectionManager)
  if (isMouseButtonPressed(MouseButton.LEFT)) {
    const hit = pickScene(getMouseX(), getMouseY());
    if (hit.hit) {
      selectedHandle = hit.handle;
      setPostFxSelected(hit.handle);
    } else {
      selectedHandle = 0;
      setPostFxSelected(0);
    }
  }

  // 3D→2D projection demo (like drei's Html component)
  const labelPos = projectToScreen(0, 3.5, 0);

  drawGrid(20, 1.0);
  endMode3D();

  // HUD
  drawText("R3F Bridge Demo (Phase 6)", 10, 10, 20, Colors.DARKGRAY);
  drawText("Scene nodes: " + String(getSceneNodeCount()), 10, 35, 16, Colors.GRAY);
  drawText("R3F elements compiled to Bloom FFI calls", 10, 55, 16, Colors.GRAY);
  drawText("Click to select | Outlines on selected", 10, 75, 16, Colors.GRAY);

  // Show projected 3D label position
  if (labelPos.visible) {
    drawText("^ Wall Top (3D projected)", labelPos.x, labelPos.y, 14, Colors.BLUE);
  }

  if (selectedHandle > 0) {
    drawText("Selected: node " + String(selectedHandle), 10, 100, 14, Colors.BLUE);
  }

  endDrawing();
}
