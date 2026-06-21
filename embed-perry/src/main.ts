// Bloom engine embedded inside a normal Perry UI app — issue #2395.
//
// The window chrome (title label, vertical stack) is plain Perry UI. The large
// viewport is a `BloomView`: Perry UI reserves a native child window and hands
// its HWND to the Bloom engine via `attachToHwnd`. From then on Bloom owns that
// surface and we drive a 3D scene every tick, while the surrounding Perry UI
// stays fully interactive.

import { App, VStack, Text, BloomView, bloomViewGetHwnd } from 'perry/ui';
import {
  attachToHwnd,
  beginDrawing, endDrawing, clearBackground,
  beginMode3D, endMode3D,
  drawCube, drawSphere,
  setAmbientLight, setDirectionalLight,
  getTime, vec3,
} from 'bloom';

const VIEW_W = 820;
const VIEW_H = 480;

// Perry UI reserves the child window; Bloom renders into it.
const view = BloomView(VIEW_W, VIEW_H);

// Drive Bloom's frame loop from the Perry UI run loop. We attach lazily on the
// first tick: by then App() has shown the window and laid the BloomView child
// out at its final size and parent, so Bloom builds its surface on a stable,
// visible window.
let attached = 0;
setInterval(() => {
  if (attached === 0) {
    attachToHwnd(bloomViewGetHwnd(view), VIEW_W, VIEW_H);
    setAmbientLight({ r: 120, g: 140, b: 180, a: 255 }, 0.40);
    setDirectionalLight(vec3(0.6, 0.9, 0.4), { r: 255, g: 235, b: 205, a: 255 }, 0.85);
    attached = 1;
  }

  const t = getTime();
  const cx = Math.cos(t * 0.6) * 9.0;
  const cz = Math.sin(t * 0.6) * 9.0;

  beginDrawing();
  clearBackground({ r: 18, g: 22, b: 34, a: 255 });

  beginMode3D({
    position: vec3(cx, 6.0, cz),
    target: vec3(0, 0.5, 0),
    up: vec3(0, 1, 0),
    fovy: 45.0,
    projection: 0.0,
  });

  // Ground plane.
  drawCube(vec3(0, -0.6, 0), 40.0, 0.4, 40.0, { r: 40, g: 50, b: 70, a: 255 });

  // A spinning ring of cubes around a glowing core.
  const N = 8;
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2.0 + t;
    const x = Math.cos(a) * 4.0;
    const z = Math.sin(a) * 4.0;
    const h = 1.2 + Math.sin(t * 2.0 + i) * 0.6;
    drawCube(
      vec3(x, h * 0.5, z),
      0.9, h, 0.9,
      { r: 120 + i * 14, g: 200 - i * 10, b: 240, a: 255 },
    );
  }

  drawSphere(vec3(0, 1.4, 0), 1.1, { r: 255, g: 210, b: 120, a: 255 });
  drawSphere(vec3(0, 1.4, 0), 1.6, { r: 255, g: 180, b: 90, a: 60 });

  endMode3D();
  endDrawing();
}, 16);

App({
  title: "Perry UI × Bloom",
  width: 880,
  height: 600,
  body: VStack(10, [
    Text("🌸  Bloom engine rendering inside a Perry UI app  —  issue #2395"),
    view,
  ]),
});
