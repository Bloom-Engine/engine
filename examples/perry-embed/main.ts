// Bloom rendering inside a normal Perry UI app — perry issue #2395 / #5519.
//
// The window chrome (title label, vertical stack) is plain Perry UI. The large
// viewport is a `BloomView`: Perry UI reserves a native view in its own view
// tree and hands out that view's platform handle. Bloom attaches its GPU
// surface to the handle and renders a live 3D scene into it, while the
// surrounding Perry UI stays fully interactive.
//
// The split is the whole point: **Perry UI does not link, or know about, Bloom.**
// It reserves a native view and exposes a handle; anything can render into it.
// Apps that never call `BloomView` pull in nothing extra. (Same shape as
// Flutter's PlatformView — Flutter never learns about Flame.)
//
// Three rules this demo exists to demonstrate, each of which is easy to get
// wrong:
//
//   1. The host owns the run loop. Drive Bloom's frame yourself via `onFrame`
//      (re-armed each frame) — NEVER `runGame`, which blocks forever and would
//      deadlock the UI.
//   2. Attach on the first frame the handle is NON-ZERO, not merely the first
//      tick. The native view exists immediately; its handle is only usable once
//      the window is actually on screen.
//   3. `attachToNativeView` is portable and returns whether it worked. Do not
//      reach for the Windows-only `attachToHwnd`/`bloomViewGetHwnd` pair — both
//      are deprecated aliases at their respective ends.
//
// Perry implements BloomView on windows / macos / ios / tvos / visionos /
// android / gtk4, and `bloomViewGetNativeHandle` returns the right thing for
// each (HWND / NSView* / UIView* / GtkWidget* / ANativeWindow*). Bloom's
// `attachToNativeView` takes any of them, so the code below is not
// Windows-specific.
//
// Run (with a stock Perry — no fork required since #2395 merged):
//   perry compile main.ts -o perry-embed && ./perry-embed

import { App, VStack, Text, BloomView, bloomViewGetNativeHandle, onFrame } from 'perry/ui';
import {
  attachToNativeView,
  beginDrawing, endDrawing, clearBackground,
  beginMode3D, endMode3D,
  drawCube, drawSphere,
  setAmbientLight, setDirectionalLight,
  getTime, vec3,
} from 'bloom';

const VIEW_W = 820;
const VIEW_H = 480;

// Perry UI reserves the native view; Bloom renders into it.
const view = BloomView(VIEW_W, VIEW_H);

let attached = false;

function frame(_timestampMs: number, _deltaMs: number): void {
  // Rule 2: the handle is only usable once the window is on screen, so keep
  // trying until it is non-zero rather than assuming the first tick is late
  // enough. Rule 3: attachToNativeView tells us whether it actually worked.
  if (!attached) {
    const handle = bloomViewGetNativeHandle(view);
    if (handle !== 0) {
      attached = attachToNativeView(handle, VIEW_W, VIEW_H);
      if (attached) {
        setAmbientLight({ r: 120, g: 140, b: 180, a: 255 }, 0.40);
        setDirectionalLight(vec3(0.6, 0.9, 0.4), { r: 255, g: 235, b: 205, a: 255 }, 0.85);
      }
    }
  }

  if (attached) {
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
  }

  onFrame(frame);   // rule 1: re-arm; the host drives us, we never drive it
}

onFrame(frame);

App({
  title: 'Perry UI × Bloom',
  width: 880,
  height: 600,
  body: VStack(10, [
    Text('🌸  Bloom engine rendering inside a Perry UI app  —  perry #2395 / #5519'),
    view,
  ]),
});
