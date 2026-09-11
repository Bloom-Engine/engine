import {
  initWindow, runGameLifecycle, clearBackground, closeWindow, setDirect2DMode,
  setTargetFPS, writeFile, getPlatform, captureFrameToPng, isFrameCaptureReady,
} from "@bloomengine/engine/core";
import { drawRect } from "@bloomengine/engine/shapes";

const BLOOM_SMOKE_FAIL_STARTUP = false;
if (BLOOM_SMOKE_FAIL_STARTUP) {
  // The browser acceptance monitor records this real FFI call and injects a
  // WebAssembly.RuntimeError across its return boundary. This is a fault
  // injection control; the normal game never writes this marker.
  writeFile("compiled-web-expected-fault", "BLOOM_EXPECTED_STARTUP_FAILURE");
}

const browser = getPlatform() === 7;
let inits = 0;
let updates = 0;
let fixedUpdates = 0;
let lastTick = 0;
let lastAlpha = 0;
let frames = 0;
let cleanups = 0;
runGameLifecycle({
  init: () => {
    inits = inits + 1;
    initWindow(128, 128, "Bloom compiled lifecycle startup");
    setTargetFPS(60);
    setDirect2DMode(true);
  },
  fixedUpdate: (_dt, tick) => { fixedUpdates = fixedUpdates + 1; lastTick = tick; },
  update: (_dt) => { updates = updates + 1; },
  draw: (alpha) => {
    lastAlpha = alpha;
    clearBackground({ r: 0, g: 0, b: 0, a: 255 });
    drawRect(32, 32, 64, 64, { r: 255, g: 255, b: 255, a: 255 });
    frames = frames + 1;
    if (frames === 8) {
      writeFile("compiled-web-frames", frames.toString());
      if (browser) closeWindow();
      else captureFrameToPng("native-startup.png");
    }
    if (!browser && frames > 8 && isFrameCaptureReady()) closeWindow();
    if (frames >= 120) closeWindow();
  },
  cleanup: () => {
    cleanups = cleanups + 1;
    writeFile("compiled-web-cleanups", cleanups.toString());
    writeFile("compiled-web-lifecycle", inits + "," + updates + "," + frames + "," + fixedUpdates + "," + lastTick + "," + lastAlpha);
    if (!browser) writeFile("native-cleanup.txt", cleanups.toString());
  },
}, { fixedStepSeconds: 0.01 });
