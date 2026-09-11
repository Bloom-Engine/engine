import {
  initWindow, runGame, clearBackground, closeWindow, setDirect2DMode,
  setTargetFPS, writeFile,
} from "@bloomengine/engine/core";
import { drawRect } from "@bloomengine/engine/shapes";

const BLOOM_SMOKE_FAIL_STARTUP = false;
if (BLOOM_SMOKE_FAIL_STARTUP) {
  // The browser acceptance monitor records this real FFI call and injects a
  // WebAssembly.RuntimeError across its return boundary. This is a fault
  // injection control; the normal game never writes this marker.
  writeFile("compiled-web-expected-fault", "BLOOM_EXPECTED_STARTUP_FAILURE");
}

initWindow(128, 128, "Bloom compiled web startup");
setTargetFPS(60);
setDirect2DMode(true);
let frames = 0;
let cleanups = 0;
runGame((_dt) => {
  clearBackground({ r: 0, g: 0, b: 0, a: 255 });
  drawRect(32, 32, 64, 64, { r: 255, g: 255, b: 255, a: 255 });
  frames = frames + 1;
  if (frames === 8) {
    writeFile("compiled-web-frames", frames.toString());
    closeWindow();
  }
}, () => {
  cleanups = cleanups + 1;
  writeFile("compiled-web-cleanups", cleanups.toString());
});
