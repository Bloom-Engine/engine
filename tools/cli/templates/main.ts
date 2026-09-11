import {
  initWindow, runGameLifecycle, closeWindow, clearBackground, setTargetFPS,
  setDirect2DMode, getScreenWidth, getScreenHeight, readFile, Colors,
} from "@bloomengine/engine/core";
import { drawRect } from "@bloomengine/engine/shapes";
import { drawText } from "@bloomengine/engine/text";

let elapsed = 0;
let previousElapsed = 0;
let greeting = "";

function init(): void {
  initWindow(800, 450, "My Bloom game");
  setTargetFPS(60);
  setDirect2DMode(true);
  greeting = readFile("assets/welcome.txt");
  if (greeting.length === 0) throw new Error("Missing starter asset: assets/welcome.txt");
}

function fixedUpdate(dt: number): void {
  previousElapsed = elapsed;
  elapsed = elapsed + dt;
}

function update(_dt: number): void {
  // Read per-frame input and update non-simulation state here.
}

function draw(alpha: number): void {
  clearBackground(Colors.BLACK);
  const displayTime = previousElapsed + (elapsed - previousElapsed) * alpha;
  const x = getScreenWidth() / 2 - 32 + Math.sin(displayTime) * 80;
  drawRect(x, getScreenHeight() / 2 - 32, 64, 64, Colors.WHITE);
  drawText(greeting, 24, 24, 24, Colors.WHITE);
}

function cleanup(): void {
  // Dispose owned textures, models, audio and physics resources here.
  closeWindow();
}

runGameLifecycle({ init, fixedUpdate, update, draw, cleanup });
