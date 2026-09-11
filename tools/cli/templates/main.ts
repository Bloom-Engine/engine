import {
  initWindow, runGame, closeWindow, clearBackground, setTargetFPS,
  setDirect2DMode, getScreenWidth, getScreenHeight, readFile, Colors,
} from "@bloomengine/engine/core";
import { drawRect } from "@bloomengine/engine/shapes";
import { drawText } from "@bloomengine/engine/text";

let elapsed = 0;
let greeting = "";

function init(): void {
  initWindow(800, 450, "My Bloom game");
  setTargetFPS(60);
  setDirect2DMode(true);
  greeting = readFile("assets/welcome.txt");
  if (greeting.length === 0) throw new Error("Missing starter asset: assets/welcome.txt");
}

function update(dt: number): void {
  // Variable update, with a cap on time accumulated while a tab is hidden.
  elapsed = elapsed + Math.min(dt, 0.1);
}

function draw(): void {
  clearBackground(Colors.BLACK);
  const x = getScreenWidth() / 2 - 32 + Math.sin(elapsed) * 80;
  drawRect(x, getScreenHeight() / 2 - 32, 64, 64, Colors.WHITE);
  drawText(greeting, 24, 24, 24, Colors.WHITE);
}

function cleanup(): void {
  // Dispose owned textures, models, audio and physics resources here.
  closeWindow();
}

init();
runGame((dt) => { update(dt); draw(); }, cleanup);
