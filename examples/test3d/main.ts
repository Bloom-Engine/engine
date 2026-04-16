import { initWindow, windowShouldClose, beginDrawing, endDrawing, clearBackground, setTargetFPS, drawText, drawCube, drawGrid, beginMode3D, endMode3D, Colors } from 'bloom';

initWindow(800, 600, "Bloom 3D Test");
setTargetFPS(60);

while (!windowShouldClose()) {
  beginDrawing();
  clearBackground(Colors.RAYWHITE);
  beginMode3D({ position: { x: 10, y: 10, z: 10 }, target: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 }, fovy: 45, projection: "perspective" });
  drawCube({ x: 0, y: 1, z: 0 }, 2, 2, 2, { r: 200, g: 50, b: 50, a: 255 });
  drawGrid(10, 1.0);
  endMode3D();
  drawText("Bloom 3D Test", 10, 10, 20, Colors.BLACK);
  endDrawing();
}
