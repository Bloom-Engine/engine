import {
  initWindow, windowShouldClose, beginDrawing, endDrawing,
  clearBackground, setTargetFPS, getDeltaTime, isKeyDown, isKeyPressed,
  isMouseButtonPressed, closeWindow, beginMode3D, endMode3D,
  disableCursor, getMouseDeltaX, getMouseDeltaY,
} from "bloom/core";
import { Color, Key, Camera3D, MouseButton } from "bloom/core";
import { drawCube, drawCubeWires } from "bloom/models";
import { drawText } from "bloom/text";
import { drawRect } from "bloom/shapes";
import {
  clamp, randomInt,
} from "bloom/math";

// Constants
const SCREEN_WIDTH = 960;
const SCREEN_HEIGHT = 540;
const CHUNK_SIZE = 16;
const WORLD_CHUNKS_X = 4;
const WORLD_CHUNKS_Z = 4;
const WORLD_HEIGHT = 32;
const BLOCK_AIR = 0;
const BLOCK_GRASS = 1;
const BLOCK_DIRT = 2;
const BLOCK_STONE = 3;
const BLOCK_WOOD = 4;
const BLOCK_LEAVES = 5;
const BLOCK_SAND = 6;
const BLOCK_WATER = 7;

// Pre-allocated block color table (avoid allocations in hot render loop)
const BLOCK_COLORS: Color[] = [
  { r: 255, g: 0, b: 255, a: 255 },   // 0: AIR (unused)
  { r: 60, g: 170, b: 60, a: 255 },    // 1: GRASS
  { r: 130, g: 90, b: 50, a: 255 },    // 2: DIRT
  { r: 128, g: 128, b: 128, a: 255 },  // 3: STONE
  { r: 120, g: 80, b: 40, a: 255 },    // 4: WOOD
  { r: 30, g: 130, b: 30, a: 200 },    // 5: LEAVES
  { r: 220, g: 200, b: 120, a: 255 },  // 6: SAND
  { r: 40, g: 80, b: 200, a: 150 },    // 7: WATER
];

// World voxel storage
const worldSizeX = WORLD_CHUNKS_X * CHUNK_SIZE;
const worldSizeZ = WORLD_CHUNKS_Z * CHUNK_SIZE;
const blocks: number[] = [];
for (let i = 0; i < worldSizeX * WORLD_HEIGHT * worldSizeZ; i++) {
  blocks.push(BLOCK_AIR);
}

function blockIndex(x: number, y: number, z: number): number {
  return (y * worldSizeX * worldSizeZ) + (z * worldSizeX) + x;
}

function getBlock(x: number, y: number, z: number): number {
  if (x < 0 || x >= worldSizeX || y < 0 || y >= WORLD_HEIGHT || z < 0 || z >= worldSizeZ) return BLOCK_AIR;
  return blocks[blockIndex(x, y, z)];
}

function setBlock(x: number, y: number, z: number, type: number): void {
  if (x < 0 || x >= worldSizeX || y < 0 || y >= WORLD_HEIGHT || z < 0 || z >= worldSizeZ) return;
  blocks[blockIndex(x, y, z)] = type;
}

// Simple terrain generation using sine waves
function generateTerrain(): void {
  for (let z = 0; z < worldSizeZ; z++) {
    for (let x = 0; x < worldSizeX; x++) {
      const height = Math.floor(
        8 + Math.sin(x * 0.1) * 3 + Math.cos(z * 0.12) * 3
        + Math.sin(x * 0.05 + z * 0.05) * 5
      );

      for (let y = 0; y < WORLD_HEIGHT; y++) {
        if (y === 0) {
          setBlock(x, y, z, BLOCK_STONE);
        } else if (y < height - 3) {
          setBlock(x, y, z, BLOCK_STONE);
        } else if (y < height) {
          setBlock(x, y, z, BLOCK_DIRT);
        } else if (y === height) {
          if (height < 6) {
            setBlock(x, y, z, BLOCK_SAND);
          } else {
            setBlock(x, y, z, BLOCK_GRASS);
          }
        } else if (y <= 5) {
          setBlock(x, y, z, BLOCK_WATER);
        }
      }
    }
  }

  // Generate some trees
  for (let t = 0; t < 20; t++) {
    const tx = randomInt(3, worldSizeX - 4);
    const tz = randomInt(3, worldSizeZ - 4);
    // Find surface height
    let surfaceY = 0;
    for (let y = WORLD_HEIGHT - 1; y >= 0; y--) {
      if (getBlock(tx, y, tz) === BLOCK_GRASS) { surfaceY = y; break; }
    }
    if (surfaceY < 7) continue;

    const trunkHeight = randomInt(4, 6);
    for (let y = 1; y <= trunkHeight; y++) {
      setBlock(tx, surfaceY + y, tz, BLOCK_WOOD);
    }
    // Canopy
    for (let dy = -2; dy <= 1; dy++) {
      const radius = dy < 0 ? 2 : 1;
      for (let dx = -radius; dx <= radius; dx++) {
        for (let dz = -radius; dz <= radius; dz++) {
          if (dx === 0 && dz === 0 && dy < 0) continue;
          setBlock(tx + dx, surfaceY + trunkHeight + dy, tz + dz, BLOCK_LEAVES);
        }
      }
    }
  }
}

// FPS camera
let camYaw = 0;
let camPitch = 0;
let camX = worldSizeX / 2;
let camY = 20;
let camZ = worldSizeZ / 2;
const MOVE_SPEED = 12;
const MOUSE_SENS = 0.003;
let selectedBlock = BLOCK_STONE;

// Currently highlighted block
let highlightX = -1;
let highlightY = -1;
let highlightZ = -1;

function getCamForward(): { x: number; y: number; z: number } {
  return {
    x: Math.cos(camPitch) * Math.sin(camYaw),
    y: Math.sin(camPitch),
    z: Math.cos(camPitch) * Math.cos(camYaw),
  };
}

function getCamRight(): { x: number; y: number; z: number } {
  return { x: Math.cos(camYaw), y: 0, z: -Math.sin(camYaw) };
}

// Simple ray march to find block under crosshair
function raycastBlock(): void {
  const dir = getCamForward();
  let rx = camX;
  let ry = camY;
  let rz = camZ;
  highlightX = -1;
  highlightY = -1;
  highlightZ = -1;

  for (let step = 0; step < 60; step++) {
    const bx = Math.floor(rx);
    const by = Math.floor(ry);
    const bz = Math.floor(rz);
    if (bx < 0 || bx >= worldSizeX || by < 0 || by >= WORLD_HEIGHT || bz < 0 || bz >= worldSizeZ) break;
    const block = getBlock(bx, by, bz);
    if (block !== BLOCK_AIR && block !== BLOCK_WATER) {
      highlightX = bx;
      highlightY = by;
      highlightZ = bz;
      break;
    }
    rx = rx + dir.x * 0.2;
    ry = ry + dir.y * 0.2;
    rz = rz + dir.z * 0.2;
  }
}

// Initialize
initWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Voxel Sandbox");
setTargetFPS(60);
disableCursor();
generateTerrain();

const camera: Camera3D = {
  position: { x: camX, y: camY, z: camZ },
  target: { x: 0, y: 0, z: 0 },
  up: { x: 0, y: 1, z: 0 },
  fovy: 70,
  projection: "perspective",
};

function handleInput(dt: number): void {
  camYaw = camYaw + getMouseDeltaX() * MOUSE_SENS;
  camPitch = clamp(camPitch - getMouseDeltaY() * MOUSE_SENS, -1.4, 1.4);

  const forward = getCamForward();
  const right = getCamRight();
  let moveX = 0;
  let moveZ = 0;
  let moveY = 0;

  if (isKeyDown(Key.W)) { moveX = moveX + forward.x; moveZ = moveZ + forward.z; }
  if (isKeyDown(Key.S)) { moveX = moveX - forward.x; moveZ = moveZ - forward.z; }
  if (isKeyDown(Key.A)) { moveX = moveX - right.x; moveZ = moveZ - right.z; }
  if (isKeyDown(Key.D)) { moveX = moveX + right.x; moveZ = moveZ + right.z; }
  if (isKeyDown(Key.SPACE)) moveY = 1;
  if (isKeyDown(Key.LEFT_SHIFT)) moveY = -1;

  const len = Math.sqrt(moveX * moveX + moveZ * moveZ);
  if (len > 0) { moveX = moveX / len; moveZ = moveZ / len; }

  camX = camX + moveX * MOVE_SPEED * dt;
  camY = camY + moveY * MOVE_SPEED * dt;
  camZ = camZ + moveZ * MOVE_SPEED * dt;

  if (isKeyPressed(Key.ONE)) selectedBlock = BLOCK_GRASS;
  if (isKeyPressed(Key.TWO)) selectedBlock = BLOCK_DIRT;
  if (isKeyPressed(Key.THREE)) selectedBlock = BLOCK_STONE;
  if (isKeyPressed(Key.FOUR)) selectedBlock = BLOCK_WOOD;
  if (isKeyPressed(Key.FIVE)) selectedBlock = BLOCK_LEAVES;
  if (isKeyPressed(Key.SIX)) selectedBlock = BLOCK_SAND;
  if (isKeyPressed(Key.SEVEN)) selectedBlock = BLOCK_WATER;

  raycastBlock();

  if (isMouseButtonPressed(MouseButton.LEFT) && highlightX >= 0) {
    setBlock(highlightX, highlightY, highlightZ, BLOCK_AIR);
  }

  if (isMouseButtonPressed(MouseButton.RIGHT) && highlightX >= 0) {
    const dir = getCamForward();
    let rx = camX;
    let ry = camY;
    let rz = camZ;
    let prevBx = Math.floor(rx);
    let prevBy = Math.floor(ry);
    let prevBz = Math.floor(rz);
    for (let step = 0; step < 60; step++) {
      const bx = Math.floor(rx);
      const by = Math.floor(ry);
      const bz = Math.floor(rz);
      if (bx === highlightX && by === highlightY && bz === highlightZ) {
        if (getBlock(prevBx, prevBy, prevBz) === BLOCK_AIR) {
          setBlock(prevBx, prevBy, prevBz, selectedBlock);
        }
        break;
      }
      prevBx = bx; prevBy = by; prevBz = bz;
      rx = rx + dir.x * 0.2; ry = ry + dir.y * 0.2; rz = rz + dir.z * 0.2;
    }
  }

  camera.position.x = camX; camera.position.y = camY; camera.position.z = camZ;
  camera.target.x = camX + forward.x;
  camera.target.y = camY + forward.y;
  camera.target.z = camZ + forward.z;
}

const renderPos = { x: 0, y: 0, z: 0 };

function renderBlocks(): void {
  const renderDist = 32;
  const minX = Math.max(0, Math.floor(camX - renderDist));
  const maxX = Math.min(worldSizeX - 1, Math.floor(camX + renderDist));
  const minZ = Math.max(0, Math.floor(camZ - renderDist));
  const maxZ = Math.min(worldSizeZ - 1, Math.floor(camZ + renderDist));

  for (let y = 0; y < WORLD_HEIGHT; y++) {
    for (let z = minZ; z <= maxZ; z++) {
      for (let x = minX; x <= maxX; x++) {
        const block = getBlock(x, y, z);
        if (block === BLOCK_AIR) continue;
        if (
          getBlock(x-1, y, z) !== BLOCK_AIR && getBlock(x+1, y, z) !== BLOCK_AIR &&
          getBlock(x, y-1, z) !== BLOCK_AIR && getBlock(x, y+1, z) !== BLOCK_AIR &&
          getBlock(x, y, z-1) !== BLOCK_AIR && getBlock(x, y, z+1) !== BLOCK_AIR
        ) continue;
        renderPos.x = x + 0.5; renderPos.y = y + 0.5; renderPos.z = z + 0.5;
        drawCube(renderPos, 1, 1, 1, BLOCK_COLORS[block]);
      }
    }
  }
}

function drawHUD(): void {
  const cx = SCREEN_WIDTH / 2;
  const cy = SCREEN_HEIGHT / 2;
  drawRect(cx - 10, cy - 1, 20, 2, Color.White);
  drawRect(cx - 1, cy - 10, 2, 20, Color.White);

  const blockNames = ["", "Grass", "Dirt", "Stone", "Wood", "Leaves", "Sand", "Water"];
  drawRect(5, SCREEN_HEIGHT - 35, 200, 30, { r: 0, g: 0, b: 0, a: 150 });
  drawText("Block: " + blockNames[selectedBlock] + " [1-7]", 10, SCREEN_HEIGHT - 30, 18, Color.White);
  drawText(
    "Pos: " + Math.floor(camX).toString() + ", " + Math.floor(camY).toString() + ", " + Math.floor(camZ).toString(),
    10, 10, 16, Color.White,
  );
}

while (!windowShouldClose()) {
  handleInput(getDeltaTime());

  beginDrawing();
  clearBackground({ r: 130, g: 200, b: 255, a: 255 });

  beginMode3D(camera);
  renderBlocks();

  if (highlightX >= 0) {
    drawCubeWires(
      { x: highlightX + 0.5, y: highlightY + 0.5, z: highlightZ + 0.5 },
      1.02, 1.02, 1.02, Color.White,
    );
  }
  endMode3D();

  drawHUD();
  endDrawing();
}

closeWindow();
