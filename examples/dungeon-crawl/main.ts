import {
  initWindow, windowShouldClose, beginDrawing, endDrawing,
  clearBackground, setTargetFPS, getDeltaTime, isKeyPressed, isKeyDown,
  getScreenWidth, getScreenHeight, closeWindow, beginMode2D, endMode2D,
} from "bloom/core";
import { Color, Key, Camera2D } from "bloom/core";
import { drawRect, drawCircle, drawRectLines } from "bloom/shapes";
import { drawText, measureText } from "bloom/text";
import { clamp, randomInt, randomFloat } from "bloom/math";

// Constants
const SCREEN_WIDTH = 800;
const SCREEN_HEIGHT = 600;
const TILE_SIZE = 32;
const MAP_WIDTH = 50;
const MAP_HEIGHT = 50;
const MAX_ROOMS = 12;
const MIN_ROOM_SIZE = 4;
const MAX_ROOM_SIZE = 10;
const MAX_ENEMIES = 20;
const FOV_RADIUS = 8;

// Tile types
const TILE_WALL = 0;
const TILE_FLOOR = 1;
const TILE_STAIRS = 2;

// Entity types
interface Entity {
  x: number;
  y: number;
  hp: number;
  maxHp: number;
  attack: number;
  active: boolean;
  name: string;
}

interface Room {
  x: number;
  y: number;
  w: number;
  h: number;
}

// Game state
const map: number[] = [];
const visible: boolean[] = [];
const explored: boolean[] = [];
let player: Entity = { x: 0, y: 0, hp: 20, maxHp: 20, attack: 5, active: true, name: "Player" };
const enemies: Entity[] = [];
let floor = 1;
let turnCount = 0;
let message = "Welcome to the dungeon!";
let messageTimer = 3;

function tileAt(x: number, y: number): number {
  if (x < 0 || x >= MAP_WIDTH || y < 0 || y >= MAP_HEIGHT) return TILE_WALL;
  return map[y * MAP_WIDTH + x];
}

function setTile(x: number, y: number, tile: number): void {
  if (x >= 0 && x < MAP_WIDTH && y >= 0 && y < MAP_HEIGHT) {
    map[y * MAP_WIDTH + x] = tile;
  }
}

function isVisible(x: number, y: number): boolean {
  if (x < 0 || x >= MAP_WIDTH || y < 0 || y >= MAP_HEIGHT) return false;
  return visible[y * MAP_WIDTH + x];
}

function isExplored(x: number, y: number): boolean {
  if (x < 0 || x >= MAP_WIDTH || y < 0 || y >= MAP_HEIGHT) return false;
  return explored[y * MAP_WIDTH + x];
}

function showMessage(msg: string): void {
  message = msg;
  messageTimer = 3;
}

function generateDungeon(): void {
  // Fill with walls
  for (let i = 0; i < MAP_WIDTH * MAP_HEIGHT; i++) {
    map[i] = TILE_WALL;
    visible[i] = false;
    explored[i] = false;
  }

  // Clear enemies
  for (let i = 0; i < MAX_ENEMIES; i++) {
    if (i < enemies.length) {
      enemies[i].active = false;
    }
  }

  // Generate rooms
  const rooms: Room[] = [];
  for (let attempt = 0; attempt < 100 && rooms.length < MAX_ROOMS; attempt++) {
    const w = randomInt(MIN_ROOM_SIZE, MAX_ROOM_SIZE);
    const h = randomInt(MIN_ROOM_SIZE, MAX_ROOM_SIZE);
    const rx = randomInt(1, MAP_WIDTH - w - 1);
    const ry = randomInt(1, MAP_HEIGHT - h - 1);

    // Check overlap
    let overlaps = false;
    for (let r = 0; r < rooms.length; r++) {
      if (rx - 1 < rooms[r].x + rooms[r].w && rx + w + 1 > rooms[r].x &&
          ry - 1 < rooms[r].y + rooms[r].h && ry + h + 1 > rooms[r].y) {
        overlaps = true;
        break;
      }
    }
    if (overlaps) continue;

    // Carve room
    for (let dy = 0; dy < h; dy++) {
      for (let dx = 0; dx < w; dx++) {
        setTile(rx + dx, ry + dy, TILE_FLOOR);
      }
    }
    rooms.push({ x: rx, y: ry, w, h });
  }

  // Connect rooms with corridors
  for (let i = 1; i < rooms.length; i++) {
    const cx1 = Math.floor(rooms[i - 1].x + rooms[i - 1].w / 2);
    const cy1 = Math.floor(rooms[i - 1].y + rooms[i - 1].h / 2);
    const cx2 = Math.floor(rooms[i].x + rooms[i].w / 2);
    const cy2 = Math.floor(rooms[i].y + rooms[i].h / 2);

    // Horizontal then vertical
    const startX = Math.min(cx1, cx2);
    const endX = Math.max(cx1, cx2);
    for (let x = startX; x <= endX; x++) {
      setTile(x, cy1, TILE_FLOOR);
    }
    const startY = Math.min(cy1, cy2);
    const endY = Math.max(cy1, cy2);
    for (let y = startY; y <= endY; y++) {
      setTile(cx2, y, TILE_FLOOR);
    }
  }

  // Place player in first room
  if (rooms.length > 0) {
    player.x = Math.floor(rooms[0].x + rooms[0].w / 2);
    player.y = Math.floor(rooms[0].y + rooms[0].h / 2);
  }

  // Place stairs in last room
  if (rooms.length > 1) {
    const lastRoom = rooms[rooms.length - 1];
    setTile(
      Math.floor(lastRoom.x + lastRoom.w / 2),
      Math.floor(lastRoom.y + lastRoom.h / 2),
      TILE_STAIRS,
    );
  }

  // Place enemies in other rooms
  let enemyIdx = 0;
  for (let r = 1; r < rooms.length - 1 && enemyIdx < MAX_ENEMIES; r++) {
    const count = randomInt(1, 2);
    for (let e = 0; e < count && enemyIdx < MAX_ENEMIES; e++) {
      const ex = randomInt(rooms[r].x + 1, rooms[r].x + rooms[r].w - 2);
      const ey = randomInt(rooms[r].y + 1, rooms[r].y + rooms[r].h - 2);
      if (enemyIdx >= enemies.length) {
        enemies.push({ x: ex, y: ey, hp: 5 + floor * 2, maxHp: 5 + floor * 2, attack: 2 + floor, active: true, name: "Goblin" });
      } else {
        enemies[enemyIdx].x = ex;
        enemies[enemyIdx].y = ey;
        enemies[enemyIdx].hp = 5 + floor * 2;
        enemies[enemyIdx].maxHp = 5 + floor * 2;
        enemies[enemyIdx].attack = 2 + floor;
        enemies[enemyIdx].active = true;
        enemies[enemyIdx].name = floor >= 3 ? "Orc" : "Goblin";
      }
      enemyIdx++;
    }
  }
}

function computeVisibility(): void {
  for (let i = 0; i < MAP_WIDTH * MAP_HEIGHT; i++) {
    visible[i] = false;
  }

  // Simple raycasting FOV
  const steps = 360;
  for (let a = 0; a < steps; a++) {
    const angle = (a / steps) * Math.PI * 2;
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    let rx = player.x + 0.5;
    let ry = player.y + 0.5;
    for (let d = 0; d < FOV_RADIUS; d++) {
      const tx = Math.floor(rx);
      const ty = Math.floor(ry);
      if (tx < 0 || tx >= MAP_WIDTH || ty < 0 || ty >= MAP_HEIGHT) break;
      const idx = ty * MAP_WIDTH + tx;
      visible[idx] = true;
      explored[idx] = true;
      if (map[idx] === TILE_WALL) break;
      rx = rx + dx;
      ry = ry + dy;
    }
  }
}

function enemyAt(x: number, y: number): number {
  for (let i = 0; i < enemies.length; i++) {
    if (enemies[i].active && enemies[i].x === x && enemies[i].y === y) return i;
  }
  return -1;
}

function tryMove(dx: number, dy: number): void {
  const nx = player.x + dx;
  const ny = player.y + dy;

  if (tileAt(nx, ny) === TILE_WALL) return;

  const ei = enemyAt(nx, ny);
  if (ei >= 0) {
    // Attack enemy
    const dmg = randomInt(player.attack - 1, player.attack + 1);
    enemies[ei].hp = enemies[ei].hp - dmg;
    if (enemies[ei].hp <= 0) {
      enemies[ei].active = false;
      showMessage("Defeated " + enemies[ei].name + "!");
    } else {
      showMessage("Hit " + enemies[ei].name + " for " + dmg.toString() + " damage");
    }
  } else {
    player.x = nx;
    player.y = ny;
  }

  // Check stairs
  if (tileAt(player.x, player.y) === TILE_STAIRS) {
    floor = floor + 1;
    generateDungeon();
    showMessage("Descended to floor " + floor.toString());
    computeVisibility();
    return;
  }

  // Enemy turns
  for (let i = 0; i < enemies.length; i++) {
    if (!enemies[i].active) continue;
    const edx = player.x - enemies[i].x;
    const edy = player.y - enemies[i].y;
    const dist = Math.abs(edx) + Math.abs(edy);

    if (dist <= 1) {
      // Attack player
      const dmg = randomInt(enemies[i].attack - 1, enemies[i].attack + 1);
      player.hp = player.hp - dmg;
      showMessage(enemies[i].name + " hits you for " + dmg.toString() + "!");
    } else if (dist <= FOV_RADIUS && isVisible(enemies[i].x, enemies[i].y)) {
      // Move toward player
      let mx = 0;
      let my = 0;
      if (Math.abs(edx) > Math.abs(edy)) {
        mx = edx > 0 ? 1 : -1;
      } else {
        my = edy > 0 ? 1 : -1;
      }
      const enx = enemies[i].x + mx;
      const eny = enemies[i].y + my;
      if (tileAt(enx, eny) !== TILE_WALL && enemyAt(enx, eny) < 0 &&
          !(enx === player.x && eny === player.y)) {
        enemies[i].x = enx;
        enemies[i].y = eny;
      }
    }
  }

  turnCount = turnCount + 1;
  computeVisibility();
}

function getTileColor(tile: number, vis: boolean, exp: boolean): Color {
  if (!vis && !exp) return { r: 0, g: 0, b: 0, a: 255 };
  const dim = vis ? 1.0 : 0.35;
  if (tile === TILE_WALL) return { r: Math.floor(80 * dim), g: Math.floor(80 * dim), b: Math.floor(100 * dim), a: 255 };
  if (tile === TILE_STAIRS) return { r: Math.floor(255 * dim), g: Math.floor(200 * dim), b: Math.floor(50 * dim), a: 255 };
  return { r: Math.floor(40 * dim), g: Math.floor(40 * dim), b: Math.floor(50 * dim), a: 255 };
}

// Initialize
initWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Dungeon Crawl");
setTargetFPS(60);

// Initialize arrays
for (let i = 0; i < MAP_WIDTH * MAP_HEIGHT; i++) {
  map.push(TILE_WALL);
  visible.push(false);
  explored.push(false);
}

generateDungeon();
computeVisibility();

const camera: Camera2D = {
  offset: { x: SCREEN_WIDTH / 2, y: SCREEN_HEIGHT / 2 },
  target: { x: player.x * TILE_SIZE + TILE_SIZE / 2, y: player.y * TILE_SIZE + TILE_SIZE / 2 },
  rotation: 0,
  zoom: 1.0,
};

// Main game loop
while (!windowShouldClose()) {
  const dt = getDeltaTime();

  if (player.hp > 0) {
    // Turn-based input
    if (isKeyPressed(Key.UP) || isKeyPressed(Key.W)) tryMove(0, -1);
    if (isKeyPressed(Key.DOWN) || isKeyPressed(Key.S)) tryMove(0, 1);
    if (isKeyPressed(Key.LEFT) || isKeyPressed(Key.A)) tryMove(-1, 0);
    if (isKeyPressed(Key.RIGHT) || isKeyPressed(Key.D)) tryMove(1, 0);
    // Wait
    if (isKeyPressed(Key.PERIOD)) tryMove(0, 0);
  } else {
    if (isKeyPressed(Key.ENTER)) {
      player.hp = player.maxHp;
      floor = 1;
      turnCount = 0;
      generateDungeon();
      computeVisibility();
      showMessage("You rise again...");
    }
  }

  // Camera zoom
  if (isKeyDown(Key.EQUAL)) camera.zoom = clamp(camera.zoom + dt, 0.5, 3.0);
  if (isKeyDown(Key.MINUS)) camera.zoom = clamp(camera.zoom - dt, 0.5, 3.0);

  // Smooth camera follow
  const targetCamX = player.x * TILE_SIZE + TILE_SIZE / 2;
  const targetCamY = player.y * TILE_SIZE + TILE_SIZE / 2;
  camera.target.x = camera.target.x + (targetCamX - camera.target.x) * 8 * dt;
  camera.target.y = camera.target.y + (targetCamY - camera.target.y) * 8 * dt;

  // Message timer
  if (messageTimer > 0) messageTimer = messageTimer - dt;

  // Drawing
  beginDrawing();
  clearBackground({ r: 10, g: 10, b: 15, a: 255 });

  beginMode2D(camera);

  // Draw tiles
  const viewTiles = Math.ceil(SCREEN_WIDTH / TILE_SIZE / camera.zoom) + 2;
  const camTileX = Math.floor(camera.target.x / TILE_SIZE);
  const camTileY = Math.floor(camera.target.y / TILE_SIZE);
  for (let dy = -viewTiles; dy <= viewTiles; dy++) {
    for (let dx = -viewTiles; dx <= viewTiles; dx++) {
      const tx = camTileX + dx;
      const ty = camTileY + dy;
      if (tx < 0 || tx >= MAP_WIDTH || ty < 0 || ty >= MAP_HEIGHT) continue;
      const tile = map[ty * MAP_WIDTH + tx];
      const vis = isVisible(tx, ty);
      const exp = isExplored(tx, ty);
      if (!vis && !exp) continue;
      const color = getTileColor(tile, vis, exp);
      drawRect(tx * TILE_SIZE, ty * TILE_SIZE, TILE_SIZE, TILE_SIZE, color);
    }
  }

  // Draw enemies
  for (let i = 0; i < enemies.length; i++) {
    if (!enemies[i].active) continue;
    if (!isVisible(enemies[i].x, enemies[i].y)) continue;
    drawRect(
      enemies[i].x * TILE_SIZE + 4,
      enemies[i].y * TILE_SIZE + 4,
      TILE_SIZE - 8, TILE_SIZE - 8,
      { r: 200, g: 50, b: 50, a: 255 },
    );
    // HP bar
    const hpRatio = enemies[i].hp / enemies[i].maxHp;
    drawRect(enemies[i].x * TILE_SIZE, enemies[i].y * TILE_SIZE - 4, Math.floor(TILE_SIZE * hpRatio), 3, Color.Red);
  }

  // Draw player
  drawRect(
    player.x * TILE_SIZE + 2,
    player.y * TILE_SIZE + 2,
    TILE_SIZE - 4, TILE_SIZE - 4,
    { r: 50, g: 150, b: 255, a: 255 },
  );

  endMode2D();

  // HUD
  drawRect(0, 0, SCREEN_WIDTH, 35, { r: 0, g: 0, b: 0, a: 180 });
  drawText("HP: " + player.hp.toString() + "/" + player.maxHp.toString(), 10, 8, 20, player.hp > player.maxHp / 3 ? Color.Green : Color.Red);
  drawText("Floor: " + floor.toString(), 200, 8, 20, Color.White);
  drawText("Turns: " + turnCount.toString(), 350, 8, 20, Color.LightGray);

  // Message log
  if (messageTimer > 0) {
    const alpha = Math.floor(clamp(messageTimer * 255, 0, 255));
    drawText(message, 10, SCREEN_HEIGHT - 30, 18, { r: 255, g: 255, b: 200, a: alpha });
  }

  // Death screen
  if (player.hp <= 0) {
    drawRect(0, SCREEN_HEIGHT / 2 - 50, SCREEN_WIDTH, 100, { r: 0, g: 0, b: 0, a: 200 });
    const deathMsg = "You have perished on floor " + floor.toString();
    drawText(deathMsg, SCREEN_WIDTH / 2 - measureText(deathMsg, 24) / 2, SCREEN_HEIGHT / 2 - 20, 24, Color.Red);
    const restartMsg = "Press ENTER to try again";
    drawText(restartMsg, SCREEN_WIDTH / 2 - measureText(restartMsg, 18) / 2, SCREEN_HEIGHT / 2 + 15, 18, Color.LightGray);
  }

  endDrawing();
}

closeWindow();
