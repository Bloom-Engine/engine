import {
  initWindow, windowShouldClose, beginDrawing, endDrawing,
  clearBackground, setTargetFPS, getDeltaTime, isKeyPressed, isKeyDown,
  getScreenWidth, getScreenHeight, closeWindow,
  getMouseX, getMouseY, isMouseButtonPressed,
  writeFile, fileExists,
} from "bloom/core";
import { Color, Key, Camera2D, MouseButton } from "bloom/core";
import {
  drawRect, drawRectLines, drawCircle, drawTriangle,
  checkCollisionPointRec,
} from "bloom/shapes";
import { drawText, measureText } from "bloom/text";
import { clamp, randomInt, randomFloat, lerp, vec2Distance } from "bloom/math";
import { Rect, Vec2 } from "bloom/core";

// Constants
const SCREEN_WIDTH = 960;
const SCREEN_HEIGHT = 640;
const TILE_W = 64;
const TILE_H = 32;
const MAP_W = 20;
const MAP_H = 20;
const MAX_NPCS = 8;
const MAX_ITEMS = 20;

// Tile types
const T_GRASS = 0;
const T_PATH = 1;
const T_WATER = 2;
const T_WALL = 3;
const T_FLOOR = 4;

// Item types
const ITEM_POTION = 0;
const ITEM_SWORD = 1;
const ITEM_SHIELD = 2;
const ITEM_KEY = 3;
const ITEM_COIN = 4;

interface Entity {
  mapX: number;
  mapY: number;
  screenX: number;
  screenY: number;
  name: string;
  hp: number;
  maxHp: number;
  attack: number;
  defense: number;
  friendly: boolean;
  dialogue: string[];
  dialogueIndex: number;
}

interface Item {
  type: number;
  mapX: number;
  mapY: number;
  active: boolean;
  name: string;
}

// Isometric conversion
function isoToScreen(mapX: number, mapY: number): Vec2 {
  return {
    x: (mapX - mapY) * (TILE_W / 2),
    y: (mapX + mapY) * (TILE_H / 2),
  };
}

function screenToIso(sx: number, sy: number): Vec2 {
  return {
    x: Math.floor((sx / (TILE_W / 2) + sy / (TILE_H / 2)) / 2),
    y: Math.floor((sy / (TILE_H / 2) - sx / (TILE_W / 2)) / 2),
  };
}

// Game state
const map: number[] = [];
for (let i = 0; i < MAP_W * MAP_H; i++) map.push(T_GRASS);

const player: Entity = {
  mapX: 5, mapY: 5, screenX: 0, screenY: 0,
  name: "Hero", hp: 30, maxHp: 30, attack: 8, defense: 3,
  friendly: true, dialogue: [], dialogueIndex: 0,
};

const npcs: Entity[] = [];
const items: Item[] = [];
const inventory: number[] = []; // item types collected
let gold = 0;
let exp = 0;
let level = 1;

// Dialogue state
let showDialogue = false;
let dialogueNpc = -1;
let dialogueText = "";

// Message log
let message = "";
let messageTimer = 0;

function showMsg(msg: string): void {
  message = msg;
  messageTimer = 3;
}

function tileAt(x: number, y: number): number {
  if (x < 0 || x >= MAP_W || y < 0 || y >= MAP_H) return T_WATER;
  return map[y * MAP_W + x];
}

function setTile(x: number, y: number, t: number): void {
  if (x >= 0 && x < MAP_W && y >= 0 && y < MAP_H) {
    map[y * MAP_W + x] = t;
  }
}

function isWalkable(x: number, y: number): boolean {
  const t = tileAt(x, y);
  return t !== T_WATER && t !== T_WALL;
}

function tileColor(t: number): Color {
  if (t === T_GRASS) return { r: 80, g: 160, b: 60, a: 255 };
  if (t === T_PATH) return { r: 180, g: 160, b: 120, a: 255 };
  if (t === T_WATER) return { r: 40, g: 90, b: 200, a: 255 };
  if (t === T_WALL) return { r: 100, g: 90, b: 80, a: 255 };
  if (t === T_FLOOR) return { r: 150, g: 130, b: 100, a: 255 };
  return { r: 100, g: 100, b: 100, a: 255 };
}

function itemColor(t: number): Color {
  if (t === ITEM_POTION) return { r: 255, g: 50, b: 50, a: 255 };
  if (t === ITEM_SWORD) return { r: 200, g: 200, b: 220, a: 255 };
  if (t === ITEM_SHIELD) return { r: 100, g: 100, b: 200, a: 255 };
  if (t === ITEM_KEY) return { r: 255, g: 220, b: 50, a: 255 };
  if (t === ITEM_COIN) return { r: 255, g: 200, b: 0, a: 255 };
  return Color.White;
}

function itemName(t: number): string {
  if (t === ITEM_POTION) return "Potion";
  if (t === ITEM_SWORD) return "Sword";
  if (t === ITEM_SHIELD) return "Shield";
  if (t === ITEM_KEY) return "Key";
  if (t === ITEM_COIN) return "Gold";
  return "???";
}

// Generate the world
function generateWorld(): void {
  // Grass everywhere
  for (let y = 0; y < MAP_H; y++) {
    for (let x = 0; x < MAP_W; x++) {
      setTile(x, y, T_GRASS);
    }
  }

  // Paths
  for (let x = 3; x < MAP_W - 3; x++) { setTile(x, 10, T_PATH); }
  for (let y = 3; y < MAP_H - 3; y++) { setTile(10, y, T_PATH); }

  // Water pond
  for (let y = 14; y < 18; y++) {
    for (let x = 14; x < 18; x++) {
      setTile(x, y, T_WATER);
    }
  }

  // Building (walls + floor)
  for (let x = 2; x < 7; x++) {
    setTile(x, 2, T_WALL);
    setTile(x, 6, T_WALL);
  }
  for (let y = 2; y < 7; y++) {
    setTile(2, y, T_WALL);
    setTile(6, y, T_WALL);
  }
  for (let y = 3; y < 6; y++) {
    for (let x = 3; x < 6; x++) {
      setTile(x, y, T_FLOOR);
    }
  }
  setTile(4, 6, T_FLOOR); // Door

  // NPCs
  npcs.push({
    mapX: 4, mapY: 4, screenX: 0, screenY: 0,
    name: "Elder", hp: 20, maxHp: 20, attack: 0, defense: 0,
    friendly: true,
    dialogue: [
      "Welcome, traveler! Our village is under threat.",
      "Goblins have been raiding from the east.",
      "If you defeat them, we'll reward you handsomely!",
    ],
    dialogueIndex: 0,
  });
  npcs.push({
    mapX: 12, mapY: 8, screenX: 0, screenY: 0,
    name: "Merchant", hp: 15, maxHp: 15, attack: 0, defense: 0,
    friendly: true,
    dialogue: ["I sell potions and shields!", "Come back when you have gold."],
    dialogueIndex: 0,
  });
  npcs.push({
    mapX: 16, mapY: 5, screenX: 0, screenY: 0,
    name: "Goblin", hp: 12, maxHp: 12, attack: 5, defense: 1,
    friendly: false, dialogue: ["Grrrr!"], dialogueIndex: 0,
  });
  npcs.push({
    mapX: 18, mapY: 7, screenX: 0, screenY: 0,
    name: "Goblin", hp: 12, maxHp: 12, attack: 5, defense: 1,
    friendly: false, dialogue: ["Grrrr!"], dialogueIndex: 0,
  });
  npcs.push({
    mapX: 17, mapY: 3, screenX: 0, screenY: 0,
    name: "Goblin Chief", hp: 25, maxHp: 25, attack: 8, defense: 3,
    friendly: false, dialogue: ["You dare challenge me?!"], dialogueIndex: 0,
  });

  // Items scattered around
  items.push({ type: ITEM_POTION, mapX: 8, mapY: 12, active: true, name: "Potion" });
  items.push({ type: ITEM_COIN, mapX: 6, mapY: 9, active: true, name: "Gold" });
  items.push({ type: ITEM_COIN, mapX: 15, mapY: 11, active: true, name: "Gold" });
  items.push({ type: ITEM_SWORD, mapX: 4, mapY: 3, active: true, name: "Iron Sword" });
  items.push({ type: ITEM_SHIELD, mapX: 12, mapY: 15, active: true, name: "Shield" });
  items.push({ type: ITEM_KEY, mapX: 18, mapY: 3, active: true, name: "Dungeon Key" });
}

function npcAt(x: number, y: number): number {
  for (let i = 0; i < npcs.length; i++) {
    if (npcs[i].hp > 0 && npcs[i].mapX === x && npcs[i].mapY === y) return i;
  }
  return -1;
}

function tryMovePlayer(dx: number, dy: number): void {
  const nx = player.mapX + dx;
  const ny = player.mapY + dy;

  if (!isWalkable(nx, ny)) return;

  // Check for NPC
  const ni = npcAt(nx, ny);
  if (ni >= 0) {
    if (npcs[ni].friendly) {
      // Talk
      dialogueNpc = ni;
      dialogueText = npcs[ni].dialogue[npcs[ni].dialogueIndex];
      showDialogue = true;
    } else {
      // Combat
      const dmg = Math.max(1, player.attack - npcs[ni].defense + randomInt(-2, 2));
      npcs[ni].hp = npcs[ni].hp - dmg;
      showMsg("Hit " + npcs[ni].name + " for " + dmg.toString() + "!");
      if (npcs[ni].hp <= 0) {
        showMsg("Defeated " + npcs[ni].name + "!");
        exp = exp + 10;
        if (exp >= level * 20) {
          level = level + 1;
          player.maxHp = player.maxHp + 5;
          player.hp = player.maxHp;
          player.attack = player.attack + 2;
          showMsg("Level up! You are now level " + level.toString());
        }
      } else {
        // Enemy counterattack
        const eDmg = Math.max(1, npcs[ni].attack - player.defense + randomInt(-1, 1));
        player.hp = player.hp - eDmg;
        showMsg(npcs[ni].name + " hits back for " + eDmg.toString() + "!");
      }
    }
    return;
  }

  player.mapX = nx;
  player.mapY = ny;

  // Check items
  for (let i = 0; i < items.length; i++) {
    if (items[i].active && items[i].mapX === nx && items[i].mapY === ny) {
      items[i].active = false;
      if (items[i].type === ITEM_COIN) {
        gold = gold + 10;
        showMsg("Found 10 gold!");
      } else if (items[i].type === ITEM_POTION) {
        player.hp = Math.min(player.hp + 10, player.maxHp);
        showMsg("Used potion! +10 HP");
      } else {
        inventory.push(items[i].type);
        showMsg("Found " + items[i].name + "!");
        if (items[i].type === ITEM_SWORD) player.attack = player.attack + 3;
        if (items[i].type === ITEM_SHIELD) player.defense = player.defense + 2;
      }
    }
  }
}

// Initialize
initWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Isometric RPG");
setTargetFPS(60);
generateWorld();

const camera: Camera2D = {
  offset: { x: SCREEN_WIDTH / 2, y: SCREEN_HEIGHT / 3 },
  target: { x: 0, y: 0 },
  rotation: 0,
  zoom: 1.0,
};

while (!windowShouldClose()) {
  const dt = getDeltaTime();

  if (showDialogue) {
    if (isKeyPressed(Key.SPACE) || isKeyPressed(Key.ENTER)) {
      const npc = npcs[dialogueNpc];
      npc.dialogueIndex = (npc.dialogueIndex + 1) % npc.dialogue.length;
      if (npc.dialogueIndex === 0) {
        showDialogue = false;
      } else {
        dialogueText = npc.dialogue[npc.dialogueIndex];
      }
    }
  } else if (player.hp > 0) {
    // Movement (turn-based)
    if (isKeyPressed(Key.UP) || isKeyPressed(Key.W)) tryMovePlayer(0, -1);
    if (isKeyPressed(Key.DOWN) || isKeyPressed(Key.S)) tryMovePlayer(0, 1);
    if (isKeyPressed(Key.LEFT) || isKeyPressed(Key.A)) tryMovePlayer(-1, 0);
    if (isKeyPressed(Key.RIGHT) || isKeyPressed(Key.D)) tryMovePlayer(1, 0);

    // Zoom
    if (isKeyDown(Key.EQUAL)) camera.zoom = clamp(camera.zoom + dt, 0.5, 2.0);
    if (isKeyDown(Key.MINUS)) camera.zoom = clamp(camera.zoom - dt, 0.5, 2.0);
  } else {
    if (isKeyPressed(Key.ENTER)) {
      // Respawn
      player.hp = player.maxHp;
      player.mapX = 5;
      player.mapY = 5;
      showMsg("You wake up at the village...");
    }
  }

  // Smooth camera
  const playerScreen = isoToScreen(player.mapX, player.mapY);
  camera.target.x = lerp(camera.target.x, playerScreen.x, 6 * dt);
  camera.target.y = lerp(camera.target.y, playerScreen.y, 6 * dt);

  if (messageTimer > 0) messageTimer = messageTimer - dt;

  // Update NPC screen positions
  for (let i = 0; i < npcs.length; i++) {
    const s = isoToScreen(npcs[i].mapX, npcs[i].mapY);
    npcs[i].screenX = s.x;
    npcs[i].screenY = s.y;
  }
  const ps = isoToScreen(player.mapX, player.mapY);
  player.screenX = ps.x;
  player.screenY = ps.y;

  // Drawing
  beginDrawing();
  clearBackground({ r: 20, g: 25, b: 30, a: 255 });

  // Use camera for world rendering
  // We'll manually offset since beginMode2D uses camera transform
  const ox = SCREEN_WIDTH / 2 - camera.target.x * camera.zoom;
  const oy = SCREEN_HEIGHT / 3 - camera.target.y * camera.zoom;

  // Draw tiles (isometric diamond)
  for (let y = 0; y < MAP_H; y++) {
    for (let x = 0; x < MAP_W; x++) {
      const tile = tileAt(x, y);
      const s = isoToScreen(x, y);
      const sx = s.x * camera.zoom + ox;
      const sy = s.y * camera.zoom + oy;
      const tw = TILE_W * camera.zoom;
      const th = TILE_H * camera.zoom;
      const color = tileColor(tile);

      // Diamond shape using a filled rect (simplified isometric)
      drawRect(sx - tw / 2, sy, tw, th, color);
      // Outline
      drawRectLines(sx - tw / 2, sy, tw, th, 1, { r: 0, g: 0, b: 0, a: 40 });
    }
  }

  // Draw items
  for (let i = 0; i < items.length; i++) {
    if (!items[i].active) continue;
    const s = isoToScreen(items[i].mapX, items[i].mapY);
    const sx = s.x * camera.zoom + ox;
    const sy = s.y * camera.zoom + oy;
    const size = 8 * camera.zoom;
    drawCircle(sx, sy + TILE_H * camera.zoom * 0.5, size, itemColor(items[i].type));
  }

  // Draw NPCs
  for (let i = 0; i < npcs.length; i++) {
    if (npcs[i].hp <= 0) continue;
    const sx = npcs[i].screenX * camera.zoom + ox;
    const sy = npcs[i].screenY * camera.zoom + oy;
    const size = 12 * camera.zoom;
    const bodyColor = npcs[i].friendly ? { r: 50, g: 150, b: 50, a: 255 } : { r: 200, g: 50, b: 50, a: 255 };
    drawRect(sx - size / 2, sy - size + TILE_H * camera.zoom * 0.3, size, size * 1.5, bodyColor);
    // HP bar
    const barW = TILE_W * camera.zoom * 0.6;
    const hpRatio = npcs[i].hp / npcs[i].maxHp;
    drawRect(sx - barW / 2, sy - size - 4 + TILE_H * camera.zoom * 0.3, barW * hpRatio, 3, Color.Red);
  }

  // Draw player
  {
    const sx = player.screenX * camera.zoom + ox;
    const sy = player.screenY * camera.zoom + oy;
    const size = 14 * camera.zoom;
    drawRect(sx - size / 2, sy - size + TILE_H * camera.zoom * 0.3, size, size * 1.5, { r: 50, g: 100, b: 255, a: 255 });
    // Head
    drawCircle(sx, sy - size + TILE_H * camera.zoom * 0.3 - 4 * camera.zoom, 5 * camera.zoom, { r: 230, g: 200, b: 170, a: 255 });
  }

  // HUD panel
  drawRect(0, 0, SCREEN_WIDTH, 45, { r: 20, g: 20, b: 30, a: 220 });
  drawText(player.name + "  Lv." + level.toString(), 10, 5, 18, Color.White);
  // HP bar
  drawRect(10, 28, 120, 10, { r: 60, g: 0, b: 0, a: 255 });
  drawRect(10, 28, Math.floor(120 * player.hp / player.maxHp), 10, Color.Red);
  drawText(player.hp.toString() + "/" + player.maxHp.toString(), 15, 27, 10, Color.White);

  drawText("ATK: " + player.attack.toString(), 150, 8, 16, { r: 255, g: 150, b: 50, a: 255 });
  drawText("DEF: " + player.defense.toString(), 240, 8, 16, { r: 50, g: 150, b: 255, a: 255 });
  drawText("Gold: " + gold.toString(), 330, 8, 16, Color.Yellow);
  drawText("EXP: " + exp.toString() + "/" + (level * 20).toString(), 430, 8, 16, { r: 150, g: 255, b: 150, a: 255 });

  // Inventory
  if (inventory.length > 0) {
    let invStr = "Items: ";
    for (let i = 0; i < inventory.length; i++) {
      if (i > 0) invStr = invStr + ", ";
      invStr = invStr + itemName(inventory[i]);
    }
    drawText(invStr, 550, 8, 14, Color.LightGray);
  }

  // Dialogue box
  if (showDialogue) {
    drawRect(50, SCREEN_HEIGHT - 120, SCREEN_WIDTH - 100, 100, { r: 10, g: 10, b: 30, a: 230 });
    drawRectLines(50, SCREEN_HEIGHT - 120, SCREEN_WIDTH - 100, 100, 2, Color.White);
    const npcName = npcs[dialogueNpc].name;
    drawText(npcName, 70, SCREEN_HEIGHT - 110, 20, Color.Yellow);
    drawText(dialogueText, 70, SCREEN_HEIGHT - 80, 18, Color.White);
    drawText("[SPACE] to continue", 70, SCREEN_HEIGHT - 35, 14, Color.LightGray);
  }

  // Message log
  if (messageTimer > 0) {
    const alpha = Math.floor(clamp(messageTimer * 255, 0, 255));
    drawText(message, 10, SCREEN_HEIGHT - 30, 16, { r: 255, g: 255, b: 200, a: alpha });
  }

  // Death
  if (player.hp <= 0) {
    drawRect(0, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH, 80, { r: 0, g: 0, b: 0, a: 200 });
    drawText("YOU DIED", SCREEN_WIDTH / 2 - measureText("YOU DIED", 50) / 2, SCREEN_HEIGHT / 2 - 30, 50, Color.Red);
    drawText("Press ENTER to respawn", SCREEN_WIDTH / 2 - measureText("Press ENTER to respawn", 18) / 2, SCREEN_HEIGHT / 2 + 25, 18, Color.LightGray);
  }

  // Controls hint
  drawText("WASD/Arrows: Move | +/-: Zoom", SCREEN_WIDTH - 310, SCREEN_HEIGHT - 20, 12, { r: 150, g: 150, b: 150, a: 150 });

  endDrawing();
}

closeWindow();
