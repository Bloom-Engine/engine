import {
  initWindow, windowShouldClose, beginDrawing, endDrawing,
  clearBackground, setTargetFPS, getDeltaTime, isKeyDown, isKeyPressed,
  closeWindow, beginMode3D, endMode3D,
} from "bloom/core";
import { Color, Key, Camera3D } from "bloom/core";
import {
  drawCube, drawPlane, drawSphere, drawCubeWires, drawGrid,
} from "bloom/models";
import { drawText, measureText } from "bloom/text";
import { drawRect } from "bloom/shapes";
import {
  clamp, lerp, vec3, vec3Add, vec3Sub, vec3Scale, vec3Normalize, vec3Length,
  randomFloat, randomInt,
} from "bloom/math";
import { Vec3 } from "bloom/core";

// Constants
const SCREEN_WIDTH = 960;
const SCREEN_HEIGHT = 540;
const TRACK_RADIUS = 60;
const TRACK_WIDTH = 14;
const NUM_CHECKPOINTS = 16;
const TOTAL_LAPS = 3;
const MAX_AI = 3;

// Physics
const ACCEL = 28;
const BRAKE_DECEL = 40;
const FRICTION = 8;
const MAX_SPEED = 45;
const TURN_SPEED = 2.2;
const DRIFT_FACTOR = 0.92;

interface Kart {
  x: number;
  z: number;
  angle: number;      // radians, 0 = +X direction
  speed: number;
  velX: number;
  velZ: number;
  lap: number;
  checkpoint: number;
  finished: boolean;
  color: Color;
  isPlayer: boolean;
  // AI
  targetCheckpoint: number;
  steerBias: number;
}

// Track: oval made of checkpoints around an ellipse
const checkpoints: Vec3[] = [];
for (let i = 0; i < NUM_CHECKPOINTS; i++) {
  const t = (i / NUM_CHECKPOINTS) * Math.PI * 2;
  checkpoints.push({
    x: Math.cos(t) * TRACK_RADIUS,
    y: 0,
    z: Math.sin(t) * TRACK_RADIUS * 0.6,
  });
}

function getTrackPos(t: number): Vec3 {
  const a = t * Math.PI * 2;
  return { x: Math.cos(a) * TRACK_RADIUS, y: 0, z: Math.sin(a) * TRACK_RADIUS * 0.6 };
}

function createKart(startIdx: number, color: Color, isPlayer: boolean): Kart {
  const cp = checkpoints[startIdx % NUM_CHECKPOINTS];
  const nextCp = checkpoints[(startIdx + 1) % NUM_CHECKPOINTS];
  const angle = Math.atan2(nextCp.z - cp.z, nextCp.x - cp.x);
  // Offset laterally so karts don't overlap
  const offset = isPlayer ? 0 : (startIdx % 2 === 0 ? 3 : -3);
  return {
    x: cp.x + Math.sin(angle) * offset,
    z: cp.z - Math.cos(angle) * offset,
    angle,
    speed: 0,
    velX: 0,
    velZ: 0,
    lap: 0,
    checkpoint: startIdx % NUM_CHECKPOINTS,
    finished: false,
    color,
    isPlayer,
    targetCheckpoint: (startIdx + 1) % NUM_CHECKPOINTS,
    steerBias: randomFloat(-0.3, 0.3),
  };
}

// Game state
let raceStarted = false;
let countdown = 3.5;
let raceTime = 0;
let bestLapTime = 0;
let lapStartTime = 0;
let raceFinished = false;
let playerPlace = 0;

const karts: Kart[] = [];
karts.push(createKart(0, { r: 50, g: 150, b: 255, a: 255 }, true));
karts.push(createKart(14, { r: 255, g: 50, b: 50, a: 255 }, false));
karts.push(createKart(12, { r: 50, g: 255, b: 50, a: 255 }, false));
karts.push(createKart(10, { r: 255, g: 200, b: 50, a: 255 }, false));

function advanceCheckpoint(kart: Kart): void {
  const nextCp = (kart.checkpoint + 1) % NUM_CHECKPOINTS;
  const cp = checkpoints[nextCp];
  const dx = cp.x - kart.x;
  const dz = cp.z - kart.z;
  const dist = Math.sqrt(dx * dx + dz * dz);
  if (dist < TRACK_WIDTH) {
    kart.checkpoint = nextCp;
    kart.targetCheckpoint = (nextCp + 1) % NUM_CHECKPOINTS;
    if (nextCp === 0 && kart.lap >= 0) {
      kart.lap = kart.lap + 1;
      if (kart.isPlayer) {
        const lapTime = raceTime - lapStartTime;
        if (bestLapTime === 0 || lapTime < bestLapTime) bestLapTime = lapTime;
        lapStartTime = raceTime;
      }
      if (kart.lap >= TOTAL_LAPS) {
        kart.finished = true;
      }
    }
  }
}

function updateKartPhysics(kart: Kart, dt: number, accel: number, steer: number): void {
  if (kart.finished) return;

  kart.speed = kart.speed + accel * dt;
  kart.speed = kart.speed - Math.sign(kart.speed) * FRICTION * dt;
  kart.speed = clamp(kart.speed, -MAX_SPEED * 0.3, MAX_SPEED);

  if (Math.abs(kart.speed) > 1) {
    kart.angle = kart.angle + steer * dt * (kart.speed > 0 ? 1 : -1);
  }

  // Velocity with drift
  const targetVelX = Math.cos(kart.angle) * kart.speed;
  const targetVelZ = Math.sin(kart.angle) * kart.speed;
  kart.velX = lerp(kart.velX, targetVelX, DRIFT_FACTOR);
  kart.velZ = lerp(kart.velZ, targetVelZ, DRIFT_FACTOR);

  kart.x = kart.x + kart.velX * dt;
  kart.z = kart.z + kart.velZ * dt;

  advanceCheckpoint(kart);
}

function updateAI(kart: Kart, dt: number): void {
  if (kart.finished || !raceStarted) return;

  const target = checkpoints[kart.targetCheckpoint];
  const dx = target.x - kart.x;
  const dz = target.z - kart.z;
  const targetAngle = Math.atan2(dz, dx);

  let angleDiff = targetAngle - kart.angle;
  while (angleDiff > Math.PI) angleDiff = angleDiff - Math.PI * 2;
  while (angleDiff < -Math.PI) angleDiff = angleDiff + Math.PI * 2;

  const steer = clamp(angleDiff * 3.0 + kart.steerBias, -TURN_SPEED, TURN_SPEED);

  // AI always accelerates, brakes on sharp turns
  let accel = ACCEL * 0.85;
  if (Math.abs(angleDiff) > 0.5) accel = accel * 0.5;

  updateKartPhysics(kart, dt, accel, steer);
}

function getPlacement(): number {
  // Sort by laps then checkpoints, player is karts[0]
  let place = 1;
  const pk = karts[0];
  const pScore = pk.lap * NUM_CHECKPOINTS + pk.checkpoint;
  for (let i = 1; i < karts.length; i++) {
    const s = karts[i].lap * NUM_CHECKPOINTS + karts[i].checkpoint;
    if (s > pScore) place = place + 1;
  }
  return place;
}

function formatTime(t: number): string {
  const mins = Math.floor(t / 60);
  const secs = Math.floor(t) % 60;
  const ms = Math.floor((t * 100) % 100);
  return mins.toString() + ":" + (secs < 10 ? "0" : "") + secs.toString() + "." + (ms < 10 ? "0" : "") + ms.toString();
}

function placeSuffix(p: number): string {
  if (p === 1) return "1st";
  if (p === 2) return "2nd";
  if (p === 3) return "3rd";
  return p.toString() + "th";
}

// Initialize
initWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Kart Racer");
setTargetFPS(60);

const camera: Camera3D = {
  position: { x: 0, y: 30, z: -50 },
  target: { x: 0, y: 0, z: 0 },
  up: { x: 0, y: 1, z: 0 },
  fovy: 55,
  projection: "perspective",
};

while (!windowShouldClose()) {
  const dt = getDeltaTime();
  const player = karts[0];

  // Countdown
  if (!raceStarted) {
    countdown = countdown - dt;
    if (countdown <= 0) {
      raceStarted = true;
      lapStartTime = 0;
    }
  }

  if (raceStarted && !raceFinished) {
    raceTime = raceTime + dt;

    // Player input
    let accel = 0;
    let steer = 0;
    if (isKeyDown(Key.UP) || isKeyDown(Key.W)) accel = ACCEL;
    if (isKeyDown(Key.DOWN) || isKeyDown(Key.S)) accel = -BRAKE_DECEL;
    if (isKeyDown(Key.LEFT) || isKeyDown(Key.A)) steer = -TURN_SPEED;
    if (isKeyDown(Key.RIGHT) || isKeyDown(Key.D)) steer = TURN_SPEED;

    updateKartPhysics(player, dt, accel, steer);

    // AI
    for (let i = 1; i < karts.length; i++) {
      updateAI(karts[i], dt);
    }

    // Simple kart-to-kart collision
    for (let i = 0; i < karts.length; i++) {
      for (let j = i + 1; j < karts.length; j++) {
        const dx = karts[j].x - karts[i].x;
        const dz = karts[j].z - karts[i].z;
        const dist = Math.sqrt(dx * dx + dz * dz);
        if (dist < 3 && dist > 0.01) {
          const push = (3 - dist) * 0.5;
          const nx = dx / dist;
          const nz = dz / dist;
          karts[i].x = karts[i].x - nx * push;
          karts[i].z = karts[i].z - nz * push;
          karts[j].x = karts[j].x + nx * push;
          karts[j].z = karts[j].z + nz * push;
        }
      }
    }

    playerPlace = getPlacement();
    if (player.finished) raceFinished = true;
  }

  // Chase camera
  const camDist = 18;
  const camHeight = 10;
  const behindX = player.x - Math.cos(player.angle) * camDist;
  const behindZ = player.z - Math.sin(player.angle) * camDist;
  camera.position.x = lerp(camera.position.x, behindX, 4 * dt);
  camera.position.y = lerp(camera.position.y, camHeight, 4 * dt);
  camera.position.z = lerp(camera.position.z, behindZ, 4 * dt);
  const lookAhead = 8;
  camera.target.x = lerp(camera.target.x, player.x + Math.cos(player.angle) * lookAhead, 6 * dt);
  camera.target.y = 1;
  camera.target.z = lerp(camera.target.z, player.z + Math.sin(player.angle) * lookAhead, 6 * dt);

  // Drawing
  beginDrawing();
  clearBackground({ r: 100, g: 180, b: 255, a: 255 });

  beginMode3D(camera);

  // Ground
  drawPlane({ x: 0, y: -0.1, z: 0 }, 200, 200, { r: 50, g: 140, b: 50, a: 255 });

  // Track segments
  for (let i = 0; i < NUM_CHECKPOINTS; i++) {
    const c1 = checkpoints[i];
    const c2 = checkpoints[(i + 1) % NUM_CHECKPOINTS];
    const mx = (c1.x + c2.x) * 0.5;
    const mz = (c1.z + c2.z) * 0.5;
    const dx = c2.x - c1.x;
    const dz = c2.z - c1.z;
    const segLen = Math.sqrt(dx * dx + dz * dz);
    const angle = Math.atan2(dz, dx);

    // Draw track surface as a flat cube
    // Track is oriented along the segment direction, width = TRACK_WIDTH
    drawCube(
      { x: mx, y: 0, z: mz },
      segLen + 2, 0.15, TRACK_WIDTH,
      { r: 80, g: 80, b: 90, a: 255 },
    );
  }

  // Start/finish line
  const startCp = checkpoints[0];
  drawCube({ x: startCp.x, y: 0.2, z: startCp.z }, 1, 0.3, TRACK_WIDTH, Color.White);

  // Checkpoint markers (small posts on track edges)
  for (let i = 0; i < NUM_CHECKPOINTS; i++) {
    const cp = checkpoints[i];
    const color = i === 0 ? Color.White : { r: 200, g: 200, b: 50, a: 255 };
    drawCube({ x: cp.x, y: 1, z: cp.z - TRACK_WIDTH * 0.5 - 1 }, 0.5, 2, 0.5, color);
    drawCube({ x: cp.x, y: 1, z: cp.z + TRACK_WIDTH * 0.5 + 1 }, 0.5, 2, 0.5, color);
  }

  // Draw karts
  for (let i = 0; i < karts.length; i++) {
    const k = karts[i];
    // Body
    drawCube({ x: k.x, y: 0.7, z: k.z }, 2.5, 1.0, 1.5, k.color);
    // Cockpit
    drawCube({ x: k.x, y: 1.4, z: k.z }, 1.2, 0.6, 1.0, { r: 200, g: 200, b: 220, a: 255 });
    // Wheels (4 spheres)
    const cos = Math.cos(k.angle);
    const sin = Math.sin(k.angle);
    const wheelPositions = [
      { fx: 1.0, fz: 0.8 },
      { fx: 1.0, fz: -0.8 },
      { fx: -1.0, fz: 0.8 },
      { fx: -1.0, fz: -0.8 },
    ];
    for (const wp of wheelPositions) {
      const wx = k.x + cos * wp.fx - sin * wp.fz;
      const wz = k.z + sin * wp.fx + cos * wp.fz;
      drawSphere({ x: wx, y: 0.3, z: wz }, 0.35, { r: 30, g: 30, b: 30, a: 255 });
    }
  }

  // Trees around the track for scenery
  for (let i = 0; i < 24; i++) {
    const t = (i / 24) * Math.PI * 2;
    const r = TRACK_RADIUS * 1.3 + Math.sin(t * 3) * 10;
    const tx = Math.cos(t) * r;
    const tz = Math.sin(t) * r * 0.6;
    // Trunk
    drawCube({ x: tx, y: 2, z: tz }, 0.8, 4, 0.8, { r: 100, g: 60, b: 30, a: 255 });
    // Foliage
    drawSphere({ x: tx, y: 5.5, z: tz }, 2.5, { r: 30, g: 120, b: 30, a: 255 });
  }

  endMode3D();

  // HUD
  drawRect(0, 0, SCREEN_WIDTH, 40, { r: 0, g: 0, b: 0, a: 150 });

  // Speed
  const speedKmh = Math.floor(Math.abs(player.speed) * 3.6);
  drawText(speedKmh.toString() + " km/h", 10, 10, 22, Color.White);

  // Position
  drawText(placeSuffix(playerPlace), 200, 10, 22, Color.Yellow);

  // Lap
  const lapText = "Lap " + Math.min(player.lap + 1, TOTAL_LAPS).toString() + "/" + TOTAL_LAPS.toString();
  drawText(lapText, 350, 10, 22, Color.White);

  // Time
  drawText(formatTime(raceTime), SCREEN_WIDTH - 150, 10, 22, Color.LightGray);

  // Best lap
  if (bestLapTime > 0) {
    drawText("Best: " + formatTime(bestLapTime), SCREEN_WIDTH - 150, 35, 16, Color.Green);
  }

  // Countdown
  if (!raceStarted && countdown > 0) {
    const countNum = Math.ceil(countdown);
    const countText = countNum > 0 ? countNum.toString() : "GO!";
    const fontSize = 80;
    drawText(countText, SCREEN_WIDTH / 2 - measureText(countText, fontSize) / 2, SCREEN_HEIGHT / 2 - 50, fontSize,
      countNum <= 1 ? Color.Green : Color.Red);
  }

  // Race finish
  if (raceFinished) {
    drawRect(0, SCREEN_HEIGHT / 2 - 60, SCREEN_WIDTH, 120, { r: 0, g: 0, b: 0, a: 200 });
    const finishText = "RACE COMPLETE!";
    drawText(finishText, SCREEN_WIDTH / 2 - measureText(finishText, 50) / 2, SCREEN_HEIGHT / 2 - 45, 50, Color.Gold);
    const resultText = "Finished " + placeSuffix(playerPlace) + " — Time: " + formatTime(raceTime);
    drawText(resultText, SCREEN_WIDTH / 2 - measureText(resultText, 24) / 2, SCREEN_HEIGHT / 2 + 15, 24, Color.White);
  }

  // Controls hint (first few seconds)
  if (raceTime < 5 && raceStarted) {
    drawText("WASD/Arrows to drive", 10, SCREEN_HEIGHT - 25, 16, { r: 200, g: 200, b: 200, a: 180 });
  }

  endDrawing();
}

closeWindow();
