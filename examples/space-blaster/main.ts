import {
  initWindow, windowShouldClose, beginDrawing, endDrawing,
  clearBackground, setTargetFPS, getDeltaTime, isKeyDown, isKeyPressed,
  getScreenWidth, getScreenHeight, closeWindow,
} from "bloom/core";
import { Color, Key } from "bloom/core";
import { drawRect, drawCircle, drawTriangle, checkCollisionRecs, checkCollisionCircleRec } from "bloom/shapes";
import { drawText, measureText } from "bloom/text";
import { initAudioDevice, closeAudioDevice } from "bloom/audio";
import { clamp, randomFloat, randomInt } from "bloom/math";
import { Rect } from "bloom/core";

// Constants
const SCREEN_WIDTH = 800;
const SCREEN_HEIGHT = 600;
const PLAYER_WIDTH = 40;
const PLAYER_HEIGHT = 30;
const PLAYER_SPEED = 350;
const BULLET_WIDTH = 4;
const BULLET_HEIGHT = 12;
const BULLET_SPEED = 600;
const BULLET_COOLDOWN = 0.12;
const MAX_BULLETS = 50;
const MAX_ENEMIES = 30;
const MAX_PARTICLES = 200;
const MAX_STARS = 100;
const ENEMY_WIDTH = 32;
const ENEMY_HEIGHT = 24;

// Types
interface Bullet {
  x: number;
  y: number;
  active: boolean;
}

interface Enemy {
  x: number;
  y: number;
  hp: number;
  speed: number;
  active: boolean;
  kind: number; // 0=basic, 1=fast, 2=tank
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  color: Color;
  active: boolean;
}

interface Star {
  x: number;
  y: number;
  speed: number;
  brightness: number;
}

// Game state
let playerX = SCREEN_WIDTH / 2;
let playerY = SCREEN_HEIGHT - 80;
let bulletCooldown = 0;
let score = 0;
let lives = 3;
let wave = 1;
let waveTimer = 0;
let enemiesSpawned = 0;
let enemiesPerWave = 5;
let gameOver = false;

// Entity pools
const bullets: Bullet[] = [];
for (let i = 0; i < MAX_BULLETS; i++) {
  bullets.push({ x: 0, y: 0, active: false });
}

const enemies: Enemy[] = [];
for (let i = 0; i < MAX_ENEMIES; i++) {
  enemies.push({ x: 0, y: 0, hp: 0, speed: 0, active: false, kind: 0 });
}

const particles: Particle[] = [];
for (let i = 0; i < MAX_PARTICLES; i++) {
  particles.push({ x: 0, y: 0, vx: 0, vy: 0, life: 0, maxLife: 0, color: Color.White, active: false });
}

// Scrolling star background
const stars: Star[] = [];
for (let i = 0; i < MAX_STARS; i++) {
  stars.push({
    x: randomFloat(0, SCREEN_WIDTH),
    y: randomFloat(0, SCREEN_HEIGHT),
    speed: randomFloat(30, 150),
    brightness: randomFloat(0.2, 1.0),
  });
}

function spawnBullet(x: number, y: number): void {
  for (let i = 0; i < MAX_BULLETS; i++) {
    if (!bullets[i].active) {
      bullets[i].x = x;
      bullets[i].y = y;
      bullets[i].active = true;
      return;
    }
  }
}

function spawnEnemy(): void {
  for (let i = 0; i < MAX_ENEMIES; i++) {
    if (!enemies[i].active) {
      const kind = wave >= 3 ? randomInt(0, 2) : (wave >= 2 ? randomInt(0, 1) : 0);
      enemies[i].x = randomFloat(ENEMY_WIDTH, SCREEN_WIDTH - ENEMY_WIDTH);
      enemies[i].y = -ENEMY_HEIGHT;
      enemies[i].kind = kind;
      if (kind === 0) {
        enemies[i].hp = 1;
        enemies[i].speed = randomFloat(80, 150);
      } else if (kind === 1) {
        enemies[i].hp = 1;
        enemies[i].speed = randomFloat(150, 250);
      } else {
        enemies[i].hp = 3;
        enemies[i].speed = randomFloat(50, 100);
      }
      enemies[i].active = true;
      return;
    }
  }
}

function spawnExplosion(x: number, y: number, count: number, color: Color): void {
  for (let n = 0; n < count; n++) {
    for (let i = 0; i < MAX_PARTICLES; i++) {
      if (!particles[i].active) {
        const angle = randomFloat(0, Math.PI * 2);
        const speed = randomFloat(50, 200);
        particles[i].x = x;
        particles[i].y = y;
        particles[i].vx = Math.cos(angle) * speed;
        particles[i].vy = Math.sin(angle) * speed;
        particles[i].life = randomFloat(0.3, 0.8);
        particles[i].maxLife = particles[i].life;
        particles[i].color = color;
        particles[i].active = true;
        break;
      }
    }
  }
}

function getEnemyColor(kind: number): Color {
  if (kind === 0) return { r: 230, g: 50, b: 50, a: 255 };
  if (kind === 1) return { r: 50, g: 230, b: 50, a: 255 };
  return { r: 100, g: 100, b: 230, a: 255 };
}

function resetGame(): void {
  playerX = SCREEN_WIDTH / 2;
  playerY = SCREEN_HEIGHT - 80;
  score = 0;
  lives = 3;
  wave = 1;
  waveTimer = 0;
  enemiesSpawned = 0;
  enemiesPerWave = 5;
  gameOver = false;
  for (let i = 0; i < MAX_BULLETS; i++) bullets[i].active = false;
  for (let i = 0; i < MAX_ENEMIES; i++) enemies[i].active = false;
  for (let i = 0; i < MAX_PARTICLES; i++) particles[i].active = false;
}

// Initialize
initWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Space Blaster");
setTargetFPS(60);
initAudioDevice();

// Main game loop
while (!windowShouldClose()) {
  const dt = getDeltaTime();

  if (gameOver) {
    if (isKeyPressed(Key.ENTER)) {
      resetGame();
    }
  } else {
    // Player movement
    if (isKeyDown(Key.LEFT) || isKeyDown(Key.A)) {
      playerX = playerX - PLAYER_SPEED * dt;
    }
    if (isKeyDown(Key.RIGHT) || isKeyDown(Key.D)) {
      playerX = playerX + PLAYER_SPEED * dt;
    }
    if (isKeyDown(Key.UP) || isKeyDown(Key.W)) {
      playerY = playerY - PLAYER_SPEED * dt;
    }
    if (isKeyDown(Key.DOWN) || isKeyDown(Key.S)) {
      playerY = playerY + PLAYER_SPEED * dt;
    }
    playerX = clamp(playerX, PLAYER_WIDTH / 2, SCREEN_WIDTH - PLAYER_WIDTH / 2);
    playerY = clamp(playerY, PLAYER_HEIGHT / 2, SCREEN_HEIGHT - PLAYER_HEIGHT / 2);

    // Shooting
    bulletCooldown = bulletCooldown - dt;
    if (isKeyDown(Key.SPACE) && bulletCooldown <= 0) {
      spawnBullet(playerX - 2, playerY - PLAYER_HEIGHT / 2);
      bulletCooldown = BULLET_COOLDOWN;
    }

    // Wave spawning
    waveTimer = waveTimer + dt;
    if (enemiesSpawned < enemiesPerWave && waveTimer > 0.6) {
      spawnEnemy();
      enemiesSpawned = enemiesSpawned + 1;
      waveTimer = 0;
    }

    // Check if wave is complete
    let activeEnemies = 0;
    for (let i = 0; i < MAX_ENEMIES; i++) {
      if (enemies[i].active) activeEnemies = activeEnemies + 1;
    }
    if (enemiesSpawned >= enemiesPerWave && activeEnemies === 0) {
      wave = wave + 1;
      enemiesPerWave = 5 + wave * 2;
      enemiesSpawned = 0;
      waveTimer = -1.5; // Pause before next wave
    }

    // Update bullets
    for (let i = 0; i < MAX_BULLETS; i++) {
      if (!bullets[i].active) continue;
      bullets[i].y = bullets[i].y - BULLET_SPEED * dt;
      if (bullets[i].y < -BULLET_HEIGHT) {
        bullets[i].active = false;
      }
    }

    // Update enemies
    for (let i = 0; i < MAX_ENEMIES; i++) {
      if (!enemies[i].active) continue;
      enemies[i].y = enemies[i].y + enemies[i].speed * dt;

      // Off screen
      if (enemies[i].y > SCREEN_HEIGHT + ENEMY_HEIGHT) {
        enemies[i].active = false;
        lives = lives - 1;
        if (lives <= 0) gameOver = true;
        continue;
      }

      // Collision with player
      const playerRect: Rect = {
        x: playerX - PLAYER_WIDTH / 2,
        y: playerY - PLAYER_HEIGHT / 2,
        width: PLAYER_WIDTH,
        height: PLAYER_HEIGHT,
      };
      const enemyRect: Rect = {
        x: enemies[i].x - ENEMY_WIDTH / 2,
        y: enemies[i].y - ENEMY_HEIGHT / 2,
        width: ENEMY_WIDTH,
        height: ENEMY_HEIGHT,
      };
      if (checkCollisionRecs(playerRect, enemyRect)) {
        spawnExplosion(enemies[i].x, enemies[i].y, 15, getEnemyColor(enemies[i].kind));
        enemies[i].active = false;
        lives = lives - 1;
        if (lives <= 0) gameOver = true;
        continue;
      }

      // Collision with bullets
      for (let j = 0; j < MAX_BULLETS; j++) {
        if (!bullets[j].active) continue;
        const bulletRect: Rect = {
          x: bullets[j].x - BULLET_WIDTH / 2,
          y: bullets[j].y - BULLET_HEIGHT / 2,
          width: BULLET_WIDTH,
          height: BULLET_HEIGHT,
        };
        if (checkCollisionRecs(bulletRect, enemyRect)) {
          bullets[j].active = false;
          enemies[i].hp = enemies[i].hp - 1;
          if (enemies[i].hp <= 0) {
            spawnExplosion(enemies[i].x, enemies[i].y, 12, getEnemyColor(enemies[i].kind));
            enemies[i].active = false;
            score = score + (enemies[i].kind === 2 ? 30 : (enemies[i].kind === 1 ? 20 : 10));
          } else {
            spawnExplosion(bullets[j].x, bullets[j].y, 3, { r: 255, g: 255, b: 100, a: 255 });
          }
          break;
        }
      }
    }

    // Update particles
    for (let i = 0; i < MAX_PARTICLES; i++) {
      if (!particles[i].active) continue;
      particles[i].x = particles[i].x + particles[i].vx * dt;
      particles[i].y = particles[i].y + particles[i].vy * dt;
      particles[i].life = particles[i].life - dt;
      if (particles[i].life <= 0) {
        particles[i].active = false;
      }
    }
  }

  // Update stars (always, even on game over)
  for (let i = 0; i < MAX_STARS; i++) {
    stars[i].y = stars[i].y + stars[i].speed * dt;
    if (stars[i].y > SCREEN_HEIGHT) {
      stars[i].y = 0;
      stars[i].x = randomFloat(0, SCREEN_WIDTH);
    }
  }

  // Drawing
  beginDrawing();
  clearBackground({ r: 5, g: 5, b: 15, a: 255 });

  // Stars
  for (let i = 0; i < MAX_STARS; i++) {
    const b = Math.floor(stars[i].brightness * 255);
    drawRect(stars[i].x, stars[i].y, 2, 2, { r: b, g: b, b: b, a: 255 });
  }

  if (!gameOver) {
    // Player ship (triangle)
    drawTriangle(
      playerX, playerY - PLAYER_HEIGHT / 2,
      playerX - PLAYER_WIDTH / 2, playerY + PLAYER_HEIGHT / 2,
      playerX + PLAYER_WIDTH / 2, playerY + PLAYER_HEIGHT / 2,
      { r: 50, g: 200, b: 255, a: 255 },
    );
    // Engine glow
    drawRect(playerX - 4, playerY + PLAYER_HEIGHT / 2, 8, 6, { r: 255, g: 150, b: 0, a: 200 });

    // Bullets
    for (let i = 0; i < MAX_BULLETS; i++) {
      if (!bullets[i].active) continue;
      drawRect(
        bullets[i].x - BULLET_WIDTH / 2,
        bullets[i].y - BULLET_HEIGHT / 2,
        BULLET_WIDTH, BULLET_HEIGHT,
        { r: 255, g: 255, b: 100, a: 255 },
      );
    }

    // Enemies
    for (let i = 0; i < MAX_ENEMIES; i++) {
      if (!enemies[i].active) continue;
      const color = getEnemyColor(enemies[i].kind);
      drawRect(
        enemies[i].x - ENEMY_WIDTH / 2,
        enemies[i].y - ENEMY_HEIGHT / 2,
        ENEMY_WIDTH, ENEMY_HEIGHT,
        color,
      );
      // Cockpit
      drawRect(
        enemies[i].x - 4,
        enemies[i].y - 4,
        8, 8,
        { r: 200, g: 200, b: 200, a: 255 },
      );
    }
  }

  // Particles
  for (let i = 0; i < MAX_PARTICLES; i++) {
    if (!particles[i].active) continue;
    const alpha = Math.floor((particles[i].life / particles[i].maxLife) * 255);
    const c = particles[i].color;
    drawRect(particles[i].x - 2, particles[i].y - 2, 4, 4, { r: c.r, g: c.g, b: c.b, a: alpha });
  }

  // HUD
  drawText("SCORE: " + score.toString(), 10, 10, 20, Color.White);
  drawText("WAVE: " + wave.toString(), SCREEN_WIDTH / 2 - 40, 10, 20, Color.White);

  // Lives
  for (let i = 0; i < lives; i++) {
    drawTriangle(
      SCREEN_WIDTH - 30 - i * 25, 12,
      SCREEN_WIDTH - 40 - i * 25, 28,
      SCREEN_WIDTH - 20 - i * 25, 28,
      { r: 50, g: 200, b: 255, a: 255 },
    );
  }

  // Wave announcement
  if (waveTimer < 0) {
    const waveText = "WAVE " + wave.toString();
    drawText(waveText, SCREEN_WIDTH / 2 - measureText(waveText, 40) / 2, SCREEN_HEIGHT / 2 - 20, 40, Color.Yellow);
  }

  // Game over screen
  if (gameOver) {
    drawText("GAME OVER", SCREEN_WIDTH / 2 - measureText("GAME OVER", 60) / 2, SCREEN_HEIGHT / 2 - 60, 60, Color.Red);
    const finalScore = "Score: " + score.toString();
    drawText(finalScore, SCREEN_WIDTH / 2 - measureText(finalScore, 30) / 2, SCREEN_HEIGHT / 2 + 10, 30, Color.White);
    const restartText = "Press ENTER to restart";
    drawText(restartText, SCREEN_WIDTH / 2 - measureText(restartText, 20) / 2, SCREEN_HEIGHT / 2 + 60, 20, Color.LightGray);
  }

  endDrawing();
}

closeAudioDevice();
closeWindow();
