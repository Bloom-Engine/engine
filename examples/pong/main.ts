import {
  initWindow, windowShouldClose, beginDrawing, endDrawing,
  clearBackground, setTargetFPS, getDeltaTime, isKeyDown,
  getScreenWidth, getScreenHeight, closeWindow,
} from "bloom/core";
import { Colors, Key } from "bloom/core";
import { drawRect, drawCircle, checkCollisionRecs } from "bloom/shapes";
import { drawText, measureText } from "bloom/text";
import { initAudio, loadSound, playSound, closeAudio } from "bloom/audio";
import { clamp } from "bloom/math";
import { Rect } from "bloom/core";

// Constants
const SCREEN_WIDTH = 800;
const SCREEN_HEIGHT = 450;
const PADDLE_WIDTH = 15;
const PADDLE_HEIGHT = 80;
const PADDLE_SPEED = 300;
const BALL_RADIUS = 8;
const BALL_SPEED = 250;
const PADDLE_MARGIN = 30;

// Game state
let leftPaddleY = SCREEN_HEIGHT / 2 - PADDLE_HEIGHT / 2;
let rightPaddleY = SCREEN_HEIGHT / 2 - PADDLE_HEIGHT / 2;
let ballX = SCREEN_WIDTH / 2;
let ballY = SCREEN_HEIGHT / 2;
let ballVelX = BALL_SPEED;
let ballVelY = BALL_SPEED * 0.5;
let leftScore = 0;
let rightScore = 0;
let paused = false;

function resetBall(direction: number): void {
  ballX = SCREEN_WIDTH / 2;
  ballY = SCREEN_HEIGHT / 2;
  ballVelX = BALL_SPEED * direction;
  ballVelY = BALL_SPEED * 0.5 * (Math.random() > 0.5 ? 1 : -1);
}

// Initialize
initWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Pong");
setTargetFPS(60);
initAudio();

// Main game loop
while (!windowShouldClose()) {
  const dt = getDeltaTime();

  // Pause toggle
  if (isKeyDown(Key.P)) {
    paused = !paused;
  }

  if (!paused) {
    // Left paddle (W/S)
    if (isKeyDown(Key.W)) {
      leftPaddleY = leftPaddleY - PADDLE_SPEED * dt;
    }
    if (isKeyDown(Key.S)) {
      leftPaddleY = leftPaddleY + PADDLE_SPEED * dt;
    }
    leftPaddleY = clamp(leftPaddleY, 0, SCREEN_HEIGHT - PADDLE_HEIGHT);

    // Right paddle (Up/Down)
    if (isKeyDown(Key.UP)) {
      rightPaddleY = rightPaddleY - PADDLE_SPEED * dt;
    }
    if (isKeyDown(Key.DOWN)) {
      rightPaddleY = rightPaddleY + PADDLE_SPEED * dt;
    }
    rightPaddleY = clamp(rightPaddleY, 0, SCREEN_HEIGHT - PADDLE_HEIGHT);

    // Ball movement
    ballX = ballX + ballVelX * dt;
    ballY = ballY + ballVelY * dt;

    // Ball collision with top/bottom walls
    if (ballY - BALL_RADIUS <= 0) {
      ballY = BALL_RADIUS;
      ballVelY = -ballVelY;
    }
    if (ballY + BALL_RADIUS >= SCREEN_HEIGHT) {
      ballY = SCREEN_HEIGHT - BALL_RADIUS;
      ballVelY = -ballVelY;
    }

    // Ball collision with paddles
    const leftPaddle: Rect = {
      x: PADDLE_MARGIN,
      y: leftPaddleY,
      width: PADDLE_WIDTH,
      height: PADDLE_HEIGHT,
    };
    const rightPaddle: Rect = {
      x: SCREEN_WIDTH - PADDLE_MARGIN - PADDLE_WIDTH,
      y: rightPaddleY,
      width: PADDLE_WIDTH,
      height: PADDLE_HEIGHT,
    };
    const ballRect: Rect = {
      x: ballX - BALL_RADIUS,
      y: ballY - BALL_RADIUS,
      width: BALL_RADIUS * 2,
      height: BALL_RADIUS * 2,
    };

    if (checkCollisionRecs(ballRect, leftPaddle) && ballVelX < 0) {
      ballVelX = -ballVelX;
      // Adjust vertical velocity based on where ball hit paddle
      const hitPos = (ballY - leftPaddleY) / PADDLE_HEIGHT;
      ballVelY = BALL_SPEED * (hitPos - 0.5) * 2;
    }
    if (checkCollisionRecs(ballRect, rightPaddle) && ballVelX > 0) {
      ballVelX = -ballVelX;
      const hitPos = (ballY - rightPaddleY) / PADDLE_HEIGHT;
      ballVelY = BALL_SPEED * (hitPos - 0.5) * 2;
    }

    // Scoring
    if (ballX < 0) {
      rightScore = rightScore + 1;
      resetBall(1);
    }
    if (ballX > SCREEN_WIDTH) {
      leftScore = leftScore + 1;
      resetBall(-1);
    }
  }

  // Drawing
  beginDrawing();
  clearBackground(Colors.BLACK);

  // Center line
  const segments = 20;
  const segHeight = SCREEN_HEIGHT / (segments * 2);
  for (let i = 0; i < segments; i = i + 1) {
    drawRect(
      SCREEN_WIDTH / 2 - 1,
      i * segHeight * 2,
      2,
      segHeight,
      Colors.DARKGRAY,
    );
  }

  // Paddles
  drawRect(PADDLE_MARGIN, leftPaddleY, PADDLE_WIDTH, PADDLE_HEIGHT, Colors.WHITE);
  drawRect(SCREEN_WIDTH - PADDLE_MARGIN - PADDLE_WIDTH, rightPaddleY, PADDLE_WIDTH, PADDLE_HEIGHT, Colors.WHITE);

  // Ball
  drawCircle(ballX, ballY, BALL_RADIUS, Colors.WHITE);

  // Scores
  const leftScoreText = leftScore.toString();
  const rightScoreText = rightScore.toString();
  drawText(leftScoreText, SCREEN_WIDTH / 4 - measureText(leftScoreText, 40) / 2, 20, 40, Colors.WHITE);
  drawText(rightScoreText, 3 * SCREEN_WIDTH / 4 - measureText(rightScoreText, 40) / 2, 20, 40, Colors.WHITE);

  // Pause text
  if (paused) {
    const pauseText = "PAUSED";
    drawText(pauseText, SCREEN_WIDTH / 2 - measureText(pauseText, 30) / 2, SCREEN_HEIGHT / 2 - 15, 30, Colors.LIGHTGRAY);
  }

  endDrawing();
}

closeAudio();
closeWindow();
