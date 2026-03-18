import { Color, Rect, Vec2 } from '../core/types';

// FFI declarations
declare function bloom_draw_line(x1: number, y1: number, x2: number, y2: number, thickness: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_rect(x: number, y: number, w: number, h: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_rect_lines(x: number, y: number, w: number, h: number, thickness: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_circle(cx: number, cy: number, radius: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_circle_lines(cx: number, cy: number, radius: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_triangle(x1: number, y1: number, x2: number, y2: number, x3: number, y3: number, r: number, g: number, b: number, a: number): void;
declare function bloom_draw_poly(cx: number, cy: number, sides: number, radius: number, rotation: number, r: number, g: number, b: number, a: number): void;

// Drawing functions

export function drawLine(startX: number, startY: number, endX: number, endY: number, thickness: number, color: Color): void {
  bloom_draw_line(startX, startY, endX, endY, thickness, color.r, color.g, color.b, color.a);
}

export function drawRect(x: number, y: number, width: number, height: number, color: Color): void {
  bloom_draw_rect(x, y, width, height, color.r, color.g, color.b, color.a);
}

export function drawRectRec(rec: Rect, color: Color): void {
  bloom_draw_rect(rec.x, rec.y, rec.width, rec.height, color.r, color.g, color.b, color.a);
}

export function drawRectLines(x: number, y: number, width: number, height: number, thickness: number, color: Color): void {
  bloom_draw_rect_lines(x, y, width, height, thickness, color.r, color.g, color.b, color.a);
}

export function drawCircle(centerX: number, centerY: number, radius: number, color: Color): void {
  bloom_draw_circle(centerX, centerY, radius, color.r, color.g, color.b, color.a);
}

export function drawCircleLines(centerX: number, centerY: number, radius: number, color: Color): void {
  bloom_draw_circle_lines(centerX, centerY, radius, color.r, color.g, color.b, color.a);
}

export function drawTriangle(x1: number, y1: number, x2: number, y2: number, x3: number, y3: number, color: Color): void {
  bloom_draw_triangle(x1, y1, x2, y2, x3, y3, color.r, color.g, color.b, color.a);
}

export function drawPoly(centerX: number, centerY: number, sides: number, radius: number, rotation: number, color: Color): void {
  bloom_draw_poly(centerX, centerY, sides, radius, rotation, color.r, color.g, color.b, color.a);
}

// Bezier curve drawing (pure TS — subdivides cubic bezier into line segments)

export function drawBezier(
  startX: number, startY: number,
  cp1X: number, cp1Y: number,
  cp2X: number, cp2Y: number,
  endX: number, endY: number,
  thickness: number, color: Color,
): void {
  const segments = 24;
  let prevX = startX;
  let prevY = startY;
  for (let i = 1; i <= segments; i++) {
    const t = i / segments;
    const u = 1 - t;
    const x = u * u * u * startX + 3 * u * u * t * cp1X + 3 * u * t * t * cp2X + t * t * t * endX;
    const y = u * u * u * startY + 3 * u * u * t * cp1Y + 3 * u * t * t * cp2Y + t * t * t * endY;
    bloom_draw_line(prevX, prevY, x, y, thickness, color.r, color.g, color.b, color.a);
    prevX = x;
    prevY = y;
  }
}

// Pure TypeScript collision detection

export function checkCollisionRecs(rec1: Rect, rec2: Rect): boolean {
  return (
    rec1.x < rec2.x + rec2.width &&
    rec1.x + rec1.width > rec2.x &&
    rec1.y < rec2.y + rec2.height &&
    rec1.y + rec1.height > rec2.y
  );
}

export function checkCollisionCircles(center1: Vec2, radius1: number, center2: Vec2, radius2: number): boolean {
  const dx = center2.x - center1.x;
  const dy = center2.y - center1.y;
  const distSq = dx * dx + dy * dy;
  const radiusSum = radius1 + radius2;
  return distSq <= radiusSum * radiusSum;
}

export function checkCollisionCircleRec(center: Vec2, radius: number, rec: Rect): boolean {
  const closestX = Math.max(rec.x, Math.min(center.x, rec.x + rec.width));
  const closestY = Math.max(rec.y, Math.min(center.y, rec.y + rec.height));
  const dx = center.x - closestX;
  const dy = center.y - closestY;
  return (dx * dx + dy * dy) <= radius * radius;
}

export function checkCollisionPointRec(point: Vec2, rec: Rect): boolean {
  return (
    point.x >= rec.x &&
    point.x <= rec.x + rec.width &&
    point.y >= rec.y &&
    point.y <= rec.y + rec.height
  );
}

export function checkCollisionPointCircle(point: Vec2, center: Vec2, radius: number): boolean {
  const dx = point.x - center.x;
  const dy = point.y - center.y;
  return (dx * dx + dy * dy) <= radius * radius;
}

export function getCollisionRec(rec1: Rect, rec2: Rect): Rect {
  const x = Math.max(rec1.x, rec2.x);
  const y = Math.max(rec1.y, rec2.y);
  const right = Math.min(rec1.x + rec1.width, rec2.x + rec2.width);
  const bottom = Math.min(rec1.y + rec1.height, rec2.y + rec2.height);
  if (right <= x || bottom <= y) {
    return { x: 0, y: 0, width: 0, height: 0 };
  }
  return { x, y, width: right - x, height: bottom - y };
}
