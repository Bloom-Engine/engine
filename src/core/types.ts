export interface Vec2 {
  x: number;
  y: number;
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Vec4 {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface Color {
  r: number;
  g: number;
  b: number;
  a: number;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Camera2D {
  offset: Vec2;
  target: Vec2;
  rotation: number;
  zoom: number;
}

export interface Camera3D {
  position: Vec3;
  target: Vec3;
  up: Vec3;
  fovy: number;
  projection: number;
}

export interface Texture {
  handle: number;
  width: number;
  height: number;
}

export interface Font {
  handle: number;
  size: number;
}

export interface Sound {
  handle: number;
}

export interface Music {
  handle: number;
}

export interface Quat {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface Ray {
  position: Vec3;
  direction: Vec3;
}

export interface BoundingBox {
  min: Vec3;
  max: Vec3;
}

export interface Model {
  handle: number;
}

export interface RayHit {
  hit: boolean;
  distance: number;
  point: Vec3;
  normal: Vec3;
}

export interface FrustumPlanes {
  planes: Vec4[];
}

export type Mat4 = number[];
