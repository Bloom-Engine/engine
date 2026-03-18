import { Vec2, Vec3, Vec4, Quat, Ray, BoundingBox, Mat4, FrustumPlanes, RayHit } from '../core/types';

// Vec2 operations

export function vec2(x: number, y: number): Vec2 {
  return { x, y };
}

export function vec2Add(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, y: a.y + b.y };
}

export function vec2Sub(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, y: a.y - b.y };
}

export function vec2Scale(v: Vec2, scalar: number): Vec2 {
  return { x: v.x * scalar, y: v.y * scalar };
}

export function vec2Length(v: Vec2): number {
  return Math.sqrt(v.x * v.x + v.y * v.y);
}

export function vec2LengthSq(v: Vec2): number {
  return v.x * v.x + v.y * v.y;
}

export function vec2Normalize(v: Vec2): Vec2 {
  const len = vec2Length(v);
  if (len === 0) return { x: 0, y: 0 };
  return { x: v.x / len, y: v.y / len };
}

export function vec2Dot(a: Vec2, b: Vec2): number {
  return a.x * b.x + a.y * b.y;
}

export function vec2Distance(a: Vec2, b: Vec2): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export function vec2Lerp(a: Vec2, b: Vec2, t: number): Vec2 {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

// Vec3 operations

export function vec3(x: number, y: number, z: number): Vec3 {
  return { x, y, z };
}

export function vec3Add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

export function vec3Sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

export function vec3Scale(v: Vec3, scalar: number): Vec3 {
  return { x: v.x * scalar, y: v.y * scalar, z: v.z * scalar };
}

export function vec3Length(v: Vec3): number {
  return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

export function vec3LengthSq(v: Vec3): number {
  return v.x * v.x + v.y * v.y + v.z * v.z;
}

export function vec3Normalize(v: Vec3): Vec3 {
  const len = vec3Length(v);
  if (len === 0) return { x: 0, y: 0, z: 0 };
  return { x: v.x / len, y: v.y / len, z: v.z / len };
}

export function vec3Dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function vec3Cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

export function vec3Distance(a: Vec3, b: Vec3): number {
  const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export function vec3Lerp(a: Vec3, b: Vec3, t: number): Vec3 {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t };
}

// Vec4 operations

export function vec4(x: number, y: number, z: number, w: number): Vec4 {
  return { x, y, z, w };
}

export function vec4Add(a: Vec4, b: Vec4): Vec4 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z, w: a.w + b.w };
}

export function vec4Scale(v: Vec4, s: number): Vec4 {
  return { x: v.x * s, y: v.y * s, z: v.z * s, w: v.w * s };
}

export function vec4Length(v: Vec4): number {
  return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}

export function vec4Normalize(v: Vec4): Vec4 {
  const len = vec4Length(v);
  if (len === 0) return { x: 0, y: 0, z: 0, w: 0 };
  return { x: v.x / len, y: v.y / len, z: v.z / len, w: v.w / len };
}

// Scalar utilities

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function clamp(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

export function remap(value: number, inMin: number, inMax: number, outMin: number, outMax: number): number {
  return outMin + (value - inMin) / (inMax - inMin) * (outMax - outMin);
}

export function randomFloat(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

export function randomInt(min: number, max: number): number {
  return Math.floor(min + Math.random() * (max - min + 1));
}

// Easing functions

export function easeInQuad(t: number): number { return t * t; }
export function easeOutQuad(t: number): number { return t * (2 - t); }
export function easeInOutQuad(t: number): number { return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; }
export function easeInCubic(t: number): number { return t * t * t; }
export function easeOutCubic(t: number): number { const t1 = t - 1; return t1 * t1 * t1 + 1; }
export function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}
export function easeInElastic(t: number): number {
  if (t === 0 || t === 1) return t;
  return -Math.pow(2, 10 * t - 10) * Math.sin((t * 10 - 10.75) * (2 * Math.PI / 3));
}
export function easeOutElastic(t: number): number {
  if (t === 0 || t === 1) return t;
  return Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * (2 * Math.PI / 3)) + 1;
}
export function easeBounce(t: number): number {
  const n1 = 7.5625, d1 = 2.75;
  if (t < 1 / d1) return n1 * t * t;
  if (t < 2 / d1) return n1 * (t -= 1.5 / d1) * t + 0.75;
  if (t < 2.5 / d1) return n1 * (t -= 2.25 / d1) * t + 0.9375;
  return n1 * (t -= 2.625 / d1) * t + 0.984375;
}

// Mat4 operations (column-major, 16-element arrays)

export function mat4Identity(): Mat4 {
  return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
}

export function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Array(16).fill(0);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] = a[row] * b[col * 4] + a[4 + row] * b[col * 4 + 1] +
                            a[8 + row] * b[col * 4 + 2] + a[12 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

export function mat4Translate(m: Mat4, v: Vec3): Mat4 {
  const out = [...m];
  for (let i = 0; i < 4; i++) {
    out[12 + i] += m[i] * v.x + m[4 + i] * v.y + m[8 + i] * v.z;
  }
  return out;
}

export function mat4Scale(m: Mat4, v: Vec3): Mat4 {
  const out = [...m];
  for (let i = 0; i < 4; i++) { out[i] *= v.x; out[4+i] *= v.y; out[8+i] *= v.z; }
  return out;
}

export function mat4RotateX(m: Mat4, angle: number): Mat4 {
  const c = Math.cos(angle), s = Math.sin(angle);
  const rot: Mat4 = [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1];
  return mat4Multiply(m, rot);
}

export function mat4RotateY(m: Mat4, angle: number): Mat4 {
  const c = Math.cos(angle), s = Math.sin(angle);
  const rot: Mat4 = [c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1];
  return mat4Multiply(m, rot);
}

export function mat4RotateZ(m: Mat4, angle: number): Mat4 {
  const c = Math.cos(angle), s = Math.sin(angle);
  const rot: Mat4 = [c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1];
  return mat4Multiply(m, rot);
}

export function mat4Perspective(fovy: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1.0 / Math.tan(fovy / 2);
  const nf = 1.0 / (near - far);
  return [f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0];
}

export function mat4Ortho(left: number, right: number, bottom: number, top: number, near: number, far: number): Mat4 {
  const lr = 1/(left-right), bt = 1/(bottom-top), nf = 1/(near-far);
  return [-2*lr,0,0,0, 0,-2*bt,0,0, 0,0,2*nf,0, (left+right)*lr,(top+bottom)*bt,(far+near)*nf,1];
}

export function mat4LookAt(eye: Vec3, center: Vec3, up: Vec3): Mat4 {
  let fx = center.x-eye.x, fy = center.y-eye.y, fz = center.z-eye.z;
  const flen = Math.sqrt(fx*fx+fy*fy+fz*fz);
  fx /= flen; fy /= flen; fz /= flen;
  let sx = fy*up.z-fz*up.y, sy = fz*up.x-fx*up.z, sz = fx*up.y-fy*up.x;
  const slen = Math.sqrt(sx*sx+sy*sy+sz*sz);
  sx /= slen; sy /= slen; sz /= slen;
  const ux = sy*fz-sz*fy, uy = sz*fx-sx*fz, uz = sx*fy-sy*fx;
  return [sx,ux,-fx,0, sy,uy,-fy,0, sz,uz,-fz,0,
    -(sx*eye.x+sy*eye.y+sz*eye.z),-(ux*eye.x+uy*eye.y+uz*eye.z),fx*eye.x+fy*eye.y+fz*eye.z,1];
}

export function mat4Invert(m: Mat4): Mat4 {
  const a00=m[0],a01=m[1],a02=m[2],a03=m[3],a10=m[4],a11=m[5],a12=m[6],a13=m[7];
  const a20=m[8],a21=m[9],a22=m[10],a23=m[11],a30=m[12],a31=m[13],a32=m[14],a33=m[15];
  const b00=a00*a11-a01*a10,b01=a00*a12-a02*a10,b02=a00*a13-a03*a10;
  const b03=a01*a12-a02*a11,b04=a01*a13-a03*a11,b05=a02*a13-a03*a12;
  const b06=a20*a31-a21*a30,b07=a20*a32-a22*a30,b08=a20*a33-a23*a30;
  const b09=a21*a32-a22*a31,b10=a21*a33-a23*a31,b11=a22*a33-a23*a32;
  let det = b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
  if (Math.abs(det) < 1e-10) return mat4Identity();
  det = 1.0 / det;
  return [
    (a11*b11-a12*b10+a13*b09)*det,(a02*b10-a01*b11-a03*b09)*det,(a31*b05-a32*b04+a33*b03)*det,(a22*b04-a21*b05-a23*b03)*det,
    (a12*b08-a10*b11-a13*b07)*det,(a00*b11-a02*b08+a03*b07)*det,(a32*b02-a30*b05-a33*b01)*det,(a20*b05-a22*b02+a23*b01)*det,
    (a10*b10-a11*b08+a13*b06)*det,(a01*b08-a00*b10-a03*b06)*det,(a30*b04-a31*b02+a33*b00)*det,(a21*b02-a20*b04-a23*b00)*det,
    (a11*b07-a10*b09-a12*b06)*det,(a00*b09-a01*b07+a02*b06)*det,(a31*b01-a30*b03-a32*b00)*det,(a20*b03-a21*b01+a22*b00)*det,
  ];
}

// Quaternion operations

export function quatIdentity(): Quat {
  return { x: 0, y: 0, z: 0, w: 1 };
}

export function quatFromEuler(pitch: number, yaw: number, roll: number): Quat {
  const cp = Math.cos(pitch * 0.5), sp = Math.sin(pitch * 0.5);
  const cy = Math.cos(yaw * 0.5), sy = Math.sin(yaw * 0.5);
  const cr = Math.cos(roll * 0.5), sr = Math.sin(roll * 0.5);
  return {
    x: sr * cp * cy - cr * sp * sy,
    y: cr * sp * cy + sr * cp * sy,
    z: cr * cp * sy - sr * sp * cy,
    w: cr * cp * cy + sr * sp * sy,
  };
}

export function quatToMat4(q: Quat): Mat4 {
  const x=q.x,y=q.y,z=q.z,w=q.w;
  const x2=x+x,y2=y+y,z2=z+z;
  const xx=x*x2,xy=x*y2,xz=x*z2,yy=y*y2,yz=y*z2,zz=z*z2;
  const wx=w*x2,wy=w*y2,wz=w*z2;
  return [1-yy-zz,xy+wz,xz-wy,0, xy-wz,1-xx-zz,yz+wx,0, xz+wy,yz-wx,1-xx-yy,0, 0,0,0,1];
}

export function quatSlerp(a: Quat, b: Quat, t: number): Quat {
  let dot = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
  let bx=b.x,by=b.y,bz=b.z,bw=b.w;
  if (dot < 0) { dot = -dot; bx=-bx; by=-by; bz=-bz; bw=-bw; }
  if (dot > 0.9995) {
    return quatNormalize({ x:a.x+(bx-a.x)*t, y:a.y+(by-a.y)*t, z:a.z+(bz-a.z)*t, w:a.w+(bw-a.w)*t });
  }
  const theta = Math.acos(dot);
  const sinT = Math.sin(theta);
  const s0 = Math.sin((1-t)*theta)/sinT;
  const s1 = Math.sin(t*theta)/sinT;
  return { x:s0*a.x+s1*bx, y:s0*a.y+s1*by, z:s0*a.z+s1*bz, w:s0*a.w+s1*bw };
}

export function quatNormalize(q: Quat): Quat {
  const len = Math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
  if (len === 0) return quatIdentity();
  return { x:q.x/len, y:q.y/len, z:q.z/len, w:q.w/len };
}

export function quatMultiply(a: Quat, b: Quat): Quat {
  return {
    x: a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
    y: a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
    z: a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
    w: a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
  };
}

// Ray / BoundingBox

export function rayIntersectsBox(ray: Ray, box_: BoundingBox): boolean {
  let tmin = -Infinity, tmax = Infinity;
  for (const axis of ['x', 'y', 'z'] as const) {
    const invD = 1.0 / ray.direction[axis];
    let t0 = (box_.min[axis] - ray.position[axis]) * invD;
    let t1 = (box_.max[axis] - ray.position[axis]) * invD;
    if (invD < 0) { const tmp = t0; t0 = t1; t1 = tmp; }
    tmin = Math.max(tmin, t0);
    tmax = Math.min(tmax, t1);
    if (tmax < tmin) return false;
  }
  return true;
}

export function rayIntersectsSphere(ray: Ray, center: Vec3, radius: number): boolean {
  const oc: Vec3 = { x: ray.position.x - center.x, y: ray.position.y - center.y, z: ray.position.z - center.z };
  const a = vec3Dot(ray.direction, ray.direction);
  const b = 2.0 * vec3Dot(oc, ray.direction);
  const c = vec3Dot(oc, oc) - radius * radius;
  return b * b - 4 * a * c >= 0;
}

export function checkCollisionSpheres(center1: Vec3, radius1: number, center2: Vec3, radius2: number): boolean {
  return vec3Distance(center1, center2) <= radius1 + radius2;
}

export function checkCollisionBoxes(a: BoundingBox, b: BoundingBox): boolean {
  return a.min.x <= b.max.x && a.max.x >= b.min.x &&
         a.min.y <= b.max.y && a.max.y >= b.min.y &&
         a.min.z <= b.max.z && a.max.z >= b.min.z;
}

// Frustum culling

export function extractFrustumPlanes(mvp: Mat4): FrustumPlanes {
  const m = mvp;
  const planes: Vec4[] = [
    // Left
    { x: m[3] + m[0], y: m[7] + m[4], z: m[11] + m[8], w: m[15] + m[12] },
    // Right
    { x: m[3] - m[0], y: m[7] - m[4], z: m[11] - m[8], w: m[15] - m[12] },
    // Bottom
    { x: m[3] + m[1], y: m[7] + m[5], z: m[11] + m[9], w: m[15] + m[13] },
    // Top
    { x: m[3] - m[1], y: m[7] - m[5], z: m[11] - m[9], w: m[15] - m[13] },
    // Near
    { x: m[3] + m[2], y: m[7] + m[6], z: m[11] + m[10], w: m[15] + m[14] },
    // Far
    { x: m[3] - m[2], y: m[7] - m[6], z: m[11] - m[10], w: m[15] - m[14] },
  ];
  // Normalize each plane
  for (let i = 0; i < 6; i++) {
    const p = planes[i];
    const len = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (len > 0) { p.x /= len; p.y /= len; p.z /= len; p.w /= len; }
  }
  return { planes };
}

export function isBoxInFrustum(box_: BoundingBox, frustum: FrustumPlanes): boolean {
  for (const plane of frustum.planes) {
    // Test the positive vertex (the corner most aligned with the plane normal)
    const px = plane.x >= 0 ? box_.max.x : box_.min.x;
    const py = plane.y >= 0 ? box_.max.y : box_.min.y;
    const pz = plane.z >= 0 ? box_.max.z : box_.min.z;
    if (plane.x * px + plane.y * py + plane.z * pz + plane.w < 0) {
      return false;
    }
  }
  return true;
}

// Ray-triangle intersection (Moller-Trumbore algorithm)

export function rayIntersectsTriangle(
  ray: Ray, v0: Vec3, v1: Vec3, v2: Vec3,
): RayHit {
  const EPSILON = 1e-8;
  const edge1: Vec3 = { x: v1.x - v0.x, y: v1.y - v0.y, z: v1.z - v0.z };
  const edge2: Vec3 = { x: v2.x - v0.x, y: v2.y - v0.y, z: v2.z - v0.z };
  const h = vec3Cross(ray.direction, edge2);
  const a = vec3Dot(edge1, h);
  const noHit: RayHit = { hit: false, distance: 0, point: { x: 0, y: 0, z: 0 }, normal: { x: 0, y: 0, z: 0 } };
  if (a > -EPSILON && a < EPSILON) return noHit;
  const f = 1.0 / a;
  const s: Vec3 = { x: ray.position.x - v0.x, y: ray.position.y - v0.y, z: ray.position.z - v0.z };
  const u = f * vec3Dot(s, h);
  if (u < 0.0 || u > 1.0) return noHit;
  const q = vec3Cross(s, edge1);
  const v = f * vec3Dot(ray.direction, q);
  if (v < 0.0 || u + v > 1.0) return noHit;
  const t = f * vec3Dot(edge2, q);
  if (t <= EPSILON) return noHit;
  const point: Vec3 = {
    x: ray.position.x + ray.direction.x * t,
    y: ray.position.y + ray.direction.y * t,
    z: ray.position.z + ray.direction.z * t,
  };
  const normal = vec3Normalize(vec3Cross(edge1, edge2));
  return { hit: true, distance: t, point, normal };
}

export function getRayCollisionBox(ray: Ray, box_: BoundingBox): RayHit {
  const noHit: RayHit = { hit: false, distance: 0, point: { x: 0, y: 0, z: 0 }, normal: { x: 0, y: 0, z: 0 } };
  let tmin = -Infinity;
  let tmax = Infinity;
  const normals: Vec3[] = [{ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }];
  for (const axis of ['x', 'y', 'z'] as const) {
    const invD = 1.0 / ray.direction[axis];
    let t0 = (box_.min[axis] - ray.position[axis]) * invD;
    let t1 = (box_.max[axis] - ray.position[axis]) * invD;
    const n0: Vec3 = { x: 0, y: 0, z: 0 };
    const n1: Vec3 = { x: 0, y: 0, z: 0 };
    if (invD >= 0) {
      n0[axis] = -1;
      n1[axis] = 1;
    } else {
      const tmp = t0; t0 = t1; t1 = tmp;
      n0[axis] = 1;
      n1[axis] = -1;
    }
    if (t0 > tmin) { tmin = t0; normals[0] = n0; }
    if (t1 < tmax) { tmax = t1; normals[1] = n1; }
    if (tmax < tmin) return noHit;
  }
  if (tmin < 0) return noHit;
  const point: Vec3 = {
    x: ray.position.x + ray.direction.x * tmin,
    y: ray.position.y + ray.direction.y * tmin,
    z: ray.position.z + ray.direction.z * tmin,
  };
  return { hit: true, distance: tmin, point, normal: normals[0] };
}
