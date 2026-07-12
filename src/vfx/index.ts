// EN-026 particles + EN-027 decals.
//
// Both are engine-simulated and game-drawn. You supply the material (so the
// look stays authorable in WGSL) and the quad mesh; the engine owns the pool
// and rewrites a dynamic instance buffer every frame. Per-frame FFI traffic is
// therefore one `update` + one `draw` per system, regardless of how many
// thousand particles are live.
//
// The instanced vertex shader receives, per instance:
//
//   @location(7)  instance_pos:   vec3<f32>   world position
//   @location(8)  instance_rot_y: f32         roll (particles) / roll about
//                                             the surface normal (decals)
//   @location(9)  instance_scale: f32         size in metres
//   @location(10) instance_tint:  vec4<f32>   rgba, alpha already faded
//   @location(11) instance_extra: vec3<f32>   particles: (age01, frame, stretch)
//                                             decals:    (frame, azimuth, elevation)
//
// For decals the normal is packed as two angles, so reconstruct it with:
//   let n = vec3(sin(el)*cos(az), cos(el), sin(el)*sin(az));

declare function bloom_particles_create(capacity: number): number;
declare function bloom_particles_configure(sys: number): void;
declare function bloom_particles_emit(sys: number, x: number, y: number, z: number, dx: number, dy: number, dz: number, count: number): void;
declare function bloom_particles_update(sys: number, dt: number): number;
declare function bloom_particles_instance_buffer(sys: number): number;
declare function bloom_particles_clear(sys: number): void;
declare function bloom_particles_live(sys: number): number;

declare function bloom_decals_init(capacity: number): number;
declare function bloom_decals_spawn(x: number, y: number, z: number, nx: number, ny: number, nz: number, size: number, roll: number): void;
declare function bloom_decals_set_style(frame: number, r: number, g: number, b: number, a: number, life: number, fade: number): void;
declare function bloom_decals_update(dt: number): number;
declare function bloom_decals_instance_buffer(): number;
declare function bloom_decals_clear(): void;

declare function bloom_mesh_scratch_reset(): void;
declare function bloom_mesh_scratch_push_f32(v: number): void;

/// How one system's particles are born, move and look. Every field has a
/// sane default; pass only what you care about.
export interface ParticleConfig {
  /// Seconds. `lifeVar` jitters it symmetrically.
  life?: number;
  lifeVar?: number;
  /// Initial speed along the emit direction, m/s.
  speed?: number;
  speedVar?: number;
  /// Half-angle of the emission cone, radians. Emitting with a zero direction
  /// ignores this and sprays into the full sphere.
  spread?: number;
  /// Constant acceleration, m/s². Default is real gravity.
  gravity?: number;
  /// Linear drag per second. Smoke wants ~2; a spark wants ~0.
  drag?: number;
  /// Size in metres at birth and at death — smoke grows, sparks shrink.
  size0?: number;
  size1?: number;
  sizeVar?: number;
  /// Colour at birth and at death. Put the fade-out in `color1`'s alpha.
  color0?: [number, number, number, number];
  color1?: [number, number, number, number];
  /// Billboard roll, rad/s.
  spin?: number;
  spinVar?: number;
  /// Spawn jitter radius, metres.
  posJitter?: number;
  /// > 0 stretches the quad along velocity by this many seconds of travel.
  /// This is the difference between a round spark and a tracer streak.
  stretch?: number;
  /// Fraction of emitter velocity inherited (reserved; the emitter is
  /// stateless today).
  inherit?: number;
  /// Atlas frames. The shader gets a frame index in `extra.y`.
  frames?: number;
  /// Bounce plane. `restitution` <= 0 (the default) means no collision;
  /// shells want ~0.35 so they clatter instead of sinking.
  floorY?: number;
  restitution?: number;
}

/// Allocate a pool. One system per *look* (smoke, sparks, blood) — each wants
/// its own material and blend bucket anyway, and one draw per look is cheap.
export function createParticleSystem(capacity: number, cfg: ParticleConfig): number {
  const sys = bloom_particles_create(capacity);
  if (sys > 0) configureParticleSystem(sys, cfg);
  return sys;
}

/// Re-tune a system at runtime. Config crosses via the mesh scratch (Perry
/// rejects JS arrays in pointer params); it is a startup-shaped call, not a
/// per-frame one.
export function configureParticleSystem(sys: number, cfg: ParticleConfig): void {
  const c0 = cfg.color0 !== undefined ? cfg.color0 : [1, 1, 1, 1];
  const c1 = cfg.color1 !== undefined ? cfg.color1 : [1, 1, 1, 0];
  const p: number[] = new Array<number>(26);
  p[0]  = cfg.life !== undefined ? cfg.life : 1.0;
  p[1]  = cfg.lifeVar !== undefined ? cfg.lifeVar : 0.0;
  p[2]  = cfg.speed !== undefined ? cfg.speed : 1.0;
  p[3]  = cfg.speedVar !== undefined ? cfg.speedVar : 0.0;
  p[4]  = cfg.spread !== undefined ? cfg.spread : 0.3;
  p[5]  = cfg.gravity !== undefined ? cfg.gravity : -9.81;
  p[6]  = cfg.drag !== undefined ? cfg.drag : 0.0;
  p[7]  = cfg.size0 !== undefined ? cfg.size0 : 0.2;
  p[8]  = cfg.size1 !== undefined ? cfg.size1 : 0.2;
  p[9]  = cfg.sizeVar !== undefined ? cfg.sizeVar : 0.0;
  p[10] = c0[0]; p[11] = c0[1]; p[12] = c0[2]; p[13] = c0[3];
  p[14] = c1[0]; p[15] = c1[1]; p[16] = c1[2]; p[17] = c1[3];
  p[18] = cfg.spin !== undefined ? cfg.spin : 0.0;
  p[19] = cfg.spinVar !== undefined ? cfg.spinVar : 0.0;
  p[20] = cfg.posJitter !== undefined ? cfg.posJitter : 0.0;
  p[21] = cfg.stretch !== undefined ? cfg.stretch : 0.0;
  p[22] = cfg.inherit !== undefined ? cfg.inherit : 0.0;
  p[23] = cfg.frames !== undefined ? cfg.frames : 1.0;
  p[24] = cfg.floorY !== undefined ? cfg.floorY : 0.0;
  p[25] = cfg.restitution !== undefined ? cfg.restitution : 0.0;

  bloom_mesh_scratch_reset();
  for (let i = 0; i < 26; i++) bloom_mesh_scratch_push_f32(p[i]);
  bloom_particles_configure(sys);
}

/// Spawn a burst. `dir` biases the cone — pass a surface normal for an impact,
/// or (0,0,0) for an omnidirectional pop.
export function emitParticles(
  sys: number,
  x: number, y: number, z: number,
  dx: number, dy: number, dz: number,
  count: number,
): void {
  bloom_particles_emit(sys, x, y, z, dx, dy, dz, count);
}

/// Integrate + upload. Returns the live count — pass it straight to
/// `drawMeshWithMaterialInstanced` as the instance count.
export function updateParticles(sys: number, dt: number): number {
  return bloom_particles_update(sys, dt);
}

export function particleInstanceBuffer(sys: number): number {
  return bloom_particles_instance_buffer(sys);
}

export function clearParticles(sys: number): void {
  bloom_particles_clear(sys);
}

export function particleCount(sys: number): number {
  return bloom_particles_live(sys);
}

// ---- Decals -----------------------------------------------------------------

/// Allocate the decal ring. One ring for the whole game: decals are all the
/// same draw (an oriented quad against an atlas), so they share a buffer and
/// the style selects which atlas cell a given spawn uses.
export function initDecals(capacity: number): number {
  return bloom_decals_init(capacity);
}

/// Look + lifetime of subsequent `spawnDecal` calls. `life <= 0` = permanent
/// (until the ring wraps); `fade` is how many trailing seconds it fades over.
export function setDecalStyle(
  frame: number,
  r: number, g: number, b: number, a: number,
  life: number, fade: number,
): void {
  bloom_decals_set_style(frame, r, g, b, a, life, fade);
}

/// Stick a decal to a surface. `n` is the surface normal (from your raycast);
/// `roll` spins it about that normal so repeated hits don't look stamped.
export function spawnDecal(
  x: number, y: number, z: number,
  nx: number, ny: number, nz: number,
  size: number, roll: number,
): void {
  bloom_decals_spawn(x, y, z, nx, ny, nz, size, roll);
}

export function updateDecals(dt: number): number {
  return bloom_decals_update(dt);
}

export function decalInstanceBuffer(): number {
  return bloom_decals_instance_buffer();
}

export function clearDecals(): void {
  bloom_decals_clear();
}
