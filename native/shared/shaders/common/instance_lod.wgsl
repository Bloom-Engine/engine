// EN-060 — distance thinning of a static instance buffer: the material's half.
//
// A buffer made with createInstanceBufferLod(data, count, start, end,
// minFraction, band, minDensity) carries a `rank` in (0, 1) per instance in
// `@location(11) instance_extra.x`, spread evenly inside each of the
// buffer's spatial tiles, and in `instance_extra.y` the `floor` of its tile:
// the level below which that tile is never thinned (how minDensity reaches
// the shader). An instance is drawn whole while
// `instance_lod_level(d, lod, floor)` is above `rank + band`, shrinks as the level
// comes down, and is gone when the level reaches its rank. The engine stops
// DRAWING the instances of a tile whose rank is at or above the level at the
// tile's nearest point, so it only ever leaves out instances the functions
// below have already taken to zero: the shader makes the hand-over smooth,
// the engine makes it cheap.
//
// In the vertex stage, with `lod = vec4(start, end, minFraction, band)`
// (send the numbers the buffer was made with as material parameters):
//
//   let d     = distance(view.camera_pos.xyz, in.instance_pos);  // full 3D
//   let level = instance_lod_level(d, lod, in.instance_extra.y);
//   let vis   = instance_lod_visible(in.instance_extra.x, level, lod.w);
//   // scale the instance by `vis`; to hold the field's coverage, widen
//   // what is left by instance_lod_fill(level, lod.w).

// The thinning level at eye distance d: `1 + band` up to `start` (every
// rank whole), a straight line down to `minFraction` at `end`, constant
// beyond, and never under the tile's floor. The Rust twin is
// `InstanceLod::level`.
fn instance_lod_level(d: f32, lod: vec4<f32>, tile_floor: f32) -> f32 {
  let t = clamp((d - lod.x) / max(lod.y - lod.x, 0.001), 0.0, 1.0);
  return max(mix(1.0 + lod.w, lod.z, t), tile_floor);
}

// 1 = draw the instance whole, 0 = it is gone (and the engine may not be
// drawing it at all).
fn instance_lod_visible(rank: f32, level: f32, band: f32) -> f32 {
  return clamp((level - rank) / max(band, 0.0001), 0.0, 1.0);
}

// 1 / (the share of the field's mass left at `level`: whole instances plus
// the part-way ones). Multiply an instance's width by it to keep the
// covered area constant. Exactly 1 where nothing is thinned.
fn instance_lod_fill(level: f32, band: f32) -> f32 {
  let b  = max(band, 0.0001);
  let hi = min(level, 1.0);
  let lo = clamp(level - b, 0.0, 1.0);
  // integral over rank in 0..1 of instance_lod_visible
  let mass = lo + ((level - lo) * (level - lo) - (level - hi) * (level - hi)) / (2.0 * b);
  return 1.0 / max(mass, 0.05);
}
