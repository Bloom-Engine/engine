// Weather that moves: a state eased over time, the rain that falls from it,
// the wet it leaves behind and the lightning in it.
//
// `setWeather` in core is the renderer's side: two numbers the sky, the cloud
// shadow and the scene shader read. This module is the game's side. It keeps
// where the weather is and where it is going, turns rain into wetness and
// lets it dry, draws the rain, and lights a flash.
//
//   weatherTo(1.0, 0.8, 40);              // close in and start raining over 40 s
//   ... each frame:
//   weatherUpdate(getDeltaTime());        // before drawing
//   beginMode3D(camera);
//   ...
//   weatherDraw(cam.x, cam.y, cam.z);     // inside the 3D pass
//   endMode3D();
//
// Perry rules apply to engine TypeScript too: module-level state, no
// reachable throw, no single-ternary helpers.

import { setWeather } from "../core/index";
import { compileAdditiveMaterial, createMeshExplicit, drawMeshWithMaterial } from "../models/index";
import { setMaterialParams } from "../core/index";
import { addDirectionalLight } from "../scene/index";
import type { Model } from "../core/index";

// ---- state ----------------------------------------------------------------
let cover = 0.3;            // cloud cover now, 0..1
let rain = 0.0;             // rain now, 0..1
let wetness = 0.0;          // surface wetness now, 0..1
let coverFrom = 0.3;
let rainFrom = 0.0;
let coverTo = 0.3;
let rainTo = 0.0;
let easeT = 1.0;            // 0..1 along the current change
let easeSeconds = 0.0;
let active = false;         // nothing is sent to the renderer until the game sets a weather
let indoors = false;
let windX = 1.0;            // the rain's slant: unit direction in XZ ...
let windZ = 0.0;
let windSpeed = 3.0;        // ... and how fast it carries a drop sideways, m/s
let flash = 0.0;            // lightning brightness now
let flashAge = 10.0;        // seconds since the last strike
let flashPower = 0.0;
let flashX = 0.3;
let flashZ = 0.2;
let strikes = 0;

/// Seconds of full rain that soak a dry surface, and seconds a soaked one
/// takes to dry once the rain has stopped.
let soakSeconds = 12.0;
let drySeconds = 120.0;

function clamp01(v: number): number {
  let r = v;
  if (r < 0.0) { r = 0.0; }
  if (r > 1.0) { r = 1.0; }
  return r;
}

/// Set the weather at once: cloud cover and rain, 0 to 1 each.
export function weatherSet(cloudCover: number, rainAmount: number): void {
  cover = clamp01(cloudCover);
  rain = clamp01(rainAmount);
  coverFrom = cover; coverTo = cover;
  rainFrom = rain; rainTo = rain;
  easeT = 1.0;
  active = true;
}

/// Move to a weather over `seconds`, from wherever it is now.
export function weatherTo(cloudCover: number, rainAmount: number, seconds: number): void {
  coverFrom = cover; rainFrom = rain;
  coverTo = clamp01(cloudCover);
  rainTo = clamp01(rainAmount);
  easeSeconds = seconds;
  easeT = 0.0;
  if (seconds <= 0.0) { easeT = 1.0; cover = coverTo; rain = rainTo; }
  active = true;
}

/// How wet surfaces are, set directly (a scene that opens after rain).
/// `weatherUpdate` carries on from here: rain adds to it, dry weather takes it away.
export function weatherSetWetness(amount: number): void {
  wetness = clamp01(amount);
  active = true;
}

/// How long full rain takes to soak a surface and a soaked surface takes to
/// dry, in seconds (12 and 120 unless set).
export function weatherSoakTimes(soak: number, dry: number): void {
  if (soak > 0.1) { soakSeconds = soak; }
  if (dry > 0.1) { drySeconds = dry; }
}

/// The wind the rain falls in: a direction in the XZ plane and a speed in m/s
/// (3 is a breeze, 10 drives the rain hard aslant). This does not move the
/// foliage or the clouds; `setWind` does, and a game usually sets both.
export function weatherWind(dirX: number, dirZ: number, speed: number): void {
  const len = Math.sqrt(dirX * dirX + dirZ * dirZ);
  if (len > 0.0001) { windX = dirX / len; windZ = dirZ / len; }
  windSpeed = speed;
  if (windSpeed < 0.0) { windSpeed = 0.0; }
}

/// Under a roof: no rain is drawn and surfaces are shown dry, while the
/// weather itself (and the wet outside) carries on.
export function weatherIndoors(on: boolean): void {
  indoors = on;
}

/// A lightning strike. `power` 1 is a strong flash close by. It lights the
/// scene from above for about half a second, flickering, and brightens the
/// rain. The thunder is the game's to play.
export function weatherLightning(power: number): void {
  flashPower = power;
  flashAge = 0.0;
  strikes = strikes + 1;
  // A different quarter of the sky each time, from a fixed sequence.
  const a = strikes * 2.399963;
  flashX = Math.cos(a) * 0.45;
  flashZ = Math.sin(a) * 0.45;
}

export function weatherCover(): number { return cover; }
export function weatherRain(): number { return rain; }
export function weatherWetness(): number { return wetness; }
export function weatherFlash(): number { return flash; }

/// Advance the weather by `dt` seconds and hand it to the renderer. Call once
/// a frame, before drawing.
export function weatherUpdate(dt: number): void {
  if (!active) { return; }
  if (easeT < 1.0) {
    easeT = easeT + dt / easeSeconds;
    if (easeT > 1.0) { easeT = 1.0; }
    const s = easeT * easeT * (3.0 - 2.0 * easeT);
    cover = coverFrom + (coverTo - coverFrom) * s;
    rain = rainFrom + (rainTo - rainFrom) * s;
  }
  if (rain > 0.02) {
    wetness = wetness + dt * rain / soakSeconds;
  } else {
    wetness = wetness - dt / drySeconds;
  }
  wetness = clamp01(wetness);

  flashAge = flashAge + dt;
  flash = 0.0;
  if (flashAge < 0.7) {
    // A sharp first stroke and one or two weaker return strokes.
    let f = Math.exp(-flashAge * 9.0);
    const again = flashAge - 0.16;
    if (again > 0.0) { f = f + 0.55 * Math.exp(-again * 14.0); }
    const third = flashAge - 0.31;
    if (third > 0.0) { f = f + 0.3 * Math.exp(-third * 16.0); }
    flash = f * flashPower;
  }

  let shownWet = wetness;
  if (indoors) { shownWet = 0.0; }
  setWeather(cover, shownWet);
}

// ---- rain -----------------------------------------------------------------
// One mesh of streak quads, placed and moved in the vertex shader: each vertex
// carries its drop's seed position in a unit cube, three random numbers in its
// colour and the quad corner in its UV. The drops live in a 24 x 16 x 24 m
// volume that follows the camera and wraps, so the rain holds still in the
// world while the camera turns. Lighter rain shows fewer of the drops.
const RAIN_DROPS = 3200;
let rainMesh: Model = { handle: 0, meshCount: 0, materialCount: 0, transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
let rainMaterial = 0;
let rainReady = false;
const rainParams: number[] = [1.0, 0.0, 3.0, 17.0, 1.0, 0.0, 0.0, 0.0];
let seed = 0x2545f491;

function nextRandom(): number {
  // xorshift32, so the rain is the same mesh on every run and every machine
  seed = seed ^ (seed << 13);
  seed = seed ^ (seed >>> 17);
  seed = seed ^ (seed << 5);
  return ((seed >>> 0) % 1000003) / 1000003.0;
}

const RAIN_WGSL = `
#include "material_abi.wgsl"

struct RainParams {
  // x = brightness, y = rain amount 0..1, z = wind speed (m/s), w = fall speed (m/s)
  rain: vec4<f32>,
  // xy = wind direction in the XZ plane
  wind: vec4<f32>,
};
@group(2) @binding(11) var<uniform> user: RainParams;

const VOLUME: vec3<f32> = vec3<f32>(24.0, 16.0, 24.0);

struct VsOut {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) quad: vec2<f32>,
  @location(1) fade: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VsOut {
  var out: VsOut;
  let seed = in.position + vec3<f32>(0.5);
  let speed = 0.8 + 0.4 * in.color.r;
  let cam = view.camera_pos.xyz;
  let fall = vec3<f32>(user.wind.x * user.rain.z, -user.rain.w, user.wind.y * user.rain.z);
  let travelled = seed * VOLUME + fall * speed * frame.time;
  let centre = (fract((travelled - cam) / VOLUME) - vec3<f32>(0.5)) * VOLUME + cam;

  let along = normalize(fall);
  let to_cam = normalize(cam - centre);
  let across = normalize(cross(along, to_cam));
  let dist = length(cam - centre);
  // A hair's width near the lens, never thinner than a pixel far away.
  let pixel = dist * 2.0 / (view.proj[1][1] * frame.render_resolution.y);
  let half_width = max(0.0045, pixel * 0.8);
  let half_length = 0.30 * speed * (0.6 + 0.4 * user.rain.y);
  let corner = in.uv * 2.0 - vec2<f32>(1.0);
  // Lighter rain is fewer drops: each drop has a threshold it falls above.
  let shown = step(in.color.b, user.rain.y);
  let world = centre + (across * corner.x * half_width + along * corner.y * half_length) * shown;
  out.clip_position = view.view_proj * vec4<f32>(world, 1.0);
  out.quad = corner;
  let near = smoothstep(1.2, 3.0, dist);
  let far = 1.0 - smoothstep(8.5, 11.5, dist);
  out.fade = shown * near * far * min(1.0, 0.0045 / half_width + 0.3) * (0.55 + 0.45 * in.color.g);
  return out;
}

@fragment
fn fs_main(in: VsOut) -> TranslucentOut {
  var out: TranslucentOut;
  let across = 1.0 - in.quad.x * in.quad.x;
  let along = smoothstep(-1.0, 0.6, in.quad.y) * (1.0 - smoothstep(0.75, 1.0, in.quad.y));
  let light = (view.ambient.rgb * view.ambient.w * 3.0 + view.fog.rgb * 2.2 + vec3<f32>(0.05)) * user.rain.x;
  let a = across * along * in.fade;
  out.hdr = vec4<f32>(light * a, a);
  return out;
}
`;

function rainInit(): void {
  rainReady = true;
  const verts: number[] = [];
  const idx: number[] = [];
  for (let i = 0; i < RAIN_DROPS; i = i + 1) {
    const sx = nextRandom() - 0.5;
    const sy = nextRandom() - 0.5;
    const sz = nextRandom() - 0.5;
    const sp = nextRandom();
    const br = nextRandom();
    const th = nextRandom();
    for (let k = 0; k < 4; k = k + 1) {
      let u = 0.0;
      let v = 0.0;
      if (k === 1 || k === 2) { u = 1.0; }
      if (k >= 2) { v = 1.0; }
      verts.push(sx); verts.push(sy); verts.push(sz);
      verts.push(0.0); verts.push(1.0); verts.push(0.0);
      verts.push(sp); verts.push(br); verts.push(th); verts.push(1.0);
      verts.push(u); verts.push(v);
    }
    const b = i * 4;
    idx.push(b); idx.push(b + 1); idx.push(b + 2);
    idx.push(b); idx.push(b + 2); idx.push(b + 3);
  }
  rainMesh = createMeshExplicit(verts, RAIN_DROPS * 4, idx, RAIN_DROPS * 6);
  rainMaterial = compileAdditiveMaterial(RAIN_WGSL);
}

/// Draw what the weather puts in the air and light its flash. Call inside the
/// 3D pass, after the scene, with the camera's position.
export function weatherDraw(camX: number, camY: number, camZ: number): void {
  if (!active) { return; }
  if (flash > 0.01) {
    addDirectionalLight(flashX, 0.85, flashZ, 0.82, 0.88, 1.0, flash * 6.0);
  }
  if (indoors || rain < 0.02) { return; }
  if (!rainReady) { rainInit(); }
  if (rainMaterial <= 0) { return; }
  // Faint by day: a streak is a glint of sky light, not a white line.
  rainParams[0] = 0.42 + flash * 2.5;
  rainParams[1] = rain;
  rainParams[2] = windSpeed;
  rainParams[3] = 15.0 + 4.0 * rain;
  rainParams[4] = windX;
  rainParams[5] = windZ;
  setMaterialParams(rainMaterial, rainParams);
  // Scale 24 puts the mesh's bounds round the camera so it is never culled;
  // the shader places the drops itself.
  drawMeshWithMaterial(rainMaterial, rainMesh, { x: camX, y: camY, z: camZ }, 24.0, { r: 255, g: 255, b: 255, a: 255 });
}
