// One cloud deck — shared by the sky that draws the clouds and the ground that
// takes their shadow.
//
// WHY THIS IS ONE FILE. These used to be two unrelated noise fields that had
// never been reconciled, and the result did not survive looking at:
//
//   - The sky's puffs were fBm over a plane pinned to the CAMERA, so they slid
//     along with the player instead of hanging over the world.
//   - The ground's "cloud shadow" was a *different*, much finer field on a
//     different noise, drifting ~80x faster (20 m/s against the sky's 0.25).
//
// So the shadow racing across the grass had no cloud above it, and the cloud
// overhead cast nothing. Worse, only the materials that happened to carry a copy
// of the ground function darkened at all: in the shooter that was the grass, but
// not the terrain under it, nor the trees standing in it — a cloud shadow that
// crosses the field and ignores the forest in the middle of it.
//
// Now there is ONE field, in WORLD space, and the shadow you are standing in
// belongs to the cloud you can look up and see.
//
// THE MODEL. The deck is a horizontal plane at `p.y` metres. A view ray and a
// sun ray are both intersected with it; whatever they hit is the same cloud.
// That is the entire trick, and it is why the apparent size of a cloud in the
// sky and the size of its shadow on the ground are no longer independently
// tunable — they are the same number seen from two directions, which is the
// point.

fn cloud_hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn cloud_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let uu = f * f * (3.0 - 2.0 * f);
    let a = cloud_hash(i);
    let b = cloud_hash(i + vec2<f32>(1.0, 0.0));
    let c = cloud_hash(i + vec2<f32>(0.0, 1.0));
    let d = cloud_hash(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, uu.x), mix(c, d, uu.x), uu.y);
}

fn cloud_fbm(p0: vec2<f32>) -> f32 {
    var s = 0.0;
    var amp = 0.5;
    var q = p0;
    for (var i = 0; i < 5; i = i + 1) {
        s = s + amp * cloud_noise(q);
        q = q * 2.03;
        amp = amp * 0.5;
    }
    return s;
}

// Cloud params, passed by each caller from whichever uniform block it happens to
// have (the sky pass and the material ABI do not share one, so this file takes
// them explicitly rather than reaching for a global):
//
//   x = shadow strength   0 = the world ignores the clouds entirely (default)
//   y = deck height       world metres
//   z = feature scale     noise units per metre — 1/z is the cloud size
//   w = drift speed       metres/second
//
// The drift DIRECTION is the wind vector — the same one that bends the grass —
// so the clouds travel the way the foliage under them is leaning. A game that
// never set a wind still gets a slow default heading rather than a frozen sky.
fn cloud_drift(wind_xy: vec2<f32>, t: f32, speed: f32) -> vec2<f32> {
    var d = wind_xy;
    if (dot(d, d) < 1e-6) { d = vec2<f32>(1.0, 0.25); }
    return normalize(d) * speed * t;
}

// Coverage of the deck at a point ON the deck. 0 = clear sky, 1 = solid cloud.
fn cloud_density(deck_xz: vec2<f32>, wind_xy: vec2<f32>, t: f32, cp: vec4<f32>) -> f32 {
    let p = (deck_xz - cloud_drift(wind_xy, t, cp.w)) * cp.z;
    // The threshold is high on purpose: it buys mostly-open sky with the
    // occasional real cloud, instead of a permanent grey smear. A low threshold
    // here is what makes a procedural deck read as fog.
    return smoothstep(0.56, 1.04, cloud_fbm(p));
}

// Where a ray from `origin` along `dir` pierces the deck.
fn cloud_deck_hit(origin: vec3<f32>, dir: vec3<f32>, deck_y: f32) -> vec2<f32> {
    let t = (deck_y - origin.y) / dir.y;
    return origin.xz + dir.xz * t;
}

// --- the two consumers -------------------------------------------------------

// SKY: coverage along a view ray. Returns (coverage, sun-alignment) — the caller
// colours the cloud from the second component so puffs facing the sun burn white
// and the ones facing away stay cool grey.
fn cloud_cover_view(cam: vec3<f32>, dir: vec3<f32>, sun_dir: vec3<f32>,
                    wind_xy: vec2<f32>, t: f32, cp: vec4<f32>) -> vec2<f32> {
    if (dir.y <= 0.02) { return vec2<f32>(0.0, 0.0); }
    var cov = cloud_density(cloud_deck_hit(cam, dir, cp.y), wind_xy, t, cp);
    // Toward the horizon a view ray runs so far through the deck that modelling
    // it as an infinitely thin plane stops being defensible — fade out instead
    // of drawing the smear the maths would give.
    let horizon_fade = smoothstep(0.03, 0.24, dir.y);
    // Thin the deck right around the sun so the disk still burns through.
    let near_sun = smoothstep(0.90, 0.999, dot(dir, sun_dir));
    cov = cov * horizon_fade * (1.0 - near_sun * 0.8) * 0.9;
    let sun_amt = clamp(dot(dir, sun_dir) * 0.5 + 0.5, 0.0, 1.0);
    return vec2<f32>(cov, sun_amt);
}

// GROUND: how much sun a world point keeps. 1.0 = full sun, lower = under cloud.
// Multiply this into direct sunlight only — a cloud blocks the sun, it does not
// stop the sky from being blue, and folding it into ambient too is what makes
// cloud shadows read as flat grey paint rather than as shade.
fn cloud_shadow_at(world_pos: vec3<f32>, sun_dir: vec3<f32>,
                   wind_xy: vec2<f32>, t: f32, cp: vec4<f32>) -> f32 {
    if (cp.x <= 0.0) { return 1.0; }
    // Sun on the horizon: the shadow ray runs nearly parallel to the deck and
    // the intersection shoots off to infinity, so the noise lookup lands a
    // kilometre away and swims. Fade the whole effect out as the sun sets.
    let up = sun_dir.y;
    if (up <= 0.02) { return 1.0; }
    let cov = cloud_density(cloud_deck_hit(world_pos, sun_dir, cp.y), wind_xy, t, cp);
    return 1.0 - cov * cp.x * smoothstep(0.02, 0.20, up);
}
