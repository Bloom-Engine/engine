// Hierarchical foliage wind — one field, shared by the scene pass that draws the
// tree and the shadow pass that casts it.
//
// WHY THREE LAYERS. A tree does not move as one thing. The trunk leans slowly
// under the whole wind load, the branches swing at their own rate, and the
// leaves flutter fast at the tips. Driving all of it from a single sine is what
// makes procedural foliage read as rippling cloth instead of wood.
//
// What was here before did less than that: the engine swayed *alpha-cut
// materials only*, so leaf cards fluttered and every trunk in every game was
// perfectly rigid — and the shadow shaders applied no wind at all, so the leaves
// moved while their shadows stayed nailed to the ground.
//
// WHERE THE WEIGHTS COME FROM. SH-013 planned to author them into vertex colours
// (R = bend, G = branch, B = flutter). That is the right answer for hand-modelled
// trees, but ours are procedurally generated, so the regions are known exactly and
// can simply be derived from where the vertex sits relative to the tree's base:
//
//   trunk bend   proportional to height^2   -- a cantilever: the crown travels, the roots do not
//   branch sway  proportional to reach out from the trunk axis
//   leaf flutter cutout cards only, small and fast
//
// So: no new vertex attribute, no GLB re-bake, no COLOR_0 (which the scene shader
// already spends on albedo tint), and it works on any foliage model the game flags.
//
// The offset is returned in WORLD space and added after the model transform.
// Displacing in local space would let each tree's per-instance yaw rotate the
// wind with it, and a stand of trees would bend in a dozen different directions.

// `rel` is the vertex's WORLD-space offset from its model origin (so it is
// already scaled and rotated -- the coefficients below are in metres and mean the
// same thing for a sapling and a giant).
// `wind` is the global wind vec4: xy = direction in the XZ plane, z = amplitude,
// w = elapsed seconds.
// `amount` scales the whole effect per draw; 0 = this is not foliage, don't move
// it. `is_leaf` is 1.0 for cutout cards, 0.0 for wood.
fn foliage_wind_world(rel: vec3<f32>, model_origin: vec3<f32>,
                      wind: vec4<f32>, amount: f32, is_leaf: f32) -> vec3<f32> {
    if (amount <= 0.0 || wind.z <= 0.0) { return vec3<f32>(0.0, 0.0, 0.0); }

    let dir = vec3<f32>(wind.x, 0.0, wind.y);
    let amp = wind.z * amount;
    let t   = wind.w;

    // Per-tree phase from the model's world origin. Without it a stand of trees
    // sways in lockstep, which is the single most obvious tell of fake foliage.
    let ph = model_origin.x * 0.137 + model_origin.z * 0.241;

    let h     = max(rel.y, 0.0);
    let reach = length(rel.xz);

    // 1. TRUNK BEND — cantilever, so travel grows with the SQUARE of height: the
    //    base is planted, the crown swings. Slow (~0.15 Hz). This is the motion
    //    you read from 30 m away, and it is the one that did not exist at all.
    // Coefficients are tuned so amount = 1.0 is a real tree in a lazy breeze:
    // with the shooter's wind amplitude (0.10) a 4 m crown travels ~35 cm.
    let bend = amp * 0.22 * h * h * sin(t * 0.95 + ph);

    // 2. BRANCH SWAY — how far the vertex reaches out from the trunk axis. Medium
    //    (~0.4 Hz), phase-offset by azimuth so opposite limbs do not swing together.
    let azim  = atan2(rel.z, rel.x);
    // Gated by height as well as reach: without it the flare at the base of the
    // trunk has some reach and would shuffle its own roots.
    let swing = amp * 0.70 * reach * clamp(h * 0.5, 0.0, 1.0)
              * sin(t * 2.4 + ph + azim * 1.7);

    // 3. LEAF FLUTTER — cutout cards only. Fast (~1 Hz), small, and keyed on the
    //    vertex's own position so neighbouring cards break up rather than shimmer
    //    as a sheet.
    let fl = amp * 0.60 * is_leaf * sin(t * 6.0 + ph + h * 3.1 + reach * 5.0);

    var o = dir * (bend + swing + fl);
    // Leaves twist rather than only sliding downwind; and a leaning crown dips a
    // little, because the tip is swinging on an arc, not a rail.
    o.y = o.y + fl * 0.45 - bend * 0.15;
    return o;
}

// Convenience wrapper: local-space vertex in, wind-displaced local-space vertex
// out. Both the scene pass and the shadow pass want exactly this, and they must
// agree exactly or a tree detaches from its own shadow.
//
// The offset is computed in WORLD space (see above) and then brought back into
// local space by inverting the model's linear part. That inverse is exact for the
// transforms these draws use -- rotation plus UNIFORM scale -- because then
// M^-1 = M^T / s^2.
fn foliage_wind_local(local_pos: vec3<f32>, model: mat4x4<f32>,
                      wind: vec4<f32>, amount: f32, is_leaf: f32) -> vec3<f32> {
    if (amount <= 0.0 || wind.z <= 0.0) { return local_pos; }
    let origin = model[3].xyz;
    let rel    = (model * vec4<f32>(local_pos, 1.0)).xyz - origin;
    let wo     = foliage_wind_world(rel, origin, wind, amount, is_leaf);
    let c0 = model[0].xyz;
    let s2 = max(dot(c0, c0), 1e-8);
    return local_pos + vec3<f32>(dot(model[0].xyz, wo),
                                 dot(model[1].xyz, wo),
                                 dot(model[2].xyz, wo)) / s2;
}
