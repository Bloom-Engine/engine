// ============================================================================
// jolt_bridge.js — web-side implementation of the bloom_physics_* FFI surface.
//
// The native Rust/Jolt backend is replaced on the web target by JoltPhysics.js
// (https://github.com/jrouwe/JoltPhysics.js). This module exports one function
// per `bloom_physics_*` entry so `bloom_web.wasm` can forward calls verbatim.
//
// Lifecycle:
//   1. Before Perry WASM boots, the host page awaits `initJolt()` below, which
//      loads the Jolt WASM module and wires allocators + types.
//   2. Once ready, all bridge calls operate on live Jolt objects.
//   3. If a bridge call arrives before init (shouldn't happen if the host page
//      sequences correctly), the bridge logs once and returns 0 / a no-op.
//
// Handle model mirrors the native Rust side (HandleRegistry<f64>):
//   - 1-based numeric handles ("slots") issued sequentially per resource type.
//   - 0 == INVALID.
//   - Separate maps for worlds / shapes / bodies / constraints.
// ============================================================================

let JoltModule = null;

const state = {
  worlds:      new Map(),   // handle → { system, bodyInterface, temp, jobs, contacts, rayHits, overlaps }
  shapes:      new Map(),   // handle → Jolt Shape
  bodies:      new Map(),   // handle → { world: number, bodyId: number }
  constraints: new Map(),   // handle → { world: number, constraint: Jolt.Constraint }
  characters:  new Map(),   // handle → { world: number, character: Jolt.CharacterVirtual, layer: number }
  nextWorld: 1, nextShape: 1, nextBody: 1, nextConstraint: 1, nextCharacter: 1,
  // Most-recent query results (drained on read)
  rayHits: [],       // array of { body, point:[x,y,z], normal:[x,y,z], fraction, subShapeId }
  overlapBodies: [], // array of body handles
  contacts: [],      // array of contact objects
  // Scratch streams for variable-size shape inputs.
  scratchF32: [],
  scratchU32: [],
  // Compound-shape builder state (cleared by compoundBegin).
  compoundChildren: [],
};

function warnUninit(fnName) {
  if (!JoltModule) {
    if (!warnUninit._warned) {
      console.warn('[jolt_bridge] Jolt not initialised; bloom_physics_* calls are no-ops until initJolt() resolves');
      warnUninit._warned = true;
    }
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Initialisation — must be awaited before Perry calls any bloom_physics_* fn.
// ---------------------------------------------------------------------------

/**
 * @param {Object} joltFactory - the default export of jolt-physics (a function
 *   returning a Promise<Jolt>). Accepting it via parameter lets the host page
 *   pick how to load it (CDN script tag, bundler, etc.).
 */
export async function initJolt(joltFactory) {
  if (JoltModule) return JoltModule;
  if (typeof joltFactory !== 'function') {
    throw new Error('initJolt requires a joltFactory function (Jolt.default or window.Jolt)');
  }
  JoltModule = await joltFactory();
  return JoltModule;
}

export function isJoltReady() { return !!JoltModule; }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function vec3(x, y, z) { return new JoltModule.Vec3(x, y, z); }
function rvec3(x, y, z) { return new JoltModule.RVec3(x, y, z); }
function quat(x, y, z, w) { return new JoltModule.Quat(x, y, z, w); }

const OBJECT_LAYER_NON_MOVING = 0;
const OBJECT_LAYER_MOVING     = 1;
const BP_LAYER_NON_MOVING     = 0;
const BP_LAYER_MOVING         = 1;

function buildSettings(J) {
  // Object-layer pair filter — 16 layers, mirrors the native C++ shim.
  const pairFilter = new J.ObjectLayerPairFilterTable(16);
  // Default: all layers collide except non-moving↔non-moving.
  for (let a = 0; a < 16; a++) {
    for (let b = 0; b < 16; b++) {
      if (a === OBJECT_LAYER_NON_MOVING && b === OBJECT_LAYER_NON_MOVING) continue;
      pairFilter.EnableCollision(a, b);
    }
  }
  // Broadphase-layer interface — 2 broadphase buckets.
  const bpInterface = new J.BroadPhaseLayerInterfaceTable(16, 2);
  bpInterface.MapObjectToBroadPhaseLayer(OBJECT_LAYER_NON_MOVING, new J.BroadPhaseLayer(BP_LAYER_NON_MOVING));
  for (let i = 1; i < 16; i++) {
    bpInterface.MapObjectToBroadPhaseLayer(i, new J.BroadPhaseLayer(BP_LAYER_MOVING));
  }
  const objectVsBp = new J.ObjectVsBroadPhaseLayerFilterTable(bpInterface, 2, pairFilter, 16);

  const settings = new J.JoltSettings();
  settings.mObjectLayerPairFilter        = pairFilter;
  settings.mBroadPhaseLayerInterface     = bpInterface;
  settings.mObjectVsBroadPhaseLayerFilter= objectVsBp;
  settings.mMaxBodies                    = 65536;
  settings.mMaxBodyPairs                 = 65536;
  settings.mMaxContactConstraints        = 10240;
  // The filter tables ride along: builds without
  // system.GetDefaultBroadPhaseLayerFilter() need them to construct the
  // per-layer query filters (see queryFilters).
  return { settings, pairFilter, objectVsBp };
}

function motionTypeFrom(t) {
  const J = JoltModule;
  switch (t | 0) {
    case 0: return J.EMotionType_Static;
    case 1: return J.EMotionType_Kinematic;
    default: return J.EMotionType_Dynamic;
  }
}

function activationFrom(flag) {
  const J = JoltModule;
  return flag !== 0 ? J.EActivation_Activate : J.EActivation_DontActivate;
}

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------

export function createWorld(gx, gy, gz, maxBodies, numThreads) {
  if (warnUninit('createWorld')) return 0;
  const J = JoltModule;
  const { settings, pairFilter, objectVsBp } = buildSettings(J);
  if ((maxBodies | 0) > 0) settings.mMaxBodies = maxBodies | 0;
  const jolt = new J.JoltInterface(settings);
  const system = jolt.GetPhysicsSystem();
  system.SetGravity(vec3(gx, gy, gz));

  const handle = state.nextWorld++;

  // Contact listener — bridges Jolt contact callbacks into the state.contacts queue.
  // JoltPhysics.js provides ContactListenerJS with virtual methods exposed as JS fields.
  const listener = new J.ContactListenerJS();
  const pushContact = (event, body1, body2, manifold, settings_out) => {
    const bodyIdA = body1.GetID();
    const bodyIdB = body2.GetID();
    const bodyAh = bodyIdToHandle(handle, bodyIdA);
    const bodyBh = bodyIdToHandle(handle, bodyIdB);
    let pointA = [0, 0, 0], pointB = [0, 0, 0], normal = [0, 1, 0], depth = 0;
    if (manifold) {
      const base = manifold.mBaseOffset;
      const n = manifold.mWorldSpaceNormal;
      normal = [n.GetX(), n.GetY(), n.GetZ()];
      depth = manifold.mPenetrationDepth;
      if (manifold.mRelativeContactPointsOn1.size() > 0) {
        const p = manifold.mRelativeContactPointsOn1.at(0);
        pointA = [base.GetX() + p.GetX(), base.GetY() + p.GetY(), base.GetZ() + p.GetZ()];
      }
      if (manifold.mRelativeContactPointsOn2.size() > 0) {
        const p = manifold.mRelativeContactPointsOn2.at(0);
        pointB = [base.GetX() + p.GetX(), base.GetY() + p.GetY(), base.GetZ() + p.GetZ()];
      }
    }
    let friction = 0, restitution = 0;
    if (settings_out) {
      friction = settings_out.mCombinedFriction ?? 0;
      restitution = settings_out.mCombinedRestitution ?? 0;
    }
    state.contacts.push({
      event, bodyA: bodyAh, bodyB: bodyBh,
      pointA, pointB, normal,
      penetrationDepth: depth,
      combinedFriction: friction,
      combinedRestitution: restitution,
    });
  };
  listener.OnContactValidate = () => J.ValidateResult_AcceptAllContactsForThisBodyPair;
  listener.OnContactAdded     = (b1, b2, m, s) => pushContact(0, b1, b2, m, s);
  listener.OnContactPersisted = (b1, b2, m, s) => pushContact(1, b1, b2, m, s);
  listener.OnContactRemoved   = (pair) => {
    const b1 = pair.GetBody1ID();
    const b2 = pair.GetBody2ID();
    state.contacts.push({
      event: 2,
      bodyA: bodyIdToHandle(handle, b1),
      bodyB: bodyIdToHandle(handle, b2),
      pointA: [0, 0, 0], pointB: [0, 0, 0], normal: [0, 1, 0],
      penetrationDepth: 0, combinedFriction: 0, combinedRestitution: 0,
    });
  };
  system.SetContactListener(listener);

  state.worlds.set(handle, {
    jolt,
    system,
    bodyInterface: system.GetBodyInterface(),
    settings,
    pairFilter,                  // for DefaultObjectLayerFilter construction
    objectVsBp,                  // for DefaultBroadPhaseLayerFilter construction
    listener,                    // hold ref so GC doesn't collect
  });
  return handle;
}

function bodyIdToHandle(worldH, bodyId) {
  for (const [h, b] of state.bodies) {
    if (b.world === worldH && b.bodyId.GetIndexAndSequenceNumber() === bodyId.GetIndexAndSequenceNumber()) {
      return h;
    }
  }
  return 0;
}

export function destroyWorld(h) {
  const w = state.worlds.get(h);
  if (!w) return;
  // Remove any bodies/constraints tied to this world first.
  for (const [bh, b] of state.bodies) {
    if (b.world === h) { destroyBodyInternal(w, b); state.bodies.delete(bh); }
  }
  for (const [ch, c] of state.constraints) {
    if (c.world === h) { w.system.RemoveConstraint(c.constraint); state.constraints.delete(ch); }
  }
  JoltModule.destroy(w.jolt);
  state.worlds.delete(h);
}

export function setGravity(h, gx, gy, gz) {
  const w = state.worlds.get(h); if (!w) return;
  w.system.SetGravity(vec3(gx, gy, gz));
}

export function getGravity(h, axis) {
  const w = state.worlds.get(h); if (!w) return 0;
  const g = w.system.GetGravity();
  if (axis === 0) return g.GetX();
  if (axis === 1) return g.GetY();
  if (axis === 2) return g.GetZ();
  return 0;
}

export function optimizeBroadphase(h) {
  const w = state.worlds.get(h); if (!w) return;
  w.system.OptimizeBroadPhase();
}

export function step(h, dt, collisionSteps) {
  const w = state.worlds.get(h); if (!w) return;
  const steps = Math.max(1, collisionSteps | 0);
  w.jolt.Step(dt, steps);
}

// --- fixed-timestep stepping (mirrors physics_jolt.rs WorldStepState) ------
// Semantics must match the native implementation exactly: clamp the frame
// dt (0.25s), simulate whole fixed steps with a catch-up cap, carry the
// remainder, snapshot before the LAST step of a batch for interpolation,
// and return alpha = remainder / fixedDt.

function stepState(w) {
  if (!w.stepState) {
    w.stepState = { fixedDt: 1 / 60, maxSteps: 4, acc: 0, alpha: 1, interpolate: false, prev: new Map() };
  }
  return w.stepState;
}

function snapshotWorldBodies(worldHandle, st) {
  st.prev.clear();
  for (const [bh, b] of state.bodies) {
    if (b.world !== worldHandle) continue;
    const bi = resolveBodyInterface(b); if (!bi) continue;
    const p = bi.GetPosition(b.bodyId), q = bi.GetRotation(b.bodyId);
    st.prev.set(bh, [p.GetX(), p.GetY(), p.GetZ(), q.GetX(), q.GetY(), q.GetZ(), q.GetW()]);
  }
}

export function stepFixed(h, frameDt, collisionSteps) {
  const w = state.worlds.get(h); if (!w) return 1;
  const st = stepState(w);
  const dt = Number.isFinite(frameDt) ? Math.min(Math.max(frameDt, 0), 0.25) : 0;
  st.acc += dt;
  let steps = Math.floor(st.acc / st.fixedDt);
  if (steps > st.maxSteps) {
    steps = st.maxSteps;
    st.acc = Math.min(st.acc, st.fixedDt * (steps + 1));
  }
  if (steps > 0) {
    const cs = Math.max(1, collisionSteps | 0);
    for (let i = 0; i < steps; i++) {
      if (st.interpolate && i + 1 === steps) snapshotWorldBodies(h, st);
      w.jolt.Step(st.fixedDt, cs);
    }
  }
  st.acc -= steps * st.fixedDt;
  st.alpha = Math.min(Math.max(st.acc / st.fixedDt, 0), 1);
  return st.alpha;
}

export function setFixedTimestep(h, hz, maxSteps) {
  const w = state.worlds.get(h); if (!w) return;
  const st = stepState(w);
  if (hz > 0 && Number.isFinite(hz)) st.fixedDt = 1 / hz;
  if (maxSteps > 0) st.maxSteps = maxSteps | 0;
}

export function setInterpolation(h, on) {
  const w = state.worlds.get(h); if (!w) return;
  const st = stepState(w);
  st.interpolate = !!on;
  if (!st.interpolate) { st.prev.clear(); st.alpha = 1; }
}

export function getStepAlpha(h) {
  const w = state.worlds.get(h);
  return w && w.stepState ? w.stepState.alpha : 1;
}

/** Interpolation blend for a body's getters, or null for raw state. */
function interpPrev(b, bh) {
  const w = state.worlds.get(b.world);
  const st = w && w.stepState;
  if (!st || !st.interpolate || st.alpha >= 1) return null;
  const prev = st.prev.get(bh);
  return prev ? { a: st.alpha, prev } : null;
}

export function setLayerCollides(h, a, b, collides) {
  const w = state.worlds.get(h); if (!w) return;
  // JoltSettings filter was already baked; runtime mutation isn't supported via
  // the JS bindings. This is a no-op for now; layer setup must happen before
  // createWorld in a future API tweak.
  void a; void b; void collides;
}
export function getLayerCollides(h, a, b) {
  void h; void a; void b;
  return 1;   // default permissive
}

export function bodyCount(h)       { const w = state.worlds.get(h); return w ? w.system.GetNumBodies() : 0; }
export function activeBodyCount(h) { const w = state.worlds.get(h); return w ? w.system.GetNumActiveBodies() : 0; }

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

function registerShape(shape) {
  if (!shape) return 0;
  const h = state.nextShape++;
  state.shapes.set(h, shape);
  return h;
}

export function shapeBox(hx, hy, hz, convexRadius) {
  if (warnUninit('shapeBox')) return 0;
  const settings = new JoltModule.BoxShapeSettings(vec3(hx, hy, hz), convexRadius);
  const result = settings.Create();
  return result.IsValid() ? registerShape(result.Get()) : 0;
}
export function shapeSphere(r) {
  if (warnUninit('shapeSphere')) return 0;
  const settings = new JoltModule.SphereShapeSettings(r);
  const result = settings.Create();
  return result.IsValid() ? registerShape(result.Get()) : 0;
}
export function shapeCapsule(h, r) {
  if (warnUninit('shapeCapsule')) return 0;
  const settings = new JoltModule.CapsuleShapeSettings(h, r);
  const result = settings.Create();
  return result.IsValid() ? registerShape(result.Get()) : 0;
}
export function shapeCylinder(h, r, cr) {
  if (warnUninit('shapeCylinder')) return 0;
  const settings = new JoltModule.CylinderShapeSettings(h, r, cr);
  const result = settings.Create();
  return result.IsValid() ? registerShape(result.Get()) : 0;
}
export function shapeScaled(base, sx, sy, sz) {
  if (warnUninit('shapeScaled')) return 0;
  const inner = state.shapes.get(base); if (!inner) return 0;
  const settings = new JoltModule.ScaledShapeSettings(inner, vec3(sx, sy, sz));
  const result = settings.Create();
  return result.IsValid() ? registerShape(result.Get()) : 0;
}
export function shapeOffsetCom(base, ox, oy, oz) {
  if (warnUninit('shapeOffsetCom')) return 0;
  const inner = state.shapes.get(base); if (!inner) return 0;
  const settings = new JoltModule.OffsetCenterOfMassShapeSettings(vec3(ox, oy, oz), inner);
  const result = settings.Create();
  return result.IsValid() ? registerShape(result.Get()) : 0;
}
export function shapeRelease(h) {
  const s = state.shapes.get(h); if (!s) return;
  state.shapes.delete(h);
  // JoltPhysics.js shapes are refcounted internally; letting GC handle it is fine.
}
export function shapeBounds(h, axis) {
  const s = state.shapes.get(h); if (!s) return 0;
  const box = s.GetLocalBounds();
  switch (axis | 0) {
    case 0: return box.mMin.GetX();
    case 1: return box.mMin.GetY();
    case 2: return box.mMin.GetZ();
    case 3: return box.mMax.GetX();
    case 4: return box.mMax.GetY();
    case 5: return box.mMax.GetZ();
    default: return 0;
  }
}
export function shapeVolume(h) {
  const s = state.shapes.get(h); if (!s) return 0;
  return s.GetVolume ? s.GetVolume() : 0;
}

// ---------------------------------------------------------------------------
// Bodies
// ---------------------------------------------------------------------------

function destroyBodyInternal(w, b) {
  if (w.bodyInterface.IsAdded(b.bodyId)) w.bodyInterface.RemoveBody(b.bodyId);
  w.bodyInterface.DestroyBody(b.bodyId);
}

function resolveBody(h) { return state.bodies.get(h); }
function resolveBodyInterface(b) {
  const w = state.worlds.get(b.world); return w ? w.bodyInterface : null;
}

export function bodyCreate(worldH, shapeH, motionType, px, py, pz, rx, ry, rz, rw, layer) {
  if (warnUninit('bodyCreate')) return 0;
  const w = state.worlds.get(worldH); if (!w) return 0;
  const s = state.shapes.get(shapeH); if (!s) return 0;
  const J = JoltModule;
  const settings = new J.BodyCreationSettings(
    s,
    rvec3(px, py, pz),
    quat(rx, ry, rz, rw),
    motionTypeFrom(motionType),
    (layer | 0),
  );
  const bodyId = w.bodyInterface.CreateAndAddBody(settings, J.EActivation_Activate);
  J.destroy(settings);
  const h = state.nextBody++;
  state.bodies.set(h, { world: worldH, bodyId });
  return h;
}

export function bodyDestroy(h) {
  const b = resolveBody(h); if (!b) return;
  const w = state.worlds.get(b.world); if (w) destroyBodyInternal(w, b);
  state.bodies.delete(h);
}

export function bodyActivate(h)   { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (bi) bi.ActivateBody(b.bodyId); }
export function bodyDeactivate(h) { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (bi) bi.DeactivateBody(b.bodyId); }
export function bodyIsActive(h)   { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); return bi && bi.IsActive(b.bodyId) ? 1 : 0; }
export function bodyIsValid(h)    { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); return bi && bi.IsAdded(b.bodyId)  ? 1 : 0; }

export function bodyGetPosition(h, axis) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return 0;
  const p = bi.GetPosition(b.bodyId);
  let x = p.GetX(), y = p.GetY(), z = p.GetZ();
  const ip = interpPrev(b, h);
  if (ip) {
    const [px, py, pz] = ip.prev;
    x = px + (x - px) * ip.a; y = py + (y - py) * ip.a; z = pz + (z - pz) * ip.a;
  }
  if (axis === 0) return x; if (axis === 1) return y; if (axis === 2) return z;
  return 0;
}
export function bodyGetRotation(h, axis) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return 0;
  const q = bi.GetRotation(b.bodyId);
  let x = q.GetX(), y = q.GetY(), z = q.GetZ(), w = q.GetW();
  const ip = interpPrev(b, h);
  if (ip) {
    // nlerp with shortest-path sign handling — matches the native getter
    let [, , , px, py, pz, pw] = ip.prev;
    const dot = px * x + py * y + pz * z + pw * w;
    const s = dot < 0 ? -1 : 1;
    px *= s; py *= s; pz *= s; pw *= s;
    const a = ip.a;
    x = px + (x - px) * a; y = py + (y - py) * a; z = pz + (z - pz) * a; w = pw + (w - pw) * a;
    const len = Math.hypot(x, y, z, w);
    if (len > 1e-6) { x /= len; y /= len; z /= len; w /= len; }
  }
  if (axis === 0) return x; if (axis === 1) return y;
  if (axis === 2) return z; if (axis === 3) return w;
  return 0;
}
export function bodySetPosition(h, x, y, z, activate) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return;
  bi.SetPosition(b.bodyId, rvec3(x, y, z), activationFrom(activate));
}
export function bodySetRotation(h, x, y, z, w, activate) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return;
  bi.SetRotation(b.bodyId, quat(x, y, z, w), activationFrom(activate));
}
export function bodySetTransform(h, px, py, pz, rx, ry, rz, rw, activate) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return;
  bi.SetPositionAndRotation(b.bodyId, rvec3(px, py, pz), quat(rx, ry, rz, rw), activationFrom(activate));
}
export function bodyMoveKinematic(h, px, py, pz, rx, ry, rz, rw, dt) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return;
  bi.MoveKinematic(b.bodyId, rvec3(px, py, pz), quat(rx, ry, rz, rw), dt);
}

export function bodyGetLinearVelocity(h, axis) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return 0;
  const v = bi.GetLinearVelocity(b.bodyId);
  if (axis === 0) return v.GetX(); if (axis === 1) return v.GetY(); if (axis === 2) return v.GetZ();
  return 0;
}
export function bodyGetAngularVelocity(h, axis) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return 0;
  const v = bi.GetAngularVelocity(b.bodyId);
  if (axis === 0) return v.GetX(); if (axis === 1) return v.GetY(); if (axis === 2) return v.GetZ();
  return 0;
}
export function bodyGetPointVelocity(h, px, py, pz, axis) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return 0;
  const v = bi.GetPointVelocity(b.bodyId, rvec3(px, py, pz));
  if (axis === 0) return v.GetX(); if (axis === 1) return v.GetY(); if (axis === 2) return v.GetZ();
  return 0;
}
export function bodySetLinearVelocity(h, x, y, z) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return;
  bi.SetLinearVelocity(b.bodyId, vec3(x, y, z));
}
export function bodySetAngularVelocity(h, x, y, z) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return;
  bi.SetAngularVelocity(b.bodyId, vec3(x, y, z));
}

export function bodyAddForce(h, x, y, z)          { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (bi) bi.AddForce(b.bodyId, vec3(x, y, z)); }
export function bodyAddImpulse(h, x, y, z)        { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (bi) bi.AddImpulse(b.bodyId, vec3(x, y, z)); }
export function bodyAddTorque(h, x, y, z)         { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (bi) bi.AddTorque(b.bodyId, vec3(x, y, z)); }
export function bodyAddAngularImpulse(h, x, y, z) { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (bi) bi.AddAngularImpulse(b.bodyId, vec3(x, y, z)); }
export function bodyAddForceAt(h, fx, fy, fz, px, py, pz)       { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (bi) bi.AddForceAt   ? bi.AddForceAt(b.bodyId, vec3(fx, fy, fz), rvec3(px, py, pz))   : bi.AddForce(b.bodyId,   vec3(fx, fy, fz), rvec3(px, py, pz)); }
export function bodyAddImpulseAt(h, ix, iy, iz, px, py, pz)     { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (bi) bi.AddImpulseAt ? bi.AddImpulseAt(b.bodyId, vec3(ix, iy, iz), rvec3(px, py, pz)) : bi.AddImpulse(b.bodyId, vec3(ix, iy, iz), rvec3(px, py, pz)); }

export function bodySetFriction(h, v)         { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (bi) bi.SetFriction(b.bodyId, v); }
export function bodySetRestitution(h, v)      { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (bi) bi.SetRestitution(b.bodyId, v); }
export function bodySetLinearDamping(h, v) {
  const b = resolveBody(h); if (!b) return;
  const w = state.worlds.get(b.world); if (!w) return;
  const J = JoltModule;
  const lock = new J.BodyLockWrite(w.system.GetBodyLockInterface(), b.bodyId);
  if (lock.SucceededAndIsInBroadPhase()) {
    const mp = lock.GetBody().GetMotionPropertiesUnchecked();
    if (mp) mp.SetLinearDamping(v);
  }
  lock.ReleaseLock();
}
export function bodySetAngularDamping(h, v) {
  const b = resolveBody(h); if (!b) return;
  const w = state.worlds.get(b.world); if (!w) return;
  const J = JoltModule;
  const lock = new J.BodyLockWrite(w.system.GetBodyLockInterface(), b.bodyId);
  if (lock.SucceededAndIsInBroadPhase()) {
    const mp = lock.GetBody().GetMotionPropertiesUnchecked();
    if (mp) mp.SetAngularDamping(v);
  }
  lock.ReleaseLock();
}
export function bodySetGravityFactor(h, v)    { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (bi && bi.SetGravityFactor) bi.SetGravityFactor(b.bodyId, v); }
export function bodySetCcd(h, enabled) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return;
  const J = JoltModule;
  bi.SetMotionQuality(b.bodyId, enabled ? J.EMotionQuality_LinearCast : J.EMotionQuality_Discrete);
}
export function bodySetMotionType(h, t, activate) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return;
  bi.SetMotionType(b.bodyId, motionTypeFrom(t), activationFrom(activate));
}
export function bodySetObjectLayer(h, layer) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (bi && bi.SetObjectLayer) bi.SetObjectLayer(b.bodyId, layer | 0);
}
export function bodySetIsSensor(h, enabled) {
  const b = resolveBody(h); if (!b) return;
  const w = state.worlds.get(b.world); if (!w) return;
  const J = JoltModule;
  const lock = new J.BodyLockWrite(w.system.GetBodyLockInterface(), b.bodyId);
  if (lock.SucceededAndIsInBroadPhase()) lock.GetBody().SetIsSensor(enabled !== 0);
  lock.ReleaseLock();
}
export function bodySetAllowSleeping(h, enabled) {
  const b = resolveBody(h); if (!b) return;
  const w = state.worlds.get(b.world); if (!w) return;
  const J = JoltModule;
  const lock = new J.BodyLockWrite(w.system.GetBodyLockInterface(), b.bodyId);
  if (lock.SucceededAndIsInBroadPhase()) lock.GetBody().SetAllowSleeping(enabled !== 0);
  lock.ReleaseLock();
}
export function bodySetShape(h, shapeH, updateMass, activate) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return;
  const s = state.shapes.get(shapeH); if (!s) return;
  bi.SetShape(b.bodyId, s, updateMass !== 0, activationFrom(activate));
}
export function bodyLockRotationAxes(h, x, y, z) {
  void h; void x; void y; void z;  // stub — requires recreating the body on JS side
}
export function bodyLockTranslationAxes(h, x, y, z) {
  void h; void x; void y; void z;
}

export function bodyGetMass(h) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi) return 0;
  const inv = bi.GetInverseMass ? bi.GetInverseMass(b.bodyId) : 0;
  return inv > 0 ? 1 / inv : 0;
}
export function bodyGetFriction(h)     { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); return bi ? bi.GetFriction(b.bodyId)    : 0; }
export function bodyGetRestitution(h)  { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); return bi ? bi.GetRestitution(b.bodyId) : 0; }
export function bodyGetObjectLayer(h)  { const b = resolveBody(h); const bi = b && resolveBodyInterface(b); return bi && bi.GetObjectLayer ? bi.GetObjectLayer(b.bodyId) : 0; }
export function bodySetUserData(h, lo, hi) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi || !bi.SetUserData) return;
  // Pack lo (uint32) + hi (uint32) into uint64. Whether a u64 crosses
  // JoltPhysics.js as BigInt or Number depends on how the build was linked
  // (-sWASM_BIGINT or not) — the CDN 1.0.0 build wants Number and THROWS on
  // BigInt, so probe once and remember. Number packing is exact here: call
  // sites keep hi = 0, far below 2^53. Most call sites use just `lo`.
  try {
    bi.SetUserData(b.bodyId, BigInt(lo >>> 0) | (BigInt(hi >>> 0) << 32n));
  } catch {
    bi.SetUserData(b.bodyId, (hi >>> 0) * 0x100000000 + (lo >>> 0));
  }
}
export function bodyGetUserData(h, part) {
  const b = resolveBody(h); const bi = b && resolveBodyInterface(b); if (!bi || !bi.GetUserData) return 0;
  const u = bi.GetUserData(b.bodyId);
  if (typeof u === 'bigint') {
    return part === 1 ? Number(u >> 32n) : Number(u & 0xFFFFFFFFn);
  }
  return Number(u);
}

// ---------------------------------------------------------------------------
// Queries  (Tier 1 — closest-hit only; all-hits/overlap = stubs for Tier 2)
// ---------------------------------------------------------------------------

export function raycast(worldH, ox, oy, oz, dx, dy, dz, maxDist, layerMask) {
  state.rayHits.length = 0;
  const w = state.worlds.get(worldH); if (!w) return 0;
  const J = JoltModule;
  const dir = vec3(dx, dy, dz);
  if (dir.LengthSq() === 0) return 0;
  const scaled = dir.Mul ? dir.Mul(maxDist / Math.sqrt(dir.LengthSq())) : dir;
  const ray = new J.RRayCast(rvec3(ox, oy, oz), scaled);
  // Layer 1 (MOVING) collides with every layer in the pair table, so its
  // Default filters are the pass-all query. Two npm-build gotchas here:
  // the abstract BroadPhaseLayerFilter/ObjectLayerFilter bases trap with a
  // null vtable, and the closest-hit `CastRay(ray, RayCastResult, ...)`
  // overload does not exist — only the collector form from the official
  // examples is bound.
  const qf = queryFilters(w, worldH, 1);
  const rcSettings = new J.RayCastSettings();
  const collector = new J.CastRayClosestHitCollisionCollector();
  w.system.GetNarrowPhaseQuery().CastRay(ray, rcSettings, collector, qf.bp, qf.obj, qf.body, qf.shape);
  const hit = collector.HadHit();
  void layerMask;  // layer filtering is handled by ObjectLayerFilter; Tier 2 adds mask support
  if (hit) {
    // Resolve body handle from BodyID.
    const result = collector.mHit;
    const bodyId = result.mBodyID;
    let bodyHandle = 0;
    for (const [h, b] of state.bodies) { if (b.world === worldH && b.bodyId.GetIndexAndSequenceNumber() === bodyId.GetIndexAndSequenceNumber()) { bodyHandle = h; break; } }
    const fraction = result.mFraction;
    state.rayHits.push({
      body: bodyHandle,
      point: [ox + (dx * maxDist) * fraction, oy + (dy * maxDist) * fraction, oz + (dz * maxDist) * fraction],
      normal: [0, 1, 0],   // TODO: world-space surface normal — needs a BodyLock
      fraction,
      subShapeId: 0,
    });
    J.destroy(collector); J.destroy(rcSettings); J.destroy(ray);
    return 1;
  }
  J.destroy(collector); J.destroy(rcSettings); J.destroy(ray);
  return 0;
}
export function raycastAll(worldH, ox, oy, oz, dx, dy, dz, maxDist, layerMask, maxHits) {
  state.rayHits.length = 0;
  void layerMask;
  const w = state.worlds.get(worldH); if (!w) return 0;
  const J = JoltModule;
  const dir = vec3(dx, dy, dz);
  if (dir.LengthSq() === 0) return 0;
  const scaled = vec3(dx * maxDist, dy * maxDist, dz * maxDist);
  const ray = new J.RRayCast(rvec3(ox, oy, oz), scaled);
  const settings = new J.RayCastSettings();
  const collector = new J.CastRayAllHitCollisionCollectorJS();
  const qf = queryFilters(w, worldH, 1);
  w.system.GetNarrowPhaseQuery().CastRay(
    ray, settings, collector,
    qf.bp, qf.obj, qf.body, qf.shape,
  );
  collector.Sort();
  const count = Math.min(maxHits | 0, collector.mHits.size());
  for (let i = 0; i < count; i++) {
    const hit = collector.mHits.at(i);
    const frac = hit.mFraction;
    let bodyHandle = 0;
    for (const [h, b] of state.bodies) {
      if (b.world === worldH && b.bodyId.GetIndexAndSequenceNumber() === hit.mBodyID.GetIndexAndSequenceNumber()) {
        bodyHandle = h; break;
      }
    }
    state.rayHits.push({
      body: bodyHandle,
      point: [ox + dx * maxDist * frac, oy + dy * maxDist * frac, oz + dz * maxDist * frac],
      normal: [0, 1, 0],   // world-space normal requires a BodyLock; skipped for now
      fraction: frac,
      subShapeId: hit.mSubShapeID2 ? hit.mSubShapeID2.GetValue() : 0,
    });
  }
  return state.rayHits.length;
}
export function rayHitCount()     { return state.rayHits.length; }
export function rayHitBody(i)     { return state.rayHits[i|0]?.body ?? 0; }
export function rayHitAxis(i, f)  {
  const h = state.rayHits[i|0]; if (!h) return 0;
  return f < 3 ? h.point[f|0] : h.normal[(f|0)-3];
}
export function rayHitFraction(i) { return state.rayHits[i|0]?.fraction ?? 0; }
export function rayHitSubShape(i) { return state.rayHits[i|0]?.subShapeId ?? 0; }

function collectOverlapBodies(w, shapeCollider, maxResults) {
  // JoltPhysics.js exposes broadphase query collectors via *_JS suffixed classes.
  const J = JoltModule;
  const collector = new J.CollideShapeBodyCollectorJS();
  shapeCollider(collector);
  const count = Math.min(maxResults | 0, collector.mHits.size());
  const out = [];
  for (let i = 0; i < count; i++) {
    const bodyId = collector.mHits.at(i);
    for (const [h, b] of state.bodies) {
      if (b.world === w && b.bodyId.GetIndexAndSequenceNumber() === bodyId.GetIndexAndSequenceNumber()) {
        out.push(h); break;
      }
    }
  }
  return out;
}

export function overlapSphere(worldH, cx, cy, cz, r, layerMask, maxResults) {
  state.overlapBodies.length = 0;
  void layerMask;
  const w = state.worlds.get(worldH); if (!w) return 0;
  const J = JoltModule;
  state.overlapBodies = collectOverlapBodies(worldH, collector => {
    w.system.GetBroadPhaseQuery().CollideSphere(
      vec3(cx, cy, cz), r, collector,
      queryFilters(w, worldH, 1).bp, queryFilters(w, worldH, 1).obj,
    );
  }, maxResults);
  return state.overlapBodies.length;
}
export function overlapPoint(worldH, px, py, pz, layerMask, maxResults) {
  state.overlapBodies.length = 0;
  void layerMask;
  const w = state.worlds.get(worldH); if (!w) return 0;
  const J = JoltModule;
  state.overlapBodies = collectOverlapBodies(worldH, collector => {
    w.system.GetBroadPhaseQuery().CollidePoint(
      vec3(px, py, pz), collector,
      queryFilters(w, worldH, 1).bp, queryFilters(w, worldH, 1).obj,
    );
  }, maxResults);
  return state.overlapBodies.length;
}
export function overlapBox(worldH, px, py, pz, rx, ry, rz, rw, hx, hy, hz, layerMask, maxResults) {
  state.overlapBodies.length = 0;
  void layerMask;
  const w = state.worlds.get(worldH); if (!w) return 0;
  const J = JoltModule;
  // Conservative AABB around the rotated box (matches native C++ shim behavior).
  const q = quat(rx, ry, rz, rw);
  const rotMat = q.GetRotation ? q.GetRotation() : null;
  // Jolt.js may not expose Quat.GetRotation; fall back to axis-aligned extents.
  const ex = Math.abs(hx) + Math.abs(hy) * 0.001 + Math.abs(hz) * 0.001;
  const ey = Math.abs(hy) + Math.abs(hx) * 0.001 + Math.abs(hz) * 0.001;
  const ez = Math.abs(hz) + Math.abs(hx) * 0.001 + Math.abs(hy) * 0.001;
  const box = new J.AABox(vec3(px - ex, py - ey, pz - ez), vec3(px + ex, py + ey, pz + ez));
  state.overlapBodies = collectOverlapBodies(worldH, collector => {
    w.system.GetBroadPhaseQuery().CollideAABox(
      box, collector,
      queryFilters(w, worldH, 1).bp, queryFilters(w, worldH, 1).obj,
    );
  }, maxResults);
  void rotMat;
  return state.overlapBodies.length;
}
export function overlapBody(i) { return state.overlapBodies[i|0] ?? 0; }

// ---------------------------------------------------------------------------
// Constraints
// ---------------------------------------------------------------------------

function resolveConstraintBodies(bodyAh, bodyBh) {
  const J = JoltModule;
  const ba = state.bodies.get(bodyAh); if (!ba) return null;
  const w = state.worlds.get(ba.world); if (!w) return null;
  // Retrieve the actual Body refs via the lock interface.
  const lockA = new J.BodyLockRead(w.system.GetBodyLockInterface(), ba.bodyId);
  if (!lockA.SucceededAndIsInBroadPhase()) { lockA.ReleaseLock(); return null; }
  const body1 = lockA.GetBody();
  let body2 = J.Body.prototype.sFixedToWorld ?? null;
  let lockB = null;
  if (bodyBh !== 0) {
    const bb = state.bodies.get(bodyBh); if (!bb) { lockA.ReleaseLock(); return null; }
    lockB = new J.BodyLockRead(w.system.GetBodyLockInterface(), bb.bodyId);
    if (!lockB.SucceededAndIsInBroadPhase()) { lockA.ReleaseLock(); lockB.ReleaseLock(); return null; }
    body2 = lockB.GetBody();
  }
  return {
    w, body1, body2,
    release: () => { lockA.ReleaseLock(); if (lockB) lockB.ReleaseLock(); },
  };
}

function registerConstraint(worldId, constraint) {
  state.worlds.get(worldId).system.AddConstraint(constraint);
  const h = state.nextConstraint++;
  state.constraints.set(h, { world: worldId, constraint });
  return h;
}

export function constraintFixed(bodyA, bodyB, ax, ay, az, bx, by, bz, worldSpace) {
  if (warnUninit('constraintFixed')) return 0;
  const ctx = resolveConstraintBodies(bodyA, bodyB); if (!ctx) return 0;
  const J = JoltModule;
  const settings = new J.FixedConstraintSettings();
  settings.mSpace = worldSpace ? J.EConstraintSpace_WorldSpace : J.EConstraintSpace_LocalToBodyCOM;
  settings.mPoint1 = rvec3(ax, ay, az);
  settings.mPoint2 = rvec3(bx, by, bz);
  settings.mAutoDetectPoint = false;
  const c = settings.Create(ctx.body1, ctx.body2);
  ctx.release();
  return registerConstraint(ctx.w === state.worlds.get(state.bodies.get(bodyA).world) ? state.bodies.get(bodyA).world : 0, c);
}

export function constraintPoint(bodyA, bodyB, ax, ay, az, bx, by, bz, worldSpace) {
  if (warnUninit('constraintPoint')) return 0;
  const ctx = resolveConstraintBodies(bodyA, bodyB); if (!ctx) return 0;
  const J = JoltModule;
  const settings = new J.PointConstraintSettings();
  settings.mSpace = worldSpace ? J.EConstraintSpace_WorldSpace : J.EConstraintSpace_LocalToBodyCOM;
  settings.mPoint1 = rvec3(ax, ay, az);
  settings.mPoint2 = rvec3(bx, by, bz);
  const c = settings.Create(ctx.body1, ctx.body2);
  ctx.release();
  return registerConstraint(state.bodies.get(bodyA).world, c);
}

export function constraintHinge(bodyA, bodyB, ax, ay, az, bx, by, bz, axX, axY, axZ, lmin, lmax, worldSpace) {
  if (warnUninit('constraintHinge')) return 0;
  const ctx = resolveConstraintBodies(bodyA, bodyB); if (!ctx) return 0;
  const J = JoltModule;
  const settings = new J.HingeConstraintSettings();
  settings.mSpace = worldSpace ? J.EConstraintSpace_WorldSpace : J.EConstraintSpace_LocalToBodyCOM;
  settings.mPoint1 = rvec3(ax, ay, az);
  settings.mPoint2 = rvec3(bx, by, bz);
  const axis = vec3(axX, axY, axZ);
  settings.mHingeAxis1 = axis;
  settings.mHingeAxis2 = axis;
  const perp = axis.GetNormalizedPerpendicular ? axis.GetNormalizedPerpendicular() : vec3(1, 0, 0);
  settings.mNormalAxis1 = perp;
  settings.mNormalAxis2 = perp;
  if (lmin < lmax) { settings.mLimitsMin = lmin; settings.mLimitsMax = lmax; }
  const c = settings.Create(ctx.body1, ctx.body2);
  ctx.release();
  return registerConstraint(state.bodies.get(bodyA).world, c);
}

export function constraintSlider(bodyA, bodyB, ax, ay, az, bx, by, bz, axX, axY, axZ, lmin, lmax, worldSpace) {
  if (warnUninit('constraintSlider')) return 0;
  const ctx = resolveConstraintBodies(bodyA, bodyB); if (!ctx) return 0;
  const J = JoltModule;
  const settings = new J.SliderConstraintSettings();
  settings.mSpace = worldSpace ? J.EConstraintSpace_WorldSpace : J.EConstraintSpace_LocalToBodyCOM;
  settings.mPoint1 = rvec3(ax, ay, az);
  settings.mPoint2 = rvec3(bx, by, bz);
  // Normalize axis.
  const len = Math.hypot(axX, axY, axZ) || 1;
  const axis = vec3(axX / len, axY / len, axZ / len);
  settings.mSliderAxis1 = axis;
  settings.mSliderAxis2 = axis;
  const perp = axis.GetNormalizedPerpendicular ? axis.GetNormalizedPerpendicular() : vec3(1, 0, 0);
  settings.mNormalAxis1 = perp;
  settings.mNormalAxis2 = perp;
  if (lmin < lmax) { settings.mLimitsMin = lmin; settings.mLimitsMax = lmax; }
  const c = settings.Create(ctx.body1, ctx.body2);
  ctx.release();
  return registerConstraint(state.bodies.get(bodyA).world, c);
}

export function constraintDistance(bodyA, bodyB, ax, ay, az, bx, by, bz, minD, maxD, worldSpace) {
  if (warnUninit('constraintDistance')) return 0;
  const ctx = resolveConstraintBodies(bodyA, bodyB); if (!ctx) return 0;
  const J = JoltModule;
  const settings = new J.DistanceConstraintSettings();
  settings.mSpace = worldSpace ? J.EConstraintSpace_WorldSpace : J.EConstraintSpace_LocalToBodyCOM;
  settings.mPoint1 = rvec3(ax, ay, az);
  settings.mPoint2 = rvec3(bx, by, bz);
  settings.mMinDistance = minD;
  settings.mMaxDistance = maxD;
  const c = settings.Create(ctx.body1, ctx.body2);
  ctx.release();
  return registerConstraint(state.bodies.get(bodyA).world, c);
}

// EN-063 (web ragdolls): six-DOF with all three translation axes locked and
// per-axis rotation limits — the articulation the engine's ragdoll builder
// uses. Mirrors the native C++ shim's convention: a rotation axis whose
// min >= max is LOCKED, otherwise limited to [min, max]. Wrapped in
// try/catch so a JoltPhysics.js build without the SixDOF surface degrades
// to "no ragdolls" instead of killing the frame.
export function constraintSixDofLockedTranslation(
  bodyA, bodyB, ax, ay, az, bx, by, bz,
  rxMin, rxMax, ryMin, ryMax, rzMin, rzMax, worldSpace,
) {
  if (warnUninit('constraintSixDofLockedTranslation')) return 0;
  const ctx = resolveConstraintBodies(bodyA, bodyB); if (!ctx) return 0;
  const J = JoltModule;
  try {
    const settings = new J.SixDOFConstraintSettings();
    settings.mSpace = worldSpace ? J.EConstraintSpace_WorldSpace : J.EConstraintSpace_LocalToBodyCOM;
    settings.mPosition1 = rvec3(ax, ay, az);
    settings.mPosition2 = rvec3(bx, by, bz);
    const AX = [
      [J.SixDOFConstraintSettings_EAxis_TranslationX, 1, -1],
      [J.SixDOFConstraintSettings_EAxis_TranslationY, 1, -1],
      [J.SixDOFConstraintSettings_EAxis_TranslationZ, 1, -1],
      [J.SixDOFConstraintSettings_EAxis_RotationX, rxMin, rxMax],
      [J.SixDOFConstraintSettings_EAxis_RotationY, ryMin, ryMax],
      [J.SixDOFConstraintSettings_EAxis_RotationZ, rzMin, rzMax],
    ];
    for (const [axis, lo, hi] of AX) {
      if (lo >= hi) settings.MakeFixedAxis(axis);
      else settings.SetLimitedAxis(axis, lo, hi);
    }
    const c = settings.Create(ctx.body1, ctx.body2);
    ctx.release();
    return registerConstraint(state.bodies.get(bodyA).world, c);
  } catch (e) {
    ctx.release();
    console.warn('[jolt_bridge] SixDOF constraint unavailable:', e);
    return 0;
  }
}

export function constraintDestroy(h) {
  const c = state.constraints.get(h); if (!c) return;
  const w = state.worlds.get(c.world); if (w) w.system.RemoveConstraint(c.constraint);
  state.constraints.delete(h);
}
export function constraintSetEnabled(h, enabled) {
  const c = state.constraints.get(h); if (!c) return;
  if (c.constraint.SetEnabled) c.constraint.SetEnabled(enabled !== 0);
}

// ---------------------------------------------------------------------------
// Contact events  (Tier 2 — listener wiring not yet ported to JS)
// ---------------------------------------------------------------------------

export function contactCount() { return state.contacts.length; }
export function contactField(i, field) {
  const c = state.contacts[i|0]; if (!c) return 0;
  switch (field|0) {
    case 0:  return c.event;
    case 1:  return c.bodyA;
    case 2:  return c.bodyB;
    case 3:  return c.pointA[0]; case 4: return c.pointA[1]; case 5: return c.pointA[2];
    case 6:  return c.pointB[0]; case 7: return c.pointB[1]; case 8: return c.pointB[2];
    case 9:  return c.normal[0]; case 10: return c.normal[1]; case 11: return c.normal[2];
    case 12: return c.penetrationDepth;
    case 13: return c.combinedFriction;
    case 14: return c.combinedRestitution;
    default: return 0;
  }
}
export function clearContacts(worldH) { void worldH; state.contacts.length = 0; }

// ============================================================================
// Phase 5 additions — complex shapes + character controller
// ============================================================================

// --- Scratch streams ---
export function scratchReset()       { state.scratchF32.length = 0; state.scratchU32.length = 0; }
export function scratchPushF32(v)    { state.scratchF32.push(v); }
export function scratchPushU32(v)    { state.scratchU32.push(v >>> 0); }

// --- Complex shape factories ---

export function shapeConvexHull(numPoints, convexRadius) {
  if (warnUninit('shapeConvexHull')) return 0;
  const J = JoltModule;
  if (state.scratchF32.length < numPoints * 3 || numPoints < 3) return 0;
  const settings = new J.ConvexHullShapeSettings();
  // JoltPhysics.js exposes mPoints as a JS-friendly Array<Vec3>.
  for (let i = 0; i < numPoints; i++) {
    const p = vec3(state.scratchF32[i*3], state.scratchF32[i*3+1], state.scratchF32[i*3+2]);
    settings.mPoints.push_back(p);
  }
  settings.mConvexRadius = convexRadius;
  const result = settings.Create();
  return result.IsValid() ? registerShape(result.Get()) : 0;
}

export function shapeMesh(vertexCount, triangleCount) {
  if (warnUninit('shapeMesh')) return 0;
  const J = JoltModule;
  if (state.scratchF32.length < vertexCount * 3 || state.scratchU32.length < triangleCount * 3) return 0;
  if (vertexCount === 0 || triangleCount === 0) return 0;
  const settings = new J.MeshShapeSettings();
  for (let i = 0; i < vertexCount; i++) {
    settings.mTriangleVertices.push_back(new J.Float3(
      state.scratchF32[i*3], state.scratchF32[i*3+1], state.scratchF32[i*3+2]
    ));
  }
  for (let t = 0; t < triangleCount; t++) {
    const idx = new J.IndexedTriangle(
      state.scratchU32[t*3], state.scratchU32[t*3+1], state.scratchU32[t*3+2], 0
    );
    settings.mIndexedTriangles.push_back(idx);
  }
  settings.Sanitize();
  const result = settings.Create();
  return result.IsValid() ? registerShape(result.Get()) : 0;
}

export function shapeHeightfield(sampleCount, ox, oy, oz, sx, sy, sz, blockSize) {
  if (warnUninit('shapeHeightfield')) return 0;
  const J = JoltModule;
  const need = sampleCount * sampleCount;
  if (state.scratchF32.length < need || sampleCount < 2) return 0;
  // The convenience constructor taking a JS Float32Array does not exist in
  // the npm/CDN JoltPhysics.js builds (Create() returns invalid — the
  // samples never reach the shape). Canonical pattern from the official
  // examples: default-construct the settings, size mHeightSamples, and
  // write the floats straight into the emscripten heap.
  const settings = new J.HeightFieldShapeSettings();
  settings.mOffset = vec3(ox, oy, oz);
  settings.mScale = vec3(sx, sy, sz);
  settings.mSampleCount = sampleCount;
  settings.mBlockSize = blockSize || 4;
  settings.mHeightSamples.resize(need);
  const heap = new Float32Array(
    J.HEAPF32.buffer,
    J.getPointer(settings.mHeightSamples.data()),
    need,
  );
  for (let i = 0; i < need; i++) heap[i] = state.scratchF32[i];
  const result = settings.Create();
  return result.IsValid() ? registerShape(result.Get()) : 0;
}

// --- Compound builder ---

export function compoundBegin() { state.compoundChildren.length = 0; }
export function compoundAddChild(shape, px, py, pz, rx, ry, rz, rw) {
  const s = state.shapes.get(shape); if (!s) return;
  state.compoundChildren.push({ shape: s, px, py, pz, rx, ry, rz, rw });
}
export function compoundEnd() {
  if (warnUninit('compoundEnd')) return 0;
  const J = JoltModule;
  if (state.compoundChildren.length === 0) return 0;
  const settings = new J.StaticCompoundShapeSettings();
  for (const c of state.compoundChildren) {
    settings.AddShape(vec3(c.px, c.py, c.pz), quat(c.rx, c.ry, c.rz, c.rw), c.shape, 0);
  }
  state.compoundChildren.length = 0;
  const result = settings.Create();
  return result.IsValid() ? registerShape(result.Get()) : 0;
}

// --- Character controller (CharacterVirtual) ---

// Query-filter compatibility + caching. Two JoltPhysics.js API generations
// exist: some builds expose `system.GetDefaultBroadPhaseLayerFilter(layer)`,
// the npm/CDN 1.0.0 build only has the constructor forms
// (`new Jolt.DefaultBroadPhaseLayerFilter(system.GetObjectVsBroadPhaseLayerFilter(), layer)`).
// Cached per (world, layer) — the character path runs every frame, and
// constructing emscripten objects per call is a real leak.
const queryFilterCache = new Map(); // "worldH:layer" → { bp, obj, body, shape }
function queryFilters(w, worldH, layer) {
  const key = worldH + ':' + (layer | 0);
  let f = queryFilterCache.get(key);
  if (f) return f;
  const J = JoltModule;
  let bp, obj;
  if (typeof w.system.GetDefaultBroadPhaseLayerFilter === 'function') {
    bp = w.system.GetDefaultBroadPhaseLayerFilter(layer);
    obj = w.system.GetDefaultLayerFilter(layer);
  } else {
    // npm/CDN builds: the canonical form from the JoltPhysics.js examples —
    // the filter getters live on the JoltInterface, and constructing the
    // Default* filters from the raw tables instead trips a null virtual
    // inside ExtendedUpdate.
    bp = new J.DefaultBroadPhaseLayerFilter(w.jolt.GetObjectVsBroadPhaseLayerFilter(), layer);
    obj = new J.DefaultObjectLayerFilter(w.jolt.GetObjectLayerPairFilter(), layer);
  }
  f = { bp, obj, body: new J.BodyFilter(), shape: new J.ShapeFilter() };
  queryFilterCache.set(key, f);
  return f;
}

export function characterCreate(
  worldH, shapeH,
  upX, upY, upZ,
  maxSlopeAngle, characterPadding,
  penetrationRecoverySpeed, predictiveContactDistance,
  maxStrength, mass, objectLayer,
  px, py, pz, rx, ry, rz, rw,
) {
  if (warnUninit('characterCreate')) return 0;
  const w = state.worlds.get(worldH); if (!w) return 0;
  const shape = state.shapes.get(shapeH); if (!shape) return 0;
  const J = JoltModule;
  const settings = new J.CharacterVirtualSettings();
  settings.mShape = shape;
  settings.mUp = vec3(upX, upY, upZ);
  settings.mMaxSlopeAngle = maxSlopeAngle;
  settings.mCharacterPadding = characterPadding;
  settings.mPenetrationRecoverySpeed = penetrationRecoverySpeed;
  settings.mPredictiveContactDistance = predictiveContactDistance;
  settings.mMaxStrength = maxStrength;
  settings.mMass = mass;
  // Two constructor generations exist. -sWASM_BIGINT builds take
  // (settings, pos, rot, userData: u64-as-BigInt, system); the npm/CDN
  // builds bind FOUR args (settings, pos, rot, system). Crucially, do NOT
  // fall back to a 5-arg call with a Number userData: emscripten dispatches
  // by declared arity, so on a 4-arg build the `0` lands in the SYSTEM slot
  // and the character is constructed around a null PhysicsSystem — nothing
  // fails until ExtendedUpdate dies with "null function". The BigInt
  // TypeError is the reliable discriminator between the generations.
  let character;
  try {
    character = new J.CharacterVirtual(
      settings, rvec3(px, py, pz), quat(rx, ry, rz, rw), 0n, w.system
    );
  } catch {
    character = new J.CharacterVirtual(
      settings, rvec3(px, py, pz), quat(rx, ry, rz, rw), w.system
    );
  }
  const h = state.nextCharacter++;
  state.characters.set(h, { world: worldH, character, layer: objectLayer | 0 });
  return h;
}

export function characterDestroy(h) {
  const e = state.characters.get(h); if (!e) return;
  JoltModule.destroy(e.character);
  state.characters.delete(h);
}

export function characterUpdate(h, dt, gx, gy, gz) {
  const e = state.characters.get(h); if (!e) return;
  const w = state.worlds.get(e.world); if (!w) return;
  const J = JoltModule;
  // Integrate gravity into velocity (match native C shim behaviour).
  const v = e.character.GetLinearVelocity();
  const newV = vec3(v.GetX() + gx * dt, v.GetY() + gy * dt, v.GetZ() + gz * dt);
  e.character.SetLinearVelocity(newV);
  const settings = new J.ExtendedUpdateSettings();
  const f = queryFilters(w, e.world, e.layer);
  e.character.ExtendedUpdate(
    dt, vec3(gx, gy, gz),
    settings,
    f.bp, f.obj, f.body, f.shape,
    w.jolt.GetTempAllocator()
  );
  J.destroy(settings);
}

export function characterGetPosition(h, axis) {
  const e = state.characters.get(h); if (!e) return 0;
  const p = e.character.GetPosition();
  if (axis === 0) return p.GetX(); if (axis === 1) return p.GetY(); if (axis === 2) return p.GetZ();
  return 0;
}
export function characterGetRotation(h, axis) {
  const e = state.characters.get(h); if (!e) return axis === 3 ? 1 : 0;
  const q = e.character.GetRotation();
  if (axis === 0) return q.GetX(); if (axis === 1) return q.GetY();
  if (axis === 2) return q.GetZ(); if (axis === 3) return q.GetW();
  return 0;
}
export function characterSetPosition(h, x, y, z) {
  const e = state.characters.get(h); if (!e) return;
  e.character.SetPosition(rvec3(x, y, z));
}
export function characterSetRotation(h, x, y, z, w) {
  const e = state.characters.get(h); if (!e) return;
  e.character.SetRotation(quat(x, y, z, w));
}
export function characterGetLinearVelocity(h, axis) {
  const e = state.characters.get(h); if (!e) return 0;
  const v = e.character.GetLinearVelocity();
  if (axis === 0) return v.GetX(); if (axis === 1) return v.GetY(); if (axis === 2) return v.GetZ();
  return 0;
}
export function characterSetLinearVelocity(h, x, y, z) {
  const e = state.characters.get(h); if (!e) return;
  e.character.SetLinearVelocity(vec3(x, y, z));
}
export function characterGetGroundState(h) {
  const e = state.characters.get(h); if (!e) return 3;
  const J = JoltModule;
  const s = e.character.GetGroundState();
  // Map Jolt enum → bj_ground_state.
  if (s === J.EGroundState_OnGround)     return 0;
  if (s === J.EGroundState_OnSteepGround) return 1;
  if (s === J.EGroundState_NotSupported) return 2;
  return 3;
}
export function characterGetGroundNormal(h, axis) {
  const e = state.characters.get(h); if (!e) return axis === 1 ? 1 : 0;
  const n = e.character.GetGroundNormal();
  if (axis === 0) return n.GetX(); if (axis === 1) return n.GetY(); if (axis === 2) return n.GetZ();
  return 0;
}
export function characterGetGroundPosition(h, axis) {
  const e = state.characters.get(h); if (!e) return 0;
  const p = e.character.GetGroundPosition();
  if (axis === 0) return p.GetX(); if (axis === 1) return p.GetY(); if (axis === 2) return p.GetZ();
  return 0;
}
export function characterGetGroundBody(h) {
  const e = state.characters.get(h); if (!e) return 0;
  const bodyId = e.character.GetGroundBodyID();
  for (const [bh, b] of state.bodies) {
    if (b.world === e.world && b.bodyId.GetIndexAndSequenceNumber() === bodyId.GetIndexAndSequenceNumber()) {
      return bh;
    }
  }
  return 0;
}
export function characterSetShape(h, shapeH) {
  const e = state.characters.get(h); if (!e) return;
  const w = state.worlds.get(e.world); if (!w) return;
  const s = state.shapes.get(shapeH); if (!s) return;
  const J = JoltModule;
  const f = queryFilters(w, e.world, e.layer);
  e.character.SetShape(
    s, Number.MAX_VALUE,
    f.bp, f.obj, f.body, f.shape,
    w.jolt.GetTempAllocator()
  );
}

// ---------------------------------------------------------------------------
// Soft bodies (Tier 2 — cloth / rope / jelly)
// ---------------------------------------------------------------------------

export function softBodyCreate(
  worldH, vertexCount, triangleCount,
  px, py, pz, rx, ry, rz, rw,
  objectLayer, edgeCompliance, gravityFactor, linearDamping, pressure,
) {
  if (warnUninit('softBodyCreate')) return 0;
  const w = state.worlds.get(worldH); if (!w) return 0;
  const needF = vertexCount * 4, needU = triangleCount * 3;
  if (state.scratchF32.length < needF || state.scratchU32.length < needU) return 0;
  if (vertexCount < 3 || triangleCount === 0) return 0;

  const J = JoltModule;
  const shared = new J.SoftBodySharedSettings();
  for (let i = 0; i < vertexCount; i++) {
    const v = new J.SoftBodySharedSettingsVertex();
    v.mPosition = new J.Float3(state.scratchF32[i*4], state.scratchF32[i*4+1], state.scratchF32[i*4+2]);
    v.mVelocity = new J.Float3(0, 0, 0);
    v.mInvMass = state.scratchF32[i*4+3];
    shared.mVertices.push_back(v);
  }
  for (let t = 0; t < triangleCount; t++) {
    const f = new J.SoftBodySharedSettingsFace(
      state.scratchU32[t*3], state.scratchU32[t*3+1], state.scratchU32[t*3+2], 0
    );
    shared.AddFace(f);
  }
  const attrs = new J.SoftBodySharedSettingsVertexAttributes();
  attrs.mCompliance = edgeCompliance;
  attrs.mShearCompliance = edgeCompliance;
  attrs.mBendCompliance = edgeCompliance;
  shared.CreateConstraints(attrs, 1, J.EBendType_Distance ?? 0);
  shared.Optimize();

  const bcs = new J.SoftBodyCreationSettings(shared, rvec3(px, py, pz), quat(rx, ry, rz, rw), objectLayer | 0);
  bcs.mGravityFactor = gravityFactor;
  bcs.mLinearDamping = linearDamping;
  bcs.mPressure = pressure;
  bcs.mUpdatePosition = true;

  const bodyId = w.bodyInterface.CreateAndAddSoftBody(bcs, J.EActivation_Activate);
  const h = state.nextBody++;
  state.bodies.set(h, { world: worldH, bodyId, isSoftBody: true });
  return h;
}

function softMotionProperties(h) {
  const b = state.bodies.get(h); if (!b) return null;
  const w = state.worlds.get(b.world); if (!w) return null;
  const J = JoltModule;
  // Use read-locking to safely access SoftBodyMotionProperties.
  const lock = new J.BodyLockRead(w.system.GetBodyLockInterface(), b.bodyId);
  if (!lock.SucceededAndIsInBroadPhase()) { lock.ReleaseLock(); return null; }
  const body = lock.GetBody();
  if (!body.IsSoftBody()) { lock.ReleaseLock(); return null; }
  const mp = J.castObject ? J.castObject(body.GetMotionPropertiesUnchecked(), J.SoftBodyMotionProperties)
                          : body.GetMotionPropertiesUnchecked();
  return { lock, body, mp };
}

export function softBodyVertexCount(h) {
  const ctx = softMotionProperties(h); if (!ctx) return 0;
  const n = ctx.mp.GetVertices ? ctx.mp.GetVertices().size() : 0;
  ctx.lock.ReleaseLock();
  return n;
}
export function softBodyGetVertex(h, idx, axis) {
  const ctx = softMotionProperties(h); if (!ctx) return 0;
  const verts = ctx.mp.GetVertices ? ctx.mp.GetVertices() : null;
  if (!verts || idx >= verts.size()) { ctx.lock.ReleaseLock(); return 0; }
  const v = verts.at(idx);
  const local = v.mPosition;     // Vec3 in body-local space
  const xform = ctx.body.GetWorldTransform();
  const worldPos = xform.Multiply3x4(local);
  ctx.lock.ReleaseLock();
  if (axis === 0) return worldPos.GetX();
  if (axis === 1) return worldPos.GetY();
  if (axis === 2) return worldPos.GetZ();
  return 0;
}
export function softBodySetVertex(h, idx, x, y, z) {
  const b = state.bodies.get(h); if (!b) return;
  const w = state.worlds.get(b.world); if (!w) return;
  const J = JoltModule;
  const lock = new J.BodyLockWrite(w.system.GetBodyLockInterface(), b.bodyId);
  if (lock.SucceededAndIsInBroadPhase()) {
    const body = lock.GetBody();
    if (body.IsSoftBody()) {
      const mp = J.castObject ? J.castObject(body.GetMotionPropertiesUnchecked(), J.SoftBodyMotionProperties)
                              : body.GetMotionPropertiesUnchecked();
      const verts = mp.GetVertices();
      if (idx < verts.size()) {
        const xform = body.GetWorldTransform();
        const inv = xform.Inversed();
        const local = inv.Multiply3x4(vec3(x, y, z));
        verts.at(idx).mPosition = local;
      }
    }
  }
  lock.ReleaseLock();
}
export function softBodySetVertexInvMass(h, idx, invMass) {
  const b = state.bodies.get(h); if (!b) return;
  const w = state.worlds.get(b.world); if (!w) return;
  const J = JoltModule;
  const lock = new J.BodyLockWrite(w.system.GetBodyLockInterface(), b.bodyId);
  if (lock.SucceededAndIsInBroadPhase()) {
    const body = lock.GetBody();
    if (body.IsSoftBody()) {
      const mp = J.castObject ? J.castObject(body.GetMotionPropertiesUnchecked(), J.SoftBodyMotionProperties)
                              : body.GetMotionPropertiesUnchecked();
      const verts = mp.GetVertices();
      if (idx < verts.size()) verts.at(idx).mInvMass = invMass;
    }
  }
  lock.ReleaseLock();
}

// ---------------------------------------------------------------------------
// Wheeled vehicles (Tier 2 — 4-wheel car)
// ---------------------------------------------------------------------------

export function vehicleCreate(
  worldH, chassisShapeH,
  upX, upY, upZ, fwX, fwY, fwZ,
  w0x, w0y, w0z, w1x, w1y, w1z, w2x, w2y, w2z, w3x, w3y, w3z,
  wheelRadius, wheelWidth, suspensionMin, suspensionMax,
  maxSteerAngle, maxBrakeTorque, maxHandbrakeTorque,
  engineMaxTorque, maxPitchRollAngle, objectLayer,
  px, py, pz, rx, ry, rz, rw,
) {
  if (warnUninit('vehicleCreate')) return 0;
  const w = state.worlds.get(worldH); if (!w) return 0;
  const shape = state.shapes.get(chassisShapeH); if (!shape) return 0;
  const J = JoltModule;

  // Chassis body.
  const chassisSettings = new J.BodyCreationSettings(
    shape, rvec3(px, py, pz), quat(rx, ry, rz, rw),
    J.EMotionType_Dynamic, objectLayer | 0,
  );
  chassisSettings.mOverrideMassProperties = J.EOverrideMassProperties_CalculateInertia;
  chassisSettings.mMassPropertiesOverride.mMass = 1500.0;
  const chassisId = w.bodyInterface.CreateAndAddBody(chassisSettings, J.EActivation_Activate);
  const chassisHandle = state.nextBody++;
  state.bodies.set(chassisHandle, { world: worldH, bodyId: chassisId });

  // VehicleConstraintSettings.
  const vcs = new J.VehicleConstraintSettings();
  vcs.mUp = vec3(upX, upY, upZ);
  vcs.mForward = vec3(fwX, fwY, fwZ);
  vcs.mMaxPitchRollAngle = maxPitchRollAngle;

  // 4 wheels — FL/FR steer, RL/RR drive + handbrake.
  const wheelPositions = [
    [w0x, w0y, w0z], [w1x, w1y, w1z], [w2x, w2y, w2z], [w3x, w3y, w3z],
  ];
  for (let i = 0; i < 4; i++) {
    const wheel = new J.WheelSettingsWV();
    wheel.mPosition = vec3(wheelPositions[i][0], wheelPositions[i][1], wheelPositions[i][2]);
    wheel.mRadius = wheelRadius;
    wheel.mWidth = wheelWidth;
    wheel.mSuspensionMinLength = suspensionMin;
    wheel.mSuspensionMaxLength = suspensionMax;
    wheel.mMaxSteerAngle = (i < 2) ? maxSteerAngle : 0.0;
    wheel.mMaxBrakeTorque = maxBrakeTorque;
    wheel.mMaxHandBrakeTorque = (i >= 2) ? maxHandbrakeTorque : 0.0;
    vcs.mWheels.push_back(wheel);
  }

  // WheeledVehicleController with a single rear-axle differential.
  const controller = new J.WheeledVehicleControllerSettings();
  controller.mEngine.mMaxTorque = engineMaxTorque;
  const diff = new J.VehicleDifferentialSettings();
  diff.mLeftWheel = 2;
  diff.mRightWheel = 3;
  controller.mDifferentials.push_back(diff);
  vcs.mController = controller;

  // Lock chassis to construct the constraint.
  const lock = new J.BodyLockWrite(w.system.GetBodyLockInterface(), chassisId);
  if (!lock.SucceededAndIsInBroadPhase()) { lock.ReleaseLock(); return 0; }
  const constraint = new J.VehicleConstraint(lock.GetBody(), vcs);
  lock.ReleaseLock();

  const tester = new J.VehicleCollisionTesterRay(OBJECT_LAYER_NON_MOVING, vec3(upX, upY, upZ));
  constraint.SetVehicleCollisionTester(tester);
  w.system.AddConstraint(constraint);
  w.system.AddStepListener(constraint);

  const vh = state.nextConstraint++;
  state.constraints.set(vh, {
    world: worldH, constraint, tester,
    isVehicle: true, chassisHandle, chassisId,
  });
  return vh;
}

export function vehicleDestroy(h) {
  const v = state.constraints.get(h); if (!v || !v.isVehicle) return;
  const w = state.worlds.get(v.world); if (!w) return;
  w.system.RemoveStepListener(v.constraint);
  w.system.RemoveConstraint(v.constraint);
  if (w.bodyInterface.IsAdded(v.chassisId)) w.bodyInterface.RemoveBody(v.chassisId);
  w.bodyInterface.DestroyBody(v.chassisId);
  state.bodies.delete(v.chassisHandle);
  state.constraints.delete(h);
}

export function vehicleGetChassis(h) {
  const v = state.constraints.get(h); return (v && v.isVehicle) ? v.chassisHandle : 0;
}

export function vehicleSetInput(h, forward, right, brake, handbrake) {
  const v = state.constraints.get(h); if (!v || !v.isVehicle) return;
  const J = JoltModule;
  const ctrl = J.castObject ? J.castObject(v.constraint.GetController(), J.WheeledVehicleController)
                            : v.constraint.GetController();
  ctrl.SetDriverInput(forward, right, brake, handbrake);
  if (forward !== 0 || brake !== 0 || handbrake !== 0 || right !== 0) {
    const w = state.worlds.get(v.world);
    if (w) w.bodyInterface.ActivateBody(v.chassisId);
  }
}

export function vehicleGetWheelTransform(h, wheelIndex, axis) {
  const v = state.constraints.get(h); if (!v || !v.isVehicle) return 0;
  const J = JoltModule;
  const xform = v.constraint.GetWheelWorldTransform(wheelIndex, J.Vec3.prototype.sAxisY(), J.Vec3.prototype.sAxisX());
  if (axis < 3) {
    const p = xform.GetTranslation();
    if (axis === 0) return p.GetX();
    if (axis === 1) return p.GetY();
    return p.GetZ();
  }
  if (axis < 7) {
    const q = xform.GetQuaternion();
    if (axis === 3) return q.GetX();
    if (axis === 4) return q.GetY();
    if (axis === 5) return q.GetZ();
    return q.GetW();
  }
  return 0;
}

export function vehicleGetEngineRpm(h) {
  const v = state.constraints.get(h); if (!v || !v.isVehicle) return 0;
  const J = JoltModule;
  const ctrl = J.castObject ? J.castObject(v.constraint.GetController(), J.WheeledVehicleController)
                            : v.constraint.GetController();
  return ctrl.GetEngine().GetCurrentRPM();
}

export function vehicleGetWheelAngularVelocity(h, wheelIndex) {
  const v = state.constraints.get(h); if (!v || !v.isVehicle) return 0;
  const J = JoltModule;
  const wheel = v.constraint.GetWheel(wheelIndex);
  if (!wheel) return 0;
  const wv = J.castObject ? J.castObject(wheel, J.WheelWV) : wheel;
  return wv.GetAngularVelocity ? wv.GetAngularVelocity() : 0;
}
