//! EN-026 — CPU-simulated, GPU-instanced particle system.
//!
//! Why this lives in the engine rather than in game code: the game can already
//! build an instance buffer and draw it, but only by pushing every float
//! across the FFI one call at a time (`bloom_mesh_scratch_push_f32`). At 2 000
//! live particles that is ~24 000 FFI calls per frame — the shooter's entire
//! current per-frame FFI budget is ~240. So the *simulation* has to sit on the
//! native side, and the game/engine traffic becomes O(spawn events) instead of
//! O(particles): a burst is one call.
//!
//! What the engine owns: the pool, the integrator, and the packed instance
//! buffer. What the game owns: the material (so the look, the atlas and the
//! blend mode stay authorable in WGSL) and the draw call. A system is created
//! per *look* — smoke, sparks, blood, shells — because each wants its own
//! texture and blend bucket anyway, and one draw per look is cheap.
//!
//! Sim is deliberately simple and closed-form-ish: position, velocity,
//! gravity, linear drag, and a lifetime; size and colour are curves over
//! normalized age evaluated in the shader from `extra.x`. No collision, no
//! sorting (additive needs none, and the cutout bucket is depth-tested).

/// Everything about how one system's particles are born, move, and look.
/// Uploaded once from TS as a flat float array — see `configure_from_slice`.
#[derive(Clone, Copy)]
pub struct ParticleConfig {
    pub life: f32,
    pub life_var: f32,
    /// Initial speed along the emit direction.
    pub speed: f32,
    pub speed_var: f32,
    /// Half-angle of the emission cone, radians. `PI` = fully spherical.
    pub spread: f32,
    /// Constant acceleration (usually negative Y).
    pub gravity: f32,
    /// Linear drag coefficient (per second). 0 = vacuum.
    pub drag: f32,
    pub size0: f32,
    pub size1: f32,
    pub size_var: f32,
    pub color0: [f32; 4],
    pub color1: [f32; 4],
    /// Roll speed, radians/sec (billboard spin).
    pub spin: f32,
    pub spin_var: f32,
    /// Spawn positions are jittered inside a sphere of this radius.
    pub pos_jitter: f32,
    /// > 0 stretches the billboard along its velocity by this many seconds of
    /// travel — the difference between a round spark and a tracer streak.
    pub stretch: f32,
    /// Fraction of the emitter's own velocity the particle inherits.
    pub inherit: f32,
    /// Number of atlas frames; the shader gets a frame index in `extra.y`.
    pub frames: f32,
    /// Bounce off the y = `floor_y` plane instead of passing through it.
    /// `restitution` <= 0 disables (the default).
    pub floor_y: f32,
    pub restitution: f32,
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self {
            life: 1.0, life_var: 0.0,
            speed: 1.0, speed_var: 0.0,
            spread: 0.3,
            gravity: -9.81,
            drag: 0.0,
            size0: 0.2, size1: 0.2, size_var: 0.0,
            color0: [1.0; 4], color1: [1.0, 1.0, 1.0, 0.0],
            spin: 0.0, spin_var: 0.0,
            pos_jitter: 0.0,
            stretch: 0.0,
            inherit: 0.0,
            frames: 1.0,
            floor_y: 0.0, restitution: 0.0,
        }
    }
}

/// Structure-of-arrays pool. Dead particles are swap-removed from the live
/// prefix, so the live set is always `[0, live)` and the instance write is one
/// contiguous memcpy with no compaction pass.
pub struct ParticleSystem {
    pub capacity: usize,
    pub live: usize,
    pub cfg: ParticleConfig,
    /// GPU instance buffer handle (dynamic, capacity-sized).
    pub instance_buffer: u32,

    px: Vec<f32>, py: Vec<f32>, pz: Vec<f32>,
    vx: Vec<f32>, vy: Vec<f32>, vz: Vec<f32>,
    age: Vec<f32>, life: Vec<f32>,
    rot: Vec<f32>, spin: Vec<f32>,
    size: Vec<f32>, seed: Vec<f32>,

    /// Packed 12-float-per-instance staging buffer, reused every frame.
    packed: Vec<f32>,
    rng: u32,
}

impl ParticleSystem {
    pub fn new(capacity: usize, instance_buffer: u32) -> Self {
        let z = || vec![0.0f32; capacity];
        Self {
            capacity,
            live: 0,
            cfg: ParticleConfig::default(),
            instance_buffer,
            px: z(), py: z(), pz: z(),
            vx: z(), vy: z(), vz: z(),
            age: z(), life: z(),
            rot: z(), spin: z(),
            size: z(), seed: z(),
            packed: vec![0.0; capacity * 12],
            rng: 0x9E3779B9,
        }
    }

    #[inline]
    fn rand(&mut self) -> f32 {
        // xorshift32 — deterministic per system, which keeps a replayed burst
        // identical frame-to-frame for screenshot diffing.
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        (x & 0x00FF_FFFF) as f32 / 16_777_215.0
    }

    /// Symmetric ±1 noise.
    #[inline]
    fn rand_s(&mut self) -> f32 { self.rand() * 2.0 - 1.0 }

    pub fn configure_from_slice(&mut self, p: &[f32]) {
        let g = |i: usize| -> f32 { p.get(i).copied().unwrap_or(0.0) };
        self.cfg = ParticleConfig {
            life: g(0).max(0.01), life_var: g(1),
            speed: g(2), speed_var: g(3),
            spread: g(4),
            gravity: g(5),
            drag: g(6),
            size0: g(7), size1: g(8), size_var: g(9),
            color0: [g(10), g(11), g(12), g(13)],
            color1: [g(14), g(15), g(16), g(17)],
            spin: g(18), spin_var: g(19),
            pos_jitter: g(20),
            stretch: g(21),
            inherit: g(22),
            frames: g(23).max(1.0),
            floor_y: g(24),
            restitution: g(25),
        };
    }

    /// Spawn `count` particles at `pos`, biased along `dir`. A zero `dir`
    /// means "no preferred direction" and emits into the full sphere, which is
    /// what a blood burst or an explosion core wants; a surface normal gives a
    /// cone, which is what an impact wants.
    pub fn emit(&mut self, pos: [f32; 3], dir: [f32; 3], count: usize) {
        let c = self.cfg;
        let dlen = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        let base = if dlen > 1e-5 {
            [dir[0] / dlen, dir[1] / dlen, dir[2] / dlen]
        } else {
            [0.0, 1.0, 0.0]
        };
        // Basis around the emit direction, for the cone sample.
        let up = if base[1].abs() > 0.99 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let right = normalize(cross(up, base));
        let realup = cross(base, right);

        let spread = if dlen > 1e-5 { c.spread } else { std::f32::consts::PI };

        for _ in 0..count {
            if self.live >= self.capacity {
                // Full: overwrite the oldest-looking slot rather than dropping
                // the spawn. A burst you asked for should always be visible;
                // it is the *stale* particle that is expendable.
                let victim = (self.rand() * self.capacity as f32) as usize % self.capacity.max(1);
                self.age[victim] = self.life[victim];
                self.kill(victim);
            }
            let i = self.live;
            self.live += 1;

            // Cone sample: cosine-ish, biased toward the axis.
            let a = self.rand() * std::f32::consts::TAU;
            let t = self.rand().sqrt() * spread;
            let (st, ct) = (t.sin(), t.cos());
            let d = [
                base[0] * ct + (right[0] * a.cos() + realup[0] * a.sin()) * st,
                base[1] * ct + (right[1] * a.cos() + realup[1] * a.sin()) * st,
                base[2] * ct + (right[2] * a.cos() + realup[2] * a.sin()) * st,
            ];
            let sp = (c.speed + self.rand_s() * c.speed_var).max(0.0);

            let jx = self.rand_s() * c.pos_jitter;
            let jy = self.rand_s() * c.pos_jitter;
            let jz = self.rand_s() * c.pos_jitter;

            self.px[i] = pos[0] + jx;
            self.py[i] = pos[1] + jy;
            self.pz[i] = pos[2] + jz;
            self.vx[i] = d[0] * sp;
            self.vy[i] = d[1] * sp;
            self.vz[i] = d[2] * sp;
            self.age[i] = 0.0;
            self.life[i] = (c.life + self.rand_s() * c.life_var).max(0.02);
            self.rot[i] = self.rand() * std::f32::consts::TAU;
            self.spin[i] = c.spin + self.rand_s() * c.spin_var;
            self.size[i] = 1.0 + self.rand_s() * c.size_var;
            self.seed[i] = self.rand();
        }
    }

    #[inline]
    fn kill(&mut self, i: usize) {
        let last = self.live - 1;
        if i != last {
            self.px.swap(i, last); self.py.swap(i, last); self.pz.swap(i, last);
            self.vx.swap(i, last); self.vy.swap(i, last); self.vz.swap(i, last);
            self.age.swap(i, last); self.life.swap(i, last);
            self.rot.swap(i, last); self.spin.swap(i, last);
            self.size.swap(i, last); self.seed.swap(i, last);
        }
        self.live = last;
    }

    /// Integrate one step and repack. Returns the live count to draw.
    pub fn update(&mut self, dt: f32) -> u32 {
        let c = self.cfg;
        let dt = dt.clamp(0.0, 0.1); // a hitch must not teleport the sim

        let mut i = 0usize;
        while i < self.live {
            self.age[i] += dt;
            if self.age[i] >= self.life[i] {
                self.kill(i);
                continue; // a live particle was swapped into i — re-test it
            }
            // Semi-implicit Euler + exponential drag.
            self.vy[i] += c.gravity * dt;
            if c.drag > 0.0 {
                let k = (1.0 - c.drag * dt).max(0.0);
                self.vx[i] *= k; self.vy[i] *= k; self.vz[i] *= k;
            }
            self.px[i] += self.vx[i] * dt;
            self.py[i] += self.vy[i] * dt;
            self.pz[i] += self.vz[i] * dt;

            if c.restitution > 0.0 && self.py[i] < c.floor_y {
                self.py[i] = c.floor_y;
                self.vy[i] = -self.vy[i] * c.restitution;
                // Kill the horizontal skid too, or shells slide forever.
                self.vx[i] *= 0.6;
                self.vz[i] *= 0.6;
                if self.vy[i].abs() < 0.4 { self.vy[i] = 0.0; }
            }

            self.rot[i] += self.spin[i] * dt;
            i += 1;
        }

        // Pack. Layout must match InstanceData3D: pos.xyz, rot_y, scale,
        // tint.rgba, extra.xyz.
        for i in 0..self.live {
            let t = (self.age[i] / self.life[i]).clamp(0.0, 1.0);
            let size = lerp(c.size0, c.size1, t) * self.size[i];
            let col = [
                lerp(c.color0[0], c.color1[0], t),
                lerp(c.color0[1], c.color1[1], t),
                lerp(c.color0[2], c.color1[2], t),
                lerp(c.color0[3], c.color1[3], t),
            ];
            let frame = if c.frames > 1.0 {
                (t * c.frames).floor().min(c.frames - 1.0)
            } else { 0.0 };
            // Velocity-stretch length in metres — the shader elongates the
            // quad along the projected velocity by this much.
            let stretch = if c.stretch > 0.0 {
                let v = (self.vx[i] * self.vx[i] + self.vy[i] * self.vy[i] + self.vz[i] * self.vz[i]).sqrt();
                v * c.stretch
            } else { 0.0 };

            let o = i * 12;
            self.packed[o]      = self.px[i];
            self.packed[o + 1]  = self.py[i];
            self.packed[o + 2]  = self.pz[i];
            self.packed[o + 3]  = self.rot[i];
            self.packed[o + 4]  = size;
            self.packed[o + 5]  = col[0];
            self.packed[o + 6]  = col[1];
            self.packed[o + 7]  = col[2];
            self.packed[o + 8]  = col[3];
            self.packed[o + 9]  = t;        // extra.x — normalized age
            self.packed[o + 10] = frame;    // extra.y — atlas frame
            self.packed[o + 11] = stretch;  // extra.z — stretch metres
        }
        self.live as u32
    }

    pub fn packed(&self) -> &[f32] { &self.packed }

    pub fn clear(&mut self) { self.live = 0; }
}

#[inline] fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }
#[inline] fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}
#[inline] fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if l > 1e-6 { [v[0]/l, v[1]/l, v[2]/l] } else { [1.0, 0.0, 0.0] }
}

/// Registry of all systems. Handles are 1-based; 0 is "no system".
pub struct ParticleManager {
    pub systems: Vec<Option<ParticleSystem>>,
}

impl ParticleManager {
    pub fn new() -> Self { Self { systems: Vec::new() } }

    pub fn create(&mut self, capacity: usize, instance_buffer: u32) -> u32 {
        self.systems.push(Some(ParticleSystem::new(capacity, instance_buffer)));
        self.systems.len() as u32
    }

    pub fn get_mut(&mut self, handle: u32) -> Option<&mut ParticleSystem> {
        if handle == 0 { return None; }
        self.systems.get_mut(handle as usize - 1)?.as_mut()
    }
}

impl Default for ParticleManager {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sys() -> ParticleSystem { ParticleSystem::new(64, 1) }

    #[test]
    fn emit_then_expire() {
        let mut s = sys();
        s.configure_from_slice(&[0.5, 0.0, 1.0, 0.0, 0.3, 0.0, 0.0, 1.0, 1.0, 0.0]);
        s.emit([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 10);
        assert_eq!(s.live, 10);
        assert_eq!(s.update(0.1), 10);
        // Past the 0.5 s lifetime everything must be reclaimed — the swap-remove
        // loop has to re-test the swapped-in particle or it leaks half the pool.
        for _ in 0..10 { s.update(0.1); }
        assert_eq!(s.live, 0);
    }

    #[test]
    fn capacity_is_never_exceeded() {
        let mut s = sys();
        s.configure_from_slice(&[10.0]);
        s.emit([0.0; 3], [0.0, 1.0, 0.0], 500);
        assert_eq!(s.live, 64);
        assert_eq!(s.update(0.016), 64);
    }

    #[test]
    fn gravity_pulls_down() {
        let mut s = sys();
        // life 10, speed 0, gravity -10
        s.configure_from_slice(&[10.0, 0.0, 0.0, 0.0, 0.0, -10.0]);
        s.emit([0.0, 5.0, 0.0], [0.0, 1.0, 0.0], 1);
        s.update(0.1);
        assert!(s.py[0] < 5.0, "expected fall, got y={}", s.py[0]);
    }

    #[test]
    fn floor_bounce_reverses_velocity() {
        let mut s = sys();
        let mut p = vec![0.0f32; 26];
        p[0] = 10.0;   // life
        p[5] = -10.0;  // gravity
        p[24] = 0.0;   // floor_y
        p[25] = 0.5;   // restitution
        s.configure_from_slice(&p);
        s.emit([0.0, 0.1, 0.0], [0.0, -1.0, 0.0], 1);
        for _ in 0..20 { s.update(0.016); }
        assert!(s.py[0] >= 0.0, "particle sank through the floor: y={}", s.py[0]);
    }
}
