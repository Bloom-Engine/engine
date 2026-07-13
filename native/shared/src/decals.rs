//! EN-027 — surface decals (bullet holes, scorch, blood splats).
//!
//! Textbook AAA decals are a deferred pass: project a box, read depth,
//! rewrite the G-buffer before lighting. That is the right long-term shape and
//! it is what a future revision should do. It is also a new pass wired into a
//! hand-ordered 559 KB frame, for a feature whose entire visible job here is
//! "the wall remembers the bullet".
//!
//! So this version is a *sticker*: an oriented quad, pushed a couple of
//! millimetres along the surface normal, drawn through the existing instanced
//! cutout material path. It depth-tests against the world, receives no
//! lighting of its own (the material multiplies the surface's own shading), and
//! costs one draw call for the whole ring. It cannot wrap around a corner —
//! that is the honest limitation, and for bullet holes on flat stone and blood
//! on flat ground nobody will ever see it.
//!
//! Packing note: the instance stride only carries a single Y rotation, but a
//! decal needs an arbitrary orientation. A unit normal is two angles, so it
//! fits in the two spare `extra` slots, leaving `rot_y` free to mean roll
//! about the normal and `extra.x` to carry the atlas frame.

/// A decal in flight. Fades out over the last `fade` seconds of its life so it
/// does not pop.
struct Decal {
    pos: [f32; 3],
    /// Surface normal, stored as azimuth/elevation to fit the instance stride.
    az: f32,
    el: f32,
    roll: f32,
    size: f32,
    color: [f32; 4],
    frame: f32,
    age: f32,
    life: f32,
    fade: f32,
}

/// Look + lifetime for the *next* spawns. Set once per decal type (bullet
/// hole, scorch, blood) rather than passed on every hit — a spawn is then 8
/// f64 args, which is exactly the ARM64 register ceiling the FFI has to
/// respect.
#[derive(Clone, Copy)]
pub struct DecalStyle {
    pub frame: f32,
    pub color: [f32; 4],
    pub life: f32,
    pub fade: f32,
}

impl Default for DecalStyle {
    fn default() -> Self {
        Self { frame: 0.0, color: [1.0; 4], life: 0.0, fade: 0.0 }
    }
}

pub struct DecalManager {
    decals: Vec<Decal>,
    capacity: usize,
    /// Ring cursor: once full, the oldest decal is the one overwritten.
    next: usize,
    pub instance_buffer: u32,
    pub style: DecalStyle,
    packed: Vec<f32>,
    pub live: u32,
}

impl DecalManager {
    pub fn new() -> Self {
        Self {
            decals: Vec::new(),
            capacity: 0,
            next: 0,
            instance_buffer: 0,
            style: DecalStyle::default(),
            packed: Vec::new(),
            live: 0,
        }
    }

    pub fn init(&mut self, capacity: usize, instance_buffer: u32) {
        self.capacity = capacity;
        self.instance_buffer = instance_buffer;
        self.decals = Vec::with_capacity(capacity);
        self.packed = vec![0.0; capacity * 12];
        self.next = 0;
        self.live = 0;
    }

    /// Place a decal using the current `style`. `n` is the surface normal (need
    /// not be normalized).
    pub fn spawn_styled(&mut self, pos: [f32; 3], n: [f32; 3], size: f32, roll: f32) {
        let st = self.style;
        self.spawn(pos, n, size, roll, st.frame, st.color, st.life, st.fade);
    }

    /// Place a decal. `n` is the surface normal (need not be normalized);
    /// `life <= 0` means permanent (well — until the ring wraps).
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        &mut self,
        pos: [f32; 3],
        n: [f32; 3],
        size: f32,
        roll: f32,
        frame: f32,
        color: [f32; 4],
        life: f32,
        fade: f32,
    ) {
        if self.capacity == 0 { return; }
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let nn = if len > 1e-5 { [n[0] / len, n[1] / len, n[2] / len] } else { [0.0, 1.0, 0.0] };
        // Spherical encode. el is the angle off +Y; az is the heading in XZ.
        let el = nn[1].clamp(-1.0, 1.0).acos();
        let az = nn[2].atan2(nn[0]);

        let d = Decal {
            // Lift off the surface: z-fighting on a coplanar quad is guaranteed
            // otherwise, and 2 mm is under the depth precision of anything the
            // player can stand close enough to notice.
            pos: [
                pos[0] + nn[0] * 0.002,
                pos[1] + nn[1] * 0.002,
                pos[2] + nn[2] * 0.002,
            ],
            az, el, roll,
            size,
            color,
            frame,
            age: 0.0,
            life: if life <= 0.0 { f32::MAX } else { life },
            fade: fade.max(0.0),
        };

        if self.decals.len() < self.capacity {
            self.decals.push(d);
        } else {
            self.decals[self.next] = d;
            self.next = (self.next + 1) % self.capacity;
        }
    }

    /// Age everything, drop the expired, repack. Returns the live count.
    pub fn update(&mut self, dt: f32) -> u32 {
        let mut i = 0usize;
        while i < self.decals.len() {
            self.decals[i].age += dt;
            if self.decals[i].age >= self.decals[i].life {
                self.decals.swap_remove(i);
                // The ring cursor indexes into a Vec that just shrank; clamp it
                // or the next spawn writes out of bounds.
                if self.next >= self.decals.len().max(1) { self.next = 0; }
                continue;
            }
            i += 1;
        }

        for (i, d) in self.decals.iter().enumerate() {
            // Fade only over the tail of the lifetime.
            let alpha = if d.fade > 0.0 && d.life != f32::MAX {
                let remaining = d.life - d.age;
                (remaining / d.fade).clamp(0.0, 1.0)
            } else { 1.0 };
            let o = i * 12;
            self.packed[o]      = d.pos[0];
            self.packed[o + 1]  = d.pos[1];
            self.packed[o + 2]  = d.pos[2];
            self.packed[o + 3]  = d.roll;
            self.packed[o + 4]  = d.size;
            self.packed[o + 5]  = d.color[0];
            self.packed[o + 6]  = d.color[1];
            self.packed[o + 7]  = d.color[2];
            self.packed[o + 8]  = d.color[3] * alpha;
            self.packed[o + 9]  = d.frame;   // extra.x
            self.packed[o + 10] = d.az;      // extra.y
            self.packed[o + 11] = d.el;      // extra.z
        }
        self.live = self.decals.len() as u32;
        self.live
    }

    pub fn packed(&self) -> &[f32] { &self.packed }

    pub fn clear(&mut self) {
        self.decals.clear();
        self.next = 0;
        self.live = 0;
    }
}

impl Default for DecalManager {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_wraps_without_growing() {
        let mut m = DecalManager::new();
        m.init(4, 1);
        for i in 0..10 {
            m.spawn([i as f32, 0.0, 0.0], [0.0, 1.0, 0.0], 0.2, 0.0, 0.0, [1.0; 4], 0.0, 0.0);
        }
        assert_eq!(m.update(0.016), 4);
    }

    #[test]
    fn expired_decals_are_reclaimed() {
        let mut m = DecalManager::new();
        m.init(8, 1);
        m.spawn([0.0; 3], [0.0, 1.0, 0.0], 0.2, 0.0, 0.0, [1.0; 4], 1.0, 0.2);
        assert_eq!(m.update(0.5), 1);
        assert_eq!(m.update(0.6), 0);
    }

    #[test]
    fn permanent_decals_survive() {
        let mut m = DecalManager::new();
        m.init(8, 1);
        m.spawn([0.0; 3], [0.0, 1.0, 0.0], 0.2, 0.0, 0.0, [1.0; 4], 0.0, 0.0);
        for _ in 0..100 { m.update(1.0); }
        assert_eq!(m.live, 1);
    }

    /// The normal must survive the spherical round-trip, or every decal on a
    /// wall silently lies flat on the floor.
    #[test]
    fn normal_encoding_round_trips() {
        let mut m = DecalManager::new();
        m.init(4, 1);
        let n = [0.0f32, 0.0, 1.0];
        m.spawn([0.0; 3], n, 1.0, 0.0, 0.0, [1.0; 4], 0.0, 0.0);
        m.update(0.0);
        let az = m.packed[10];
        let el = m.packed[11];
        let rx = el.sin() * az.cos();
        let ry = el.cos();
        let rz = el.sin() * az.sin();
        assert!((rx - n[0]).abs() < 1e-4, "x {rx}");
        assert!((ry - n[1]).abs() < 1e-4, "y {ry}");
        assert!((rz - n[2]).abs() < 1e-4, "z {rz}");
    }
}
