//! Frame callback system for Bloom Engine.
//!
//! Allows Perry-compiled TypeScript to register callbacks that run every frame,
//! ordered by priority. This replaces React Three Fiber's useFrame(cb, priority).
//!
//! Callbacks are stored as function pointers (Perry compiles TS functions to native
//! function pointers). Lower priority numbers run first.

/// A registered frame callback.
struct FrameCallback {
    priority: i32,
    callback: extern "C" fn(f64), // receives delta_time as argument
    active: bool,
}

/// Manages frame callbacks with priority ordering.
pub struct FrameCallbackSystem {
    callbacks: Vec<FrameCallback>,
    sorted: bool,
    next_id: u64,
    ids: Vec<u64>, // parallel to callbacks, for removal
}

impl FrameCallbackSystem {
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
            sorted: true,
            next_id: 1,
            ids: Vec::new(),
        }
    }

    /// Register a callback with a priority. Returns an ID for removal.
    /// Lower priority numbers run first (matching R3F convention).
    pub fn register(&mut self, priority: i32, callback: extern "C" fn(f64)) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.callbacks.push(FrameCallback {
            priority,
            callback,
            active: true,
        });
        self.ids.push(id);
        self.sorted = false;
        id
    }

    /// Unregister a callback by ID.
    pub fn unregister(&mut self, id: u64) {
        if let Some(idx) = self.ids.iter().position(|&i| i == id) {
            self.callbacks.remove(idx);
            self.ids.remove(idx);
        }
    }

    /// Run all active callbacks in priority order.
    pub fn run_all(&mut self, delta_time: f64) {
        if !self.sorted {
            // Sort by priority (stable sort preserves insertion order for equal priorities)
            let mut indices: Vec<usize> = (0..self.callbacks.len()).collect();
            indices.sort_by_key(|&i| self.callbacks[i].priority);

            let old_callbacks: Vec<_> = self.callbacks.drain(..).collect();
            let old_ids: Vec<_> = self.ids.drain(..).collect();
            for &i in &indices {
                self.callbacks.push(FrameCallback {
                    priority: old_callbacks[i].priority,
                    callback: old_callbacks[i].callback,
                    active: old_callbacks[i].active,
                });
                self.ids.push(old_ids[i]);
            }
            self.sorted = true;
        }

        for cb in &self.callbacks {
            if cb.active {
                (cb.callback)(delta_time);
            }
        }
    }

    /// Get number of registered callbacks.
    pub fn count(&self) -> usize {
        self.callbacks.len()
    }
}
