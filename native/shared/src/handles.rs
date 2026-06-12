//! Generic handle registry: maps f64 handles to owned values.
//!
//! Handles are **generational**: the f64 encodes a 1-based slot index in
//! the low 32 bits and a generation counter above it. Freeing a slot
//! bumps its generation, so a stale handle held by game code after a
//! free/realloc cycle fails the lookup instead of silently aliasing
//! whatever object reused the slot — the classic use-after-free-by-index
//! bug this registry used to permit.
//!
//! Encoding: `value = (generation << 32) | (slot + 1)`, stored in an f64.
//! f64 represents integers exactly up to 2^53, generations wrap at 2^21
//! (a slot must be freed two million times, and the stale handle
//! retained throughout, to produce a false positive). Generation 0
//! handles are plain small integers — the representation games saw
//! before generations existed.
//!
//! `0` is never a valid handle.

const SLOT_MASK: u64 = 0xFFFF_FFFF;
const GEN_SHIFT: u32 = 32;
/// Generations wrap below 2^21 so the full encoded value stays under
/// 2^53 (exact f64 integer range).
const GEN_WRAP: u64 = 1 << 21;

pub struct HandleRegistry<T> {
    items: Vec<Option<T>>,
    generations: Vec<u64>,
    free_list: Vec<usize>,
}

impl<T> Default for HandleRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> HandleRegistry<T> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            generations: Vec::new(),
            free_list: Vec::new(),
        }
    }

    fn encode(&self, idx: usize) -> f64 {
        ((self.generations[idx] << GEN_SHIFT) | (idx as u64 + 1)) as f64
    }

    /// Decode a handle to a live slot index, or None if the handle is
    /// invalid, out of range, freed, or from a previous generation.
    fn decode(&self, handle: f64) -> Option<usize> {
        if !(handle.is_finite() && handle > 0.0) {
            return None;
        }
        let v = handle as u64;
        let slot = v & SLOT_MASK;
        let gen = v >> GEN_SHIFT;
        if slot == 0 || slot as usize > self.items.len() {
            return None;
        }
        let idx = slot as usize - 1;
        if self.generations[idx] != gen {
            return None; // stale handle from before a free()
        }
        self.items[idx].as_ref()?;
        Some(idx)
    }

    /// Allocate a handle for the given item.
    pub fn alloc(&mut self, item: T) -> f64 {
        if let Some(idx) = self.free_list.pop() {
            self.items[idx] = Some(item);
            self.encode(idx)
        } else {
            self.items.push(Some(item));
            self.generations.push(0);
            self.encode(self.items.len() - 1)
        }
    }

    /// Get a reference to the item at the given handle.
    pub fn get(&self, handle: f64) -> Option<&T> {
        self.decode(handle).and_then(|idx| self.items[idx].as_ref())
    }

    /// Get a mutable reference to the item at the given handle.
    pub fn get_mut(&mut self, handle: f64) -> Option<&mut T> {
        let idx = self.decode(handle)?;
        self.items[idx].as_mut()
    }

    /// Free the item at the given handle. The slot's generation is bumped
    /// so every outstanding copy of this handle becomes invalid.
    pub fn free(&mut self, handle: f64) -> Option<T> {
        let idx = self.decode(handle)?;
        let item = self.items[idx].take();
        if item.is_some() {
            self.generations[idx] = (self.generations[idx] + 1) % GEN_WRAP;
            self.free_list.push(idx);
        }
        item
    }

    /// Number of slots (including freed). Use for iteration bounds.
    pub fn capacity(&self) -> usize {
        self.items.len()
    }

    /// Iterate over all live (handle, &T) pairs. Handles carry their
    /// current generation — they compare equal to what alloc() returned.
    pub fn iter(&self) -> impl Iterator<Item = (f64, &T)> {
        self.items.iter().enumerate().filter_map(|(idx, slot)| {
            slot.as_ref()
                .map(|item| (((self.generations[idx] << GEN_SHIFT) | (idx as u64 + 1)) as f64, item))
        })
    }

    /// Iterate over all live (handle, &mut T) pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (f64, &mut T)> {
        let generations = &self.generations;
        self.items.iter_mut().enumerate().filter_map(move |(idx, slot)| {
            slot.as_mut()
                .map(|item| (((generations[idx] << GEN_SHIFT) | (idx as u64 + 1)) as f64, item))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_get_free() {
        let mut r = HandleRegistry::new();
        let h = r.alloc("a");
        assert_eq!(r.get(h), Some(&"a"));
        assert_eq!(r.free(h), Some("a"));
        assert_eq!(r.get(h), None);
    }

    #[test]
    fn first_generation_handles_are_small_integers() {
        let mut r = HandleRegistry::new();
        assert_eq!(r.alloc("a"), 1.0);
        assert_eq!(r.alloc("b"), 2.0);
    }

    #[test]
    fn stale_handle_does_not_alias_reused_slot() {
        let mut r = HandleRegistry::new();
        let old = r.alloc("old");
        r.free(old);
        let new = r.alloc("new"); // reuses slot 0 with generation 1
        assert_ne!(old.to_bits(), new.to_bits());
        assert_eq!(r.get(old), None, "stale handle resolved after slot reuse");
        assert_eq!(r.get(new), Some(&"new"));
        // double-free through the stale handle is a no-op
        assert_eq!(r.free(old), None);
        assert_eq!(r.get(new), Some(&"new"));
    }

    #[test]
    fn iter_returns_current_generation_handles() {
        let mut r = HandleRegistry::new();
        let a = r.alloc(1);
        r.free(a);
        let b = r.alloc(2);
        let collected: Vec<(f64, &i32)> = r.iter().collect();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].0.to_bits(), b.to_bits());
        assert_eq!(r.get(collected[0].0), Some(&2));
    }

    #[test]
    fn garbage_handles_rejected() {
        let r: HandleRegistry<i32> = HandleRegistry::new();
        for h in [0.0, -1.0, 0.5, f64::NAN, f64::INFINITY, 1e300] {
            assert!(r.get(h).is_none());
        }
    }
}
