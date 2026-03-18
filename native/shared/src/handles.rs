/// Generic handle registry: maps f64 handles to owned values.
/// Handles are small integer indices (1-based) cast to f64 for FFI.
pub struct HandleRegistry<T> {
    items: Vec<Option<T>>,
    free_list: Vec<usize>,
}

impl<T> HandleRegistry<T> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            free_list: Vec::new(),
        }
    }

    /// Allocate a handle for the given item. Returns a 1-based index as f64.
    pub fn alloc(&mut self, item: T) -> f64 {
        if let Some(idx) = self.free_list.pop() {
            self.items[idx] = Some(item);
            (idx + 1) as f64
        } else {
            self.items.push(Some(item));
            self.items.len() as f64
        }
    }

    /// Get a reference to the item at the given handle.
    pub fn get(&self, handle: f64) -> Option<&T> {
        let idx = handle as usize;
        if idx == 0 || idx > self.items.len() {
            return None;
        }
        self.items[idx - 1].as_ref()
    }

    /// Get a mutable reference to the item at the given handle.
    pub fn get_mut(&mut self, handle: f64) -> Option<&mut T> {
        let idx = handle as usize;
        if idx == 0 || idx > self.items.len() {
            return None;
        }
        self.items[idx - 1].as_mut()
    }

    /// Free the item at the given handle.
    pub fn free(&mut self, handle: f64) -> Option<T> {
        let idx = handle as usize;
        if idx == 0 || idx > self.items.len() {
            return None;
        }
        let item = self.items[idx - 1].take();
        if item.is_some() {
            self.free_list.push(idx - 1);
        }
        item
    }
}
