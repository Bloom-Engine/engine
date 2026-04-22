/// Platform-agnostic input state.

const MAX_KEYS: usize = 512;
const MAX_MOUSE_BUTTONS: usize = 8;
const MAX_GAMEPAD_AXES: usize = 8;
const MAX_GAMEPAD_BUTTONS: usize = 16;
const MAX_TOUCH_POINTS: usize = 10;

pub struct TouchPoint {
    pub x: f64,
    pub y: f64,
    pub active: bool,
}

pub struct InputState {
    keys_pressed: [bool; MAX_KEYS],
    keys_down: [bool; MAX_KEYS],
    keys_released: [bool; MAX_KEYS],
    prev_keys_down: [bool; MAX_KEYS],

    pub mouse_x: f64,
    pub mouse_y: f64,
    pub mouse_delta_x: f64,
    pub mouse_delta_y: f64,
    prev_mouse_x: f64,
    prev_mouse_y: f64,
    raw_delta_x: f64,
    raw_delta_y: f64,
    /// Scroll wheel accumulator. Platform event loops call `accumulate_mouse_wheel`
    /// when a scroll event arrives; games/editors call `consume_mouse_wheel` once
    /// per frame to read the total delta (and reset it to 0).
    mouse_wheel_y: f64,
    /// Digital Crown rotation accumulator (watchOS). Platforms push rotation
    /// deltas via `accumulate_crown_rotation`; games call `consume_crown_rotation`
    /// once per frame. Units are radians, positive = clockwise (away from user).
    crown_rotation: f64,
    /// Character input ring buffer for text-entry support (E3b). Platform
    /// event loops push Unicode codepoints here on key-down; editors pop them
    /// one at a time via `pop_char`.
    char_queue: [u32; 32],
    char_queue_head: usize,
    char_queue_tail: usize,
    /// Cursor shape requested by the editor. The platform event loop polls
    /// this and applies the appropriate native cursor.
    /// 0=default, 1=hand, 2=move, 3=text, 4=resize_h, 5=resize_v, 6=crosshair
    pub cursor_shape: u32,
    pub cursor_disabled: bool,
    mouse_pressed: [bool; MAX_MOUSE_BUTTONS],
    mouse_down: [bool; MAX_MOUSE_BUTTONS],
    mouse_released: [bool; MAX_MOUSE_BUTTONS],
    prev_mouse_down: [bool; MAX_MOUSE_BUTTONS],

    // Gamepad
    pub gamepad_available: bool,
    gamepad_axes: [f32; MAX_GAMEPAD_AXES],
    gamepad_buttons_down: [bool; MAX_GAMEPAD_BUTTONS],
    gamepad_buttons_pressed: [bool; MAX_GAMEPAD_BUTTONS],
    gamepad_buttons_released: [bool; MAX_GAMEPAD_BUTTONS],
    prev_gamepad_buttons: [bool; MAX_GAMEPAD_BUTTONS],
    pub gamepad_axis_count: usize,

    // Touch
    pub touch_points: [TouchPoint; MAX_TOUCH_POINTS],
    pub touch_count: usize,
    // Release-deferral: an ACTION_UP that arrives in the same frame as its
    // ACTION_DOWN would otherwise zero out `touch_count` before the game has a
    // chance to poll. Mark the slot here instead and clear at end_frame, so a
    // fast tap is still visible for one full frame.
    touch_pending_release: [bool; MAX_TOUCH_POINTS],
}

impl InputState {
    pub fn new() -> Self {
        const EMPTY_TOUCH: TouchPoint = TouchPoint { x: 0.0, y: 0.0, active: false };
        Self {
            keys_pressed: [false; MAX_KEYS],
            keys_down: [false; MAX_KEYS],
            keys_released: [false; MAX_KEYS],
            prev_keys_down: [false; MAX_KEYS],
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_delta_x: 0.0,
            mouse_delta_y: 0.0,
            prev_mouse_x: 0.0,
            prev_mouse_y: 0.0,
            raw_delta_x: 0.0,
            raw_delta_y: 0.0,
            mouse_wheel_y: 0.0,
            crown_rotation: 0.0,
            char_queue: [0; 32],
            char_queue_head: 0,
            char_queue_tail: 0,
            cursor_shape: 0,
            cursor_disabled: false,
            mouse_pressed: [false; MAX_MOUSE_BUTTONS],
            mouse_down: [false; MAX_MOUSE_BUTTONS],
            mouse_released: [false; MAX_MOUSE_BUTTONS],
            prev_mouse_down: [false; MAX_MOUSE_BUTTONS],
            gamepad_available: false,
            gamepad_axes: [0.0; MAX_GAMEPAD_AXES],
            gamepad_buttons_down: [false; MAX_GAMEPAD_BUTTONS],
            gamepad_buttons_pressed: [false; MAX_GAMEPAD_BUTTONS],
            gamepad_buttons_released: [false; MAX_GAMEPAD_BUTTONS],
            prev_gamepad_buttons: [false; MAX_GAMEPAD_BUTTONS],
            gamepad_axis_count: 0,
            touch_points: [EMPTY_TOUCH; MAX_TOUCH_POINTS],
            touch_count: 0,
            touch_pending_release: [false; MAX_TOUCH_POINTS],
        }
    }

    pub fn begin_frame(&mut self) {
        if self.cursor_disabled {
            self.mouse_delta_x = self.raw_delta_x;
            self.mouse_delta_y = self.raw_delta_y;
            self.raw_delta_x = 0.0;
            self.raw_delta_y = 0.0;
        } else {
            self.mouse_delta_x = self.mouse_x - self.prev_mouse_x;
            self.mouse_delta_y = self.mouse_y - self.prev_mouse_y;
        }
        for i in 0..MAX_KEYS {
            self.keys_pressed[i] = self.keys_down[i] && !self.prev_keys_down[i];
            self.keys_released[i] = !self.keys_down[i] && self.prev_keys_down[i];
        }
        for i in 0..MAX_MOUSE_BUTTONS {
            self.mouse_pressed[i] = self.mouse_down[i] && !self.prev_mouse_down[i];
            self.mouse_released[i] = !self.mouse_down[i] && self.prev_mouse_down[i];
        }
        for i in 0..MAX_GAMEPAD_BUTTONS {
            self.gamepad_buttons_pressed[i] = self.gamepad_buttons_down[i] && !self.prev_gamepad_buttons[i];
            self.gamepad_buttons_released[i] = !self.gamepad_buttons_down[i] && self.prev_gamepad_buttons[i];
        }
    }

    pub fn end_frame(&mut self) {
        self.prev_keys_down = self.keys_down;
        self.prev_mouse_down = self.mouse_down;
        self.prev_gamepad_buttons = self.gamepad_buttons_down;
        self.prev_mouse_x = self.mouse_x;
        self.prev_mouse_y = self.mouse_y;
        let mut flushed = false;
        for i in 0..MAX_TOUCH_POINTS {
            if self.touch_pending_release[i] {
                self.touch_points[i].active = false;
                self.touch_pending_release[i] = false;
                flushed = true;
            }
        }
        if flushed {
            self.touch_count = self.touch_points.iter().filter(|t| t.active).count();
        }
    }

    // Keyboard
    pub fn set_key_down(&mut self, key: usize) { if key < MAX_KEYS { self.keys_down[key] = true; } }
    pub fn set_key_up(&mut self, key: usize) { if key < MAX_KEYS { self.keys_down[key] = false; } }

    pub fn is_key_pressed(&self, key: usize) -> bool { key < MAX_KEYS && self.keys_pressed[key] }
    pub fn is_key_down(&self, key: usize) -> bool { key < MAX_KEYS && self.keys_down[key] }
    pub fn is_key_released(&self, key: usize) -> bool { key < MAX_KEYS && self.keys_released[key] }

    // Mouse
    pub fn set_mouse_position(&mut self, x: f64, y: f64) { self.mouse_x = x; self.mouse_y = y; }
    pub fn accumulate_mouse_delta(&mut self, dx: f64, dy: f64) { self.raw_delta_x += dx; self.raw_delta_y += dy; }
    pub fn clear_mouse_delta(&mut self) { self.raw_delta_x = 0.0; self.raw_delta_y = 0.0; self.mouse_delta_x = 0.0; self.mouse_delta_y = 0.0; }
    pub fn accumulate_mouse_wheel(&mut self, dy: f64) { self.mouse_wheel_y += dy; }
    /// Returns the accumulated scroll delta since the last call, and resets it.
    /// Games/editors call this once per frame.
    pub fn consume_mouse_wheel(&mut self) -> f64 {
        let v = self.mouse_wheel_y;
        self.mouse_wheel_y = 0.0;
        v
    }
    /// Push a Unicode codepoint into the character input queue. Called by
    /// platform event loops when a key-down event produces a printable char.
    pub fn push_char(&mut self, c: u32) {
        let next = (self.char_queue_head + 1) % self.char_queue.len();
        if next != self.char_queue_tail {
            self.char_queue[self.char_queue_head] = c;
            self.char_queue_head = next;
        }
        // If the queue is full, silently drop the character.
    }

    /// Pop the next queued character. Returns 0 when the queue is empty.
    /// Called once per frame by the editor's text-input widget.
    pub fn pop_char(&mut self) -> u32 {
        if self.char_queue_tail == self.char_queue_head {
            return 0;
        }
        let c = self.char_queue[self.char_queue_tail];
        self.char_queue_tail = (self.char_queue_tail + 1) % self.char_queue.len();
        c
    }

    pub fn set_mouse_button_down(&mut self, button: usize) { if button < MAX_MOUSE_BUTTONS { self.mouse_down[button] = true; } }
    pub fn set_mouse_button_up(&mut self, button: usize) { if button < MAX_MOUSE_BUTTONS { self.mouse_down[button] = false; } }

    pub fn is_mouse_button_pressed(&self, button: usize) -> bool { button < MAX_MOUSE_BUTTONS && self.mouse_pressed[button] }
    pub fn is_mouse_button_down(&self, button: usize) -> bool { button < MAX_MOUSE_BUTTONS && self.mouse_down[button] }
    pub fn is_mouse_button_released(&self, button: usize) -> bool { button < MAX_MOUSE_BUTTONS && self.mouse_released[button] }

    // Gamepad
    pub fn set_gamepad_axis(&mut self, axis: usize, value: f32) {
        if axis < MAX_GAMEPAD_AXES { self.gamepad_axes[axis] = value; }
    }
    pub fn set_gamepad_button_down(&mut self, button: usize) {
        if button < MAX_GAMEPAD_BUTTONS { self.gamepad_buttons_down[button] = true; }
    }
    pub fn set_gamepad_button_up(&mut self, button: usize) {
        if button < MAX_GAMEPAD_BUTTONS { self.gamepad_buttons_down[button] = false; }
    }

    pub fn is_gamepad_available(&self) -> bool { self.gamepad_available }
    pub fn get_gamepad_axis(&self, axis: usize) -> f32 {
        if axis < MAX_GAMEPAD_AXES { self.gamepad_axes[axis] } else { 0.0 }
    }
    pub fn is_gamepad_button_pressed(&self, button: usize) -> bool {
        button < MAX_GAMEPAD_BUTTONS && self.gamepad_buttons_pressed[button]
    }
    pub fn is_gamepad_button_down(&self, button: usize) -> bool {
        button < MAX_GAMEPAD_BUTTONS && self.gamepad_buttons_down[button]
    }
    pub fn is_gamepad_button_released(&self, button: usize) -> bool {
        button < MAX_GAMEPAD_BUTTONS && self.gamepad_buttons_released[button]
    }
    pub fn get_gamepad_axis_count(&self) -> usize { self.gamepad_axis_count }

    // Touch
    pub fn set_touch(&mut self, index: usize, x: f64, y: f64, active: bool) {
        if index < MAX_TOUCH_POINTS {
            self.touch_points[index] = TouchPoint { x, y, active };
            if active {
                self.touch_pending_release[index] = false;
            }
            self.touch_count = self.touch_points.iter().filter(|t| t.active).count();
        }
    }
    /// Deferred release: keeps the slot active for the current frame and
    /// clears it at `end_frame`. Call from platform event loops on ACTION_UP
    /// so a DOWN+UP that falls inside one frame remains visible to the game.
    pub fn release_touch(&mut self, index: usize, x: f64, y: f64) {
        if index < MAX_TOUCH_POINTS {
            self.touch_points[index].x = x;
            self.touch_points[index].y = y;
            self.touch_pending_release[index] = true;
        }
    }
    pub fn get_touch_x(&self, index: usize) -> f64 {
        if index < MAX_TOUCH_POINTS { self.touch_points[index].x } else { 0.0 }
    }
    pub fn get_touch_y(&self, index: usize) -> f64 {
        if index < MAX_TOUCH_POINTS { self.touch_points[index].y } else { 0.0 }
    }
    pub fn get_touch_count(&self) -> usize { self.touch_count }

    // Digital Crown (watchOS)
    pub fn accumulate_crown_rotation(&mut self, delta: f64) { self.crown_rotation += delta; }
    pub fn consume_crown_rotation(&mut self) -> f64 {
        let v = self.crown_rotation;
        self.crown_rotation = 0.0;
        v
    }

    // Any input
    pub fn is_any_input_pressed(&self) -> bool {
        self.keys_pressed.iter().any(|&k| k)
            || self.mouse_pressed.iter().any(|&m| m)
            || self.gamepad_buttons_pressed.iter().any(|&g| g)
            || self.touch_count > 0
    }
}
