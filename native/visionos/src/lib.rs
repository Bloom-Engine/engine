use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;
use bloom_shared::string_header::str_from_header;

use objc2::encode::{Encode, Encoding, RefEncode};
use objc2::rc::{Allocated, Retained};
use objc2::runtime::{AnyClass, AnyObject, Bool, Sel};
use objc2::{msg_send, sel};

use raw_window_handle::{RawDisplayHandle, RawWindowHandle, UiKitDisplayHandle, UiKitWindowHandle};

use std::ffi::c_void;
use std::sync::OnceLock;

// ============================================================
// objc_msg_lookup shim for the `objc` v0.2 crate
// ============================================================
// The `objc` v0.2 crate (used by `metal` via wgpu-hal) only recognizes
// macOS and iOS as Apple platforms. On tvOS it falls through to the GNUstep
// codepath which expects `objc_msg_lookup`. We shim it to return
// `objc_msgSend`, which on arm64 is the universal message dispatcher.
extern "C" {
    fn objc_msgSend();
    fn objc_msgSendSuper();
}

#[repr(C)]
struct ObjcSuper {
    receiver: *mut c_void,
    super_class: *const c_void,
}

#[no_mangle]
pub unsafe extern "C" fn objc_msg_lookup(
    _receiver: *mut c_void, _sel: *const c_void,
) -> unsafe extern "C" fn() {
    objc_msgSend
}

#[no_mangle]
pub unsafe extern "C" fn objc_msg_lookup_super(
    _sup: *const ObjcSuper, _sel: *const c_void,
) -> unsafe extern "C" fn() {
    objc_msgSendSuper
}

static mut ENGINE: OnceLock<EngineState> = OnceLock::new();
static mut UI_WINDOW: Option<Retained<AnyObject>> = None;
static mut UI_VIEW: Option<Retained<AnyObject>> = None;
static mut TOUCH_MAP: [*const c_void; 10] = [std::ptr::null(); 10];
static mut BUNDLE_PATH: Option<String> = None;
static mut SCREEN_SCALE: f64 = 1.0;
static SCENE_PTR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// Atomic key event buffer: main thread writes, game thread reads before begin_frame
// Bit layout: bits 0-511 = key down, bits 512-1023 = key up (pending)
static PENDING_KEY_DOWN: [std::sync::atomic::AtomicU64; 8] = {
    const INIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    [INIT; 8] // 8 * 64 = 512 bits
};
static PENDING_KEY_UP: [std::sync::atomic::AtomicU64; 8] = {
    const INIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    [INIT; 8]
};

fn pending_key_down(key: usize) {
    if key < 512 {
        PENDING_KEY_DOWN[key / 64].fetch_or(1u64 << (key % 64), std::sync::atomic::Ordering::Release);
    }
}
fn pending_key_up(key: usize) {
    if key < 512 {
        PENDING_KEY_UP[key / 64].fetch_or(1u64 << (key % 64), std::sync::atomic::Ordering::Release);
    }
}
fn drain_pending_keys(eng: &mut EngineState) {
    for i in 0..8 {
        let down = PENDING_KEY_DOWN[i].swap(0, std::sync::atomic::Ordering::Acquire);
        let up = PENDING_KEY_UP[i].swap(0, std::sync::atomic::Ordering::Acquire);
        if down != 0 || up != 0 {
            static DRAIN_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = DRAIN_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 20 {
                std::io::Write::write_all(&mut std::io::stderr(),
                    format!("[bloom-visionos] DRAIN: down=0x{:x} up=0x{:x} bucket={}\n", down, up, i).as_bytes()).ok();
            }
            for bit in 0..64 {
                let key = i * 64 + bit;
                let is_down = down & (1u64 << bit) != 0;
                let is_up = up & (1u64 << bit) != 0;
                if is_down && is_up {
                    // Both pressed and released in same frame — register as down now,
                    // re-queue the up for next frame
                    eng.input.set_key_down(key);
                    pending_key_up(key);
                } else if is_down {
                    eng.input.set_key_down(key);
                } else if is_up {
                    eng.input.set_key_up(key);
                }
            }
        }
    }
}
#[no_mangle]
static SCREEN_DIMS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// Resolve a relative asset path to the app bundle path.
fn resolve_path(path: &str) -> String {
    if path.starts_with('/') {
        return path.to_string();
    }
    unsafe {
        if let Some(ref base) = BUNDLE_PATH {
            format!("{}/{}", base, path)
        } else {
            path.to_string()
        }
    }
}

fn engine() -> &'static mut EngineState {
    unsafe { ENGINE.get_mut().expect("Engine not initialized") }
}
/// Asset-path hook for define_core_ffi! — routes through this platform's
/// resolve_path (relative paths don't resolve from the app working dir here).
fn bloom_resolve_asset_path(path: &str) -> std::borrow::Cow<'_, str> {
    std::borrow::Cow::Owned(resolve_path(path))
}

// The full shared (non-physics) FFI surface. See bloom_shared::ffi_core
// docs for the contract; tools/validate-ffi.js checks parity in CI.
bloom_shared::define_core_ffi!();


// ============================================================
// CG types with objc2 Encode
// ============================================================

#[repr(C)]
#[derive(Copy, Clone)]
struct CGPoint { x: f64, y: f64 }

unsafe impl Encode for CGPoint {
    const ENCODING: Encoding = Encoding::Struct("CGPoint", &[Encoding::Double, Encoding::Double]);
}
unsafe impl RefEncode for CGPoint {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

#[repr(C)]
#[derive(Copy, Clone)]
struct CGSize { width: f64, height: f64 }

unsafe impl Encode for CGSize {
    const ENCODING: Encoding = Encoding::Struct("CGSize", &[Encoding::Double, Encoding::Double]);
}
unsafe impl RefEncode for CGSize {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

#[repr(C)]
#[derive(Copy, Clone)]
struct CGRect { origin: CGPoint, size: CGSize }

unsafe impl Encode for CGRect {
    const ENCODING: Encoding = Encoding::Struct("CGRect", &[CGPoint::ENCODING, CGSize::ENCODING]);
}
unsafe impl RefEncode for CGRect {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

// ============================================================
// ObjC runtime
// ============================================================

extern "C" {
    fn objc_allocateClassPair(superclass: *const AnyClass, name: *const u8, extra_bytes: usize) -> *mut AnyClass;
    fn objc_registerClassPair(cls: *mut AnyClass);
    fn class_addMethod(cls: *mut AnyClass, sel: Sel, imp: *const c_void, types: *const u8) -> bool;

    fn CFRunLoopRunInMode(mode: *const c_void, seconds: f64, return_after: u8) -> i32;
    static kCFRunLoopDefaultMode: *const c_void;
}

fn pump_run_loop(seconds: f64) {
    unsafe { CFRunLoopRunInMode(kCFRunLoopDefaultMode, seconds, 0); }
}

// ============================================================
// Register BloomMetalView — UIView subclass with +layerClass = CAMetalLayer
// ============================================================

unsafe extern "C" fn bloom_layer_class(_cls: *const c_void, _sel: Sel) -> *const c_void {
    AnyClass::get(c"CAMetalLayer").unwrap() as *const AnyClass as *const c_void
}

unsafe extern "C" fn bloom_touches_began(_this: *mut c_void, _sel: Sel, touches: *const AnyObject, _event: *const AnyObject) {
    handle_touches(touches, TouchPhase::Began);
}

unsafe extern "C" fn bloom_touches_moved(_this: *mut c_void, _sel: Sel, touches: *const AnyObject, _event: *const AnyObject) {
    handle_touches(touches, TouchPhase::Moved);
}

unsafe extern "C" fn bloom_touches_ended(_this: *mut c_void, _sel: Sel, touches: *const AnyObject, _event: *const AnyObject) {
    handle_touches(touches, TouchPhase::Ended);
}

unsafe extern "C" fn bloom_touches_cancelled(_this: *mut c_void, _sel: Sel, touches: *const AnyObject, _event: *const AnyObject) {
    handle_touches(touches, TouchPhase::Ended);
}

enum TouchPhase { Began, Moved, Ended }

unsafe fn handle_touches(touches: *const AnyObject, phase: TouchPhase) {
    if touches.is_null() { return; }

    let view_ptr: *const AnyObject = match UI_VIEW.as_ref() {
        Some(v) => Retained::as_ptr(v),
        None => std::ptr::null(),
    };

    let enumerator: Retained<AnyObject> = msg_send![&*touches, objectEnumerator];
    loop {
        let touch: *const AnyObject = msg_send![&*enumerator, nextObject];
        if touch.is_null() { break; }

        let touch_id = touch as *const c_void;
        let loc: CGPoint = msg_send![&*touch, locationInView: view_ptr];

        let index = match phase {
            TouchPhase::Began => {
                let mut slot = None;
                for i in 0..10 {
                    if TOUCH_MAP[i].is_null() {
                        TOUCH_MAP[i] = touch_id;
                        slot = Some(i);
                        break;
                    }
                }
                match slot {
                    Some(i) => i,
                    None => continue,
                }
            }
            TouchPhase::Moved => {
                match TOUCH_MAP.iter().position(|&p| p == touch_id) {
                    Some(i) => i,
                    None => continue,
                }
            }
            TouchPhase::Ended => {
                match TOUCH_MAP.iter().position(|&p| p == touch_id) {
                    Some(i) => {
                        TOUCH_MAP[i] = std::ptr::null();
                        i
                    }
                    None => continue,
                }
            }
        };

        if let Some(eng) = ENGINE.get_mut() {
            let active = !matches!(phase, TouchPhase::Ended);
            // Scale touch from points to pixels to match getScreenWidth/Height
            let sx = loc.x * SCREEN_SCALE;
            let sy = loc.y * SCREEN_SCALE;
            eng.input.set_touch(index, sx, sy, active);

            if index == 0 {
                eng.input.set_mouse_position(sx, sy);
                if active {
                    eng.input.set_mouse_button_down(0);
                } else {
                    eng.input.set_mouse_button_up(0);
                }
            }
        }
    }
}

// tvOS: allow the metal view to become focused (required for remote events)
unsafe extern "C" fn bloom_can_become_focused(_this: *mut c_void, _sel: Sel) -> Bool {
    Bool::YES
}

// tvOS: handle Siri Remote / game controller press events
// UIPressType values: 0=UpArrow, 1=DownArrow, 2=LeftArrow, 3=RightArrow,
//                     4=Select, 5=Menu, 6=PlayPause
unsafe extern "C" fn bloom_presses_began(_this: *mut c_void, _sel: Sel, presses: *const AnyObject, _event: *const AnyObject) {
    // Log to file since stderr doesn't always capture
    static PRESS_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = PRESS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let _ = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/bloom_press_log.txt")
        .and_then(|mut f| std::io::Write::write_all(&mut f, format!("pressesBegan #{}\n", n).as_bytes()));
    handle_presses(presses, true);
}

unsafe extern "C" fn bloom_presses_ended(_this: *mut c_void, _sel: Sel, presses: *const AnyObject, _event: *const AnyObject) {
    handle_presses(presses, false);
}

unsafe fn handle_presses(presses: *const AnyObject, down: bool) {
    if presses.is_null() { return; }

    extern "C" { fn objc_msgSend(); }
    let send_ptr: unsafe extern "C" fn(*const AnyObject, Sel) -> *const AnyObject =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    let send_i64: unsafe extern "C" fn(*const AnyObject, Sel) -> i64 =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());

    let enumerator = send_ptr(presses, sel!(objectEnumerator));
    if enumerator.is_null() { return; }
    loop {
        let press = send_ptr(enumerator, sel!(nextObject));
        if press.is_null() { break; }

        let press_type = send_i64(press, Sel::register(c"type"));
        // Map press types to Bloom Key codes
        // Map press types to Bloom Key codes
        // Remote: 0=Up 1=Down 2=Left 3=Right 4=Select 5=Menu 6=PlayPause
        // Keyboard: 2040=Enter 2041=Escape 2044=Space
        //           2079=Right 2080=Left 2081=Down 2082=Up
        //           2000+HID for other keys
        let key = match press_type {
            0 => Some(256), 1 => Some(257), 2 => Some(258), 3 => Some(259),
            4 => Some(265), 5 => Some(27), 6 => Some(27),
            2040 => Some(265), // Enter
            2041 => Some(27),  // Escape
            2044 => Some(32),  // Space
            2080 => Some(258), // Left arrow
            2079 => Some(259), // Right arrow
            2081 => Some(257), // Down arrow
            2082 => Some(256), // Up arrow
            _ => None,
        };
        if let Some(k) = key {
            if down { pending_key_down(k); } else { pending_key_up(k); }
        } else {
            static UNK_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if UNK_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 30 {
                std::io::Write::write_all(&mut std::io::stderr(),
                    format!("[bloom-visionos] UNKNOWN press type={} down={}\n", press_type, down).as_bytes()).ok();
            }
        }
        // Enter/Select also triggers Space (for jump)
        if press_type == 4 || press_type == 2040 || press_type == 2044 {
            if down { pending_key_down(32); } else { pending_key_up(32); }
        }
    }
}

/// Returns an NSArray containing just the VC's view, so the focus system focuses it.
unsafe extern "C" fn bloom_vc_preferred_focus(this: *mut c_void, _sel: Sel) -> *const AnyObject {
    let view: Retained<AnyObject> = msg_send![&*(this as *const AnyObject), view];
    let arr_cls = AnyClass::get(c"NSArray").unwrap();
    let arr: Retained<AnyObject> = msg_send![arr_cls, arrayWithObject: &*view];
    let ptr = Retained::as_ptr(&arr);
    std::mem::forget(arr);
    ptr
}

static ORIG_SEND_EVENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// BloomApplication.sendEvent: override — intercepts ALL events at the app level
unsafe extern "C" fn bloom_app_send_event(this: *mut c_void, _sel: Sel, event: *const AnyObject) {
    let event_type: i64 = msg_send![&*event, type];
    static APP_EVENT_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let count = APP_EVENT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if count < 100 {
        let subtype: i64 = msg_send![&*event, subtype];
        std::io::Write::write_all(&mut std::io::stderr(),
            format!("[bloom-visionos] APP sendEvent #{} type={} subtype={}\n", count, event_type, subtype).as_bytes()).ok();
    }
    // Let ALL events flow through to the responder chain (UIWindow → VC).
    // BloomViewController's pressesBegan handles press events.
    // BloomWindow's sendEvent handles keyboard events (type 4) by eating them.

    // For non-press/keyboard events, call super
    extern "C" { fn objc_msgSendSuper(); }
    #[repr(C)]
    struct ObjcSuperCall { receiver: *mut c_void, super_class: *const c_void }
    let superclass = AnyClass::get(c"UIApplication").unwrap();
    let sup = ObjcSuperCall { receiver: this, super_class: superclass as *const AnyClass as *const c_void };
    let send_super: unsafe extern "C" fn(*const ObjcSuperCall, Sel, *const AnyObject) = std::mem::transmute(objc_msgSendSuper as unsafe extern "C" fn());
    send_super(&sup, _sel, event);
}

fn register_bloom_application_class() {
    if AnyClass::get(c"BloomApplication").is_some() { return; }

    unsafe {
        let superclass = AnyClass::get(c"UIApplication").unwrap();
        let cls = objc_allocateClassPair(superclass as *const AnyClass, b"BloomApplication\0".as_ptr(), 0);
        if cls.is_null() { return; }

        class_addMethod(cls, sel!(sendEvent:), bloom_app_send_event as *const c_void, b"v24@0:8@16\0".as_ptr());

        objc_registerClassPair(cls);
    }
}

/// Window-level sendEvent: override. Intercepts ALL events (keyboard type 4, presses type 3)
/// and maps them to Bloom key input. Eats keyboard/press events to prevent system dismissal.
unsafe extern "C" fn bloom_window_send_event(this: *mut c_void, sel: Sel, event: *const AnyObject) {
    let event_type: i64 = msg_send![&*event, type];

    // Type 3 = UIPresses, Type 4 = keyboard events from simulator
    if event_type == 3 || event_type == 4 {
        static WIN_EVENT_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let count = WIN_EVENT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count < 10 {
            std::io::Write::write_all(&mut std::io::stderr(),
                format!("[bloom-visionos] window sendEvent type={}\n", event_type).as_bytes()).ok();
        }

        // For press events, extract key info
        if event_type == 3 {
            let all_presses: *const AnyObject = msg_send![&*event, allPresses];
            if !all_presses.is_null() {
                let enumerator: Retained<AnyObject> = msg_send![&*all_presses, objectEnumerator];
                loop {
                    let press: *const AnyObject = msg_send![&*enumerator, nextObject];
                    if press.is_null() { break; }
                    let phase: i64 = msg_send![&*press, phase];
                    let press_type: i64 = {
                        let sel_type = Sel::register(c"type");
                        let send_i64: unsafe extern "C" fn(*const AnyObject, Sel) -> i64 =
                            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
                        send_i64(press, sel_type)
                    };
                    let down = phase == 0;
                    let up = phase == 3 || phase == 4;
                    let key = match press_type {
                        0 => Some(256), 1 => Some(257), 2 => Some(258), 3 => Some(259),
                        4 => Some(265), 5 => Some(27), 6 => Some(27),
                        2040 => Some(265), 2041 => Some(27), 2044 => Some(32),
                        2080 => Some(258), 2079 => Some(259), 2081 => Some(257), 2082 => Some(256),
                        _ => None,
                    };
                    if let Some(k) = key {
                        if down { pending_key_down(k); }
                        if up { pending_key_up(k); }
                    }
                    if press_type == 4 {
                        if down { pending_key_down(32); }
                        if up { pending_key_up(32); }
                    }
                }
            }
        }

        // For keyboard events (type 4), call super to dispatch through the
        // responder chain. BloomViewController.pressesBegan will handle the presses
        // and NOT call super, which prevents the system from dismissing the app.
        // (This only works now that the r#type selector bug is fixed.)
        // For type 4 keyboard events, try multiple selectors to extract key info
        if event_type == 4 {
            // Try _key to get the UIKey object
            let responds_key: Bool = msg_send![&*event, respondsToSelector: sel!(_key)];
            if responds_key.as_bool() {
                let key_obj: *const AnyObject = msg_send![&*event, _key];
                if !key_obj.is_null() {
                    let responds_keycode: Bool = msg_send![&*key_obj, respondsToSelector: sel!(keyCode)];
                    if responds_keycode.as_bool() {
                        let keycode: i64 = msg_send![&*key_obj, keyCode];
                        let is_down: Bool = msg_send![&*event, _isKeyDown];
                        let bloom_key = match keycode {
                            79 => Some(259), 80 => Some(258), 81 => Some(257), 82 => Some(256),
                            40 => Some(265), 41 => Some(27), 44 => Some(32), _ => None,
                        };
                        if let Some(k) = bloom_key {
                            if is_down.as_bool() { pending_key_down(k); }
                            else { pending_key_up(k); }
                        }
                        if keycode == 40 || keycode == 44 {
                            if is_down.as_bool() { pending_key_down(32); }
                            else { pending_key_up(32); }
                        }
                    }
                }
            }
            // If _key worked, we're done. If not, fall through to call super
            // so the responder chain (BloomViewController.pressesBegan) handles it.
            if responds_key.as_bool() { return; }
            // Call super for type 4 — dispatches to VC's pressesBegan
            extern "C" { fn objc_msgSendSuper(); }
            #[repr(C)]
            struct Sup2 { receiver: *mut c_void, super_class: *const c_void }
            let sc2 = AnyClass::get(c"UIWindow").unwrap();
            let s2 = Sup2 { receiver: this, super_class: sc2 as *const AnyClass as *const c_void };
            let f2: unsafe extern "C" fn(*const Sup2, Sel, *const AnyObject) =
                std::mem::transmute(objc_msgSendSuper as unsafe extern "C" fn());
            f2(&s2, sel, event);
            return;
        }

        if false {
            let all_presses: *const AnyObject = msg_send![&*event, allPresses];
            if !all_presses.is_null() {
                let enumerator: Retained<AnyObject> = msg_send![&*all_presses, objectEnumerator];
                loop {
                    let press: *const AnyObject = msg_send![&*enumerator, nextObject];
                    if press.is_null() { break; }
                    let phase: i64 = msg_send![&*press, phase];
                    let press_type: i64 = {
                        let sel_type = Sel::register(c"type");
                        let send_i64: unsafe extern "C" fn(*const AnyObject, Sel) -> i64 =
                            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
                        send_i64(press, sel_type)
                    };
                    let eng_ptr = ENGINE.get().map(|e| e as *const EngineState as *mut EngineState);
                    if let Some(eng) = eng_ptr.map(|p| &mut *p) {
                        let down = phase == 0;
                        let up = phase == 3 || phase == 4;
                        // press_type for keyboard keys are large numbers (e.g. 2227)
                        // Map common ones: arrows, enter, escape, space
                        let key = match press_type {
                            0 => Some(256),    // Up arrow (remote)
                            1 => Some(257),    // Down arrow (remote)
                            2 => Some(258),    // Left arrow (remote)
                            3 => Some(259),    // Right arrow (remote)
                            4 => Some(265),    // Select (remote) → Enter
                            5 => Some(27),     // Menu (remote) → Escape
                            6 => Some(27),     // PlayPause → Escape
                            2227 => Some(265), // Keyboard Enter
                            2233 => Some(27),  // Keyboard Escape
                            2232 => Some(32),  // Keyboard Space
                            2228 => Some(9),   // Keyboard Tab
                            2103 => Some(256), // Keyboard Up
                            2105 => Some(257), // Keyboard Down
                            2104 => Some(258), // Keyboard Left
                            2106 => Some(259), // Keyboard Right
                            _ => None,
                        };
                        if let Some(k) = key {
                            if down { eng.input.set_key_down(k); }
                            if up { eng.input.set_key_up(k); }
                        }
                        // Select/Enter also maps to Space for jump
                        if press_type == 4 || press_type == 2227 {
                            if down { eng.input.set_key_down(32); }
                            if up { eng.input.set_key_up(32); }
                        }
                    }
                }
            }
        }

        return; // Don't call super for type 3 — we extracted keys above
    }

    // For other events, call super
    extern "C" { fn objc_msgSendSuper(); }
    #[repr(C)]
    struct ObjcSuperCall { receiver: *mut c_void, super_class: *const c_void }
    let superclass = AnyClass::get(c"UIWindow").unwrap();
    let sup = ObjcSuperCall { receiver: this, super_class: superclass as *const AnyClass as *const c_void };
    let send_super: unsafe extern "C" fn(*const ObjcSuperCall, Sel, *const AnyObject) =
        std::mem::transmute(objc_msgSendSuper as unsafe extern "C" fn());
    send_super(&sup, sel, event);
}

fn register_window_class() {
    if AnyClass::get(c"BloomWindow").is_some() { return; }

    unsafe {
        let superclass = AnyClass::get(c"UIWindow").unwrap();
        let cls = objc_allocateClassPair(superclass as *const AnyClass, b"BloomWindow\0".as_ptr(), 0);
        if cls.is_null() { return; }

        // Override sendEvent: to intercept ALL events at the window level
        class_addMethod(cls, sel!(sendEvent:), bloom_window_send_event as *const c_void, b"v24@0:8@16\0".as_ptr());

        // Also override press events for responder chain
        let press_types = b"v32@0:8@16@24\0".as_ptr();
        class_addMethod(cls, sel!(pressesBegan:withEvent:), bloom_presses_began as *const c_void, press_types);
        class_addMethod(cls, sel!(pressesEnded:withEvent:), bloom_presses_ended as *const c_void, press_types);
        class_addMethod(cls, sel!(pressesCancelled:withEvent:), bloom_presses_ended as *const c_void, press_types);

        objc_registerClassPair(cls);
    }
}

fn register_view_controller_class() {
    if AnyClass::get(c"BloomViewController").is_some() { return; }

    unsafe {
        // Plain UIViewController subclass — matches what works in the Swift test
        let superclass = AnyClass::get(c"UIViewController").unwrap();
        let cls = objc_allocateClassPair(superclass as *const AnyClass, b"BloomViewController\0".as_ptr(), 0);
        if cls.is_null() { return; }

        // Override pressesBegan/pressesEnded to capture remote/keyboard events
        let press_types = b"v32@0:8@16@24\0".as_ptr();
        class_addMethod(cls, sel!(pressesBegan:withEvent:), bloom_presses_began as *const c_void, press_types);
        class_addMethod(cls, sel!(pressesEnded:withEvent:), bloom_presses_ended as *const c_void, press_types);
        class_addMethod(cls, sel!(pressesCancelled:withEvent:), bloom_presses_ended as *const c_void, press_types);

        objc_registerClassPair(cls);
    }
}

fn register_metal_view_class() {
    if AnyClass::get(c"BloomMetalView").is_some() { return; }

    unsafe {
        let superclass = AnyClass::get(c"UIView").unwrap();
        let cls = objc_allocateClassPair(superclass as *const AnyClass, b"BloomMetalView\0".as_ptr(), 0);
        if cls.is_null() { return; }

        extern "C" { fn object_getClass(obj: *const c_void) -> *mut AnyClass; }
        let meta = object_getClass(cls as *const c_void);

        class_addMethod(meta, sel!(layerClass), bloom_layer_class as *const c_void, b"#8@0:8\0".as_ptr());

        let touch_types = b"v32@0:8@16@24\0".as_ptr();
        class_addMethod(cls, sel!(touchesBegan:withEvent:), bloom_touches_began as *const c_void, touch_types);
        class_addMethod(cls, sel!(touchesMoved:withEvent:), bloom_touches_moved as *const c_void, touch_types);
        class_addMethod(cls, sel!(touchesEnded:withEvent:), bloom_touches_ended as *const c_void, touch_types);
        class_addMethod(cls, sel!(touchesCancelled:withEvent:), bloom_touches_cancelled as *const c_void, touch_types);

        // tvOS focus engine — view must be focusable to receive remote events
        class_addMethod(cls, sel!(canBecomeFocused), bloom_can_become_focused as *const c_void, b"B8@0:8\0".as_ptr());
        class_addMethod(cls, sel!(canBecomeFirstResponder), bloom_can_become_focused as *const c_void, b"B8@0:8\0".as_ptr());

        // tvOS press events — Siri Remote physical buttons
        let press_types = b"v32@0:8@16@24\0".as_ptr();
        class_addMethod(cls, sel!(pressesBegan:withEvent:), bloom_presses_began as *const c_void, press_types);
        class_addMethod(cls, sel!(pressesEnded:withEvent:), bloom_presses_ended as *const c_void, press_types);

        objc_registerClassPair(cls);
    }
}

// ============================================================
// Scene delegate — creates UIWindow + Metal view + wgpu engine
// ============================================================

unsafe extern "C" fn scene_will_connect(
    _this: *mut c_void,
    _sel: Sel,
    scene: *const AnyObject,
    _session: *const AnyObject,
    _options: *const AnyObject,
) {
    let _ = std::fs::write("/tmp/bloom_scene_connect.txt", format!("scene_will_connect called, scene={:?}\n", scene));
    if scene.is_null() { return; }
    std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] scene_will_connect called\n").ok();

    // Get screen bounds
    let screen_cls = AnyClass::get(c"UIScreen").unwrap();
    let screen: Retained<AnyObject> = msg_send![screen_cls, mainScreen];
    let bounds: CGRect = msg_send![&*screen, bounds];
    let scale: f64 = msg_send![&*screen, scale];
    let pixel_width = (bounds.size.width * scale) as u32;
    let pixel_height = (bounds.size.height * scale) as u32;
    SCREEN_SCALE = scale;

    // Create BloomWindow (captures press events) attached to the scene
    let window_cls = AnyClass::get(c"BloomWindow").unwrap();
    let window: Allocated<AnyObject> = msg_send![window_cls, alloc];
    let window: Retained<AnyObject> = msg_send![window, initWithWindowScene: scene];

    // Use GCEventViewController directly — it prevents the system from
    // intercepting remote/keyboard events when controllerUserInteractionEnabled=NO
    let gc_vc_cls = AnyClass::get(c"GCEventViewController");
    std::io::Write::write_all(&mut std::io::stderr(),
        format!("[bloom-visionos] GCEventViewController available: {}\n", gc_vc_cls.is_some()).as_bytes()).ok();
    let vc_cls = gc_vc_cls.unwrap_or_else(|| AnyClass::get(c"UIViewController").unwrap());
    let vc: Allocated<AnyObject> = msg_send![vc_cls, alloc];
    let vc: Retained<AnyObject> = msg_send![vc, init];
    if gc_vc_cls.is_some() {
        let _: () = msg_send![&*vc, setControllerUserInteractionEnabled: Bool::NO];
        std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] controllerUserInteractionEnabled = NO\n").ok();
    }

    // Create BloomMetalView
    let view_cls = AnyClass::get(c"BloomMetalView").unwrap();
    let view: Allocated<AnyObject> = msg_send![view_cls, alloc];
    let view: Retained<AnyObject> = msg_send![view, initWithFrame: bounds];

    // Set background to black
    let color_cls = AnyClass::get(c"UIColor").unwrap();
    let black: Retained<AnyObject> = msg_send![color_cls, blackColor];
    let _: () = msg_send![&*view, setBackgroundColor: &*black];

    // Configure CAMetalLayer - set framebufferOnly=NO for screenshot capture
    let layer: Retained<AnyObject> = msg_send![&*view, layer];
    let drawable_size = CGSize { width: pixel_width as f64, height: pixel_height as f64 };
    let _: () = msg_send![&*layer, setDrawableSize: drawable_size];
    let _: () = msg_send![&*layer, setContentsScale: scale];
    let _: () = msg_send![&*layer, setOpaque: Bool::YES];
    let _: () = msg_send![&*layer, setFramebufferOnly: Bool::NO];
    // presentsWithTransaction MUST stay NO (the default). wgpu presents its
    // drawable asynchronously via -presentDrawable: on the command buffer;
    // setting presentsWithTransaction:YES makes CoreAnimation wait for a
    // synchronous CATransaction commit that wgpu never performs, so the layer
    // never displays rendered frames (the screen stays black behind UIKit subviews).
    let _: () = msg_send![&*layer, setPresentsWithTransaction: Bool::NO];

    // Enable touches & focus
    let _: () = msg_send![&*view, setUserInteractionEnabled: Bool::YES];
    let _: () = msg_send![&*view, setMultipleTouchEnabled: Bool::YES];

    // Set up window hierarchy
    let _: () = msg_send![&*vc, setView: &*view];
    let _: () = msg_send![&*window, setRootViewController: &*vc];
    let _: () = msg_send![&*window, makeKeyAndVisible];

    UI_VIEW = Some(view.clone());
    UI_WINDOW = Some(window);

    // Create wgpu surface and engine on the main thread (like iOS)
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let view_ptr = Retained::as_ptr(&view) as *mut c_void;
    let handle = UiKitWindowHandle::new(
        std::ptr::NonNull::new(view_ptr).unwrap(),
    );
    let raw = RawWindowHandle::UiKit(handle);
    std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] creating wgpu surface\n").ok();
    let surface = match instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
        raw_display_handle: Some(RawDisplayHandle::UiKit(UiKitDisplayHandle::new())),
        raw_window_handle: raw,
    }) {
        Ok(s) => s,
        Err(e) => panic!("[bloom-visionos] Failed to create wgpu surface: {e}"),
    };

    let adapter = match pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })) {
        Ok(a) => a,
        Err(_) => panic!("[bloom-visionos] No GPU adapter found"),
    };

    // Ticket 007b: HW ray-query on RT-capable tvOS hardware (A13+).
    let supported = adapter.features();
    let force_sw_gi = std::env::var("BLOOM_FORCE_SW_GI")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let rt_mask = wgpu::Features::EXPERIMENTAL_RAY_QUERY;
    let mut required_features = wgpu::Features::empty();
    // Ticket 011: request TIMESTAMP_QUERY when supported so the profiler
    // can record GPU timings. Optional — profiler falls back to CPU-only
    // when the adapter doesn't grant it.
    if supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required_features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    // Cooked BC7 textures (bloom-cook) upload compressed when the
    // adapter has BC support; without it they CPU-decode at load.
    if supported.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        required_features |= wgpu::Features::TEXTURE_COMPRESSION_BC;
    }
    if !force_sw_gi && supported.contains(rt_mask) {
        required_features |= rt_mask;
    }
    let experimental_features = if required_features.intersects(rt_mask) {
        unsafe { wgpu::ExperimentalFeatures::enabled() }
    } else {
        wgpu::ExperimentalFeatures::disabled()
    };
    // Base the requested limits on what the adapter actually advertises so we
    // never ask for more than the backend can grant. The tvOS/iOS *simulators*
    // cap several limits below wgpu's desktop default() — notably
    // max_inter_stage_shader_variables (15 vs 16) — which makes request_device
    // fail with LimitsExceeded. On real Apple TV hardware adapter.limits()
    // meets or exceeds default(), so behaviour there is unchanged.
    let adapter_limits = adapter.limits();
    let mut required_limits = wgpu::Limits::default();
    required_limits.max_inter_stage_shader_variables = required_limits
        .max_inter_stage_shader_variables
        .min(adapter_limits.max_inter_stage_shader_variables);
    if required_features.intersects(rt_mask) {
        required_limits = required_limits
            .using_minimum_supported_acceleration_structure_values();
    }
    let (device, queue) = match pollster_block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("bloom_device"),
            required_features,
            required_limits,
            experimental_features,
            ..Default::default()
        },
    )) {
        Ok(dq) => dq,
        Err(e) => panic!("[bloom-visionos] Failed to create device: {e}"),
    };

    let surface_caps = surface.get_capabilities(&adapter);
    // Use non-sRGB format to match game's sRGB color space (colors are specified as sRGB 0-255)
    let format = surface_caps.formats.iter()
        .find(|f| !f.is_srgb()).copied()
        .unwrap_or(surface_caps.formats[0]);

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: pixel_width,
        height: pixel_height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &surface_config);

    let renderer = Renderer::new(device, queue, surface, surface_config, pixel_width, pixel_height);
    let _ = ENGINE.set(EngineState::new(renderer));
    std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] ENGINE created on main thread\n").ok();
}

/// Called by the perry runtime's main() before UIApplicationMain to register
/// ObjC classes needed for the scene lifecycle.
#[no_mangle]
unsafe extern "C" fn configuration_for_connecting_scene(
    _this: *mut c_void, _sel: Sel,
    _app: *const AnyObject,
    scene_session: *const AnyObject,
    _options: *const AnyObject,
) -> *const AnyObject {
    let _ = std::fs::write("/tmp/bloom_config_scene.txt", "configurationForConnecting called\n");
    // Get the session's role
    let role: *const AnyObject = msg_send![&*scene_session, role];
    // Create UISceneConfiguration
    let config_cls = AnyClass::get(c"UISceneConfiguration").unwrap();
    let ns_cls = AnyClass::get(c"NSString").unwrap();
    let name_str: Retained<AnyObject> = msg_send![ns_cls, stringWithUTF8String: b"Default Configuration\0".as_ptr()];
    let config: Allocated<AnyObject> = msg_send![config_cls, alloc];
    let config: Retained<AnyObject> = msg_send![config, initWithName: &*name_str sessionRole: role];
    // Use PerryGameLoopAppDelegate as scene delegate (it has scene:willConnectToSession: added)
    let delegate_cls = AnyClass::get(c"PerryGameLoopAppDelegate").unwrap();
    let _: () = msg_send![&*config, setDelegateClass: delegate_cls];
    std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] returning scene config with delegate\n").ok();
    // Leak the config — UIKit retains it. We can't use autorelease pool from extern C.
    let ptr = Retained::as_ptr(&config);
    std::mem::forget(config);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn perry_register_native_classes() {
    let _ = std::fs::write("/tmp/bloom_register_classes.txt", "perry_register_native_classes called\n");
    register_bloom_application_class();
    register_metal_view_class();
    register_window_class();
    register_view_controller_class();
    register_scene_delegate();

    // Add press event handlers AND configurationForConnectingSceneSession to the app delegate
    if let Some(app_delegate_cls) = AnyClass::get(c"PerryGameLoopAppDelegate") {
        let press_types = b"v32@0:8@16@24\0".as_ptr();
        class_addMethod(
            app_delegate_cls as *const AnyClass as *mut AnyClass,
            sel!(pressesBegan:withEvent:),
            bloom_presses_began as *const c_void,
            press_types,
        );
        class_addMethod(
            app_delegate_cls as *const AnyClass as *mut AnyClass,
            sel!(pressesEnded:withEvent:),
            bloom_presses_ended as *const c_void,
            press_types,
        );
        let sel = Sel::register(c"application:configurationForConnectingSceneSession:options:");
        let types = b"@48@0:8@16@24@32\0".as_ptr();
        class_addMethod(
            app_delegate_cls as *const AnyClass as *mut AnyClass,
            sel,
            configuration_for_connecting_scene as *const c_void,
            types,
        );

        // Also add scene:willConnectToSession:connectionOptions: to the app delegate
        // so it can act as its own scene delegate (PerrySceneDelegate dispatch never fires)
        let scene_sel = Sel::register(c"scene:willConnectToSession:connectionOptions:");
        let scene_types = b"v48@0:8@16@24@32\0".as_ptr();
        class_addMethod(
            app_delegate_cls as *const AnyClass as *mut AnyClass,
            scene_sel,
            scene_will_connect as *const c_void,
            scene_types,
        );

        // Add UIWindowSceneDelegate protocol to app delegate
        extern "C" { fn objc_getProtocol(name: *const u8) -> *const c_void; }
        extern "C" { fn class_addProtocol(cls: *mut c_void, protocol: *const c_void) -> bool; }
        let protocol = objc_getProtocol(b"UIWindowSceneDelegate\0".as_ptr());
        if !protocol.is_null() {
            class_addProtocol(app_delegate_cls as *const AnyClass as *mut c_void, protocol);
        }
    }
    std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] classes registered, scene delegate ready\n").ok();
}

/// Called by the runtime's scene delegate when the UIWindowScene connects.
/// This runs on the main thread — safe for all UIKit operations.
#[no_mangle]
unsafe extern "C" fn deferred_init(_ctx: *mut c_void) {
    let _ = std::fs::write("/tmp/bloom_deferred_init.txt", "deferred_init called\n");

    // Find the connected scene
    let app_cls = AnyClass::get(c"UIApplication").unwrap();
    let app: *const AnyObject = msg_send![app_cls, sharedApplication];
    let scenes: *const AnyObject = msg_send![&*app, connectedScenes];
    let count: usize = msg_send![&*scenes, count];
    if count == 0 {
        let _ = std::fs::write("/tmp/bloom_deferred_2.txt", "scenes=0, retrying in 500ms\n");
        // Retry after a delay
        extern "C" {
            static _dispatch_main_q: c_void;
            fn dispatch_after_f(when: u64, queue: *const c_void, context: *mut c_void, work: unsafe extern "C" fn(*mut c_void));
            fn dispatch_time(when: u64, delta: i64) -> u64;
        }
        let when = dispatch_time(0, 500_000_000); // 500ms
        dispatch_after_f(when, &_dispatch_main_q as *const _, std::ptr::null_mut(), deferred_init);
        return;
    }

    let scene: Retained<AnyObject> = msg_send![&*scenes, anyObject];
    // Log the scene class name for debugging
    extern "C" { fn class_getName(cls: *const c_void) -> *const u8; }
    let scene_class: *const c_void = msg_send![&*scene, class];
    let scene_class_name = std::ffi::CStr::from_ptr(class_getName(scene_class) as *const i8).to_str().unwrap_or("?");
    let scene_state: i64 = msg_send![&*scene, activationState];
    let _ = std::fs::write("/tmp/bloom_deferred_2.txt", format!("scenes={}\nclass={}\nactivationState={}\n", count, scene_class_name, scene_state));

    // Get screen dimensions
    let screen_cls = AnyClass::get(c"UIScreen").unwrap();
    let screen: Retained<AnyObject> = msg_send![screen_cls, mainScreen];
    let bounds: CGRect = msg_send![&*screen, bounds];
    let scale: f64 = msg_send![&*screen, scale];
    let pixel_width = (bounds.size.width * scale) as u32;
    let pixel_height = (bounds.size.height * scale) as u32;
    SCREEN_SCALE = scale;

    // Create window WITH the scene (required on tvOS for visibility)
    let window_cls = AnyClass::get(c"BloomWindow").unwrap();
    let w: Allocated<AnyObject> = msg_send![window_cls, alloc];
    let window: Retained<AnyObject> = msg_send![w, initWithWindowScene: &*scene];

    // Create view controller
    let gc_vc_cls = AnyClass::get(c"GCEventViewController");
    let vc_cls = gc_vc_cls.unwrap_or_else(|| AnyClass::get(c"UIViewController").unwrap());
    let vc: Allocated<AnyObject> = msg_send![vc_cls, alloc];
    let vc: Retained<AnyObject> = msg_send![vc, init];
    if gc_vc_cls.is_some() {
        let _: () = msg_send![&*vc, setControllerUserInteractionEnabled: Bool::NO];
    }

    // Create BloomMetalView
    let view_cls = AnyClass::get(c"BloomMetalView").unwrap();
    let v: Allocated<AnyObject> = msg_send![view_cls, alloc];
    let view: Retained<AnyObject> = msg_send![v, initWithFrame: bounds];

    let color_cls = AnyClass::get(c"UIColor").unwrap();
    let black: Retained<AnyObject> = msg_send![color_cls, blackColor];
    let _: () = msg_send![&*view, setBackgroundColor: &*black];
    let _: () = msg_send![&*view, setUserInteractionEnabled: Bool::YES];

    // Configure CAMetalLayer
    let layer: Retained<AnyObject> = msg_send![&*view, layer];
    let drawable_size = CGSize { width: pixel_width as f64, height: pixel_height as f64 };
    let _: () = msg_send![&*layer, setDrawableSize: drawable_size];
    let _: () = msg_send![&*layer, setContentsScale: scale];
    let _: () = msg_send![&*layer, setOpaque: Bool::YES];
    let _: () = msg_send![&*layer, setFramebufferOnly: Bool::NO];
    // presentsWithTransaction MUST stay NO (the default). wgpu presents its
    // drawable asynchronously via -presentDrawable: on the command buffer;
    // setting presentsWithTransaction:YES makes CoreAnimation wait for a
    // synchronous CATransaction commit that wgpu never performs, so the layer
    // never displays rendered frames (the screen stays black behind UIKit subviews).
    let _: () = msg_send![&*layer, setPresentsWithTransaction: Bool::NO];

    // Set up window hierarchy
    let _: () = msg_send![&*vc, setView: &*view];
    let _: () = msg_send![&*window, setRootViewController: &*vc];
    let _: () = msg_send![&*window, makeKeyAndVisible];

    UI_VIEW = Some(view.clone());
    UI_WINDOW = Some(window.clone());

    // Verify window state
    let is_key: Bool = msg_send![&*window, isKeyWindow];
    let is_hidden: Bool = msg_send![&*window, isHidden];
    let win_scene: *const AnyObject = msg_send![&*window, windowScene];
    let win_frame: CGRect = msg_send![&*window, frame];
    let win_alpha: f64 = msg_send![&*window, alpha];
    let root_vc: *const AnyObject = msg_send![&*window, rootViewController];
    let vc_view: *const AnyObject = if !root_vc.is_null() { msg_send![&*root_vc, view] } else { std::ptr::null() };

    // Check scene's windows
    let scene_windows: *const AnyObject = msg_send![&*scene, windows];
    let scene_win_count: usize = msg_send![&*scene_windows, count];

    let debug = format!(
        "window+view created with scene, {}x{}\n\
         isKey={} isHidden={} alpha={}\n\
         windowScene={:?} (expected={:?})\n\
         frame=({},{},{},{})\n\
         rootVC={:?} vcView={:?}\n\
         scene.windows.count={}\n",
        pixel_width, pixel_height,
        is_key.as_bool(), is_hidden.as_bool(), win_alpha,
        win_scene, Retained::as_ptr(&scene),
        win_frame.origin.x, win_frame.origin.y, win_frame.size.width, win_frame.size.height,
        root_vc, vc_view,
        scene_win_count,
    );
    let _ = std::fs::write("/tmp/bloom_deferred_3.txt", &debug);

    // Create wgpu engine using CAMetalLayer
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let layer_ptr = Retained::as_ptr(&layer) as *mut c_void;
    let surface = instance.create_surface_unsafe(
        wgpu::SurfaceTargetUnsafe::CoreAnimationLayer(layer_ptr)
    ).expect("[bloom-visionos] Failed to create wgpu surface");

    let adapter = pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })).expect("[bloom-visionos] No GPU adapter found");

    // Ticket 007b: HW ray-query on RT-capable tvOS hardware (A13+).
    let supported = adapter.features();
    let force_sw_gi = std::env::var("BLOOM_FORCE_SW_GI")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let rt_mask = wgpu::Features::EXPERIMENTAL_RAY_QUERY;
    let mut required_features = wgpu::Features::empty();
    // Ticket 011: request TIMESTAMP_QUERY when supported so the profiler
    // can record GPU timings. Optional — profiler falls back to CPU-only
    // when the adapter doesn't grant it.
    if supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required_features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    // Cooked BC7 textures (bloom-cook) upload compressed when the
    // adapter has BC support; without it they CPU-decode at load.
    if supported.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        required_features |= wgpu::Features::TEXTURE_COMPRESSION_BC;
    }
    if !force_sw_gi && supported.contains(rt_mask) {
        required_features |= rt_mask;
    }
    let experimental_features = if required_features.intersects(rt_mask) {
        unsafe { wgpu::ExperimentalFeatures::enabled() }
    } else {
        wgpu::ExperimentalFeatures::disabled()
    };
    // Base the requested limits on what the adapter actually advertises so we
    // never ask for more than the backend can grant. The tvOS/iOS *simulators*
    // cap several limits below wgpu's desktop default() — notably
    // max_inter_stage_shader_variables (15 vs 16) — which makes request_device
    // fail with LimitsExceeded. On real Apple TV hardware adapter.limits()
    // meets or exceeds default(), so behaviour there is unchanged.
    let adapter_limits = adapter.limits();
    let mut required_limits = wgpu::Limits::default();
    required_limits.max_inter_stage_shader_variables = required_limits
        .max_inter_stage_shader_variables
        .min(adapter_limits.max_inter_stage_shader_variables);
    if required_features.intersects(rt_mask) {
        required_limits = required_limits
            .using_minimum_supported_acceleration_structure_values();
    }
    let (device, queue) = pollster_block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("bloom_device"),
            required_features,
            required_limits,
            experimental_features,
            ..Default::default()
        },
    )).expect("[bloom-visionos] Failed to create device");

    let surface_caps = surface.get_capabilities(&adapter);
    let format = surface_caps.formats.iter()
        .find(|f| !f.is_srgb()).copied()
        .unwrap_or(surface_caps.formats[0]);

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: pixel_width,
        height: pixel_height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &surface_config);

    let renderer = Renderer::new(device, queue, surface, surface_config, pixel_width, pixel_height);
    let _ = ENGINE.set(EngineState::new(renderer));
    let _ = std::fs::write("/tmp/bloom_tvos_debug.txt", format!("ENGINE created\nformat={:?}\nsize={}x{}\n", format, pixel_width, pixel_height));
}

#[no_mangle]
pub unsafe extern "C" fn perry_scene_will_connect(scene: *const c_void) {
    std::io::Write::write_all(&mut std::io::stderr(), format!("[bloom-visionos] perry_scene_will_connect scene={:?}\n", scene).as_bytes()).ok();
    // When scene is null (called from didFinishLaunchingWithOptions), create the
    // window synchronously so UIKit knows we handle events. Then dispatch async
    // to attach to the scene (needed for Metal rendering).
    if scene.is_null() {
        // On tvOS, windows MUST be attached to a UIWindowScene to be visible.
        // Dispatch deferred_init with a delay so the scene is fully connected.
        extern "C" {
            static _dispatch_main_q: c_void;
            fn dispatch_after_f(when: u64, queue: *const c_void, context: *mut c_void, work: unsafe extern "C" fn(*mut c_void));
            fn dispatch_time(when: u64, delta: i64) -> u64;
        }
        let when = dispatch_time(0, 500_000_000); // 500ms delay
        dispatch_after_f(when, &_dispatch_main_q as *const _, std::ptr::null_mut(), deferred_init);
        return;
    }

    let screen_cls = AnyClass::get(c"UIScreen").expect("[bloom-visionos] UIScreen class not found");
    let screen: Retained<AnyObject> = msg_send![screen_cls, mainScreen];
    let bounds: CGRect = msg_send![&*screen, bounds];
    let scale: f64 = msg_send![&*screen, scale];
    eprintln!("[bloom-visionos] screen bounds: {}x{}, scale={}", bounds.size.width, bounds.size.height, scale);

    let pixel_width = (bounds.size.width * scale) as u32;
    let pixel_height = (bounds.size.height * scale) as u32;

    // Store scale for touch coordinate conversion (points → pixels)
    SCREEN_SCALE = scale;

    // Create BloomWindow — attached to scene if available, otherwise plain
    let window_cls = AnyClass::get(c"BloomWindow").unwrap();
    let window: Retained<AnyObject> = if !scene.is_null() {
        let w: Allocated<AnyObject> = msg_send![window_cls, alloc];
        msg_send![w, initWithWindowScene: scene as *const AnyObject]
    } else {
        let w: Allocated<AnyObject> = msg_send![window_cls, alloc];
        msg_send![w, initWithFrame: bounds]
    };

    // Use GCEventViewController to prevent system from intercepting remote events
    let gc_vc_cls = AnyClass::get(c"GCEventViewController");
    std::io::Write::write_all(&mut std::io::stderr(),
        format!("[bloom-visionos] GCEventViewController available: {}\n", gc_vc_cls.is_some()).as_bytes()).ok();
    let vc_cls = gc_vc_cls.unwrap_or_else(|| AnyClass::get(c"UIViewController").unwrap());
    let vc: Allocated<AnyObject> = msg_send![vc_cls, alloc];
    let vc: Retained<AnyObject> = msg_send![vc, init];
    if gc_vc_cls.is_some() {
        let _: () = msg_send![&*vc, setControllerUserInteractionEnabled: Bool::NO];
        std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] controllerUserInteractionEnabled = NO\n").ok();
    }

    // Create BloomMetalView
    eprintln!("[bloom-visionos] creating BloomMetalView");
    let view_cls = AnyClass::get(c"BloomMetalView").expect("[bloom-visionos] BloomMetalView class not found");
    let view: Allocated<AnyObject> = msg_send![view_cls, alloc];
    let view: Retained<AnyObject> = msg_send![view, initWithFrame: bounds];

    // Set background to black
    let color_cls = AnyClass::get(c"UIColor").unwrap();
    let black: Retained<AnyObject> = msg_send![color_cls, blackColor];
    let _: () = msg_send![&*view, setBackgroundColor: &*black];

    // Configure CAMetalLayer
    let layer: Retained<AnyObject> = msg_send![&*view, layer];
    let drawable_size = CGSize { width: pixel_width as f64, height: pixel_height as f64 };
    let _: () = msg_send![&*layer, setDrawableSize: drawable_size];
    let _: () = msg_send![&*layer, setContentsScale: scale];
    let _: () = msg_send![&*layer, setOpaque: Bool::YES];

    // Enable touches
    let _: () = msg_send![&*view, setUserInteractionEnabled: Bool::YES];
    let _: () = msg_send![&*view, setMultipleTouchEnabled: Bool::YES];

    // Set up window hierarchy
    let _: () = msg_send![&*vc, setView: &*view];
    let _: () = msg_send![&*window, setRootViewController: &*vc];
    let _: () = msg_send![&*window, makeKeyAndVisible];

    // Store references
    UI_VIEW = Some(view.clone());
    UI_WINDOW = Some(window);
    // Add a transparent focusable button so the tvOS focus engine has something to focus
    // Without a focused element, tvOS suspends the app on any remote button press
    let btn_cls = AnyClass::get(c"UIButton").unwrap();
    let btn: Retained<AnyObject> = msg_send![btn_cls, buttonWithType: 0i64]; // UIButtonTypeCustom
    let _: () = msg_send![&*btn, setFrame: bounds];
    let _: () = msg_send![&*btn, setAlpha: 0.0f64]; // fully transparent
    let _: () = msg_send![&*view, addSubview: &*btn];

    // Add a menu gesture recognizer to prevent the system from dismissing on Menu press
    let tap_cls = AnyClass::get(c"UITapGestureRecognizer").unwrap();
    let menu_tap: Allocated<AnyObject> = msg_send![tap_cls, alloc];
    let menu_tap: Retained<AnyObject> = msg_send![menu_tap, initWithTarget: std::ptr::null::<AnyObject>() action: std::ptr::null::<c_void>()];
    // allowedPressTypes = @[@(UIPressTypeMenu)] = @[@5]
    let num_cls = AnyClass::get(c"NSNumber").unwrap();
    let menu_num: Retained<AnyObject> = msg_send![num_cls, numberWithInteger: 5i64];
    let arr_cls = AnyClass::get(c"NSArray").unwrap();
    let press_types_arr: Retained<AnyObject> = msg_send![arr_cls, arrayWithObject: &*menu_num];
    let _: () = msg_send![&*menu_tap, setAllowedPressTypes: &*press_types_arr];
    let _: () = msg_send![&*view, addGestureRecognizer: &*menu_tap];

    // Trigger focus
    let _: () = msg_send![&*vc, setNeedsFocusUpdate];
    let _: () = msg_send![&*vc, updateFocusIfNeeded];
    let _: () = msg_send![&*view, becomeFirstResponder];
    let is_focused: Bool = msg_send![&*view, isFocused];
    std::io::Write::write_all(&mut std::io::stderr(),
        format!("[bloom-visionos] view.isFocused={}\n", is_focused.as_bool()).as_bytes()).ok();
    // Verify GCEventViewController state
    {
        extern "C" { fn class_getName(cls: *const c_void) -> *const u8; }
        let vc_class: *const c_void = msg_send![&*vc, class];
        let vc_name = std::ffi::CStr::from_ptr(class_getName(vc_class) as *const i8).to_str().unwrap_or("?");
        // Check if responds to controllerUserInteractionEnabled
        let responds: Bool = msg_send![&*vc, respondsToSelector: sel!(controllerUserInteractionEnabled)];
        let ctrl_ui: Bool = if responds.as_bool() { msg_send![&*vc, controllerUserInteractionEnabled] } else { Bool::NO };
        std::io::Write::write_all(&mut std::io::stderr(),
            format!("[bloom-visionos] rootVC class={}, respondsToControllerUI={}, controllerUserInteractionEnabled={}\n",
                vc_name, responds.as_bool(), ctrl_ui.as_bool()).as_bytes()).ok();
    }
    // Check window state
    {
        let app_cls2 = AnyClass::get(c"UIApplication").unwrap();
        let app2: *const AnyObject = msg_send![app_cls2, sharedApplication];
        let key_win: *const AnyObject = msg_send![&*app2, keyWindow];
        let is_key = UI_WINDOW.as_ref().map(|w| Retained::as_ptr(w) as *const AnyObject == key_win).unwrap_or(false);
        // Count all windows
        let windows: *const AnyObject = msg_send![&*app2, windows];
        let win_count: usize = msg_send![&*windows, count];
        std::io::Write::write_all(&mut std::io::stderr(),
            format!("[bloom-visionos] windows={}, keyWindow==ours={}\n", win_count, is_key).as_bytes()).ok();
    }

    eprintln!("[bloom-visionos] window hierarchy set up, signaling game thread");
    // Store the layer pointer and screen dimensions for the game thread to create wgpu
    SCENE_PTR.store(Retained::as_ptr(&layer) as u64, std::sync::atomic::Ordering::Release);
    // Store dimensions packed into u64
    SCREEN_DIMS.store(((pixel_width as u64) << 32) | (pixel_height as u64), std::sync::atomic::Ordering::Release);
}

fn register_scene_delegate() {
    if AnyClass::get(c"PerrySceneDelegate").is_some() { return; }

    unsafe {
        let superclass = AnyClass::get(c"UIResponder").unwrap();
        let cls = objc_allocateClassPair(superclass as *const AnyClass, b"PerrySceneDelegate\0".as_ptr(), 0);
        if cls.is_null() { return; }

        // scene:willConnectToSession:connectionOptions:
        let sel = Sel::register(c"scene:willConnectToSession:connectionOptions:");
        let types = b"v48@0:8@16@24@32\0".as_ptr();
        class_addMethod(cls, sel, scene_will_connect as *const c_void, types);

        // Add UIWindowSceneDelegate protocol
        extern "C" { fn objc_getProtocol(name: *const u8) -> *const c_void; }
        let protocol = objc_getProtocol(b"UIWindowSceneDelegate\0".as_ptr());
        if !protocol.is_null() {
            extern "C" { fn class_addProtocol(cls: *mut AnyClass, protocol: *const c_void) -> bool; }
            class_addProtocol(cls, protocol);
        }

        objc_registerClassPair(cls);
    }
}

// ============================================================
// Minimal pollster
// ============================================================

fn pollster_block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, Wake, Waker};
    use std::pin::Pin;
    use std::sync::Arc;
    struct NoopWaker;
    impl Wake for NoopWaker { fn wake(self: Arc<Self>) {} }
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut future = unsafe { Pin::new_unchecked(Box::new(future)) };
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(result) => return result,
            Poll::Pending => {
                pump_run_loop(0.001);
            }
        }
    }
}

// ============================================================
// GCController — Siri Remote and game controller monitoring
// ============================================================

/// Poll connected game controllers and feed their state into the engine input system.
/// Called once during init and also from bloom_begin_drawing to poll controller state.
fn poll_game_controllers() {
    unsafe {
        let gc_cls = match AnyClass::get(c"GCController") {
            Some(c) => c,
            None => return,
        };
        let controllers: Retained<AnyObject> = msg_send![gc_cls, controllers];
        let count: usize = msg_send![&*controllers, count];
        if count == 0 { return; }

        // Use the first connected controller
        let controller: Retained<AnyObject> = msg_send![&*controllers, objectAtIndex: 0usize];

        // Try extended gamepad profile first (MFi, PS, Xbox controllers)
        let extended: *const AnyObject = msg_send![&*controller, extendedGamepad];
        if !extended.is_null() {
            if let Some(eng) = ENGINE.get_mut() {
                eng.input.gamepad_available = true;

                // Left thumbstick
                let left_stick: Retained<AnyObject> = msg_send![&*extended, leftThumbstick];
                let lx: f64 = msg_send![&*left_stick, xAxis_value];
                let ly: f64 = msg_send![&*left_stick, yAxis_value];
                eng.input.set_gamepad_axis(0, lx as f32);
                eng.input.set_gamepad_axis(1, -ly as f32); // Invert Y

                // Right thumbstick
                let right_stick: Retained<AnyObject> = msg_send![&*extended, rightThumbstick];
                let rx: f64 = msg_send![&*right_stick, xAxis_value];
                let ry: f64 = msg_send![&*right_stick, yAxis_value];
                eng.input.set_gamepad_axis(2, rx as f32);
                eng.input.set_gamepad_axis(3, -ry as f32);

                // Buttons: A(0), B(1), X(2), Y(3)
                let btn_a: Retained<AnyObject> = msg_send![&*extended, buttonA];
                let btn_b: Retained<AnyObject> = msg_send![&*extended, buttonB];
                let btn_x: Retained<AnyObject> = msg_send![&*extended, buttonX];
                let btn_y: Retained<AnyObject> = msg_send![&*extended, buttonY];
                let a_pressed: Bool = msg_send![&*btn_a, isPressed];
                let b_pressed: Bool = msg_send![&*btn_b, isPressed];
                let x_pressed: Bool = msg_send![&*btn_x, isPressed];
                let y_pressed: Bool = msg_send![&*btn_y, isPressed];
                if a_pressed.as_bool() { eng.input.set_gamepad_button_down(0); }
                if b_pressed.as_bool() { eng.input.set_gamepad_button_down(1); }
                if x_pressed.as_bool() { eng.input.set_gamepad_button_down(2); }
                if y_pressed.as_bool() { eng.input.set_gamepad_button_down(3); }

                // D-pad
                let dpad: Retained<AnyObject> = msg_send![&*extended, dpad];
                let up: Retained<AnyObject> = msg_send![&*dpad, up];
                let down: Retained<AnyObject> = msg_send![&*dpad, down];
                let left: Retained<AnyObject> = msg_send![&*dpad, left];
                let right: Retained<AnyObject> = msg_send![&*dpad, right];
                let up_p: Bool = msg_send![&*up, isPressed];
                let down_p: Bool = msg_send![&*down, isPressed];
                let left_p: Bool = msg_send![&*left, isPressed];
                let right_p: Bool = msg_send![&*right, isPressed];
                if up_p.as_bool() { eng.input.set_gamepad_button_down(12); }
                if down_p.as_bool() { eng.input.set_gamepad_button_down(13); }
                if left_p.as_bool() { eng.input.set_gamepad_button_down(14); }
                if right_p.as_bool() { eng.input.set_gamepad_button_down(15); }

                // Shoulders and triggers
                let l_shoulder: Retained<AnyObject> = msg_send![&*extended, leftShoulder];
                let r_shoulder: Retained<AnyObject> = msg_send![&*extended, rightShoulder];
                let ls_p: Bool = msg_send![&*l_shoulder, isPressed];
                let rs_p: Bool = msg_send![&*r_shoulder, isPressed];
                if ls_p.as_bool() { eng.input.set_gamepad_button_down(4); }
                if rs_p.as_bool() { eng.input.set_gamepad_button_down(5); }

                let l_trigger: Retained<AnyObject> = msg_send![&*extended, leftTrigger];
                let r_trigger: Retained<AnyObject> = msg_send![&*extended, rightTrigger];
                eng.input.set_gamepad_axis(4, { let v: f64 = msg_send![&*l_trigger, value]; v as f32 });
                eng.input.set_gamepad_axis(5, { let v: f64 = msg_send![&*r_trigger, value]; v as f32 });
            }
            return;
        }

        // Fall back to micro gamepad profile (Siri Remote)
        let micro: *const AnyObject = msg_send![&*controller, microGamepad];
        if !micro.is_null() {
            if let Some(eng) = ENGINE.get_mut() {
                eng.input.gamepad_available = true;

                // Siri Remote touchpad → axes 0/1
                let dpad: Retained<AnyObject> = msg_send![&*micro, dpad];
                let x_val: f64 = { let axis: Retained<AnyObject> = msg_send![&*dpad, xAxis]; msg_send![&*axis, value] };
                let y_val: f64 = { let axis: Retained<AnyObject> = msg_send![&*dpad, yAxis]; msg_send![&*axis, value] };
                eng.input.set_gamepad_axis(0, x_val as f32);
                eng.input.set_gamepad_axis(1, -y_val as f32);

                // Button A (select/click) and Button X (play/pause)
                let btn_a: Retained<AnyObject> = msg_send![&*micro, buttonA];
                let btn_x: Retained<AnyObject> = msg_send![&*micro, buttonX];
                let a_pressed: Bool = msg_send![&*btn_a, isPressed];
                let x_pressed: Bool = msg_send![&*btn_x, isPressed];
                if a_pressed.as_bool() { eng.input.set_gamepad_button_down(0); }
                if x_pressed.as_bool() { eng.input.set_gamepad_button_down(9); } // start/pause
            }
        }
    }
}

fn setup_game_controllers() {
    // Register for GCController connection notifications and set up input handlers.
    // This is the correct way to handle Siri Remote input on tvOS — it claims
    // the controller at the system level, preventing the OS from dismissing the app.
    unsafe {
        extern "C" {
            static _dispatch_main_q: c_void;
            fn dispatch_async_f(queue: *const c_void, context: *mut c_void, work: unsafe extern "C" fn(*mut c_void));
        }
        unsafe extern "C" fn setup_gc(_: *mut c_void) {
            let gc_cls = match AnyClass::get(c"GCController") {
                Some(c) => c,
                None => return,
            };

            // Start wireless controller discovery (finds Siri Remote in simulator)
            let _: () = msg_send![gc_cls, startWirelessControllerDiscoveryWithCompletionHandler: std::ptr::null::<c_void>()];

            // Check for already-connected controllers
            let controllers: Retained<AnyObject> = msg_send![gc_cls, controllers];
            let count: usize = msg_send![&*controllers, count];
            std::io::Write::write_all(&mut std::io::stderr(),
                format!("[bloom-visionos] GCControllers found: {}\n", count).as_bytes()).ok();

            // Set up value-changed handlers on connected controllers
            for i in 0..count {
                let ctrl: Retained<AnyObject> = msg_send![&*controllers, objectAtIndex: i as usize];
                let micro: *const AnyObject = msg_send![&*ctrl, microGamepad];
                if !micro.is_null() {
                    // Set reportsAbsoluteDpadValues so polled values are absolute position
                    let _: () = msg_send![&*micro, setReportsAbsoluteDpadValues: Bool::YES];
                }
            }

            // visionOS: do NOT create a GCVirtualController. Its on-screen
            // overlay sits on top of the window and swallows indirect
            // (eye+pinch) input as its own d-pad/button presses, so the
            // game's touch handlers (touchesBegan -> bloom_get_touch_*) never
            // see the pinch. We want pinches to reach the Metal view as plain
            // UITouches at the gaze location so touch/pinch controls work. A
            // *physical* Bluetooth controller (count > 0) is still picked up
            // below. (tvOS keeps the virtual controller; visionOS omits it.)

            for i in 0..count {
                let controller: Retained<AnyObject> = msg_send![&*controllers, objectAtIndex: i as usize];

                // Check micro gamepad (Siri Remote)
                let micro: *const AnyObject = msg_send![&*controller, microGamepad];
                if !micro.is_null() {
                    std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] Found micro gamepad (Siri Remote)\n").ok();
                    // Set reportsAbsoluteDpadValues so we get position, not delta
                    let _: () = msg_send![&*micro, setReportsAbsoluteDpadValues: Bool::YES];
                    // allowsRotation for landscape usage
                    let _: () = msg_send![&*micro, setAllowsRotation: Bool::YES];
                }

                // Check extended gamepad (MFi, PS, Xbox)
                let extended: *const AnyObject = msg_send![&*controller, extendedGamepad];
                if !extended.is_null() {
                    std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] Found extended gamepad\n").ok();
                }
            }
        }
        dispatch_async_f(&_dispatch_main_q as *const _, std::ptr::null_mut(), setup_gc);
    }
}

// ============================================================
// FFI entry points
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_init_window(_width: f64, _height: f64, title_ptr: *const u8, _fullscreen: f64) {
    let _title = str_from_header(title_ptr);

    // Register ObjC classes for the scene delegate (window/view creation)
    register_metal_view_class();
    register_window_class();
    register_view_controller_class();
    register_scene_delegate();

    // Signal the main thread that our ObjC classes are ready.
    // UIApplicationMain (on main thread) waits for this before starting.
    extern "C" { fn perry_ios_classes_registered(); }
    unsafe { perry_ios_classes_registered(); }

    // Debug: write marker to confirm we reached this point
    let _ = std::fs::write("/tmp/bloom_checkpoint_1.txt", "classes registered\n");

    // Get app bundle path for resolving relative asset paths
    unsafe {
        let bundle_cls = AnyClass::get(c"NSBundle").unwrap();
        let main_bundle: Retained<AnyObject> = msg_send![bundle_cls, mainBundle];
        let resource_path: *const AnyObject = msg_send![&*main_bundle, resourcePath];
        if !resource_path.is_null() {
            let utf8: *const u8 = msg_send![&*resource_path, UTF8String];
            if !utf8.is_null() {
                let cstr = std::ffi::CStr::from_ptr(utf8 as *const i8);
                if let Ok(s) = cstr.to_str() {
                    BUNDLE_PATH = Some(s.to_string());
                }
            }
        }
    }

    // With --features ios-game-loop, this function runs on the game thread.
    // The engine is created on the main thread by scene_will_connect (like iOS).
    // Just wait for it here.
    std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] waiting for ENGINE...\n").ok();
    unsafe {
        for i in 0..3000 {
            if ENGINE.get().is_some() { break; }
            if i % 100 == 0 && i > 0 {
                let msg = format!("[bloom-visionos] still waiting for ENGINE... {}s\n", i / 100);
                std::io::Write::write_all(&mut std::io::stderr(), msg.as_bytes()).ok();
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        if ENGINE.get().is_none() {
            panic!("[bloom-visionos] ENGINE not available after 30s");
        }
    }
    std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] ENGINE ready on game thread\n").ok();

    // Set up GCController monitoring for Siri Remote and game controllers
    setup_game_controllers();
}

#[no_mangle]
pub extern "C" fn bloom_close_window() {
    unsafe { UI_VIEW = None; UI_WINDOW = None; }
}

#[no_mangle]
pub extern "C" fn bloom_window_should_close() -> f64 {
    if engine().should_close { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_begin_drawing() {
    static FRAME_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let frame = FRAME_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if frame == 0 {
        std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] first bloom_begin_drawing\n").ok();
    }
    // Poll GCController synchronously on the game thread.
    // GCController value reading is thread-safe.
    unsafe {
        if let Some(gc_cls) = AnyClass::get(c"GCController") {
            let controllers: Retained<AnyObject> = msg_send![gc_cls, controllers];
            let count: usize = msg_send![&*controllers, count];
            {
                static POLL_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let n = POLL_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 3 || (n % 300 == 0) {
                    std::io::Write::write_all(&mut std::io::stderr(),
                        format!("[bloom-visionos] GC poll: {} controllers\n", count).as_bytes()).ok();
                }
            }
            if count > 0 {
                let controller: Retained<AnyObject> = msg_send![&*controllers, objectAtIndex: 0usize];
                let eng = engine();
                eng.input.gamepad_available = true;

                // Micro gamepad (Siri Remote)
                let micro: *const AnyObject = msg_send![&*controller, microGamepad];
                let extended_check: *const AnyObject = msg_send![&*controller, extendedGamepad];
                {
                    static PROFILE_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !PROFILE_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        std::io::Write::write_all(&mut std::io::stderr(),
                            format!("[bloom-visionos] micro={} extended={}\n", !micro.is_null(), !extended_check.is_null()).as_bytes()).ok();
                    }
                }
                if !micro.is_null() {
                    let dpad: Retained<AnyObject> = msg_send![&*micro, dpad];
                    let x_axis: Retained<AnyObject> = msg_send![&*dpad, xAxis];
                    let y_axis: Retained<AnyObject> = msg_send![&*dpad, yAxis];
                    let x_val: f32 = msg_send![&*x_axis, value];
                    let y_val: f32 = msg_send![&*y_axis, value];
                    eng.input.set_gamepad_axis(0, x_val);
                    eng.input.set_gamepad_axis(1, -y_val);
                    let btn_a: Retained<AnyObject> = msg_send![&*micro, buttonA];
                    let btn_x: Retained<AnyObject> = msg_send![&*micro, buttonX];
                    let a_val: f32 = msg_send![&*btn_a, value];
                    let x_btn_val: f32 = msg_send![&*btn_x, value];
                    if a_val > 0.01 || x_btn_val > 0.01 || x_val.abs() > 0.01 || y_val.abs() > 0.01 {
                        static BTN_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        let n = BTN_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if n < 20 {
                            std::io::Write::write_all(&mut std::io::stderr(),
                                format!("[bloom-visionos] micro: x={:.2} y={:.2} a={:.2} x_btn={:.2}\n", x_val, y_val, a_val, x_btn_val).as_bytes()).ok();
                        }
                    }
                    if a_val > 0.5 { eng.input.set_gamepad_button_down(0); }
                    if x_btn_val > 0.5 { eng.input.set_gamepad_button_down(7); }
                }

                // Extended gamepad (MFi/PS/Xbox)
                let extended: *const AnyObject = msg_send![&*controller, extendedGamepad];
                if !extended.is_null() {
                    let left_stick: Retained<AnyObject> = msg_send![&*extended, leftThumbstick];
                    let lx_axis: Retained<AnyObject> = msg_send![&*left_stick, xAxis];
                    let ly_axis: Retained<AnyObject> = msg_send![&*left_stick, yAxis];
                    let lx: f32 = msg_send![&*lx_axis, value];
                    let ly: f32 = msg_send![&*ly_axis, value];
                    eng.input.set_gamepad_axis(0, lx);
                    eng.input.set_gamepad_axis(1, -ly);
                    let btn_a: Retained<AnyObject> = msg_send![&*extended, buttonA];
                    let a_val: f32 = msg_send![&*btn_a, value];
                    if lx.abs() > 0.01 || ly.abs() > 0.01 || a_val > 0.01 {
                        static EXT_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        let n = EXT_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if n < 20 {
                            std::io::Write::write_all(&mut std::io::stderr(),
                                format!("[bloom-visionos] ext: lx={:.2} ly={:.2} a={:.2}\n", lx, ly, a_val).as_bytes()).ok();
                        }
                    }
                    if a_val > 0.5 { eng.input.set_gamepad_button_down(0); }
                }
            }
        }
    }
    // Drain pending key events from main thread BEFORE begin_frame snapshots
    drain_pending_keys(engine());

    // Check if any pending keys were drained
    {
        static DRAIN_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        // Check if any PENDING_KEY_DOWN bits are set (before they were drained)
        let mut any = false;
        for i in 0..8 {
            if PENDING_KEY_DOWN[i].load(std::sync::atomic::Ordering::Relaxed) != 0 { any = true; }
        }
        if any {
            let n = DRAIN_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 10 {
                std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] PENDING KEYS FOUND!\n").ok();
            }
        }
    }

    if frame == 0 {
        std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] calling begin_frame\n").ok();
    }
    engine().begin_frame();
    if frame == 0 {
        std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] begin_frame OK\n").ok();
    }
}

#[no_mangle]
pub extern "C" fn bloom_end_drawing() {
    static END_FRAME_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let frame = END_FRAME_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if frame == 0 {
        std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] first bloom_end_drawing\n").ok();
    }
    engine().end_frame();
    if frame == 0 {
        std::io::Write::write_all(&mut std::io::stderr(), b"[bloom-visionos] end_frame OK\n").ok();
    }
}

// ============================================================
// Keyboard input
// ============================================================

// ============================================================
// Mouse input
// ============================================================

// ============================================================
// Shape drawing
// ============================================================

// ============================================================
// Text
// ============================================================

// ============================================================
// Textures
// ============================================================

// ============================================================
// Camera 2D
// ============================================================

// ============================================================
// 3D Camera and Drawing
// ============================================================

// ============================================================
// Joint test (skeletal animation debug)
// ============================================================

// ============================================================
// Lighting
// ============================================================

// ============================================================
// Models
// ============================================================

// ============================================================
// Phase 1c — material system FFI
// ============================================================

// ============================================================
mod audio_backend;


// ============================================================
// Music
// ============================================================

// ============================================================
// Staging / commit (thread-safe asset loading for ios-game-loop)
// ============================================================

// ============================================================
// Gamepad input
// ============================================================

// ============================================================
// Touch input
// ============================================================

// ============================================================
// Utility
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_toggle_fullscreen() {}

#[no_mangle]
pub extern "C" fn bloom_set_window_title(_title_ptr: *const u8) {}

#[no_mangle]
pub extern "C" fn bloom_set_window_icon(_path_ptr: *const u8) {}

#[no_mangle]
pub extern "C" fn bloom_disable_cursor() {
    engine().input.cursor_disabled = true;
}

#[no_mangle]
pub extern "C" fn bloom_enable_cursor() {
    engine().input.cursor_disabled = false;
}

// E4: Clipboard (stub on this platform)
#[no_mangle]
pub extern "C" fn bloom_set_clipboard_text(_text_ptr: *const u8) {}
#[no_mangle]
pub extern "C" fn bloom_get_clipboard_text() -> *const u8 { std::ptr::null() }

// E5b: File dialogs (stub on this platform)
#[no_mangle]
pub extern "C" fn bloom_open_file_dialog(_filter_ptr: *const u8, _title_ptr: *const u8) -> *const u8 { std::ptr::null() }
#[no_mangle]
pub extern "C" fn bloom_save_file_dialog(_default_name_ptr: *const u8, _title_ptr: *const u8) -> *const u8 { std::ptr::null() }

// ============================================================
// Input injection + platform detection
// ============================================================
#[no_mangle]
pub extern "C" fn bloom_get_platform() -> f64 { 9.0 }

/// Preferred OS language packed as `c0*256+c1` (ISO-639 primary subtag). See macos lib for format.
#[no_mangle]
pub extern "C" fn bloom_get_language() -> f64 {
    fn pack(code: &str) -> f64 { let l = code.to_ascii_lowercase(); let b = l.as_bytes(); if b.len() >= 2 { (b[0] as f64) * 256.0 + (b[1] as f64) } else { 25966.0 } }
    let langs = objc2_foundation::NSLocale::preferredLanguages();
    match langs.firstObject() { Some(s) => pack(&s.to_string()), None => 25966.0 }
}


// Q6: Multi-hit picking
// ============================================================

// ============================================================
// Render quality toggles (individual + preset) — ticket 011
// Mirror of the macOS FFI surface added in commit 95da6af; previously
// macOS-only, now exposed on every native platform so non-macOS builds
// don't fail at runtime (missing symbol) when the TS API invokes them.
// ============================================================

// ============================================================
// Profiler — CPU phase timings (always available) + GPU timestamps
// (when the adapter supports TIMESTAMP_QUERY). Disabled by default.
// ============================================================

// ============================================================
// Physics (Jolt 5.x) — FFI surface generated from shared macro
// ============================================================

#[cfg(feature = "jolt")]
#[inline]
fn bloom_jolt_ffi_physics() -> &'static mut bloom_shared::physics_jolt::JoltPhysics {
    &mut engine().jolt
}

#[cfg(feature = "jolt")]
bloom_shared::define_physics_ffi!();

// ============================================================
// Screenshot + HDR env + Post-FX / resolution FFI
// ------------------------------------------------------------
// Ported from native/macos/src/lib.rs. These delegate to the shared
// bloom_shared renderer (identical type used here), so they are real
// implementations, not stubs. They were present on macOS/linux/windows
// but missing on tvOS, which caused `ld64.lld: undefined symbol: _bloom_*`
// link errors for any app using the post-processing API on tvOS.
// ============================================================

// --- Post-FX knobs (heuristic visual layer; default-off) ---
