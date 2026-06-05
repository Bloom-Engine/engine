use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;
use bloom_shared::string_header::{str_from_header, alloc_perry_string};
use bloom_shared::audio::{parse_wav, parse_ogg, parse_mp3, SoundData};

use objc2::encode::{Encode, Encoding, RefEncode};
use objc2::rc::{Allocated, Retained};
use objc2::runtime::{AnyClass, AnyObject, Bool, Sel};
use objc2::{msg_send, sel};

use raw_window_handle::{RawDisplayHandle, RawWindowHandle, UiKitDisplayHandle, UiKitWindowHandle};

use std::ffi::c_void;
use std::sync::OnceLock;

static mut ENGINE: OnceLock<EngineState> = OnceLock::new();
static mut UI_WINDOW: Option<Retained<AnyObject>> = None;
static mut UI_VIEW: Option<Retained<AnyObject>> = None;
static mut TOUCH_MAP: [*const c_void; 10] = [std::ptr::null(); 10];
static mut BUNDLE_PATH: Option<String> = None;
static mut SCREEN_SCALE: f64 = 1.0;
/// UIInterfaceOrientationMask for the root view controller.
/// Default: all (0x1E). Set to landscape (0x18) from bloom_init_window when width > height.
static mut ORIENTATION_MASK: u64 = 0x1E; // UIInterfaceOrientationMaskAll


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

        objc_registerClassPair(cls);
    }
}

// ============================================================
// Register BloomViewController — UIViewController with orientation lock
// ============================================================

unsafe extern "C" fn bloom_vc_supported_orientations(_this: *const c_void, _sel: Sel) -> u64 {
    unsafe { ORIENTATION_MASK }
}

fn register_view_controller_class() {
    if AnyClass::get(c"BloomViewController").is_some() { return; }

    unsafe {
        let superclass = AnyClass::get(c"UIViewController").unwrap();
        let cls = objc_allocateClassPair(superclass as *const AnyClass, b"BloomViewController\0".as_ptr(), 0);
        if cls.is_null() { return; }

        let sel = Sel::register(c"supportedInterfaceOrientations");
        class_addMethod(cls, sel, bloom_vc_supported_orientations as *const c_void, b"Q16@0:8\0".as_ptr());

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
    if scene.is_null() { return; }

    // Get screen bounds from UIScreen.mainScreen (simpler than scene.screen)
    let screen_cls = AnyClass::get(c"UIScreen").unwrap();
    let screen: Retained<AnyObject> = msg_send![screen_cls, mainScreen];
    let bounds: CGRect = msg_send![&*screen, bounds];
    let scale: f64 = msg_send![&*screen, scale];

    let pixel_width = (bounds.size.width * scale) as u32;
    let pixel_height = (bounds.size.height * scale) as u32;

    // Create UIWindow attached to the scene
    let window_cls = AnyClass::get(c"UIWindow").unwrap();
    let window: Allocated<AnyObject> = msg_send![window_cls, alloc];
    let window: Retained<AnyObject> = msg_send![window, initWithWindowScene: scene];

    // Create BloomViewController (with orientation lock support)
    let vc_cls = AnyClass::get(c"BloomViewController")
        .unwrap_or_else(|| AnyClass::get(c"UIViewController").unwrap());
    let vc: Allocated<AnyObject> = msg_send![vc_cls, alloc];
    let vc: Retained<AnyObject> = msg_send![vc, init];

    // Create BloomMetalView
    let view_cls = AnyClass::get(c"BloomMetalView").unwrap();
    let view: Allocated<AnyObject> = msg_send![view_cls, alloc];
    let view: Retained<AnyObject> = msg_send![view, initWithFrame: bounds];

    // Set background to green (diagnostic — visible if Metal isn't rendering)
    let color_cls = AnyClass::get(c"UIColor").unwrap();
    let green: Retained<AnyObject> = msg_send![color_cls, greenColor];
    let _: () = msg_send![&*view, setBackgroundColor: &*green];

    // Configure CAMetalLayer
    let layer: Retained<AnyObject> = msg_send![&*view, layer];
    let drawable_size = CGSize { width: pixel_width as f64, height: pixel_height as f64 };
    let _: () = msg_send![&*layer, setDrawableSize: drawable_size];
    let _: () = msg_send![&*layer, setContentsScale: scale];
    let _: () = msg_send![&*layer, setOpaque: Bool::NO];  // Non-opaque so green shows if Metal fails

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

    // Create wgpu surface
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let view_ptr = Retained::as_ptr(&view) as *mut c_void;
    let handle = UiKitWindowHandle::new(
        std::ptr::NonNull::new(view_ptr).unwrap(),
    );
    let raw = RawWindowHandle::UiKit(handle);
    let surface = instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
        raw_display_handle: Some(RawDisplayHandle::UiKit(UiKitDisplayHandle::new())),
        raw_window_handle: raw,
    }).expect("Failed to create wgpu surface");

    let adapter = pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })).expect("No GPU adapter found");

    // Ticket 007b: opt into HW ray-query on RT-capable Apple Silicon
    // devices. `BLOOM_FORCE_SW_GI=1` forces the SW path for bench parity.
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
    if !force_sw_gi && supported.contains(rt_mask) {
        required_features |= rt_mask;
    }
    let experimental_features = if required_features.intersects(rt_mask) {
        unsafe { wgpu::ExperimentalFeatures::enabled() }
    } else {
        wgpu::ExperimentalFeatures::disabled()
    };
    let mut required_limits = wgpu::Limits::default();
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
        Err(e) => {
            // Some real devices (e.g. iPhone 16 Pro / A18) advertise
            // EXPERIMENTAL_RAY_QUERY on the adapter but reject it at device
            // creation through wgpu's Metal backend, which would otherwise
            // abort the app on launch. Retry with a minimal, always-supported
            // configuration. The renderer keys off device.features() (see
            // renderer/mod.rs), so it transparently falls back to the non-RT
            // path when ray-query isn't granted.
            eprintln!("[bloom-ios] request_device with preferred features failed ({e:?}); retrying without ray-query/experimental features");
            pollster_block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("bloom_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                ..Default::default()
            })).expect("Failed to create device (minimal fallback)")
        }
    };

    let surface_caps = surface.get_capabilities(&adapter);
    let format = surface_caps.formats.iter()
        .find(|f| f.is_srgb()).copied()
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

    // Write debug info to app container tmp
    {
        let ns_cls = AnyClass::get(c"NSTemporaryDirectory").unwrap_or(AnyClass::get(c"NSString").unwrap());
        // Use NSFileManager to get tmp dir
        extern "C" { fn NSTemporaryDirectory() -> *const AnyObject; }
        let tmp_dir: *const AnyObject = NSTemporaryDirectory();
        if !tmp_dir.is_null() {
            let utf8: *const u8 = msg_send![&*tmp_dir, UTF8String];
            if !utf8.is_null() {
                let cstr_val = std::ffi::CStr::from_ptr(utf8 as *const i8);
                if let Ok(tmp_path) = cstr_val.to_str() {
                    let _ = std::fs::write(format!("{}bloom_debug.txt", tmp_path),
                        format!("scene_will_connect OK\npixels={}x{} scale={}\nwindow_scene={}\n",
                            pixel_width, pixel_height, scale, !scene.is_null()));
                }
            }
        }
    }

    // ENGINE is now set — bloom_init_window polls for this on the game thread
}

/// Called by the perry runtime's main() before UIApplicationMain to register
/// ObjC classes needed for the scene lifecycle.
#[no_mangle]
pub unsafe extern "C" fn perry_register_native_classes() {
    register_metal_view_class();
    register_view_controller_class();
    register_scene_delegate();
}

/// Called by the runtime's scene delegate when the UIWindowScene connects.
/// This runs on the main thread — safe for all UIKit operations.
#[no_mangle]
pub unsafe extern "C" fn perry_scene_will_connect(scene: *const c_void) {
    let screen_cls = AnyClass::get(c"UIScreen").unwrap();
    let screen: Retained<AnyObject> = msg_send![screen_cls, mainScreen];
    let bounds: CGRect = msg_send![&*screen, bounds];
    let scale: f64 = msg_send![&*screen, scale];

    let pixel_width = (bounds.size.width * scale) as u32;
    let pixel_height = (bounds.size.height * scale) as u32;

    // Store scale for touch coordinate conversion (points → pixels)
    SCREEN_SCALE = scale;

    // Create UIWindow — attached to scene if available, otherwise plain
    let window_cls = AnyClass::get(c"UIWindow").unwrap();
    let window: Retained<AnyObject> = if !scene.is_null() {
        let w: Allocated<AnyObject> = msg_send![window_cls, alloc];
        msg_send![w, initWithWindowScene: scene as *const AnyObject]
    } else {
        let w: Allocated<AnyObject> = msg_send![window_cls, alloc];
        msg_send![w, initWithFrame: bounds]
    };

    // Create BloomViewController (with orientation lock support)
    let vc_cls = AnyClass::get(c"BloomViewController")
        .unwrap_or_else(|| AnyClass::get(c"UIViewController").unwrap());
    let vc: Allocated<AnyObject> = msg_send![vc_cls, alloc];
    let vc: Retained<AnyObject> = msg_send![vc, init];

    // Create BloomMetalView
    let view_cls = AnyClass::get(c"BloomMetalView").unwrap();
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

    // Create wgpu surface and engine
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let view_ptr = Retained::as_ptr(&view) as *mut c_void;
    let handle = UiKitWindowHandle::new(
        std::ptr::NonNull::new(view_ptr).unwrap(),
    );
    let raw = RawWindowHandle::UiKit(handle);
    let surface = instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
        raw_display_handle: Some(RawDisplayHandle::UiKit(UiKitDisplayHandle::new())),
        raw_window_handle: raw,
    }).expect("Failed to create wgpu surface");

    let adapter = pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })).expect("No GPU adapter found");

    // Ticket 007b: opt into HW ray-query on RT-capable Apple Silicon
    // devices. `BLOOM_FORCE_SW_GI=1` forces the SW path for bench parity.
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
    if !force_sw_gi && supported.contains(rt_mask) {
        required_features |= rt_mask;
    }
    let experimental_features = if required_features.intersects(rt_mask) {
        unsafe { wgpu::ExperimentalFeatures::enabled() }
    } else {
        wgpu::ExperimentalFeatures::disabled()
    };
    let mut required_limits = wgpu::Limits::default();
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
        Err(e) => {
            // Some real devices (e.g. iPhone 16 Pro / A18) advertise
            // EXPERIMENTAL_RAY_QUERY on the adapter but reject it at device
            // creation through wgpu's Metal backend, which would otherwise
            // abort the app on launch. Retry with a minimal, always-supported
            // configuration. The renderer keys off device.features() (see
            // renderer/mod.rs), so it transparently falls back to the non-RT
            // path when ray-query isn't granted.
            eprintln!("[bloom-ios] request_device with preferred features failed ({e:?}); retrying without ray-query/experimental features");
            pollster_block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("bloom_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                ..Default::default()
            })).expect("Failed to create device (minimal fallback)")
        }
    };

    let surface_caps = surface.get_capabilities(&adapter);
    let format = surface_caps.formats.iter()
        .find(|f| f.is_srgb()).copied()
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
// FFI entry points
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_init_window(_width: f64, _height: f64, title_ptr: *const u8, _fullscreen: f64) {
    let _title = str_from_header(title_ptr);

    // Set orientation mask based on requested dimensions
    // width > height → landscape, otherwise all orientations
    if _width > _height {
        unsafe { ORIENTATION_MASK = 0x18; } // UIInterfaceOrientationMaskLandscape
    }

    // Register ObjC classes for the scene delegate (window/view creation)
    register_metal_view_class();
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
    // UIApplicationMain runs on the main thread. The runtime's scene delegate
    // calls perry_scene_will_connect() which creates the window, view, wgpu
    // surface, and ENGINE — all on the main thread. We just wait for it.
    unsafe {
        for _ in 0..1000 {
            if ENGINE.get().is_some() { break; }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    if unsafe { ENGINE.get().is_none() } {
        panic!("[bloom-ios] Engine not initialized after 10s. \
                Compile with: perry compile --target ios-simulator --features ios-game-loop");
    }
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
    // No run loop pumping needed — UIApplicationMain handles the main run loop
    // on its own thread. The game runs on the game thread.

    // Update drawable size to match actual view bounds (handles orientation changes)
    unsafe {
        if let Some(view) = &UI_VIEW {
            let view_ptr = Retained::as_ptr(view);
            let layer: Retained<AnyObject> = msg_send![&*view_ptr, layer];
            let view_bounds: CGRect = msg_send![&*view_ptr, bounds];
            let scale = SCREEN_SCALE;
            let pw = (view_bounds.size.width * scale) as u32;
            let ph = (view_bounds.size.height * scale) as u32;
            let eng = engine();
            if pw > 0 && ph > 0 && (pw != eng.renderer.width() || ph != eng.renderer.height()) {
                let ds = CGSize { width: pw as f64, height: ph as f64 };
                let _: () = msg_send![&*layer, setDrawableSize: ds];
                eng.renderer.resize(pw, ph, pw, ph);
            } else {
                eng.begin_frame();
                return;
            }
        }
    }
    engine().begin_frame();
}

#[no_mangle]
pub extern "C" fn bloom_end_drawing() {
    engine().end_frame();
}

#[no_mangle]
pub extern "C" fn bloom_clear_background(r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.set_clear_color(r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_set_target_fps(fps: f64) { engine().target_fps = fps; }

#[no_mangle]
pub extern "C" fn bloom_set_direct_2d_mode(on: f64) { engine().direct_2d_mode = on > 0.5; }

#[no_mangle]
pub extern "C" fn bloom_get_delta_time() -> f64 { engine().delta_time }

#[no_mangle]
pub extern "C" fn bloom_get_fps() -> f64 { engine().get_fps() }

#[no_mangle]
pub extern "C" fn bloom_get_screen_width() -> f64 { engine().screen_width() }

#[no_mangle]
pub extern "C" fn bloom_get_screen_height() -> f64 { engine().screen_height() }

// ============================================================
// Keyboard input
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_is_key_pressed(key: f64) -> f64 {
    if engine().input.is_key_pressed(key as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_key_down(key: f64) -> f64 {
    if engine().input.is_key_down(key as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_key_released(key: f64) -> f64 {
    if engine().input.is_key_released(key as usize) { 1.0 } else { 0.0 }
}

// ============================================================
// Mouse input
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_get_mouse_x() -> f64 { engine().input.mouse_x }

#[no_mangle]
pub extern "C" fn bloom_get_mouse_y() -> f64 { engine().input.mouse_y }

#[no_mangle]
pub extern "C" fn bloom_is_mouse_button_pressed(btn: f64) -> f64 {
    if engine().input.is_mouse_button_pressed(btn as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_mouse_button_down(btn: f64) -> f64 {
    if engine().input.is_mouse_button_down(btn as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_mouse_button_released(btn: f64) -> f64 {
    if engine().input.is_mouse_button_released(btn as usize) { 1.0 } else { 0.0 }
}

// ============================================================
// Shape drawing
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_draw_line(x1: f64, y1: f64, x2: f64, y2: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_line(x1, y1, x2, y2, thickness, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_rect(x: f64, y: f64, w: f64, h: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_rect(x, y, w, h, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_rect_lines(x: f64, y: f64, w: f64, h: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_rect_lines(x, y, w, h, thickness, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_circle(cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_circle(cx, cy, radius, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_circle_lines(cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_circle_lines(cx, cy, radius, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_triangle(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_triangle(x1, y1, x2, y2, x3, y3, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_poly(cx: f64, cy: f64, sides: f64, radius: f64, rotation: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_poly(cx, cy, sides, radius, rotation, r, g, b, a);
}

// ============================================================
// Text
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_draw_text(text_ptr: *const u8, x: f64, y: f64, size: f64, r: f64, g: f64, b: f64, a: f64) {
    let text = str_from_header(text_ptr);
    let eng = engine();
    let mut text_renderer = std::mem::replace(&mut eng.text, bloom_shared::text_renderer::TextRenderer::empty());
    text_renderer.draw_text(&mut eng.renderer, text, x, y, size as u32, r, g, b, a);
    eng.text = text_renderer;
}

#[no_mangle]
pub extern "C" fn bloom_measure_text(text_ptr: *const u8, size: f64) -> f64 {
    let text = str_from_header(text_ptr);
    engine().text.measure_text(text, size as u32)
}

#[no_mangle]
pub extern "C" fn bloom_load_font(path_ptr: *const u8, _size: f64) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) { Ok(data) => engine().text.load_font(&data) as f64, Err(_) => 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_unload_font(font_handle: f64) {
    engine().text.unload_font(font_handle as usize);
}

#[no_mangle]
pub extern "C" fn bloom_draw_text_ex(font_handle: f64, text_ptr: *const u8, x: f64, y: f64, size: f64, spacing: f64, r: f64, g: f64, b: f64, a: f64) {
    let text = str_from_header(text_ptr);
    let eng = engine();
    let mut text_renderer = std::mem::replace(&mut eng.text, bloom_shared::text_renderer::TextRenderer::empty());
    text_renderer.draw_text_ex(&mut eng.renderer, font_handle as usize, text, x, y, size as u32, spacing as f32, r, g, b, a);
    eng.text = text_renderer;
}

#[no_mangle]
pub extern "C" fn bloom_measure_text_ex(font_handle: f64, text_ptr: *const u8, size: f64, spacing: f64) -> f64 {
    let text = str_from_header(text_ptr);
    engine().text.measure_text_ex(font_handle as usize, text, size as u32, spacing as f32)
}

// ============================================================
// Textures
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_load_texture(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) {
        Ok(data) => {
            let eng = engine();
            let renderer_ptr = &mut eng.renderer as *mut Renderer;
            eng.textures.load_texture(unsafe { &mut *renderer_ptr }, &data)
        }
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_unload_texture(handle: f64) {
    let eng = engine();
    let renderer_ptr = &mut eng.renderer as *mut Renderer;
    eng.textures.unload_texture(handle, unsafe { &mut *renderer_ptr });
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture(handle: f64, x: f64, y: f64, r: f64, g: f64, b: f64, a: f64) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let idx = tex.bind_group_idx;
        eng.renderer.draw_texture(idx, x, y, r, g, b, a);
    }
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture_pro(
    handle: f64,
    src_x: f64, src_y: f64, src_w: f64, src_h: f64,
    dst_x: f64, dst_y: f64, dst_w: f64, dst_h: f64,
    origin_x: f64, origin_y: f64, rotation: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let idx = tex.bind_group_idx;
        eng.renderer.draw_texture_pro(idx, src_x, src_y, src_w, src_h, dst_x, dst_y, dst_w, dst_h, origin_x, origin_y, rotation, r, g, b, a);
    }
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture_rec(
    handle: f64,
    src_x: f64, src_y: f64, src_w: f64, src_h: f64,
    dst_x: f64, dst_y: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let idx = tex.bind_group_idx;
        eng.renderer.draw_texture_rec(idx, src_x, src_y, src_w, src_h, dst_x, dst_y, r, g, b, a);
    }
}

#[no_mangle]
pub extern "C" fn bloom_get_texture_width(handle: f64) -> f64 {
    engine().textures.get(handle).map(|t| t.width as f64).unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn bloom_get_texture_height(handle: f64) -> f64 {
    engine().textures.get(handle).map(|t| t.height as f64).unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn bloom_gen_texture_mipmaps(_handle: f64) {}

#[no_mangle]
pub extern "C" fn bloom_set_texture_filter(handle: f64, mode: f64) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let bind_group_idx = tex.bind_group_idx;
        eng.renderer.set_texture_filter(bind_group_idx, mode > 0.5);
    }
}

#[no_mangle]
pub extern "C" fn bloom_load_image(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) {
        Ok(data) => engine().textures.load_image(&data),
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_image_resize(handle: f64, w: f64, h: f64) {
    engine().textures.image_resize(handle, w as u32, h as u32);
}

#[no_mangle]
pub extern "C" fn bloom_image_crop(handle: f64, x: f64, y: f64, w: f64, h: f64) {
    engine().textures.image_crop(handle, x as u32, y as u32, w as u32, h as u32);
}

#[no_mangle]
pub extern "C" fn bloom_image_flip_h(handle: f64) {
    engine().textures.image_flip_h(handle);
}

#[no_mangle]
pub extern "C" fn bloom_image_flip_v(handle: f64) {
    engine().textures.image_flip_v(handle);
}

#[no_mangle]
pub extern "C" fn bloom_load_texture_from_image(handle: f64) -> f64 {
    let eng = engine();
    let renderer_ptr = &mut eng.renderer as *mut Renderer;
    eng.textures.load_texture_from_image(handle, unsafe { &mut *renderer_ptr })
}

// ============================================================
// Camera 2D
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_begin_mode_2d(offset_x: f64, offset_y: f64, target_x: f64, target_y: f64, rotation: f64, zoom: f64) {
    engine().renderer.begin_mode_2d(
        offset_x as f32, offset_y as f32,
        target_x as f32, target_y as f32,
        rotation as f32, zoom as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_end_mode_2d() {
    engine().renderer.end_mode_2d();
}

// ============================================================
// 3D Camera and Drawing
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_begin_mode_3d(
    pos_x: f64, pos_y: f64, pos_z: f64,
    target_x: f64, target_y: f64, target_z: f64,
    up_x: f64, up_y: f64, up_z: f64,
    fovy: f64, projection: f64,
) {
    engine().renderer.begin_mode_3d(
        pos_x as f32, pos_y as f32, pos_z as f32,
        target_x as f32, target_y as f32, target_z as f32,
        up_x as f32, up_y as f32, up_z as f32,
        fovy as f32, projection as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_end_mode_3d() {
    engine().renderer.end_mode_3d();
}

#[no_mangle]
pub extern "C" fn bloom_draw_cube(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cube(x, y, z, w, h, d, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_cube_wires(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cube_wires(x, y, z, w, h, d, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_sphere(cx: f64, cy: f64, cz: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_sphere(cx, cy, cz, radius, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_sphere_wires(cx: f64, cy: f64, cz: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_sphere_wires(cx, cy, cz, radius, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_cylinder(x: f64, y: f64, z: f64, radius_top: f64, radius_bottom: f64, height: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cylinder(x, y, z, radius_top, radius_bottom, height, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_plane(cx: f64, cy: f64, cz: f64, w: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_plane(cx, cy, cz, w, d, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_grid(slices: f64, spacing: f64) {
    engine().renderer.draw_grid(slices as i32, spacing);
}

#[no_mangle]
pub extern "C" fn bloom_draw_ray(origin_x: f64, origin_y: f64, origin_z: f64, dir_x: f64, dir_y: f64, dir_z: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_ray(origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, r, g, b, a);
}

// ============================================================
// Joint test (skeletal animation debug)
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_set_joint_test(_joint: f64, _angle: f64) {
    // No-op for now — skeletal animation testing
}

// ============================================================
// Lighting
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_set_ambient_light(r: f64, g: f64, b: f64, intensity: f64) {
    engine().renderer.set_ambient_light(r, g, b, intensity);
}

#[no_mangle]
pub extern "C" fn bloom_set_directional_light(dx: f64, dy: f64, dz: f64, r: f64, g: f64, b: f64, intensity: f64) {
    engine().renderer.set_directional_light(dx, dy, dz, r, g, b, intensity);
}

#[no_mangle]
pub extern "C" fn bloom_set_procedural_sky(enabled: f64, rayleigh_density: f64, mie_density: f64, ground_albedo: f64) {
    engine().renderer.set_procedural_sky(
        enabled != 0.0,
        rayleigh_density as f32,
        mie_density as f32,
        ground_albedo as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_set_sun_direction(dx: f64, dy: f64, dz: f64, intensity: f64) {
    engine().renderer.set_sun_direction(dx as f32, dy as f32, dz as f32, intensity as f32);
}

// ============================================================
// Models
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_load_model(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) {
        Ok(data) => {
            let eng = engine();
            let renderer_ptr = &mut eng.renderer as *mut crate::Renderer;
            eng.models.load_model_with_textures(&data, unsafe { &mut *renderer_ptr })
        }
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_draw_model(handle: f64, x: f64, y: f64, z: f64, scale: f64, r: f64, g: f64, b: f64, a: f64) {
    let eng = engine();
    if let Some(model) = eng.models.get(handle) {
        let tint = [(r / 255.0) as f32, (g / 255.0) as f32, (b / 255.0) as f32, (a / 255.0) as f32];
        let position = [x as f32, y as f32, z as f32];
        let handle_bits = handle.to_bits();
        if eng.renderer.cache_model_if_static(handle_bits, &model.meshes) {
            eng.renderer.draw_model_cached(handle_bits, position, scale as f32, tint);
        } else {
            for mesh in &model.meshes {
                let tex_idx = mesh.texture_idx.unwrap_or(0);
                eng.renderer.draw_model_mesh_tinted(&mesh.vertices, &mesh.indices, position, scale as f32, tint, tex_idx);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_draw_model_rotated(
    handle: f64, x: f64, y: f64, z: f64,
    scale: f64, rot_y: f64,
    color_packed_argb: f64,
) {
    let bits = color_packed_argb as u32;
    let a = ((bits >> 24) & 0xff) as f32 / 255.0;
    let r = ((bits >> 16) & 0xff) as f32 / 255.0;
    let g = ((bits >>  8) & 0xff) as f32 / 255.0;
    let b = ( bits        & 0xff) as f32 / 255.0;
    let eng = engine();
    if let Some(model) = eng.models.get(handle) {
        let position = [x as f32, y as f32, z as f32];
        let scale = scale as f32;
        let tint = [r, g, b, a];
        for mesh in &model.meshes {
            let tex_idx = mesh.texture_idx.unwrap_or(0);
            eng.renderer.draw_model_mesh_tinted_rotated(
                &mesh.vertices, &mesh.indices, position, scale, tint, tex_idx, rot_y as f32,
            );
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_unload_model(handle: f64) {
    engine().models.unload_model(handle);
}

#[no_mangle]
pub extern "C" fn bloom_gen_mesh_cube(w: f64, h: f64, d: f64) -> f64 {
    engine().models.gen_mesh_cube(w as f32, h as f32, d as f32)
}

#[no_mangle]
pub extern "C" fn bloom_gen_mesh_heightmap(image_handle: f64, size_x: f64, size_y: f64, size_z: f64) -> f64 {
    let eng = engine();
    if let Some(img) = eng.textures.images.get(image_handle) {
        let data = img.data.clone();
        let w = img.width;
        let h = img.height;
        eng.models.gen_mesh_heightmap(&data, w, h, size_x as f32, size_y as f32, size_z as f32)
    } else {
        0.0
    }
}

#[no_mangle]
pub extern "C" fn bloom_load_shader(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    engine().renderer.load_custom_shader(source) as f64
}

#[no_mangle]
pub extern "C" fn bloom_create_mesh(vertex_ptr: *const f32, vertex_count: f64, index_ptr: *const u32, index_count: f64) -> f64 {
    if vertex_ptr.is_null() || index_ptr.is_null() { return 0.0; }
    let vcount = vertex_count as usize;
    let icount = index_count as usize;
    let vertex_data = unsafe { std::slice::from_raw_parts(vertex_ptr, vcount * 12) };
    let index_data = unsafe { std::slice::from_raw_parts(index_ptr, icount) };
    engine().models.create_mesh(vertex_data, index_data)
}

// ============================================================
// Phase 1c — material system FFI
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_set_material_params(
    handle: f64,
    params_ptr: *const f64,
    param_count: f64,
) {
    let count = param_count as usize;
    if count > 64 {
        eprintln!("[material] set_material_params: param_count {} > 64 (256-byte UBO cap)", count);
        return;
    }
    let mut bytes = vec![0u8; count * 4];
    if !params_ptr.is_null() && count > 0 {
        let slots = unsafe { std::slice::from_raw_parts(params_ptr, count) };
        for (i, &v) in slots.iter().enumerate() {
            let f = v as f32;
            bytes[i*4..i*4+4].copy_from_slice(&f.to_le_bytes());
        }
    }
    let eng = engine();
    if let Err(e) = eng.renderer.material_system.set_user_params(
        &eng.renderer.device, &eng.renderer.queue,
        handle as u32, &bytes,
    ) {
        eprintln!("[material] set_material_params failed: {}", e);
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material(source) {
        Ok(handle) => handle as f64,
        Err(e) => {
            eprintln!("[material] compile failed: {:?}", e);
            0.0
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_refractive(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Refractive, true, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[refractive] compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_transparent(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Transparent, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_additive(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Additive, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_cutout(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Opaque, Bucket::Cutout, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_instanced(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_instanced(source) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] instanced compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_create_instance_buffer(
    data_ptr: *const f64, instance_count: f64,
) -> f64 {
    if data_ptr.is_null() || instance_count <= 0.0 { return 0.0; }
    let count = instance_count as u32;
    let slot_count = (count as usize) * 9;
    let raw_f64 = unsafe { std::slice::from_raw_parts(data_ptr, slot_count) };
    let raw_f32: Vec<f32> = raw_f64.iter().map(|&v| v as f32).collect();
    engine().renderer.create_instance_buffer(&raw_f32, count) as f64
}

#[no_mangle]
pub extern "C" fn bloom_submit_material_draw_instanced(
    material: f64, mesh_handle: f64, mesh_idx: f64,
    instance_buffer: f64, instance_count: f64,
) {
    let eng = engine();
    let handle_bits = mesh_handle.to_bits();
    if let Some(model) = eng.models.get(mesh_handle) {
        eng.renderer.cache_model_if_static(handle_bits, &model.meshes);
    }
    eng.renderer.submit_material_draw_instanced(
        material as u32,
        handle_bits,
        mesh_idx as usize,
        instance_buffer as u32,
        instance_count as u32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_destroy_instance_buffer(handle: f64) {
    engine().renderer.destroy_instance_buffer(handle as u32);
}

/// EN-011 — create a planar reflection probe. See macOS lib.rs for the
/// full doc comment; this entry-point exists on every native platform
/// so games can target the same FFI surface across iOS/tvOS/Windows/
/// Linux/Android.
#[no_mangle]
pub extern "C" fn bloom_create_planar_reflection(
    plane_y: f64, nx: f64, ny: f64, nz: f64, resolution: f64,
) -> f64 {
    engine().renderer.create_planar_reflection(
        plane_y as f32,
        [nx as f32, ny as f32, nz as f32],
        resolution as u32,
    ) as f64
}

/// EN-011 — link a material to a planar reflection probe. `probe = 0`
/// reverts the binding to the engine's default 1×1 black texture.
#[no_mangle]
pub extern "C" fn bloom_set_material_reflection_probe(
    material: f64, probe: f64,
) {
    engine().renderer.set_material_reflection_probe(material as u32, probe as u32);
}

/// EN-014 — create a texture array from concatenated RGBA8 byte data.
/// See macOS lib.rs for the full doc comment; this entry-point exists
/// on every native platform so a TS game targets the same FFI across
/// iOS / tvOS / Windows / Linux / Android.
#[no_mangle]
pub extern "C" fn bloom_create_texture_array(
    data_ptr:    *const u8,
    data_len:    f64,
    width:       f64,
    height:      f64,
    layer_count: f64,
) -> f64 {
    // EN-014 V2 — V1 forwards to _ex with default sRGB / no mips.
    bloom_create_texture_array_ex(data_ptr, data_len, width, height, layer_count, 0.0, 1.0)
}

/// EN-014 V2 — explicit format + mip control. See macOS lib.rs for the
/// reference implementation; this entry-point exists on every native
/// platform so a TS game targets the same FFI surface.
#[no_mangle]
pub extern "C" fn bloom_create_texture_array_ex(
    data_ptr:    *const u8,
    data_len:    f64,
    width:       f64,
    height:      f64,
    layer_count: f64,
    format:      f64,
    mip_levels:  f64,
) -> f64 {
    if data_ptr.is_null() || data_len <= 0.0 { return 0.0; }
    let w = width as u32;
    let h = height as u32;
    if w == 0 || h == 0 { return 0.0; }
    let layers_count = (layer_count as u32)
        .min(bloom_shared::renderer::material_system::MAX_TEXTURE_ARRAY_LAYERS);
    if layers_count == 0 { return 0.0; }
    let layer_size = (w as usize) * (h as usize) * 4;
    let total_bytes = (data_len as usize)
        .min(layers_count as usize * layer_size);
    let bytes = unsafe { std::slice::from_raw_parts(data_ptr, total_bytes) };
    let mut layers: Vec<(&[u8], u32, u32)> = Vec::with_capacity(layers_count as usize);
    for i in 0..(layers_count as usize) {
        let start = i * layer_size;
        let end   = start + layer_size;
        if end > bytes.len() { break; }
        layers.push((&bytes[start..end], w, h));
    }
    engine().renderer.create_texture_array_ex(&layers, format as u32, mip_levels as u32) as f64
}

/// EN-014 — link a texture-array handle to a material at one of three
/// slots: 0 = albedo (binding 14), 1 = normal (binding 15),
/// 2 = MR (binding 16). Pass `array = 0` to revert to the stub.
#[no_mangle]
pub extern "C" fn bloom_set_material_texture_array(
    material: f64, slot: f64, array: f64,
) {
    engine().renderer.set_material_texture_array(
        material as u32, slot as u32, array as u32,
    );
}

/// EN-012 — set the shading model for a material (0=default lit,
/// 1=foliage, 2=subsurface V2 stub).
#[no_mangle]
pub extern "C" fn bloom_set_material_shading_model(
    material: f64, model: f64,
) {
    engine().renderer.set_material_shading_model(material as u32, model as u32);
}

/// EN-012 — set the foliage shading parameters for a material.
/// Only takes effect when shading_model == 1 (foliage).
#[no_mangle]
pub extern "C" fn bloom_set_material_foliage(
    material: f64,
    trans_r: f64, trans_g: f64, trans_b: f64,
    trans_amount: f64, wrap_factor: f64,
) {
    engine().renderer.set_material_foliage(
        material as u32,
        [trans_r as f32, trans_g as f32, trans_b as f32],
        trans_amount as f32, wrap_factor as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_from_file(
    path_ptr: *const u8,
    bucket_kind: f64,
) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let path = str_from_header(path_ptr);
    let (profile, bucket, reads_scene) = match bucket_kind as u32 {
        0 => (FragmentProfile::Opaque,      Bucket::Opaque,      false),
        1 => (FragmentProfile::Translucent, Bucket::Transparent, false),
        2 => (FragmentProfile::Translucent, Bucket::Refractive,  true),
        3 => (FragmentProfile::Translucent, Bucket::Additive,    false),
        4 => (FragmentProfile::Opaque,      Bucket::Cutout,      false),
        _ => {
            eprintln!("[material] from_file: unknown bucket_kind {bucket_kind}");
            return 0.0;
        }
    };
    match engine().renderer.compile_material_from_file(
        std::path::Path::new(path), profile, bucket, reads_scene,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] from_file failed: {e}"); 0.0 }
    }
}

/// EN-017 — compile + install a fullscreen post-pass material.
/// See `bloom-macos::bloom_set_post_pass` for the full ABI.
#[no_mangle]
pub extern "C" fn bloom_set_post_pass(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    match engine().renderer.set_post_pass(source) {
        Ok(()) => 1.0,
        Err(e) => { eprintln!("[post_pass] compile failed: {:?}", e); 0.0 }
    }
}

/// EN-017 — uninstall the active post-pass.
#[no_mangle]
pub extern "C" fn bloom_clear_post_pass() {
    engine().renderer.clear_post_pass();
}

/// EN-017 V2 — append a post-pass to the stack.
/// See `bloom-macos::bloom_add_post_pass` for the full ABI.
#[no_mangle]
pub extern "C" fn bloom_add_post_pass(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    match engine().renderer.add_post_pass(source) {
        Ok(h) => h as f64,
        Err(e) => { eprintln!("[post_pass] compile failed: {:?}", e); 0.0 }
    }
}

/// EN-017 V2 — wipe the entire post-pass stack.
#[no_mangle]
pub extern "C" fn bloom_clear_all_post_passes() {
    engine().renderer.clear_all_post_passes();
}

#[no_mangle]
pub extern "C" fn bloom_draw_material(
    material: f64,
    mesh_handle: f64,
    mesh_idx: f64,
    x: f64, y: f64, z: f64, scale: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    let eng = engine();
    let handle_bits = mesh_handle.to_bits();
    if let Some(model) = eng.models.get(mesh_handle) {
        eng.renderer.cache_model_if_static(handle_bits, &model.meshes);
    }
    eng.renderer.submit_material_draw(
        material as u32,
        handle_bits,
        mesh_idx as usize,
        [x as f32, y as f32, z as f32],
        scale as f32,
        [(r / 255.0) as f32, (g / 255.0) as f32, (b / 255.0) as f32, (a / 255.0) as f32],
    );
}

#[no_mangle]
pub extern "C" fn bloom_load_model_animation(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) {
        Ok(data) => engine().models.load_model_animation(&data),
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_update_model_animation(handle: f64, anim_index: f64, time: f64, scale: f64, px: f64, py: f64, pz: f64, rot_sin: f64, rot_cos: f64) {
    let eng = engine();
    eng.models.update_model_animation(handle, anim_index as usize, time as f32);
    if let Some(anim) = eng.models.get_animation(handle) {
        if !anim.joint_matrices.is_empty() {
            eng.renderer.set_joint_matrices_scaled(&anim.joint_matrices, scale as f32, [px as f32, py as f32, pz as f32], rot_sin as f32, rot_cos as f32);
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_get_model_mesh_count(handle: f64) -> f64 {
    match engine().models.get(handle) {
        Some(model) => model.meshes.len() as f64,
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_get_model_material_count(handle: f64) -> f64 {
    match engine().models.get(handle) {
        Some(model) => model.meshes.len() as f64,
        None => 0.0,
    }
}

// ============================================================
// CoreAudio (iOS) — RemoteIO Audio Unit
// ============================================================

type AudioUnit = *mut c_void;
type OSStatus = i32;
type AudioUnitPropertyID = u32;
type AudioUnitScope = u32;
type AudioUnitElement = u32;

#[repr(C)]
#[derive(Clone, Copy)]
struct AudioComponentDescription {
    component_type: u32,
    component_sub_type: u32,
    component_manufacturer: u32,
    component_flags: u32,
    component_flags_mask: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AudioStreamBasicDescription {
    sample_rate: f64,
    format_id: u32,
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
    reserved: u32,
}

#[repr(C)]
struct AudioBufferList {
    number_buffers: u32,
    buffers: [AudioBufferData; 1],
}

#[repr(C)]
struct AudioBufferData {
    number_channels: u32,
    data_byte_size: u32,
    data: *mut c_void,
}

type AURenderCallback = unsafe extern "C" fn(
    in_ref_con: *mut c_void,
    io_action_flags: *mut u32,
    in_time_stamp: *const c_void,
    in_bus_number: u32,
    in_number_frames: u32,
    io_data: *mut AudioBufferList,
) -> OSStatus;

#[repr(C)]
struct AURenderCallbackStruct {
    input_proc: AURenderCallback,
    input_proc_ref_con: *mut c_void,
}

type AudioComponent = *mut c_void;

#[link(name = "AudioToolbox", kind = "framework")]
extern "C" {
    fn AudioComponentFindNext(component: AudioComponent, desc: *const AudioComponentDescription) -> AudioComponent;
    fn AudioComponentInstanceNew(component: AudioComponent, out: *mut AudioUnit) -> OSStatus;
    fn AudioUnitSetProperty(
        unit: AudioUnit,
        property_id: AudioUnitPropertyID,
        scope: AudioUnitScope,
        element: AudioUnitElement,
        data: *const c_void,
        data_size: u32,
    ) -> OSStatus;
    fn AudioUnitInitialize(unit: AudioUnit) -> OSStatus;
    fn AudioOutputUnitStart(unit: AudioUnit) -> OSStatus;
    fn AudioOutputUnitStop(unit: AudioUnit) -> OSStatus;
    fn AudioComponentInstanceDispose(unit: AudioUnit) -> OSStatus;
}

const K_AUDIO_UNIT_TYPE_OUTPUT: u32 = u32::from_be_bytes(*b"auou");
const K_AUDIO_UNIT_SUB_TYPE_REMOTE_IO: u32 = u32::from_be_bytes(*b"rioc");
const K_AUDIO_UNIT_MANUFACTURER_APPLE: u32 = u32::from_be_bytes(*b"appl");

const K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT: AudioUnitPropertyID = 8;
const K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK: AudioUnitPropertyID = 23;
const K_AUDIO_UNIT_SCOPE_INPUT: AudioUnitScope = 1;

const K_AUDIO_FORMAT_LINEAR_PCM: u32 = u32::from_be_bytes(*b"lpcm");
const K_AUDIO_FORMAT_FLAG_IS_FLOAT: u32 = 1;
const K_AUDIO_FORMAT_FLAG_IS_PACKED: u32 = 8;

struct AudioUnitInstance { unit: AudioUnit }
unsafe impl Send for AudioUnitInstance {}
unsafe impl Sync for AudioUnitInstance {}

static mut AUDIO_UNIT: Option<AudioUnitInstance> = None;

unsafe extern "C" fn audio_render_callback(
    _in_ref_con: *mut c_void,
    _io_action_flags: *mut u32,
    _in_time_stamp: *const c_void,
    _in_bus_number: u32,
    in_number_frames: u32,
    io_data: *mut AudioBufferList,
) -> OSStatus {
    let buffer_list = &mut *io_data;
    let buffer = &mut buffer_list.buffers[0];
    let num_samples = in_number_frames as usize * 2;
    let output = std::slice::from_raw_parts_mut(buffer.data as *mut f32, num_samples);
    ENGINE.get_mut().map(|eng| { eng.audio.mix_output(output); });
    0
}

#[no_mangle]
pub extern "C" fn bloom_init_audio() {
    unsafe {
        let desc = AudioComponentDescription {
            component_type: K_AUDIO_UNIT_TYPE_OUTPUT,
            component_sub_type: K_AUDIO_UNIT_SUB_TYPE_REMOTE_IO,
            component_manufacturer: K_AUDIO_UNIT_MANUFACTURER_APPLE,
            component_flags: 0,
            component_flags_mask: 0,
        };

        let component = AudioComponentFindNext(std::ptr::null_mut(), &desc);
        if component.is_null() { return; }

        let mut unit: AudioUnit = std::ptr::null_mut();
        if AudioComponentInstanceNew(component, &mut unit) != 0 { return; }

        let stream_desc = AudioStreamBasicDescription {
            sample_rate: 44100.0,
            format_id: K_AUDIO_FORMAT_LINEAR_PCM,
            format_flags: K_AUDIO_FORMAT_FLAG_IS_FLOAT | K_AUDIO_FORMAT_FLAG_IS_PACKED,
            bytes_per_packet: 8,
            frames_per_packet: 1,
            bytes_per_frame: 8,
            channels_per_frame: 2,
            bits_per_channel: 32,
            reserved: 0,
        };

        AudioUnitSetProperty(
            unit, K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT, K_AUDIO_UNIT_SCOPE_INPUT, 0,
            &stream_desc as *const _ as *const c_void,
            std::mem::size_of::<AudioStreamBasicDescription>() as u32,
        );

        let callback_struct = AURenderCallbackStruct {
            input_proc: audio_render_callback,
            input_proc_ref_con: std::ptr::null_mut(),
        };

        AudioUnitSetProperty(
            unit, K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK, K_AUDIO_UNIT_SCOPE_INPUT, 0,
            &callback_struct as *const _ as *const c_void,
            std::mem::size_of::<AURenderCallbackStruct>() as u32,
        );

        AudioUnitInitialize(unit);
        AudioOutputUnitStart(unit);
        AUDIO_UNIT = Some(AudioUnitInstance { unit });
    }
}

#[no_mangle]
pub extern "C" fn bloom_close_audio() {
    unsafe {
        if let Some(au) = AUDIO_UNIT.take() {
            AudioOutputUnitStop(au.unit);
            AudioComponentInstanceDispose(au.unit);
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_load_sound(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) {
        Ok(data) => {
            if let Some(s) = parse_wav(&data) {
                engine().audio.load_sound(s)
            } else if let Some(s) = parse_ogg(&data) {
                engine().audio.load_sound(s)
            } else if let Some(s) = parse_mp3(&data) {
                engine().audio.load_sound(s)
            } else {
                0.0
            }
        }
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_play_sound(handle: f64) { engine().audio.play_sound(handle); }
#[no_mangle]
pub extern "C" fn bloom_stop_sound(handle: f64) { engine().audio.stop_sound(handle); }
#[no_mangle]
pub extern "C" fn bloom_set_sound_volume(handle: f64, volume: f64) { engine().audio.set_sound_volume(handle, volume as f32); }
#[no_mangle]
pub extern "C" fn bloom_set_master_volume(volume: f64) { engine().audio.master_volume = volume as f32; }

#[no_mangle]
pub extern "C" fn bloom_play_sound_3d(handle: f64, x: f64, y: f64, z: f64) {
    engine().audio.play_sound_3d(handle, x as f32, y as f32, z as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_listener_position(x: f64, y: f64, z: f64, fx: f64, fy: f64, fz: f64) {
    engine().audio.set_listener_position(x as f32, y as f32, z as f32, fx as f32, fy as f32, fz as f32);
}

// ============================================================
// Music
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_load_music(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) {
        Ok(data) => {
            if let Some(s) = parse_ogg(&data) {
                engine().audio.load_music(s)
            } else if let Some(s) = parse_wav(&data) {
                engine().audio.load_music(s)
            } else if let Some(s) = parse_mp3(&data) {
                engine().audio.load_music(s)
            } else {
                0.0
            }
        }
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_play_music(handle: f64) { engine().audio.play_music(handle); }
#[no_mangle]
pub extern "C" fn bloom_stop_music(handle: f64) { engine().audio.stop_music(handle); }
#[no_mangle]
pub extern "C" fn bloom_update_music_stream(handle: f64) { engine().audio.update_music_stream(handle); }
#[no_mangle]
pub extern "C" fn bloom_set_music_volume(handle: f64, volume: f64) { engine().audio.set_music_volume(handle, volume as f32); }
#[no_mangle]
pub extern "C" fn bloom_is_music_playing(handle: f64) -> f64 {
    if engine().audio.is_music_playing(handle) { 1.0 } else { 0.0 }
}

// ============================================================
// Gamepad input
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_is_gamepad_available(_gamepad: f64) -> f64 {
    if engine().input.is_gamepad_available() { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_get_gamepad_axis(_gamepad: f64, axis: f64) -> f64 {
    engine().input.get_gamepad_axis(axis as usize) as f64
}

#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_pressed(_gamepad: f64, button: f64) -> f64 {
    if engine().input.is_gamepad_button_pressed(button as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_down(_gamepad: f64, button: f64) -> f64 {
    if engine().input.is_gamepad_button_down(button as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_released(_gamepad: f64, button: f64) -> f64 {
    if engine().input.is_gamepad_button_released(button as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_get_gamepad_axis_count(_gamepad: f64) -> f64 {
    engine().input.get_gamepad_axis_count() as f64
}

// ============================================================
// Touch input
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_get_touch_x(index: f64) -> f64 {
    engine().input.get_touch_x(index as usize)
}

#[no_mangle]
pub extern "C" fn bloom_get_touch_y(index: f64) -> f64 {
    engine().input.get_touch_y(index as usize)
}

#[no_mangle]
pub extern "C" fn bloom_get_touch_count() -> f64 {
    engine().input.get_touch_count() as f64
}

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

#[no_mangle]
pub extern "C" fn bloom_get_mouse_delta_x() -> f64 {
    engine().input.mouse_delta_x
}

#[no_mangle]
pub extern "C" fn bloom_get_mouse_delta_y() -> f64 {
    engine().input.mouse_delta_y
}

// Accumulated scroll wheel delta since the last call. Reading consumes the
// value (returns 0 on the next call until the user scrolls again). Used by
// the editor's orbit camera and any scrollable UI panel.
#[no_mangle]
pub extern "C" fn bloom_get_mouse_wheel() -> f64 {
    engine().input.consume_mouse_wheel()
}

#[no_mangle]
pub extern "C" fn bloom_get_char_pressed() -> f64 {
    engine().input.pop_char() as f64
}

// Q2: Cursor shape
#[no_mangle]
pub extern "C" fn bloom_set_cursor_shape(shape: f64) {
    engine().input.cursor_shape = shape as u32;
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

// Model bounds accessors. Return the axis-aligned bounding box of a loaded
// model in model-local coordinates. Editors use these to size gizmos, auto-
// frame the camera on selection, and snap placed entities onto terrain.
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_min_x(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).0[0] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_min_y(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).0[1] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_min_z(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).0[2] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_max_x(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).1[0] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_max_y(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).1[1] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_max_z(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).1[2] as f64
}

#[no_mangle]
pub extern "C" fn bloom_write_file(path_ptr: *const u8, data_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    let data = str_from_header(data_ptr);
    match std::fs::write(path, data.as_bytes()) {
        Ok(_) => 1.0,
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_file_exists(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    let resolved = resolve_path(path);
    if std::path::Path::new(&resolved).exists() { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_read_file(path_ptr: *const u8) -> *const u8 {
    let path = str_from_header(path_ptr);
    match std::fs::read_to_string(resolve_path(path)) {
        Ok(contents) => {
            // Return Perry-format string: StringHeader (length u32 + capacity u32 + refcount u32) followed by UTF-8 data
            let bytes = contents.as_bytes();
            let len = bytes.len();
            let total = 12 + len; // 12 bytes header (3 × u32) + data
            let layout = std::alloc::Layout::from_size_align(total, 4).unwrap();
            unsafe {
                let ptr = std::alloc::alloc(layout);
                if ptr.is_null() { return std::ptr::null(); }
                *(ptr as *mut u32) = len as u32;           // length
                *(ptr.add(4) as *mut u32) = len as u32;    // capacity
                *(ptr.add(8) as *mut u32) = 1;             // refcount (unique)
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(12), len);
                ptr
            }
        }
        Err(_) => std::ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn bloom_get_time() -> f64 {
    engine().get_time()
}

// ============================================================
// Input injection + platform detection
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_inject_key_down(key: f64) {
    engine().input.set_key_down(key as usize);
}
#[no_mangle]
pub extern "C" fn bloom_inject_key_up(key: f64) {
    engine().input.set_key_up(key as usize);
}
#[no_mangle]
pub extern "C" fn bloom_inject_gamepad_axis(axis: f64, value: f64) {
    engine().input.set_gamepad_axis(axis as usize, value as f32);
}
#[no_mangle]
pub extern "C" fn bloom_inject_gamepad_button_down(button: f64) {
    engine().input.set_gamepad_button_down(button as usize);
}
#[no_mangle]
pub extern "C" fn bloom_inject_gamepad_button_up(button: f64) {
    engine().input.set_gamepad_button_up(button as usize);
}
#[no_mangle]
pub extern "C" fn bloom_get_platform() -> f64 { 2.0 }
#[no_mangle]
pub extern "C" fn bloom_is_any_input_pressed() -> f64 {
    if engine().input.is_any_input_pressed() { 1.0 } else { 0.0 }
}
#[no_mangle]
pub extern "C" fn bloom_get_crown_rotation() -> f64 {
    engine().input.consume_crown_rotation()
}

// ============================================================
// Thread-safe staging (for async asset loading via Perry threads)
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_stage_texture(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(resolve_path(path)) {
        Ok(data) => bloom_shared::staging::decode_and_stage_texture(&data),
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_stage_model(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    let data = match std::fs::read(resolve_path(path)) {
        Ok(d) => d,
        Err(_) => return 0.0,
    };
    match bloom_shared::models::load_gltf_staged(&data) {
        Some(staged) => bloom_shared::staging::stage_model(staged),
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_stage_sound(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    let data = match std::fs::read(resolve_path(path)) {
        Ok(d) => d,
        Err(_) => return 0.0,
    };
    let sound_data = if path.ends_with(".ogg") || path.ends_with(".OGG") {
        parse_ogg(&data)
    } else if path.ends_with(".mp3") || path.ends_with(".MP3") {
        parse_mp3(&data)
    } else {
        parse_wav(&data)
    };
    match sound_data {
        Some(sd) => bloom_shared::staging::stage_sound(sd),
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_commit_texture(staging_handle: f64) -> f64 {
    let staged = match bloom_shared::staging::take_texture(staging_handle) {
        Some(s) => s,
        None => return 0.0,
    };
    let eng = engine();
    let bind_group_idx = eng.renderer.register_texture(staged.width, staged.height, &staged.data);
    eng.textures.textures.alloc(bloom_shared::textures::TextureData {
        bind_group_idx, width: staged.width, height: staged.height,
    })
}

#[no_mangle]
pub extern "C" fn bloom_commit_model(staging_handle: f64) -> f64 {
    let staged = match bloom_shared::staging::take_model(staging_handle) {
        Some(s) => s,
        None => return 0.0,
    };
    let eng = engine();
    let mut tex_map: Vec<u32> = Vec::with_capacity(staged.textures.len());
    for tex in &staged.textures {
        tex_map.push(eng.renderer.register_texture(tex.width, tex.height, &tex.data));
    }
    let mut model = staged.model;
    for mesh in &mut model.meshes {
        if let Some(ref mut idx) = mesh.texture_idx {
            let staged_idx = *idx as usize;
            if staged_idx > 0 && staged_idx <= tex_map.len() {
                *idx = tex_map[staged_idx - 1];
            } else {
                mesh.texture_idx = None;
            }
        }
    }
    eng.models.models.alloc(model)
}

#[no_mangle]
pub extern "C" fn bloom_commit_sound(staging_handle: f64) -> f64 {
    match bloom_shared::staging::take_sound(staging_handle) {
        Some(sd) => engine().audio.load_sound(sd),
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_commit_music(staging_handle: f64) -> f64 {
    match bloom_shared::staging::take_sound(staging_handle) {
        Some(sd) => engine().audio.load_music(sd),
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_run_game(_callback: extern "C" fn(f64)) {
    // No-op on native. The TypeScript runGame() helper provides the while loop.
}


// Q6: Multi-hit picking
static mut LAST_PICK_ALL: Vec<bloom_shared::picking::PickResult> = Vec::new();

#[no_mangle]
pub extern "C" fn bloom_scene_pick_all(screen_x: f64, screen_y: f64, max_results: f64) -> f64 {
    let eng = engine();
    let inv_vp = eng.renderer.inverse_vp_matrix();
    let cam_pos = eng.renderer.camera_pos();
    let w = eng.renderer.width() as f32;
    let h = eng.renderer.height() as f32;
    let (origin, direction) = bloom_shared::picking::screen_to_ray(
        screen_x as f32, screen_y as f32, w, h, &inv_vp, &cam_pos,
    );
    let results = bloom_shared::picking::raycast_scene_all(&eng.scene, &origin, &direction, max_results as usize);
    let count = results.len();
    unsafe { LAST_PICK_ALL = results; }
    count as f64
}
#[no_mangle]
pub extern "C" fn bloom_pick_all_handle(index: f64) -> f64 {
    let i = index as usize;
    unsafe { LAST_PICK_ALL.get(i).map(|r| r.handle).unwrap_or(0.0) }
}
#[no_mangle]
pub extern "C" fn bloom_pick_all_distance(index: f64) -> f64 {
    let i = index as usize;
    unsafe { LAST_PICK_ALL.get(i).map(|r| r.distance as f64).unwrap_or(0.0) }
}
// ============================================================

// ============================================================
// Render quality toggles (individual + preset) — ticket 011
// Mirror of the macOS FFI surface added in commit 95da6af; previously
// macOS-only, now exposed on every native platform so non-macOS builds
// don't fail at runtime (missing symbol) when the TS API invokes them.
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_set_quality_preset(preset: f64) {
    engine().renderer.apply_quality_preset(preset as u32);
}
#[no_mangle]
pub extern "C" fn bloom_set_shadows_enabled(on: f64) {
    engine().renderer.set_shadows_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_shadows_always_fresh(on: f64) {
    engine().renderer.set_shadows_always_fresh(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_bloom_enabled(on: f64) {
    engine().renderer.set_bloom_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_ssao_enabled(on: f64) {
    engine().renderer.set_ssao_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_ssao_intensity(value: f64) {
    engine().renderer.set_ssao_strength(value as f32);
}
#[no_mangle]
pub extern "C" fn bloom_set_ssao_radius(world_radius: f64) {
    engine().renderer.set_ssao_radius(world_radius as f32);
}
#[no_mangle]
pub extern "C" fn bloom_set_wind(dir_x: f64, dir_z: f64, amplitude: f64, frequency: f64) {
    engine().renderer.set_wind(dir_x as f32, dir_z as f32, amplitude as f32, frequency as f32);
}
#[no_mangle]
pub extern "C" fn bloom_set_ssr_enabled(on: f64) {
    engine().renderer.set_ssr_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_motion_blur_enabled(on: f64) {
    engine().renderer.set_motion_blur_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_sss_enabled(on: f64) {
    engine().renderer.set_sss_enabled(on != 0.0);
}

// ============================================================
// Profiler — CPU phase timings (always available) + GPU timestamps
// (when the adapter supports TIMESTAMP_QUERY). Disabled by default.
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_set_profiler_enabled(on: f64) {
    engine().profiler.set_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_get_profiler_frame_cpu_us() -> f64 {
    engine().profiler.avg_frame_cpu_us()
}
#[no_mangle]
pub extern "C" fn bloom_get_profiler_frame_gpu_us() -> f64 {
    engine().profiler.avg_frame_gpu_us()
}
#[no_mangle]
pub extern "C" fn bloom_print_profiler_summary() {
    print!("{}", engine().profiler.summary());
}

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

// ---- FFI parity port from macos (iOS crate drift fix) ----
#[no_mangle]
pub extern "C" fn bloom_take_screenshot(path_ptr: *const u8) {
    let path = str_from_header(path_ptr).to_string();
    let eng = engine();
    eng.renderer.screenshot_requested = true;
    eng.renderer.pending_screenshot_path = Some(path);
}

#[no_mangle]
pub extern "C" fn bloom_set_env_clear_from_hdr(path_ptr: *const u8) {
    use image::ImageDecoder;
    let path = str_from_header(path_ptr).to_string();
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let decoder = match image::codecs::hdr::HdrDecoder::new(std::io::BufReader::new(file)) {
        Ok(d) => d,
        Err(_) => return,
    };
    let (w, h) = decoder.dimensions();
    let byte_len = (w as usize) * (h as usize) * 3 * 4;
    let mut buf = vec![0u8; byte_len];
    if decoder.read_image(&mut buf).is_err() {
        return;
    }
    // Reinterpret the byte buffer as f32 RGB triples for the renderer.
    let rgb_f32: Vec<f32> = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    engine().renderer.load_env_from_hdr(w, h, &rgb_f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_fog(r: f64, g: f64, b: f64, density: f64, height_ref: f64, height_falloff: f64) {
    let r_ = engine();
    r_.renderer.set_fog_color(r as f32, g as f32, b as f32);
    r_.renderer.set_fog_density(density as f32);
    r_.renderer.set_fog_height_falloff(height_ref as f32, height_falloff as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_chromatic_aberration(strength: f64) {
    engine().renderer.set_chromatic_aberration(strength as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_vignette(strength: f64, softness: f64) {
    engine().renderer.set_vignette(strength as f32, softness as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_film_grain(strength: f64) {
    engine().renderer.set_film_grain(strength as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_sun_shafts(strength: f64, decay: f64, r: f64, g: f64, b: f64) {
    let eng = engine();
    eng.renderer.set_sun_shaft_strength(strength as f32);
    eng.renderer.set_sun_shaft_decay(decay as f32);
    eng.renderer.set_sun_shaft_color(r as f32, g as f32, b as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_auto_exposure(on: f64) {
    engine().renderer.set_auto_exposure(on != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_set_taa_enabled(on: f64) {
    engine().renderer.set_taa_enabled(on != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_set_render_scale(scale: f64) {
    engine().renderer.set_render_scale(scale as f32);
}

#[no_mangle]
pub extern "C" fn bloom_get_render_scale() -> f64 {
    engine().renderer.render_scale() as f64
}

#[no_mangle]
pub extern "C" fn bloom_set_upscale_mode(mode: f64) {
    engine().renderer.set_upscale_mode(mode as u32);
}

#[no_mangle]
pub extern "C" fn bloom_set_cas_strength(strength: f64) {
    engine().renderer.set_cas_strength(strength as f32);
}

#[no_mangle]
pub extern "C" fn bloom_get_physical_width() -> f64 {
    engine().renderer.physical_width() as f64
}

#[no_mangle]
pub extern "C" fn bloom_get_physical_height() -> f64 {
    engine().renderer.physical_height() as f64
}

#[no_mangle]
pub extern "C" fn bloom_set_auto_resolution(target_hz: f64, enabled: f64) {
    let eng = engine();
    if enabled != 0.0 {
        let current = eng.renderer.render_scale();
        eng.drs.enable(target_hz as f32, current);
    } else {
        eng.drs.disable();
    }
}

#[no_mangle]
pub extern "C" fn bloom_set_manual_exposure(value: f64) {
    engine().renderer.set_manual_exposure(value as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_env_intensity(intensity: f64) {
    engine().renderer.set_env_intensity(intensity as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_ssgi_enabled(enabled: f64) {
    engine().renderer.set_ssgi_enabled(enabled != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_set_ssgi_intensity(intensity: f64) {
    engine().renderer.set_ssgi_intensity(intensity as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_ssgi_radius(radius: f64) {
    engine().renderer.set_ssgi_radius(radius as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_dof(enabled: f64, focus_distance: f64, aperture: f64) {
    let r = &mut engine().renderer;
    r.set_dof_enabled(enabled != 0.0);
    r.set_dof_focus_distance(focus_distance as f32);
    r.set_dof_aperture(aperture as f32);
}

#[no_mangle]
pub extern "C" fn bloom_splat_impulse(x: f64, z: f64, radius: f64, strength: f64) {
    engine().renderer.impulse_field.submit_splat(
        x as f32, z as f32, radius as f32, strength as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_profiler_frame_history() -> *const u8 {
    let hist = engine().profiler.frame_history();
    let mut s = String::with_capacity(hist.len() * 24);
    for (cpu, gpu) in &hist {
        s.push_str(&format!("{:.2}|{:.2}\n", cpu, gpu));
    }
    alloc_perry_string(&s)
}

#[no_mangle]
pub extern "C" fn bloom_profiler_overlay_text() -> *const u8 {
    let snap = engine().profiler.snapshot();
    let mut s = String::with_capacity(snap.len() * 48);
    for (label, cpu, gpu) in &snap {
        s.push_str(label);
        s.push('|');
        s.push_str(&format!("{:.2}", cpu));
        s.push('|');
        match gpu {
            Some(g) => s.push_str(&format!("{:.2}", g)),
            None    => s.push_str("-1"),
        }
        s.push('\n');
    }
    alloc_perry_string(&s)
}

#[no_mangle]
pub extern "C" fn bloom_register_frame_callback(priority: f64, callback: extern "C" fn(f64)) -> f64 {
    engine().frame_callbacks.register(priority as i32, callback) as f64
}

#[no_mangle]
pub extern "C" fn bloom_unregister_frame_callback(id: f64) {
    engine().frame_callbacks.unregister(id as u64);
}

#[no_mangle]
pub extern "C" fn bloom_add_directional_light(
    dx: f64, dy: f64, dz: f64,
    r: f64, g: f64, b: f64,
    intensity: f64,
) {
    engine().renderer.add_directional_light(
        dx as f32, dy as f32, dz as f32,
        r as f32, g as f32, b as f32,
        intensity as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_add_point_light(
    x: f64, y: f64, z: f64, range: f64,
    r: f64, g: f64, b: f64,
    intensity: f64,
) {
    engine().renderer.add_point_light(
        x as f32, y as f32, z as f32, range as f32,
        r as f32, g as f32, b as f32,
        intensity as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_scene_create_node() -> f64 {
    engine().scene.create_node()
}

#[no_mangle]
pub extern "C" fn bloom_scene_destroy_node(handle: f64) {
    engine().scene.destroy_node(handle);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_visible(handle: f64, visible: f64) {
    engine().scene.set_visible(handle, visible != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_cast_shadow(handle: f64, cast: f64) {
    engine().scene.set_cast_shadow(handle, cast != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_receive_shadow(handle: f64, receive: f64) {
    engine().scene.set_receive_shadow(handle, receive != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_parent(handle: f64, parent: f64) {
    engine().scene.set_parent(handle, parent);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_transform(handle: f64, mat_ptr: *const f64) {
    if mat_ptr.is_null() { return; }
    let slice = unsafe { std::slice::from_raw_parts(mat_ptr, 16) };
    let mut mat = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            mat[col][row] = slice[col * 4 + row] as f32;
        }
    }
    engine().scene.set_transform(handle, mat);
}

#[no_mangle]
pub extern "C" fn bloom_scene_update_geometry(
    handle: f64,
    vert_ptr: *const f64,
    vert_count: f64,
    idx_ptr: *const f64,
    idx_count: f64,
) {
    if vert_ptr.is_null() || idx_ptr.is_null() { return; }
    let nv = vert_count as usize;
    let ni = idx_count as usize;

    let vert_floats = unsafe { std::slice::from_raw_parts(vert_ptr, nv * 12) };
    let idx_floats = unsafe { std::slice::from_raw_parts(idx_ptr, ni) };

    let mut vertices = Vec::with_capacity(nv);
    for i in 0..nv {
        let base = i * 12;
        vertices.push(bloom_shared::renderer::Vertex3D {
            position: [vert_floats[base] as f32, vert_floats[base+1] as f32, vert_floats[base+2] as f32],
            normal: [vert_floats[base+3] as f32, vert_floats[base+4] as f32, vert_floats[base+5] as f32],
            color: [vert_floats[base+6] as f32, vert_floats[base+7] as f32, vert_floats[base+8] as f32, vert_floats[base+9] as f32],
            uv: [vert_floats[base+10] as f32, vert_floats[base+11] as f32],
            joints: [0.0; 4],
            weights: [0.0; 4],
            tangent: [0.0; 4],
        });
    }

    let indices: Vec<u32> = idx_floats.iter().map(|&v| v as u32).collect();

    engine().scene.update_geometry(handle, vertices, indices);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_material_color(handle: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().scene.set_material_color(handle, r as f32, g as f32, b as f32, a as f32);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_material_pbr(handle: f64, roughness: f64, metalness: f64) {
    engine().scene.set_material_pbr(handle, roughness as f32, metalness as f32);
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_material_texture(handle: f64, texture_idx: f64) {
    engine().scene.set_material_texture(handle, texture_idx as u32);
}

#[no_mangle]
pub extern "C" fn bloom_scene_node_count() -> f64 {
    engine().scene.node_count() as f64
}

#[no_mangle]
pub extern "C" fn bloom_scene_node_vertex_count(handle: f64) -> f64 {
    match engine().scene.nodes.get(handle) {
        Some(node) => node.vertices.len() as f64,
        None => -1.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_scene_node_index_count(handle: f64) -> f64 {
    match engine().scene.nodes.get(handle) {
        Some(node) => node.indices.len() as f64,
        None => -1.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_scene_set_material_water(handle: f64, wave_amp: f64, wave_speed: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().scene.set_material_water(handle, wave_amp as f32, wave_speed as f32, r as f32, g as f32, b as f32, a as f32);
}

#[no_mangle]
pub extern "C" fn bloom_gen_mesh_spline_ribbon(points_ptr: *const u8, point_count: f64, widths_ptr: *const u8, width_count: f64) -> f64 {
    let n = point_count as usize;
    let wn = width_count as usize;
    let points = unsafe { std::slice::from_raw_parts(points_ptr as *const f32, n * 3) };
    let widths = unsafe { std::slice::from_raw_parts(widths_ptr as *const f32, wn) };
    engine().models.gen_mesh_spline_ribbon(points, widths)
}

#[no_mangle]
pub extern "C" fn bloom_load_render_texture(width: f64, height: f64) -> f64 {
    let w = width as u32;
    let h = height as u32;
    let eng = engine();
    let rt_handle = eng.textures.load_render_texture(w, h);

    // Create the GPU texture via the renderer's public method.
    let (bind_group_idx, _tex_vec_idx) = eng.renderer.create_render_texture(w, h);

    // Register as a texture handle so drawTexture can sample it.
    let tex_handle = eng.textures.textures.alloc(bloom_shared::textures::TextureData {
        bind_group_idx, width: w, height: h,
    });
    eng.textures.set_render_texture_handle(rt_handle, tex_handle);

    rt_handle
}

#[no_mangle]
pub extern "C" fn bloom_unload_render_texture(handle: f64) {
    engine().textures.unload_render_texture(handle);
}

#[no_mangle]
pub extern "C" fn bloom_begin_texture_mode(handle: f64) {
    let eng = engine();
    let (w, h, bg_idx) = match eng.textures.render_textures.get(handle) {
        Some(rt) => {
            let tex_handle = rt.texture_handle;
            match eng.textures.textures.get(tex_handle) {
                Some(td) => (rt.width, rt.height, td.bind_group_idx as usize),
                None => return,
            }
        }
        None => return,
    };
    if let Some(texture) = eng.renderer.get_texture_ref(bg_idx) {
        // We need to call begin_texture_mode with a reference to the texture,
        // but get_texture_ref borrows renderer immutably. Clone the texture view
        // data we need first, then call the mutable method.
        let color_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        // Create depth texture for this RT.
        let depth_tex = eng.renderer.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rt_depth"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float, usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        eng.renderer.rt_color_view = Some(color_view);
        eng.renderer.rt_depth_view = Some(depth_view);
        eng.renderer.rt_depth_texture = Some(depth_tex);
        eng.renderer.rt_width = w;
        eng.renderer.rt_height = h;
    }
}

#[no_mangle]
pub extern "C" fn bloom_end_texture_mode() {
    engine().renderer.end_texture_mode();
}

#[no_mangle]
pub extern "C" fn bloom_get_render_texture_texture(handle: f64) -> f64 {
    engine().textures.get_render_texture_texture(handle)
}

#[no_mangle]
pub extern "C" fn bloom_scene_get_transform(handle: f64, index: f64) -> f64 {
    let mat = engine().scene.get_transform(handle);
    let i = index as usize;
    let col = i / 4;
    let row = i % 4;
    if col < 4 && row < 4 { mat[col][row] as f64 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_min_x(handle: f64) -> f64 { engine().scene.get_bounds(handle).0[0] as f64 }

#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_min_y(handle: f64) -> f64 { engine().scene.get_bounds(handle).0[1] as f64 }

#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_min_z(handle: f64) -> f64 { engine().scene.get_bounds(handle).0[2] as f64 }

#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_max_x(handle: f64) -> f64 { engine().scene.get_bounds(handle).1[0] as f64 }

#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_max_y(handle: f64) -> f64 { engine().scene.get_bounds(handle).1[1] as f64 }

#[no_mangle]
pub extern "C" fn bloom_scene_get_bounds_max_z(handle: f64) -> f64 { engine().scene.get_bounds(handle).1[2] as f64 }

#[no_mangle]
pub extern "C" fn bloom_scene_set_user_data(handle: f64, data: f64) {
    engine().scene.set_user_data(handle, data as i64);
}

#[no_mangle]
pub extern "C" fn bloom_scene_get_user_data(handle: f64) -> f64 {
    engine().scene.get_user_data(handle) as f64
}

#[no_mangle]
pub extern "C" fn bloom_scene_extrude_polygon(
    handle: f64,
    polygon_ptr: *const f64,
    polygon_count: f64,
    depth: f64,
) {
    if polygon_ptr.is_null() { return; }
    let n = polygon_count as usize;
    let polygon = unsafe { std::slice::from_raw_parts(polygon_ptr, n * 2) };

    let geo = bloom_shared::geometry::extrude_polygon(polygon, &[], depth);
    engine().scene.update_geometry(handle, geo.vertices, geo.indices);
}

#[no_mangle]
pub extern "C" fn bloom_scene_subtract_box(
    handle: f64,
    min_x: f64, min_y: f64, min_z: f64,
    max_x: f64, max_y: f64, max_z: f64,
) {
    let eng = engine();
    if let Some(node) = eng.scene.nodes.get(handle) {
        let current = bloom_shared::geometry::GeometryData {
            vertices: node.vertices.clone(),
            indices: node.indices.clone(),
        };
        let result = bloom_shared::geometry::subtract_box(
            &current,
            [min_x as f32, min_y as f32, min_z as f32],
            [max_x as f32, max_y as f32, max_z as f32],
        );
        eng.scene.update_geometry(handle, result.vertices, result.indices);
    }
}

#[no_mangle]
pub extern "C" fn bloom_enable_shadows() {
    engine().renderer.shadow_map.enable();
}

#[no_mangle]
pub extern "C" fn bloom_disable_shadows() {
    engine().renderer.shadow_map.disable();
}

#[no_mangle]
pub extern "C" fn bloom_dump_shadow_map(path_ptr: *const u8) {
    let path = str_from_header(path_ptr).to_string();
    engine().renderer.dump_shadow_map(&path);
}

#[no_mangle]
pub extern "C" fn bloom_enable_postfx() {
    let eng = engine();
    let w = eng.renderer.width();
    let h = eng.renderer.height();
    let fmt = eng.renderer.surface_format();
    eng.postfx = Some(bloom_shared::postfx::PostFxPipeline::new(
        &eng.renderer.device, w, h, fmt,
    ));
}

#[no_mangle]
pub extern "C" fn bloom_disable_postfx() {
    engine().postfx = None;
}

#[no_mangle]
pub extern "C" fn bloom_postfx_set_selected(handle: f64) {
    if let Some(pfx) = &mut engine().postfx {
        if handle == 0.0 {
            pfx.set_selected(Vec::new());
        } else {
            pfx.set_selected(vec![handle]);
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_postfx_set_hovered(handle: f64) {
    if let Some(pfx) = &mut engine().postfx {
        pfx.set_hovered(handle);
    }
}

#[no_mangle]
pub extern "C" fn bloom_postfx_set_outline_color(r: f64, g: f64, b: f64, a: f64) {
    if let Some(pfx) = &mut engine().postfx {
        pfx.outline_params.color_selected = [r as f32, g as f32, b as f32, a as f32];
    }
}

#[no_mangle]
pub extern "C" fn bloom_postfx_set_outline_thickness(thickness: f64) {
    if let Some(pfx) = &mut engine().postfx {
        pfx.outline_params.thickness[0] = thickness as f32;
    }
}

#[no_mangle]
pub extern "C" fn bloom_scene_attach_model(node_handle: f64, model_handle: f64, mesh_index: f64) {
    let eng = engine();
    let mi = mesh_index as usize;

    // Get model mesh data
    let model_data = match eng.models.models.get(model_handle) {
        Some(md) => md,
        None => return,
    };
    if mi >= model_data.meshes.len() { return; }
    let mesh = &model_data.meshes[mi];

    // Copy vertices and indices to scene node
    let vertices = mesh.vertices.clone();
    let indices = mesh.indices.clone();
    let base_color_tex = mesh.texture_idx;
    let normal_tex = mesh.normal_texture_idx;
    let mr_tex = mesh.metallic_roughness_texture_idx;
    let emissive_tex = mesh.emissive_texture_idx;
    let emissive_factor = mesh.emissive_factor;
    eng.scene.update_geometry(node_handle, vertices, indices);

    // Pipe PBR textures through to the scene node material so the
    // renderer's scene pipeline can sample them.
    if let Some(tex_idx) = base_color_tex {
        eng.scene.set_material_texture(node_handle, tex_idx);
    }
    if let Some(tex_idx) = normal_tex {
        eng.scene.set_material_normal_texture(node_handle, tex_idx);
    }
    if let Some(tex_idx) = mr_tex {
        eng.scene.set_material_metallic_roughness_texture(node_handle, tex_idx);
    }
    if let Some(tex_idx) = emissive_tex {
        eng.scene.set_material_emissive_texture(node_handle, tex_idx);
    }
    eng.scene.set_material_emissive_factor(
        node_handle,
        emissive_factor[0],
        emissive_factor[1],
        emissive_factor[2],
    );
}

