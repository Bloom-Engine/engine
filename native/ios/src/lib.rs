// `static mut` is intentional throughout this FFI surface — Perry calls
// us from a single OS thread (the UIKit main run-loop), so the engine
// singleton + view/window scratch state never race. The 2024 lint
// flagging `&UI_VIEW`-style accesses is a real concern in
// multi-threaded code, but inapplicable here. Suppress at the crate
// root to avoid a dozen noise lines in every build. Mirrors
// native/macos/src/lib.rs.
#![allow(static_mut_refs)]

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
            eprintln!("[bloom-ios] request_device with preferred features failed ({e:?}); retrying with adapter limits and no ray-query/experimental features");
            // Request exactly the adapter's reported limits (not wgpu's
            // Limits::default()): some iOS GPUs cap individual limits — e.g.
            // max_inter_stage_shader_variables — below wgpu's defaults, so
            // request_device rejects the default-limits request too. Asking for
            // adapter.limits() can never exceed what the device supports.
            pollster_block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("bloom_device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
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
            eprintln!("[bloom-ios] request_device with preferred features failed ({e:?}); retrying with adapter limits and no ray-query/experimental features");
            // Request exactly the adapter's reported limits (not wgpu's
            // Limits::default()): some iOS GPUs cap individual limits — e.g.
            // max_inter_stage_shader_variables — below wgpu's defaults, so
            // request_device rejects the default-limits request too. Asking for
            // adapter.limits() can never exceed what the device supports.
            pollster_block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("bloom_device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
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

/// Attach the engine to a host-owned `UIView*` instead of creating its
/// own UIWindow (PerryTS/perry#5519). `handle` is the raw `UIView*` the
/// host (Perry UI's `BloomView`) owns; `width`/`height` are its size in
/// points. Returns 1.0 on success, 0.0 on a null/invalid handle or if
/// surface bring-up failed. Idempotent once attached.
///
/// HiDPI: callers wanting full backing resolution should pass the pixel
/// size (points × `UIScreen.scale`); this path uses `width`/`height` as
/// the drawable size directly.
#[no_mangle]
pub extern "C" fn bloom_attach_native(handle: i64, width: f64, height: f64) -> f64 {
    if handle == 0 {
        return 0.0;
    }
    if unsafe { ENGINE.get() }.is_some() {
        return 1.0;
    }
    let Some(view_nn) = std::ptr::NonNull::new(handle as *mut c_void) else {
        return 0.0;
    };
    let target = {
        let h = UiKitWindowHandle::new(view_nn);
        wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(RawDisplayHandle::UiKit(UiKitDisplayHandle::new())),
            raw_window_handle: RawWindowHandle::UiKit(h),
        }
    };
    match unsafe {
        bloom_shared::attach::attach_engine(
            target,
            bloom_shared::attach::AttachParams {
                backends: wgpu::Backends::METAL,
                logical_w: width as u32,
                logical_h: height as u32,
                physical_w: (width as u32).max(1),
                physical_h: (height as u32).max(1),
                format: bloom_shared::attach::FormatPreference::Srgb,
            },
        )
    } {
        Ok(es) => {
            unsafe {
                let _ = ENGINE.set(es);
            }
            1.0
        }
        Err(_) => 0.0,
    }
}

/// Resize the engine's surface (#70 parity; used by host-driven
/// BloomViews on layout changes). `phys_*` physical px, `log_*` logical.
#[no_mangle]
pub extern "C" fn bloom_resize(phys_w: f64, phys_h: f64, log_w: f64, log_h: f64) {
    if let Some(eng) = unsafe { ENGINE.get_mut() } {
        eng.renderer.resize(phys_w as u32, phys_h as u32, log_w as u32, log_h as u32);
    }
}

/// HWND host-embed (#70) — Windows only; a no-op here for FFI-manifest
/// parity. Non-Windows hosts attach via `bloom_attach_native`.
#[no_mangle]
pub extern "C" fn bloom_attach_hwnd(_hwnd_bits: f64, _width: f64, _height: f64) {}

#[no_mangle]
pub extern "C" fn bloom_close_window() {
    unsafe { UI_VIEW = None; UI_WINDOW = None; }
}

#[no_mangle]
pub extern "C" fn bloom_window_should_close() -> f64 {
    if engine().should_close { 1.0 } else { 0.0 }
}

/// Poll the first connected game controller (MFi / Xbox / PlayStation) through
/// the GameController framework and feed it into the shared input state.
///
/// iOS had NO gamepad support before this despite iPhone/iPad supporting MFi and
/// console controllers — the shared gamepad FFI read `engine().input`, which
/// nothing populated. Resolved dynamically via the ObjC runtime so no
/// GameController link dep is needed; a no-op when absent. Identical to the
/// macOS poll (compile-verified there); button/axis indices match the
/// tvOS/visionOS map. Values are ObjC `float`, read as `f32` (reading them as
/// `f64` is an ABI mismatch).
///
/// NOTE: compile-verified on the macOS twin; not yet run on an iOS device.
fn poll_game_controllers() {
    use objc2::runtime::{AnyClass, AnyObject, Bool};
    macro_rules! btn_down {
        ($btn:expr, $eng:expr, $idx:expr) => {{
            let b: Retained<AnyObject> = $btn;
            let pressed: Bool = msg_send![&*b, isPressed];
            if pressed.as_bool() { $eng.input.set_gamepad_button_down($idx); }
        }};
    }
    unsafe {
        let gc_cls = match AnyClass::get(c"GCController") {
            Some(c) => c,
            None => return,
        };
        let controllers: Retained<AnyObject> = msg_send![gc_cls, controllers];
        let count: usize = msg_send![&*controllers, count];
        if count == 0 {
            return; // no pad: leave any injected gamepad state untouched
        }
        let controller: Retained<AnyObject> = msg_send![&*controllers, objectAtIndex: 0usize];
        let extended: *mut AnyObject = msg_send![&*controller, extendedGamepad];
        if extended.is_null() {
            return;
        }
        let extended: &AnyObject = &*extended;

        let eng = engine();
        eng.input.reset_gamepad();
        eng.input.gamepad_available = true;

        let ls: Retained<AnyObject> = msg_send![extended, leftThumbstick];
        let ls_x: Retained<AnyObject> = msg_send![&*ls, xAxis];
        let ls_y: Retained<AnyObject> = msg_send![&*ls, yAxis];
        let lx: f32 = msg_send![&*ls_x, value];
        let ly: f32 = msg_send![&*ls_y, value];
        eng.input.set_gamepad_axis(0, lx);
        eng.input.set_gamepad_axis(1, -ly);
        let rs: Retained<AnyObject> = msg_send![extended, rightThumbstick];
        let rs_x: Retained<AnyObject> = msg_send![&*rs, xAxis];
        let rs_y: Retained<AnyObject> = msg_send![&*rs, yAxis];
        let rx: f32 = msg_send![&*rs_x, value];
        let ry: f32 = msg_send![&*rs_y, value];
        eng.input.set_gamepad_axis(2, rx);
        eng.input.set_gamepad_axis(3, -ry);

        btn_down!(msg_send![extended, buttonA], eng, 0);
        btn_down!(msg_send![extended, buttonB], eng, 1);
        btn_down!(msg_send![extended, buttonX], eng, 2);
        btn_down!(msg_send![extended, buttonY], eng, 3);
        btn_down!(msg_send![extended, leftShoulder], eng, 4);
        btn_down!(msg_send![extended, rightShoulder], eng, 5);
        let dpad: Retained<AnyObject> = msg_send![extended, dpad];
        btn_down!(msg_send![&*dpad, up], eng, 12);
        btn_down!(msg_send![&*dpad, down], eng, 13);
        btn_down!(msg_send![&*dpad, left], eng, 14);
        btn_down!(msg_send![&*dpad, right], eng, 15);

        let lt: Retained<AnyObject> = msg_send![extended, leftTrigger];
        let rt: Retained<AnyObject> = msg_send![extended, rightTrigger];
        let ltv: f32 = msg_send![&*lt, value];
        let rtv: f32 = msg_send![&*rt, value];
        eng.input.set_gamepad_axis(4, ltv);
        eng.input.set_gamepad_axis(5, rtv);
    }
}

#[no_mangle]
pub extern "C" fn bloom_begin_drawing() {
    // No run loop pumping needed — UIApplicationMain handles the main run loop
    // on its own thread. The game runs on the game thread.

    // Poll a connected controller before either begin_frame path below.
    poll_game_controllers();

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
// Render half of the audio system — owned by the CoreAudio render thread
// after bloom_init_audio hands it off via AudioMixer::take_renderer.
// See native/shared/src/audio/mod.rs for the threading contract.
static mut AUDIO_RENDERER: Option<bloom_shared::audio::AudioRenderer> = None;

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
    match AUDIO_RENDERER.as_mut() {
        Some(r) => r.mix(output),
        None => output.iter_mut().for_each(|s| *s = 0.0),
    }
    0
}

#[no_mangle]
pub extern "C" fn bloom_init_audio() {
    unsafe {
        // Configure the audio session BEFORE bringing up RemoteIO. Without a
        // category the session defaults to SoloAmbient: audio is silenced by
        // the ring/silent switch and stops when the screen locks (audit gap).
        // `Playback` keeps game audio audible over the silent switch. Done via
        // the ObjC runtime so the crate needs no AVFAudio link dependency; the
        // category constants' NSString VALUES equal their symbol names, so we
        // build the string directly rather than linking the extern constant.
        // NOTE: written blind (no iOS SDK here) — verify on device.
        if let Some(av_cls) = objc2::runtime::AnyClass::get(c"AVAudioSession") {
            let session: Retained<objc2::runtime::AnyObject> = msg_send![av_cls, sharedInstance];
            let cat = objc2_foundation::NSString::from_str("AVAudioSessionCategoryPlayback");
            let null_err: *mut *mut objc2::runtime::AnyObject = std::ptr::null_mut();
            let _: objc2::runtime::Bool = msg_send![&*session, setCategory: &*cat, error: null_err];
            let _: objc2::runtime::Bool = msg_send![&*session, setActive: true, error: null_err];
        }

        // Hand the render half to the audio thread before the callback
        // can fire. Idempotent: a second init keeps the existing renderer.
        if AUDIO_RENDERER.is_none() {
            AUDIO_RENDERER = engine().audio.take_renderer();
        }
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

// ============================================================
// Music
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
pub extern "C" fn bloom_get_platform() -> f64 { 2.0 }

/// Preferred OS language packed as `c0*256+c1` (ISO-639 primary subtag). See macos lib for format.
#[no_mangle]
pub extern "C" fn bloom_get_language() -> f64 {
    fn pack(code: &str) -> f64 { let l = code.to_ascii_lowercase(); let b = l.as_bytes(); if b.len() >= 2 { (b[0] as f64) * 256.0 + (b[1] as f64) } else { 25966.0 } }
    let langs = objc2_foundation::NSLocale::preferredLanguages();
    match langs.firstObject() { Some(s) => pack(&s.to_string()), None => 25966.0 }
}

// ============================================================
// Thread-safe staging (for async asset loading via Perry threads)
// ============================================================


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

