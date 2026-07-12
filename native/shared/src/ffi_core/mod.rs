//! `define_core_ffi!` — the shared, non-physics `bloom_*` FFI surface.
//!
//! # Why this exists
//!
//! Every platform crate must export the full FFI surface declared in
//! `package.json` (`perry.nativeLibrary.functions`); a missing symbol is a
//! link error or dlopen crash in shipped games. Before this macro the ~250
//! non-physics functions were hand-copied into six platform crates
//! (~9,000 duplicated lines) and drifted constantly:
//!
//!   - Android shipped 60 functions behind (dlopen crash, PR #59), then
//!     "fixed" part of the gap with silent no-op stubs.
//!   - Windows stubbed the entire scene-graph / lighting / picking /
//!     post-FX surface with silent no-ops.
//!   - iOS/tvOS declared gamepad functions with an extra leading
//!     `gamepad` parameter the manifest doesn't have, so every axis and
//!     button read was off by one argument register.
//!   - `bloom_create_mesh` / `bloom_gen_mesh_spline_ribbon` read `*const
//!     f32` on some platforms; Perry passes arrays as pointers to inline
//!     f64 data (see perry-codegen `lower_call/extern_func.rs`), so those
//!     reads were garbage.
//!
//! One macro, expanded once per platform crate, makes per-platform drift
//! in this surface structurally impossible — the same fix the physics
//! surface got with `define_physics_ffi!`. `tools/validate-ffi.js`
//! cross-checks the macro, the platform crates, and package.json in CI.
//!
//! # Contract for the invoking crate
//!
//! The platform crate must define, before invoking the macro:
//!
//! ```ignore
//! /// Engine-state accessor. FFI is single-threaded (Perry calls in on
//! /// the run-loop thread); see the audio module for the one exception.
//! fn engine() -> &'static mut bloom_shared::engine::EngineState { ... }
//!
//! /// Asset-path resolver: identity on desktop; prepends the app asset
//! /// dir on Android/iOS/tvOS where relative paths don't resolve.
//! fn bloom_resolve_asset_path(path: &str) -> std::borrow::Cow<'_, str> { ... }
//!
//! bloom_shared::define_core_ffi!();
//! ```
//!
//! Functions NOT generated here (platform crates implement them by hand,
//! validated by tools/validate-ffi.js): window + event loop
//! (`bloom_init_window`, `bloom_begin/end_drawing`, ...), audio backend
//! init/teardown, fullscreen, cursor capture, clipboard, file dialogs,
//! window title/icon, `bloom_get_platform`, `bloom_get_language`.
//!
//! # Conventions inside the macro
//!
//!   - Every body is wrapped in [`crate::ffi::guard`] — panics log once
//!     and return a default instead of crossing the C boundary.
//!   - `models3d` / `image-extras` gated functions compile to a
//!     once-warning stub when the feature is off. Symbols never silently
//!     vanish and never silently no-op.
//!   - String params are Perry StringHeader pointers
//!     (`crate::string_header`); array params are pointers to inline f64
//!     data (the compiler skips the 8-byte ArrayHeader at the callsite).

mod game_loop;
mod input;
mod draw;
mod assets;
mod audio_ffi;
mod models;
mod scene;
mod visual;
mod vfx;

/// Expand the full shared (non-physics) FFI surface. Composed from the
/// per-subsystem section macros in this directory; platform crates invoke
/// only this. Nested `$crate::__bloom_ffi_*` invocations expand at THIS
/// macro's call site, so the `engine()` / `bloom_resolve_asset_path()`
/// hooks still resolve in the invoking crate.
#[macro_export]
macro_rules! define_core_ffi {
    () => {
        $crate::__bloom_ffi_game_loop!();
        $crate::__bloom_ffi_input!();
        $crate::__bloom_ffi_draw!();
        $crate::__bloom_ffi_assets!();
        $crate::__bloom_ffi_audio_ffi!();
        $crate::__bloom_ffi_models!();
        $crate::__bloom_ffi_scene!();
        $crate::__bloom_ffi_visual!();
        $crate::__bloom_ffi_vfx!();
    };
}


// Compile-coverage for the macro body: expand it against mock hooks so
// `cargo test -p bloom-shared` catches breakage without building any
// platform crate. Nothing here runs — the hooks panic if called.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod macro_expansion_compile_check {
    #![allow(dead_code, unused_variables)]

    fn engine() -> &'static mut crate::engine::EngineState {
        unreachable!("compile-coverage mock — never called")
    }

    fn bloom_resolve_asset_path(path: &str) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed(path)
    }

    crate::define_core_ffi!();
}
