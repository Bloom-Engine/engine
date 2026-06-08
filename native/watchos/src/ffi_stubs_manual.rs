// Hand-written no-op stubs for bloom_* FFI functions that the engine's
// TypeScript (src/core/index.ts) `declare function`s but that are ABSENT from
// the package.json `perry.nativeLibrary.functions` manifest — so gen_stubs.js
// (which reads the manifest) cannot generate them.
//
// The watchOS proof-of-life crate does not implement these advanced
// 3D / post-fx / profiler features (SCNTechnique/SCNRenderer is absent from the
// watchOS SDK). Bloom Jump (2D) never calls them, but Perry still emits their
// wrapper symbols from the TS declarations, so the native symbols must resolve
// at link time.
//
// Signatures mirror the `declare function` lines in core/index.ts:
//   - `number` params  -> f64 (Perry passes these in float registers)
//   - the `any` pointer -> i64 (a pointer carried in an integer register)
//   - `string` returns  -> i64 (a pointer carried in an integer register)
//
// If these four are ever added to the manifest, delete this file and let
// gen_stubs.js generate them instead.
#![allow(unused_variables, non_snake_case)]

#[no_mangle]
pub extern "C" fn bloom_set_material_params(_handle: f64, _params_ptr: i64, _count: f64) {}

#[no_mangle]
pub extern "C" fn bloom_splat_impulse(_x: f64, _z: f64, _radius: f64, _strength: f64) {}

#[no_mangle]
pub extern "C" fn bloom_profiler_frame_history() -> i64 {
    0
}

#[no_mangle]
pub extern "C" fn bloom_profiler_overlay_text() -> i64 {
    0
}
