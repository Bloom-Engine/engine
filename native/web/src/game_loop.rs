use wasm_bindgen::prelude::*;

// The JavaScript FFI bridge owns the browser scheduler and Perry closure
// handles. These exports keep the shared native/web manifest complete.
#[wasm_bindgen]
pub fn bloom_run_game(_callback: f64) {}

#[wasm_bindgen]
pub fn bloom_run_game_with_cleanup(_callback: f64, _cleanup: f64) {}
