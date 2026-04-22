pub mod string_header;
pub mod handles;
pub mod input;
pub mod renderer;
pub mod text_renderer;
pub mod audio;
pub mod textures;
pub mod models;
pub mod scene;
pub mod frame_callbacks;
pub mod geometry;
pub mod picking;
pub mod shadows;
pub mod postfx;
pub mod custom_shaders;
pub mod staging;
pub mod profiler;
// Jolt C ABI + Rust wrapper live on native only. On wasm32 the web crate
// routes bloom_physics_* calls through wasm_bindgen to JoltPhysics.js;
// no Rust-side Jolt integration is needed.
#[cfg(all(feature = "jolt", not(target_arch = "wasm32")))]
pub mod jolt_sys;
#[cfg(all(feature = "jolt", not(target_arch = "wasm32")))]
pub mod physics_jolt;
pub mod engine;

pub use engine::EngineState;
pub use renderer::Renderer;
pub use string_header::str_from_header;
pub use audio::{AudioMixer, SoundData, parse_wav, parse_ogg};
#[cfg(feature = "mp3")]
pub use audio::parse_mp3;
pub use textures::TextureManager;
pub use models::ModelManager;
pub use scene::SceneGraph;
pub use frame_callbacks::FrameCallbackSystem;
