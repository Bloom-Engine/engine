// `static mut` is intentional in this engine — single-threaded FFI
// surface, no contention to worry about. Suppress the 2024 lint at
// the crate root rather than leaving 16+ warnings in every build.
#![allow(static_mut_refs)]

pub mod string_header;
pub mod ffi;
#[cfg(not(target_arch = "wasm32"))]
pub mod ffi_core;
pub mod handles;
pub mod input;
pub mod renderer;
pub mod text_renderer;
pub mod audio;
pub mod textures;
// Not gated on models3d: the mixer is pure per-instance state embedded in
// ModelAnimation (always compiled); only the gltf/image_dds LOADERS are
// behind the feature, in models_gltf.rs (EN-063).
pub mod anim_mixer;
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
pub mod particles;
pub mod decals;
#[cfg(all(feature = "models3d", feature = "jolt", not(target_arch = "wasm32")))]
pub mod ragdoll;
pub mod sdf_cache;
// Jolt C ABI + Rust wrapper live on native only. On wasm32 the web crate
// routes bloom_physics_* calls through wasm_bindgen to JoltPhysics.js;
// no Rust-side Jolt integration is needed.
#[cfg(all(feature = "jolt", not(target_arch = "wasm32")))]
pub mod jolt_sys;
#[cfg(all(feature = "jolt", not(target_arch = "wasm32")))]
pub mod physics_jolt;
pub mod engine;
pub mod drs;
// Host-surface attach path (PerryTS/perry#5519). Pulls in wgpu's
// raw-surface API; web builds its surface from a canvas id instead, so
// this is native-only.
#[cfg(not(target_arch = "wasm32"))]
pub mod attach;

pub use engine::EngineState;
pub use renderer::Renderer;
pub use string_header::str_from_header;
pub use audio::{AudioMixer, SoundData, parse_wav, parse_ogg};
#[cfg(feature = "mp3")]
pub use audio::parse_mp3;
pub use textures::TextureManager;
#[cfg(feature = "models3d")]
pub use models::ModelManager;
pub use scene::SceneGraph;
pub use frame_callbacks::FrameCallbackSystem;
