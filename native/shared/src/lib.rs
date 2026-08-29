// `static mut` is intentional in this engine — single-threaded FFI
// surface, no contention to worry about. Suppress the 2024 lint at
// the crate root rather than leaving 16+ warnings in every build.
#![allow(static_mut_refs)]

#[cfg(any(feature = "image-extras", feature = "models3d"))]
mod adapter_profile;
pub mod audio;
#[cfg(all(feature = "image-extras", not(target_arch = "wasm32")))]
pub mod cooked_texture_store;
pub mod ffi;
#[cfg(not(target_arch = "wasm32"))]
pub mod ffi_core;
pub mod handles;
pub mod input;
pub mod renderer;
pub mod string_header;
pub mod text_renderer;
pub mod textures;
#[cfg(feature = "models3d")]
pub mod virtual_geometry;
// Not gated on models3d: the mixer is pure per-instance state embedded in
// ModelAnimation (always compiled); only the gltf/image_dds LOADERS are
// behind the feature, in models_gltf.rs (EN-063).
pub mod anim_mixer;
pub mod custom_shaders;
pub mod decals;
pub mod frame_callbacks;
pub mod geometry;
pub mod models;
pub mod particles;
pub mod picking;
pub mod postfx;
pub mod profiler;
#[cfg(all(feature = "models3d", feature = "jolt"))]
pub mod ragdoll;
pub mod scene;
pub mod sdf_cache;
pub mod shadows;
pub mod staging;
pub(crate) mod virtual_shadows;
// Jolt C ABI + Rust wrapper live on native only. On wasm32 the web crate
// routes bloom_physics_* calls through wasm_bindgen to JoltPhysics.js;
// no Rust-side Jolt integration is needed.
pub mod drs;
pub mod engine;
#[cfg(all(feature = "jolt", not(target_arch = "wasm32")))]
pub mod jolt_sys;
#[cfg(all(feature = "jolt", not(target_arch = "wasm32")))]
pub mod physics_jolt;
// Host-surface attach path (PerryTS/perry#5519). Pulls in wgpu's
// raw-surface API; web builds its surface from a canvas id instead, so
// this is native-only.
#[cfg(not(target_arch = "wasm32"))]
pub mod attach;

#[cfg(feature = "mp3")]
pub use audio::parse_mp3;
pub use audio::{parse_ogg, parse_wav, AudioMixer, SoundData};
pub use engine::EngineState;
pub use frame_callbacks::FrameCallbackSystem;
#[cfg(feature = "models3d")]
pub use models::ModelManager;
pub use renderer::Renderer;
pub use scene::SceneGraph;
pub use string_header::str_from_header;
pub use textures::TextureManager;
