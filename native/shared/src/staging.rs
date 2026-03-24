use std::sync::{Mutex, OnceLock};
use crate::models::ModelData;
use crate::audio::SoundData;

pub struct StagedTexture {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub struct StagedModel {
    pub model: ModelData,
    pub textures: Vec<StagedTexture>,
}

// Thread-safe staging stores using Mutex<Vec<Option<T>>>.
// The lock is only held during insert/remove (microseconds), not during decode.

fn texture_store() -> &'static Mutex<Vec<Option<StagedTexture>>> {
    static INSTANCE: OnceLock<Mutex<Vec<Option<StagedTexture>>>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(Vec::new()))
}

fn model_store() -> &'static Mutex<Vec<Option<StagedModel>>> {
    static INSTANCE: OnceLock<Mutex<Vec<Option<StagedModel>>>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(Vec::new()))
}

fn sound_store() -> &'static Mutex<Vec<Option<SoundData>>> {
    static INSTANCE: OnceLock<Mutex<Vec<Option<SoundData>>>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(Vec::new()))
}

fn stage_into<T>(store: &Mutex<Vec<Option<T>>>, item: T) -> f64 {
    let mut vec = store.lock().unwrap();
    // Reuse freed slots
    for (i, slot) in vec.iter_mut().enumerate() {
        if slot.is_none() {
            *slot = Some(item);
            return (i + 1) as f64;
        }
    }
    vec.push(Some(item));
    vec.len() as f64
}

fn take_from<T>(store: &Mutex<Vec<Option<T>>>, handle: f64) -> Option<T> {
    let idx = handle as usize;
    if idx == 0 { return None; }
    let mut vec = store.lock().unwrap();
    if idx > vec.len() { return None; }
    vec[idx - 1].take()
}

// Public API

/// Decode image bytes (PNG/JPEG/etc) and stage the result. Thread-safe.
pub fn decode_and_stage_texture(file_data: &[u8]) -> f64 {
    let img = match image::load_from_memory(file_data) {
        Ok(img) => img.to_rgba8(),
        Err(_) => return 0.0,
    };
    let width = img.width();
    let height = img.height();
    stage_texture(StagedTexture { data: img.into_raw(), width, height })
}

pub fn stage_texture(tex: StagedTexture) -> f64 {
    stage_into(texture_store(), tex)
}

pub fn take_texture(handle: f64) -> Option<StagedTexture> {
    take_from(texture_store(), handle)
}

pub fn stage_model(model: StagedModel) -> f64 {
    stage_into(model_store(), model)
}

pub fn take_model(handle: f64) -> Option<StagedModel> {
    take_from(model_store(), handle)
}

pub fn stage_sound(sound: SoundData) -> f64 {
    stage_into(sound_store(), sound)
}

pub fn take_sound(handle: f64) -> Option<SoundData> {
    take_from(sound_store(), handle)
}
