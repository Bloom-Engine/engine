//! The low-tier GPU decision (SH-069): which phones get the low-tier scene shading
//! (`shaders/scene_lowtier.wgsl`, spliced by `mobile_shading.rs`) and, through
//! `bloom_is_low_tier_gpu`, the game's own low-tier content budget.
//!
//! Only Android classifies its adapter: an iPhone, a desktop or a browser is never low tier
//! unless `BLOOM_LOW_TIER=1` forces it (to preview the low-tier look on a desktop).
//! `BLOOM_LOW_TIER=0` turns it off on a phone that would qualify; on Android the property
//! `debug.bloom.lowtier` sets the same variable (JNI_OnLoad).
//!
//! The classification is by GPU family and generation, the way engines ship device
//! profiles. Low tier is roughly "below a Snapdragon 730G": the reference phone is a Pixel 4a
//! (Adreno 618), which ran the mobile profile at 4 fps.

use std::sync::atomic::{AtomicU8, Ordering};

/// 0 = not decided (no Android adapter seen), 1 = low tier, 2 = not low tier.
static DECIDED: AtomicU8 = AtomicU8::new(0);

/// Whether a GPU of this adapter name belongs to the low tier.
pub fn is_low_tier_gpu_name(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    if let Some(model) = number_after(&n, "adreno") {
        // Adreno 3xx-5xx all; 6xx below 640 (605-630: Snapdragon 4xx-7xx of 2018-2021);
        // 7xx below 725 (710/720: the 2022 budget parts).
        return model < 640 || (700..725).contains(&model);
    }
    if n.contains("mali") {
        // Midgard (Mali-T) and the Bifrost / early Valhall budget cores. G76+ (and the
        // G6xx/G7xx Valhall generation, Immortalis) are not low tier.
        if n.contains("mali-t") || n.contains("mali-4") {
            return true;
        }
        if let Some(model) = number_after(&n, "mali-g") {
            if matches!(model, 31 | 51 | 52 | 57 | 68 | 71 | 72 | 310 | 510) {
                return true;
            }
            // G76/G77 span both tiers by core count: an MC4 (Helio G90) is a Pixel 4a, an
            // MP12 (Exynos 9820) is not.
            if matches!(model, 76 | 77) {
                let cores = number_after(&n, " mc").or_else(|| number_after(&n, " mp"));
                return cores.is_some_and(|c| c <= 5);
            }
        }
        return false;
    }
    // PowerVR Rogue / IMG in phones are all budget parts.
    n.contains("powervr") || n.starts_with("img ")
}

/// The first run of digits after `key` in `s` (`"adreno (tm) 618"` -> 618).
fn number_after(s: &str, key: &str) -> Option<u32> {
    let rest = &s[s.find(key)? + key.len()..];
    let digits: String = rest
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

fn env_override() -> Option<bool> {
    match std::env::var("BLOOM_LOW_TIER").as_deref() {
        Ok("1") => Some(true),
        Ok("0") => Some(false),
        _ => None,
    }
}

/// Called by the Android platform crate once its adapter is chosen, before the renderer
/// builds a pipeline.
pub fn note_android_adapter(name: &str) {
    let low = env_override().unwrap_or_else(|| is_low_tier_gpu_name(name));
    DECIDED.store(if low { 1 } else { 2 }, Ordering::SeqCst);
}

/// Whether this run uses the low tier.
pub fn low_tier() -> bool {
    match DECIDED.load(Ordering::SeqCst) {
        1 => true,
        2 => false,
        _ => env_override().unwrap_or(false),
    }
}

#[cfg(test)]
mod tests {
    use super::is_low_tier_gpu_name as low;

    #[test]
    fn budget_and_older_gpus_are_low_tier() {
        for name in [
            "Adreno (TM) 618",
            "Adreno (TM) 610",
            "Adreno (TM) 506",
            "Adreno (TM) 619",
            "Adreno (TM) 710",
            "Mali-G52 MC2",
            "Mali-G57 MC2",
            "Mali-G72 MP3",
            "Mali-G76 MC4",
            "Mali-T880",
            "PowerVR Rogue GE8320",
        ] {
            assert!(low(name), "{name}");
        }
    }

    #[test]
    fn current_and_flagship_gpus_are_not() {
        for name in [
            "Adreno (TM) 640",
            "Adreno (TM) 660",
            "Adreno (TM) 730",
            "Adreno (TM) 740",
            "Adreno (TM) 750",
            "Mali-G78 MP20",
            "Mali-G76 MP12",
            "Mali-G710 MC10",
            "Mali-G715-Immortalis MC11",
            "Samsung Xclipse 920",
            "Apple M1 Max",
            "NVIDIA GeForce RTX 4090",
        ] {
            assert!(!low(name), "{name}");
        }
    }
}
