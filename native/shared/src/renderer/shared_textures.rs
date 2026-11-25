//! Textures shared between models by content.
//!
//! Models embed their textures, and many embed the same one: the Jonah game's
//! chapters held 41 to 60 % of their texture memory twice or more (a material
//! set used by a village, a harbour and a ship; two sailors with one skin).
//! The cook tool (`bloom-cook glb-compress`) writes the SHA-256 of every KTX2
//! image into its `extras.bloomContentHash`. An image whose hash, and use
//! (normal map, sRGB), the renderer already holds is given that texture slot
//! instead of a new one, and the slot counts the models that hold it: a
//! model's unload frees a shared slot only when no other model holds it.
//! Images without a hash (older files, PNG, alpha-mask variants) load as
//! before, one slot per model.

use std::collections::HashMap;

use super::Renderer;

/// Content hash (128 bits), normal map, sRGB.
pub(crate) type SharedTextureKey = (u128, bool, bool);

#[derive(Default)]
pub(crate) struct SharedTextures {
    by_key: HashMap<SharedTextureKey, (u32, u32)>,
    key_of: HashMap<u32, SharedTextureKey>,
}

/// The `bloomContentHash` of an image's `extras` (32 hex digits), if any.
pub(crate) fn content_hash(extras: Option<&str>) -> Option<u128> {
    let extras = extras?;
    let at = extras.find("\"bloomContentHash\"")?;
    let rest = &extras[at + "\"bloomContentHash\"".len()..];
    let start = rest.find('"')? + 1;
    let hex = rest.get(start..start + 32)?;
    if rest.as_bytes().get(start + 32) != Some(&b'"') {
        return None;
    }
    u128::from_str_radix(hex, 16).ok()
}

impl Renderer {
    /// The slot already holding the texture `key`, now held once more.
    pub(crate) fn shared_texture_take(&mut self, key: SharedTextureKey) -> Option<u32> {
        let (slot, holders) = self.shared_textures.by_key.get_mut(&key)?;
        *holders += 1;
        Some(*slot)
    }

    /// `slot` now holds the texture `key`, for one holder.
    pub(crate) fn shared_texture_add(&mut self, key: SharedTextureKey, slot: u32) {
        if slot == 0 || self.shared_textures.key_of.contains_key(&slot) {
            return;
        }
        self.shared_textures.by_key.insert(key, (slot, 1));
        self.shared_textures.key_of.insert(slot, key);
    }

    /// A model that held `slot` lets it go: the texture is unloaded unless it
    /// is shared and another model still holds it.
    pub fn release_model_texture(&mut self, slot: u32) {
        if let Some(key) = self.shared_textures.key_of.get(&slot).copied() {
            if let Some((_, holders)) = self.shared_textures.by_key.get_mut(&key) {
                if *holders > 1 {
                    *holders -= 1;
                    return;
                }
            }
            self.shared_textures.by_key.remove(&key);
            self.shared_textures.key_of.remove(&slot);
        }
        self.unload_texture(slot);
    }
}

#[cfg(test)]
mod tests {
    use super::content_hash;

    #[test]
    fn the_hash_is_read_from_the_extras() {
        let h = "0123456789abcdef0123456789abcdef";
        assert_eq!(
            content_hash(Some(&format!("{{\"bloomContentHash\":\"{h}\"}}"))),
            Some(0x0123456789abcdef0123456789abcdef)
        );
        assert_eq!(
            content_hash(Some(&format!(
                "{{\"a\": 1, \"bloomContentHash\" : \"{h}\" }}"
            ))),
            Some(0x0123456789abcdef0123456789abcdef)
        );
        assert_eq!(content_hash(Some("{\"bloomContentHash\":\"0123\"}")), None);
        assert_eq!(content_hash(Some("{\"other\":1}")), None);
        assert_eq!(content_hash(None), None);
    }
}
