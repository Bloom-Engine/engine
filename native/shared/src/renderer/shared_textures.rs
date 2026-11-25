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
    /// Slots the import in progress holds (taken or added), released again
    /// if the import fails.
    importing: Vec<u32>,
}

/// The shared textures one model's import holds, by key (an image repeated
/// in one model is held once).
#[derive(Default)]
pub(crate) struct ModelShares(HashMap<SharedTextureKey, u32>);

/// The key an image is shared by: its content hash and use. `None` without a
/// hash, and for an image that feeds an alpha mask (it takes the RGBA path).
pub(crate) fn image_key(
    extras: Option<&str>,
    is_normal: bool,
    srgb: bool,
    feeds_alpha_mask: bool,
) -> Option<SharedTextureKey> {
    if feeds_alpha_mask {
        return None;
    }
    content_hash(extras).map(|hash| (hash, is_normal, srgb))
}

/// The top-level `bloomContentHash` of an image's `extras` object (32 hex
/// digits), if any. A key of that name nested deeper is not it.
pub(crate) fn content_hash(extras: Option<&str>) -> Option<u128> {
    let bytes = extras?.as_bytes();
    let (mut depth, mut i) = (0usize, 0usize);
    // the string starting at `at` (a quote): its contents and the index after it
    let string = |at: usize| -> Option<(&[u8], usize)> {
        let mut j = at + 1;
        while j < bytes.len() {
            match bytes[j] {
                b'\\' => j += 2,
                b'"' => return Some((&bytes[at + 1..j], j + 1)),
                _ => j += 1,
            }
        }
        None
    };
    while i < bytes.len() {
        match bytes[i] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth = depth.saturating_sub(1),
            b'"' => {
                let (text, after) = string(i)?;
                let mut k = after;
                while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                    k += 1;
                }
                let is_key = bytes.get(k) == Some(&b':');
                if depth == 1 && is_key && text == b"bloomContentHash" {
                    k += 1;
                    while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                        k += 1;
                    }
                    if bytes.get(k) != Some(&b'"') {
                        return None;
                    }
                    let (value, _) = string(k)?;
                    if value.len() != 32 {
                        return None;
                    }
                    return u128::from_str_radix(std::str::from_utf8(value).ok()?, 16).ok();
                }
                i = after;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

impl Renderer {
    /// A model's import begins: what it takes is recorded until it ends.
    pub(crate) fn shared_import_begin(&mut self) {
        self.shared_textures.importing.clear();
    }

    /// The import ended; a failed one lets go of the shared slots it held.
    pub(crate) fn shared_import_end(&mut self, loaded: bool) {
        let held = std::mem::take(&mut self.shared_textures.importing);
        if !loaded {
            for slot in held {
                self.release_model_texture(slot);
            }
        }
    }

    /// The slot holding the texture `key` if one does, held by this model
    /// (once, however often the model uses it).
    pub(crate) fn shared_texture_hold(
        &mut self,
        shares: &mut ModelShares,
        key: Option<SharedTextureKey>,
    ) -> Option<u32> {
        let key = key?;
        if let Some(&slot) = shares.0.get(&key) {
            return Some(slot);
        }
        let slot = self.shared_texture_take(key)?;
        shares.0.insert(key, slot);
        Some(slot)
    }

    /// `slot` was just made for this model's image `key`: others may share it.
    pub(crate) fn shared_texture_register(
        &mut self,
        shares: &mut ModelShares,
        key: Option<SharedTextureKey>,
        slot: u32,
    ) {
        if let Some(key) = key {
            self.shared_texture_add(key, slot);
            shares.0.insert(key, slot);
        }
    }

    /// The slot already holding the texture `key`, now held once more.
    pub(crate) fn shared_texture_take(&mut self, key: SharedTextureKey) -> Option<u32> {
        let (slot, holders) = self.shared_textures.by_key.get_mut(&key)?;
        *holders += 1;
        let slot = *slot;
        self.shared_textures.importing.push(slot);
        Some(slot)
    }

    /// `slot` now holds the texture `key`, for one holder.
    pub(crate) fn shared_texture_add(&mut self, key: SharedTextureKey, slot: u32) {
        if slot == 0 || self.shared_textures.key_of.contains_key(&slot) {
            return;
        }
        self.shared_textures.by_key.insert(key, (slot, 1));
        self.shared_textures.key_of.insert(slot, key);
        self.shared_textures.importing.push(slot);
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
        // only the top level counts, and a value that merely equals the name is not a key
        let nested = format!(
            "{{\"a\":{{\"bloomContentHash\":\"{}\"}},\"bloomContentHash\":\"{h}\"}}",
            "0".repeat(32)
        );
        assert_eq!(
            content_hash(Some(&nested)),
            Some(0x0123456789abcdef0123456789abcdef)
        );
        assert_eq!(
            content_hash(Some(
                "{\"a\":{\"bloomContentHash\":\"0123456789abcdef0123456789abcdef\"}}"
            )),
            None
        );
        assert_eq!(content_hash(Some("{\"x\":\"bloomContentHash\"}")), None);
        assert_eq!(content_hash(None), None);
    }
}
