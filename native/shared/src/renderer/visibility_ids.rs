//! Collision-free draw namespace for the shared packed visibility target.

pub(crate) const VISIBILITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
pub(crate) const VISIBILITY_BYTES_PER_PIXEL: u64 = 8;
pub(crate) const INVALID_DRAW_ID: u32 = u32::MAX;
pub(crate) const VIRTUAL_DRAW_BIT: u32 = 1 << 31;
pub(crate) const DRAW_INDEX_MASK: u32 = VIRTUAL_DRAW_BIT - 1;
pub(crate) const FRONT_FACE_BIT: u32 = 1 << 31;
pub(crate) const PRIMITIVE_ID_MASK: u32 = FRONT_FACE_BIT - 1;
const MAX_VIRTUAL_DRAW_INDEX: u32 = DRAW_INDEX_MASK - 1;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum VisibilityDraw {
    Compatibility(u32),
    Virtual(u32),
}

/// One `Rg32Uint` texel. Draw bit 31 selects compatibility or virtual
/// geometry; primitive bit 31 independently carries raster face orientation.
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct VisibilityRecord {
    pub draw_id: u32,
    pub primitive_and_face: u32,
}

impl VisibilityRecord {
    pub(crate) const BACKGROUND: Self = Self {
        draw_id: INVALID_DRAW_ID,
        primitive_and_face: u32::MAX,
    };

    pub(crate) const fn encode(
        draw_id: u32,
        primitive_id: u32,
        front_facing: bool,
    ) -> Option<Self> {
        Self::encode_draw(
            VisibilityDraw::Compatibility(draw_id),
            primitive_id,
            front_facing,
        )
    }

    pub(crate) const fn encode_virtual(
        draw_index: u32,
        primitive_id: u32,
        front_facing: bool,
    ) -> Option<Self> {
        Self::encode_draw(
            VisibilityDraw::Virtual(draw_index),
            primitive_id,
            front_facing,
        )
    }

    pub(crate) const fn encode_draw(
        draw: VisibilityDraw,
        primitive_id: u32,
        front_facing: bool,
    ) -> Option<Self> {
        if primitive_id > PRIMITIVE_ID_MASK {
            return None;
        }
        let draw_id = match draw {
            VisibilityDraw::Compatibility(index) if index <= DRAW_INDEX_MASK => index,
            VisibilityDraw::Virtual(index) if index <= MAX_VIRTUAL_DRAW_INDEX => {
                VIRTUAL_DRAW_BIT | index
            }
            _ => return None,
        };
        Some(Self {
            draw_id,
            primitive_and_face: primitive_id | if front_facing { FRONT_FACE_BIT } else { 0 },
        })
    }

    pub(crate) const fn decode(self) -> Option<(u32, u32, bool)> {
        if self.draw_id == INVALID_DRAW_ID {
            return None;
        }
        Some((
            self.draw_id,
            self.primitive_and_face & PRIMITIVE_ID_MASK,
            (self.primitive_and_face & FRONT_FACE_BIT) != 0,
        ))
    }

    pub(crate) const fn decode_draw(self) -> Option<(VisibilityDraw, u32, bool)> {
        let (_, primitive, front) = match self.decode() {
            Some(decoded) => decoded,
            None => return None,
        };
        let draw = if self.draw_id & VIRTUAL_DRAW_BIT == 0 {
            VisibilityDraw::Compatibility(self.draw_id)
        } else {
            VisibilityDraw::Virtual(self.draw_id & DRAW_INDEX_MASK)
        };
        Some((draw, primitive, front))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_and_virtual_draws_never_alias_or_hit_background() {
        let compatibility =
            VisibilityRecord::encode(DRAW_INDEX_MASK, PRIMITIVE_ID_MASK, false).unwrap();
        let virtual_draw =
            VisibilityRecord::encode_virtual(MAX_VIRTUAL_DRAW_INDEX, 19, true).unwrap();
        assert_eq!(
            compatibility.decode_draw(),
            Some((
                VisibilityDraw::Compatibility(DRAW_INDEX_MASK),
                PRIMITIVE_ID_MASK,
                false,
            ))
        );
        assert_eq!(
            virtual_draw.decode_draw(),
            Some((VisibilityDraw::Virtual(MAX_VIRTUAL_DRAW_INDEX), 19, true,))
        );
        assert_ne!(compatibility.draw_id, virtual_draw.draw_id);
        assert_ne!(virtual_draw.draw_id, INVALID_DRAW_ID);
        assert_eq!(VisibilityRecord::BACKGROUND.decode_draw(), None);
        assert_eq!(VisibilityRecord::encode(VIRTUAL_DRAW_BIT, 0, true), None);
        assert_eq!(
            VisibilityRecord::encode_virtual(DRAW_INDEX_MASK, 0, true),
            None
        );
    }
}
