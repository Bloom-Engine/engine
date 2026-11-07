use super::{PhysicalPage, VirtualShadowPageCache, VSM_CLIP_LEVELS, VSM_VIRTUAL_PAGES_PER_AXIS};

pub(super) const FREE: [u8; 3] = [8, 8, 8];
pub(super) const MISS_UNRENDERED: [u8; 3] = [255, 180, 35];
pub(super) const INVALIDATED: [u8; 3] = [255, 55, 190];
pub(super) const LEVELS: [[u8; 3]; VSM_CLIP_LEVELS as usize] =
    [[70, 210, 110], [70, 150, 255], [190, 100, 255]];

fn page_color(slot: &PhysicalPage, page: super::VirtualShadowPage) -> [u8; 3] {
    if slot.dirty && slot.rendered_frame == 0 {
        MISS_UNRENDERED
    } else if slot.dirty {
        INVALIDATED
    } else {
        // Local cube faces share the established diagnostic palette in
        // repeating XYZ pairs. This keeps the v1 capture contract stable
        // while avoiding an out-of-range clip-level lookup.
        LEVELS[page.level as usize % LEVELS.len()]
    }
}

pub(super) fn virtual_rgb(
    cache: &VirtualShadowPageCache,
    light: u16,
    scale: u32,
) -> (u32, u32, Vec<u8>) {
    let scale = scale.max(1);
    let axis = VSM_VIRTUAL_PAGES_PER_AXIS as u32;
    let width = axis * scale;
    let height = axis * VSM_CLIP_LEVELS as u32 * scale;
    let mut rgb = vec![8u8; (width * height * 3) as usize];
    for (&page, &physical_page) in &cache.mapping {
        if page.light != light {
            continue;
        }
        paint_cell(
            &mut rgb,
            width,
            scale,
            u32::from(page.x),
            u32::from(page.y) + u32::from(page.level) * axis,
            page_color(&cache.physical[physical_page as usize], page),
        );
    }
    (width, height, rgb)
}

pub(super) fn physical_rgb(cache: &VirtualShadowPageCache, scale: u32) -> (u32, u32, Vec<u8>) {
    let scale = scale.max(1);
    let columns = 16u32.min(cache.physical.len().max(1) as u32);
    let rows = (cache.physical.len() as u32).div_ceil(columns);
    let width = columns * scale;
    let height = rows.max(1) * scale;
    let mut rgb = vec![8u8; (width * height * 3) as usize];
    for (index, slot) in cache.physical.iter().enumerate() {
        let Some(owner) = slot.owner else {
            continue;
        };
        paint_cell(
            &mut rgb,
            width,
            scale,
            index as u32 % columns,
            index as u32 / columns,
            page_color(slot, owner),
        );
    }
    (width, height, rgb)
}

pub(super) fn legend_rgb(scale: u32) -> (u32, u32, Vec<u8>) {
    let scale = scale.max(1);
    let colors = [
        FREE,
        MISS_UNRENDERED,
        INVALIDATED,
        LEVELS[0],
        LEVELS[1],
        LEVELS[2],
    ];
    let width = colors.len() as u32 * scale;
    let mut rgb = vec![0u8; (width * scale * 3) as usize];
    for (x, color) in colors.into_iter().enumerate() {
        paint_cell(&mut rgb, width, scale, x as u32, 0, color);
    }
    (width, scale, rgb)
}

fn paint_cell(rgb: &mut [u8], width: u32, scale: u32, cell_x: u32, cell_y: u32, color: [u8; 3]) {
    for y in 0..scale {
        for x in 0..scale {
            let pixel_x = cell_x * scale + x;
            let pixel_y = cell_y * scale + y;
            let offset = ((pixel_y * width + pixel_x) * 3) as usize;
            rgb[offset..offset + 3].copy_from_slice(&color);
        }
    }
}
