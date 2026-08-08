use super::*;

pub(super) const PAGES_PER_LEVEL: usize =
    VSM_VIRTUAL_PAGES_PER_AXIS as usize * VSM_VIRTUAL_PAGES_PER_AXIS as usize;
pub(super) const COVERAGE_ENTRIES: usize = PAGES_PER_LEVEL * VSM_CLIP_LEVELS as usize;

/// Mark directional virtual pages touched by camera-visible receiver bounds.
///
/// The fixed R32 coverage array is both the low-overhead CPU oracle and the
/// exact storage shape used by GPU request marking. Bounds receive a one-page
/// guard for PCF, jitter, and small camera motion. Coverage wins over center
/// distance so shared ground/walls rank before isolated geometry. Per-level
/// caps bound compaction, residency, and uploads; omitted pages use CSM.
pub fn directional_receiver_demand(
    level_vps: [[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize],
    receiver_bounds: &[([f32; 3], [f32; 3])],
    light: u16,
) -> Vec<VirtualShadowPage> {
    let axis = VSM_VIRTUAL_PAGES_PER_AXIS as usize;
    let center = i32::from(VSM_VIRTUAL_PAGES_PER_AXIS);
    let per_level: [Vec<VirtualShadowPage>; VSM_CLIP_LEVELS as usize] =
        std::array::from_fn(|level| {
            let planes = crate::scene::extract_frustum_planes(&level_vps[level]);
            let mut coverage = [0u32; PAGES_PER_LEVEL];
            let mut touched = Vec::with_capacity(
                receiver_bounds
                    .len()
                    .saturating_mul(16)
                    .min(PAGES_PER_LEVEL),
            );
            for &(bmin, bmax) in receiver_bounds {
                let Some((min_x, min_y, max_x, max_y)) =
                    projected_directional_page_rect(&level_vps[level], &planes, bmin, bmax, 1)
                else {
                    continue;
                };
                for y in min_y..=max_y {
                    for x in min_x..=max_x {
                        let index = usize::from(y) * axis + usize::from(x);
                        if coverage[index] == 0 {
                            touched.push(index as u16);
                        }
                        coverage[index] = coverage[index].saturating_add(1);
                    }
                }
            }

            rank_level(
                &coverage,
                touched.into_iter().map(usize::from),
                light,
                level,
                axis,
                center,
            )
        });

    interleave_levels(per_level)
}

pub(super) fn compact_directional_coverage(coverage: &[u32], light: u16) -> Vec<VirtualShadowPage> {
    if coverage.len() != COVERAGE_ENTRIES {
        return Vec::new();
    }
    let axis = VSM_VIRTUAL_PAGES_PER_AXIS as usize;
    let center = i32::from(VSM_VIRTUAL_PAGES_PER_AXIS);
    let per_level = std::array::from_fn(|level| {
        let offset = level * PAGES_PER_LEVEL;
        let counts = &coverage[offset..offset + PAGES_PER_LEVEL];
        rank_level(counts, 0..PAGES_PER_LEVEL, light, level, axis, center)
    });
    interleave_levels(per_level)
}

fn rank_level(
    coverage: &[u32],
    indices: impl IntoIterator<Item = usize>,
    light: u16,
    level: usize,
    axis: usize,
    center: i32,
) -> Vec<VirtualShadowPage> {
    let mut ranked = Vec::new();
    for index in indices {
        let count = coverage[index];
        if count == 0 {
            continue;
        }
        ranked.push((
            VirtualShadowPage {
                light,
                level: level as u8,
                x: (index % axis) as u16,
                y: (index / axis) as u16,
            },
            count,
        ));
    }
    ranked.sort_unstable_by_key(|(page, count)| {
        let dx = (i32::from(page.x) * 2 + 1 - center).unsigned_abs();
        let dy = (i32::from(page.y) * 2 + 1 - center).unsigned_abs();
        (
            std::cmp::Reverse(*count),
            dx.max(dy),
            dx + dy,
            page.y,
            page.x,
        )
    });
    ranked
        .into_iter()
        .take(VSM_DIRECTIONAL_LEVEL_PAGE_CAPS[level])
        .map(|(page, _)| page)
        .collect()
}

fn interleave_levels(
    per_level: [Vec<VirtualShadowPage>; VSM_CLIP_LEVELS as usize],
) -> Vec<VirtualShadowPage> {
    let mut pages = Vec::with_capacity(per_level.iter().map(Vec::len).sum());
    let mut next = [0usize; VSM_CLIP_LEVELS as usize];
    loop {
        let mut appended = false;
        for level in 0..VSM_CLIP_LEVELS as usize {
            if next[level] < per_level[level].len() {
                pages.push(per_level[level][next[level]]);
                next[level] += 1;
                appended = true;
            }
        }
        if !appended {
            return pages;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn hash_oracle(
        level_vps: [[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize],
        receiver_bounds: &[([f32; 3], [f32; 3])],
        light: u16,
    ) -> Vec<VirtualShadowPage> {
        let center = i32::from(VSM_VIRTUAL_PAGES_PER_AXIS);
        let per_level = std::array::from_fn(|level| {
            let planes = crate::scene::extract_frustum_planes(&level_vps[level]);
            let mut coverage = HashMap::<VirtualShadowPage, u32>::new();
            for &(bmin, bmax) in receiver_bounds {
                let Some((min_x, min_y, max_x, max_y)) =
                    projected_directional_page_rect(&level_vps[level], &planes, bmin, bmax, 1)
                else {
                    continue;
                };
                for y in min_y..=max_y {
                    for x in min_x..=max_x {
                        let page = VirtualShadowPage {
                            light,
                            level: level as u8,
                            x,
                            y,
                        };
                        coverage
                            .entry(page)
                            .and_modify(|count| *count = count.saturating_add(1))
                            .or_insert(1);
                    }
                }
            }
            let mut ranked: Vec<_> = coverage.into_iter().collect();
            ranked.sort_unstable_by_key(|(page, count)| {
                let dx = (i32::from(page.x) * 2 + 1 - center).unsigned_abs();
                let dy = (i32::from(page.y) * 2 + 1 - center).unsigned_abs();
                (
                    std::cmp::Reverse(*count),
                    dx.max(dy),
                    dx + dy,
                    page.y,
                    page.x,
                )
            });
            ranked
                .into_iter()
                .take(VSM_DIRECTIONAL_LEVEL_PAGE_CAPS[level])
                .map(|(page, _)| page)
                .collect()
        });
        interleave_levels(per_level)
    }

    #[test]
    fn fixed_coverage_compaction_matches_hash_oracle_exactly() {
        let mut bounds = Vec::new();
        for y in 0..9 {
            for x in 0..12 {
                let min_x = -0.92 + x as f32 * 0.13;
                let min_y = -0.88 + y as f32 * 0.17;
                bounds.push(([min_x, min_y, 0.2], [min_x + 0.31, min_y + 0.28, 0.8]));
            }
        }
        bounds.push(([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]));
        bounds.push(([1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]));
        let vps = [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize];
        let expected = hash_oracle(vps, &bounds, 7);
        assert_eq!(directional_receiver_demand(vps, &bounds, 7), expected);

        let axis = VSM_VIRTUAL_PAGES_PER_AXIS as usize;
        let mut dense = vec![0u32; COVERAGE_ENTRIES];
        for level in 0..VSM_CLIP_LEVELS as usize {
            let planes = crate::scene::extract_frustum_planes(&vps[level]);
            for &(bmin, bmax) in &bounds {
                let Some((min_x, min_y, max_x, max_y)) =
                    projected_directional_page_rect(&vps[level], &planes, bmin, bmax, 1)
                else {
                    continue;
                };
                for y in min_y..=max_y {
                    for x in min_x..=max_x {
                        let index =
                            level * PAGES_PER_LEVEL + usize::from(y) * axis + usize::from(x);
                        dense[index] = dense[index].saturating_add(1);
                    }
                }
            }
        }
        assert_eq!(compact_directional_coverage(&dense, 7), expected);
    }

    #[test]
    fn coverage_storage_matches_one_r32_virtual_level() {
        assert_eq!(PAGES_PER_LEVEL, 1024);
        assert_eq!(
            std::mem::size_of::<[u32; PAGES_PER_LEVEL]>(),
            PAGES_PER_LEVEL * std::mem::size_of::<u32>(),
        );
    }
}
