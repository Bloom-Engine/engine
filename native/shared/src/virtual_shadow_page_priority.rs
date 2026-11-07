use super::VirtualShadowPage;

/// Prioritize the guarded pages nearest each projected caster footprint.
///
/// Near clip levels win ties so a strict page budget improves the highest
/// detail shadow first. Integer doubled coordinates keep the order stable.
pub(super) fn center_first(
    mut pages: Vec<VirtualShadowPage>,
    priority: &[(u16, u16)],
) -> Vec<VirtualShadowPage> {
    pages.sort_unstable_by_key(|page| {
        let (radius, distance) = priority[page.table_index()];
        (page.level, radius, distance, page.y, page.x)
    });
    pages
}

#[cfg(test)]
mod tests {
    use super::super::{directional_dynamic_fallback_pages, VSM_CLIP_LEVELS};

    #[test]
    fn separated_casters_prioritize_both_cores_before_the_gap() {
        let pages = directional_dynamic_fallback_pages(
            [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize],
            &[
                ([-0.8, -0.02, 0.4], [-0.7, 0.02, 0.6]),
                ([0.7, -0.02, 0.4], [0.8, 0.02, 0.6]),
            ],
            0,
        );
        assert!(pages[..8].iter().all(|page| {
            page.level == 0
                && ((3..=4).contains(&page.x) || (27..=28).contains(&page.x))
                && (15..=16).contains(&page.y)
        }));
    }
}
