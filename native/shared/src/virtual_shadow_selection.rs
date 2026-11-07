use std::sync::OnceLock;

use crate::renderer::capabilities::RendererCapabilityTier;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct VirtualShadowSelection {
    pub(super) requested_by_user: bool,
    pub(super) capability_eligible: bool,
    pub(super) enabled: bool,
    pub(super) reason: &'static str,
}

static CONFIGURED_CAPABILITY_TIER: OnceLock<RendererCapabilityTier> = OnceLock::new();

pub(crate) fn configure_capability_tier(tier: RendererCapabilityTier) {
    let _ = CONFIGURED_CAPABILITY_TIER.set(tier);
}

fn selection_for(
    requested_by_user: bool,
    tier: Option<RendererCapabilityTier>,
) -> VirtualShadowSelection {
    let capability_eligible = tier.is_none_or(|tier| tier >= RendererCapabilityTier::HighEnd);
    let (enabled, reason) = match (requested_by_user, capability_eligible) {
        (false, _) => (false, "not-requested"),
        (true, false) => (false, "lower-tier-csm-fallback"),
        (true, true) => (true, "high-tier-vsm"),
    };
    VirtualShadowSelection {
        requested_by_user,
        capability_eligible,
        enabled,
        reason,
    }
}

fn requested_by_user() -> bool {
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        std::env::var("BLOOM_VSM")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "on" | "true" | "enabled"
                )
            })
            .unwrap_or(false)
    })
}

pub(super) fn selection() -> VirtualShadowSelection {
    selection_for(
        requested_by_user(),
        CONFIGURED_CAPABILITY_TIER.get().copied(),
    )
}

pub(crate) fn virtual_shadows_requested() -> bool {
    selection().enabled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_vsm_is_enabled_on_high_tier_only() {
        let disabled = selection_for(false, Some(RendererCapabilityTier::HighEnd));
        assert!(!disabled.requested_by_user);
        assert!(disabled.capability_eligible);
        assert!(!disabled.enabled);
        assert_eq!(disabled.reason, "not-requested");

        for tier in [
            RendererCapabilityTier::Baseline,
            RendererCapabilityTier::Modern,
        ] {
            let fallback = selection_for(true, Some(tier));
            assert!(fallback.requested_by_user);
            assert!(!fallback.capability_eligible);
            assert!(!fallback.enabled);
            assert_eq!(fallback.reason, "lower-tier-csm-fallback");
        }

        let high = selection_for(true, Some(RendererCapabilityTier::HighEnd));
        assert!(high.requested_by_user);
        assert!(high.capability_eligible);
        assert!(high.enabled);
        assert_eq!(high.reason, "high-tier-vsm");
    }
}
