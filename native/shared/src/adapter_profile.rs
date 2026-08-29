//! One capability-owned package profile decision shared by cooked assets.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AdapterAssetProfilePlan {
    runtime_platform: &'static str,
    bc_supported: bool,
    native_profile_selected: bool,
}

impl AdapterAssetProfilePlan {
    pub(crate) fn from_features(features: wgpu::Features) -> Self {
        Self::from_bc_support(features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC))
    }

    pub(crate) fn from_bc_support(bc_supported: bool) -> Self {
        let runtime_platform = runtime_platform_profile();
        let native_profile_selected = desktop_bc_profile(runtime_platform) && bc_supported;
        Self {
            runtime_platform,
            bc_supported,
            native_profile_selected,
        }
    }

    pub(crate) const fn runtime_platform(self) -> &'static str {
        self.runtime_platform
    }

    pub(crate) const fn bc_supported(self) -> bool {
        self.bc_supported
    }

    pub(crate) const fn native_profile_selected(self) -> bool {
        self.native_profile_selected
    }

    pub(crate) const fn selected_platform(self) -> &'static str {
        if self.native_profile_selected {
            self.runtime_platform
        } else {
            "portable"
        }
    }

    pub(crate) const fn has_portable_fallback(self) -> bool {
        self.native_profile_selected
    }
}

const fn runtime_platform_profile() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "android") {
        "android"
    } else if cfg!(target_os = "ios") {
        "ios"
    } else if cfg!(target_os = "tvos") {
        "tvos"
    } else if cfg!(target_os = "visionos") {
        "visionos"
    } else {
        "portable"
    }
}

fn desktop_bc_profile(platform: &str) -> bool {
    matches!(platform, "macos" | "windows" | "linux")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_plan_never_assigns_bc_to_a_capability_neutral_profile() {
        let portable = AdapterAssetProfilePlan::from_bc_support(false);
        assert_eq!(portable.selected_platform(), "portable");
        assert!(!portable.has_portable_fallback());

        let bc = AdapterAssetProfilePlan::from_bc_support(true);
        if matches!(bc.runtime_platform(), "macos" | "windows" | "linux") {
            assert_eq!(bc.selected_platform(), bc.runtime_platform());
            assert!(bc.has_portable_fallback());
        } else {
            assert_eq!(bc.selected_platform(), "portable");
            assert!(!bc.has_portable_fallback());
        }
    }
}
