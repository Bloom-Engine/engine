use super::VirtualGeometryAsset;
use bloom_geometry_format::{GeometryArchive, NO_RELATION};
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClusterGroup {
    pub first_cluster: u32,
    pub cluster_count: u32,
}

impl ClusterGroup {
    fn range(self) -> Range<usize> {
        let start = self.first_cluster as usize;
        start..start + self.cluster_count as usize
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PageTransition {
    pub group: ClusterGroup,
    pub upload_pages: Vec<u32>,
    pub evict_pages: Vec<u32>,
    pub resident_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedClusterGroup {
    pub group: ClusterGroup,
    pub lod_level: u32,
    pub requested_lod_level: u32,
    pub fallback_levels: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResidencyTelemetry {
    pub budget_bytes: u64,
    pub resident_bytes: u64,
    pub pinned_bytes: u64,
    pub resident_pages: u32,
    pub pinned_pages: u32,
    pub uploads: u64,
    pub evictions: u64,
    pub exact_resolutions: u64,
    pub fallback_resolutions: u64,
    pub unresolved_requests: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct PageState {
    resident: bool,
    pinned: bool,
    last_use: u64,
}

/// Deterministic fixed-budget page state for a future GPU geometry cache.
///
/// Root pages are pinned at construction. `make_group_resident` performs an
/// atomic metadata transition: either every page needed by the requested
/// cluster group fits, or no residency state changes.
#[derive(Debug)]
pub struct VirtualGeometryResidency {
    asset: Arc<VirtualGeometryAsset>,
    pages: Vec<PageState>,
    telemetry: ResidencyTelemetry,
    clock: u64,
}

impl VirtualGeometryResidency {
    pub fn new(
        asset: Arc<VirtualGeometryAsset>,
        budget_bytes: u64,
    ) -> Result<Self, ResidencyError> {
        let archive = asset.archive();
        let pinned_pages = archive.coarse_root_page_count();
        let pinned_bytes = archive.coarse_root_page_bytes();
        if pinned_bytes > budget_bytes {
            return Err(ResidencyError::RootBudgetExceeded {
                required_bytes: pinned_bytes,
                budget_bytes,
            });
        }
        let mut pages = vec![PageState::default(); archive.pages.len()];
        for page in pages.iter_mut().take(pinned_pages) {
            page.resident = true;
            page.pinned = true;
        }
        Ok(Self {
            asset,
            pages,
            telemetry: ResidencyTelemetry {
                budget_bytes,
                resident_bytes: pinned_bytes,
                pinned_bytes,
                resident_pages: pinned_pages as u32,
                pinned_pages: pinned_pages as u32,
                ..ResidencyTelemetry::default()
            },
            clock: 0,
        })
    }

    pub fn asset(&self) -> &Arc<VirtualGeometryAsset> {
        &self.asset
    }

    pub fn telemetry(&self) -> ResidencyTelemetry {
        self.telemetry
    }

    pub fn is_page_resident(&self, page_index: u32) -> bool {
        self.pages
            .get(page_index as usize)
            .is_some_and(|page| page.resident)
    }

    pub fn pinned_upload_pages(&self) -> Range<u32> {
        0..self.telemetry.pinned_pages
    }

    pub fn group_containing(&self, cluster_index: u32) -> Result<ClusterGroup, ResidencyError> {
        group_containing(self.asset.archive(), cluster_index)
    }

    pub fn make_group_resident(
        &mut self,
        cluster_index: u32,
    ) -> Result<PageTransition, ResidencyError> {
        let group = self.group_containing(cluster_index)?;
        let required_pages = group_pages(self.asset.archive(), group);
        let upload_pages = required_pages
            .iter()
            .copied()
            .filter(|page| !self.pages[*page as usize].resident)
            .collect::<Vec<_>>();
        let upload_bytes = page_bytes(self.asset.archive(), &upload_pages);
        let required_streamable_pages = required_pages
            .iter()
            .copied()
            .filter(|page| !self.pages[*page as usize].pinned)
            .collect::<Vec<_>>();
        let required_streamable_bytes =
            page_bytes(self.asset.archive(), &required_streamable_pages);
        if self.telemetry.pinned_bytes + required_streamable_bytes > self.telemetry.budget_bytes {
            return Err(ResidencyError::GroupBudgetExceeded {
                group,
                required_bytes: required_streamable_bytes,
                available_bytes: self
                    .telemetry
                    .budget_bytes
                    .saturating_sub(self.telemetry.pinned_bytes),
            });
        }

        let mut projected = self.telemetry.resident_bytes + upload_bytes;
        let mut candidates = self
            .pages
            .iter()
            .enumerate()
            .filter(|(index, state)| {
                state.resident && !state.pinned && !required_pages.contains(&(*index as u32))
            })
            .map(|(index, state)| (state.last_use, index as u32))
            .collect::<Vec<_>>();
        candidates.sort_unstable();
        let mut evict_pages = Vec::new();
        for (_, page_index) in candidates {
            if projected <= self.telemetry.budget_bytes {
                break;
            }
            projected -= self.asset.archive().pages[page_index as usize].payload_bytes as u64;
            evict_pages.push(page_index);
        }
        if projected > self.telemetry.budget_bytes {
            return Err(ResidencyError::GroupBudgetExceeded {
                group,
                required_bytes: required_streamable_bytes,
                available_bytes: self
                    .telemetry
                    .budget_bytes
                    .saturating_sub(self.telemetry.pinned_bytes),
            });
        }

        for page_index in &evict_pages {
            self.pages[*page_index as usize].resident = false;
        }
        for page_index in &upload_pages {
            self.pages[*page_index as usize].resident = true;
        }
        self.clock = self.clock.saturating_add(1);
        for page_index in &required_pages {
            self.pages[*page_index as usize].last_use = self.clock;
        }
        self.telemetry.resident_bytes = projected;
        self.telemetry.resident_pages =
            self.pages.iter().filter(|page| page.resident).count() as u32;
        self.telemetry.uploads += upload_pages.len() as u64;
        self.telemetry.evictions += evict_pages.len() as u64;
        Ok(PageTransition {
            group,
            upload_pages,
            evict_pages,
            resident_bytes: projected,
        })
    }

    pub fn resolve_cluster(
        &mut self,
        cluster_index: u32,
    ) -> Result<Option<ResolvedClusterGroup>, ResidencyError> {
        let archive = self.asset.archive_arc();
        let requested_group = group_containing(&archive, cluster_index)?;
        let requested_lod_level = group_lod(&archive, requested_group);
        let mut group = requested_group;
        loop {
            if group_pages(&archive, group)
                .into_iter()
                .all(|page| self.is_page_resident(page))
            {
                self.touch_group(&archive, group);
                let lod_level = group_lod(&archive, group);
                let fallback_levels = lod_level.saturating_sub(requested_lod_level);
                if fallback_levels == 0 {
                    self.telemetry.exact_resolutions += 1;
                } else {
                    self.telemetry.fallback_resolutions += 1;
                }
                return Ok(Some(ResolvedClusterGroup {
                    group,
                    lod_level,
                    requested_lod_level,
                    fallback_levels,
                }));
            }
            let Some(parent) = parent_group(&archive, group) else {
                self.telemetry.unresolved_requests += 1;
                return Ok(None);
            };
            group = parent;
        }
    }

    fn touch_group(&mut self, archive: &GeometryArchive, group: ClusterGroup) {
        self.clock = self.clock.saturating_add(1);
        for page in group_pages(archive, group) {
            self.pages[page as usize].last_use = self.clock;
        }
    }
}

fn group_containing(
    archive: &GeometryArchive,
    cluster_index: u32,
) -> Result<ClusterGroup, ResidencyError> {
    let cluster = archive
        .clusters
        .get(cluster_index as usize)
        .ok_or(ResidencyError::MissingCluster(cluster_index))?;
    if cluster.parent != NO_RELATION {
        let parent = &archive.clusters[cluster.parent as usize];
        return Ok(ClusterGroup {
            first_cluster: parent.first_child,
            cluster_count: parent.child_count,
        });
    }
    if cluster.child_count != 0 {
        let child = &archive.clusters[cluster.first_child as usize];
        return Ok(ClusterGroup {
            first_cluster: child.parent,
            cluster_count: child.parent_count,
        });
    }
    Ok(ClusterGroup {
        first_cluster: cluster_index,
        cluster_count: 1,
    })
}

fn parent_group(archive: &GeometryArchive, group: ClusterGroup) -> Option<ClusterGroup> {
    let first = &archive.clusters[group.first_cluster as usize];
    (first.parent != NO_RELATION).then_some(ClusterGroup {
        first_cluster: first.parent,
        cluster_count: first.parent_count,
    })
}

fn group_lod(archive: &GeometryArchive, group: ClusterGroup) -> u32 {
    archive.clusters[group.first_cluster as usize].lod_level
}

fn group_pages(archive: &GeometryArchive, group: ClusterGroup) -> Vec<u32> {
    let mut pages = archive.clusters[group.range()]
        .iter()
        .map(|cluster| cluster.page_index)
        .collect::<Vec<_>>();
    pages.sort_unstable();
    pages.dedup();
    pages
}

fn page_bytes(archive: &GeometryArchive, pages: &[u32]) -> u64 {
    pages
        .iter()
        .map(|page| archive.pages[*page as usize].payload_bytes as u64)
        .sum()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResidencyError {
    MissingCluster(u32),
    RootBudgetExceeded {
        required_bytes: u64,
        budget_bytes: u64,
    },
    GroupBudgetExceeded {
        group: ClusterGroup,
        required_bytes: u64,
        available_bytes: u64,
    },
}

impl fmt::Display for ResidencyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCluster(index) => write!(formatter, "cluster {index} does not exist"),
            Self::RootBudgetExceeded {
                required_bytes,
                budget_bytes,
            } => write!(
                formatter,
                "coarse roots require {required_bytes} bytes but the residency budget is {budget_bytes}"
            ),
            Self::GroupBudgetExceeded {
                group,
                required_bytes,
                available_bytes,
            } => write!(
                formatter,
                "cluster group {}+{} requires {required_bytes} streamable bytes but only \
                 {available_bytes} are available",
                group.first_cluster, group.cluster_count
            ),
        }
    }
}

impl std::error::Error for ResidencyError {}
