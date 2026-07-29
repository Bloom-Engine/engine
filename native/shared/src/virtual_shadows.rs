//! Deterministic virtual-shadow page residency for issue #132.
//!
//! Owns bounded virtual-to-physical mapping, invalidation, and page tables.
//! CSM remains the fallback for absent, dirty, over-budget, or unrendered pages.

use std::collections::HashMap;
#[path = "directional_shadow_clipmap.rs"]
mod clipmap;
#[path = "virtual_shadow_debug.rs"]
mod debug;
#[path = "virtual_shadow_gpu_receiver.rs"]
mod gpu_receiver;
#[path = "virtual_shadow_local_lights.rs"]
mod local_lights;
#[path = "virtual_shadow_page_priority.rs"]
mod page_priority;
#[path = "virtual_shadow_receiver_demand.rs"]
mod receiver_demand;
#[path = "virtual_shadow_report.rs"]
mod report;
#[path = "virtual_shadow_selection.rs"]
mod selection;

pub(crate) use local_lights::{LocalShadowAdmissionStats, LocalShadowRequest};
use selection::selection as virtual_shadow_selection;
pub(crate) use selection::{configure_capability_tier, virtual_shadows_requested};

pub const VSM_CLIP_LEVELS: u8 = 3;
pub const VSM_VIRTUAL_PAGES_PER_AXIS: u16 = 32;
pub const VSM_PAGE_INTERIOR: u16 = 128;
pub const VSM_PAGE_BORDER: u16 = 2;
pub const VSM_PHYSICAL_PAGE_SIZE: u16 = VSM_PAGE_INTERIOR + VSM_PAGE_BORDER * 2;
pub const VSM_DEFAULT_PHYSICAL_PAGES: u16 = 256;
pub const VSM_MAX_PAGE_RENDER_BUDGET: u16 = 64;
pub const VSM_DYNAMIC_OVERLAY_PAGE_BUDGET: usize = 4;
pub const VSM_DYNAMIC_OVERLAY_DRAW_BUDGET: usize = 64;
pub const VSM_LOCAL_FACES: u8 = 6;
pub const VSM_MAX_LOCAL_SHADOW_LIGHTS: usize = 5;
pub const VSM_MAX_LOCAL_SHADOW_REQUESTS: usize = 256;
const VSM_DIRECTIONAL_LEVEL_PAGE_CAPS: [usize; VSM_CLIP_LEVELS as usize] = [144, 64, 16];

pub(crate) use clipmap::{
    projection as directional_clipmap_projection, DirectionalClipmapCacheKey,
};
pub use receiver_demand::directional_receiver_demand;

#[repr(C, align(16))]
#[derive(Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct LocalVsmSamplingSlot {
    face_vps: [[[f32; 4]; 4]; VSM_LOCAL_FACES as usize],
    face_pages_0_3: [u32; 4],
    face_pages_4_5: [u32; 4],
}

#[repr(C, align(16))]
#[derive(Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct DirectionalVsmSamplingParams {
    level_vps: [[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize],
    words: [u32; 4],
    local_light_meta: [[u32; 4]; VSM_MAX_LOCAL_SHADOW_REQUESTS],
    local_slots: [LocalVsmSamplingSlot; VSM_MAX_LOCAL_SHADOW_LIGHTS],
}

pub(crate) const VSM_SAMPLING_PARAMS_BYTES: u64 =
    std::mem::size_of::<DirectionalVsmSamplingParams>() as u64;

/// Page-table value zero means "sample the conventional shadow fallback".
///
/// Resident entries store physical page + 1 in the low 16 bits and a
/// saturating residency age in the high 16 bits. The shader can therefore
/// cross-fade a newly rendered VSM page over the CSM result without another
/// texture or buffer.
pub const VSM_PAGE_TABLE_MISSING: u32 = 0;

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct PreparedLocalShadowLight {
    pub request: LocalShadowRequest,
    pub shading_index: u16,
    pub face_vps: [[[f32; 4]; 4]; VSM_LOCAL_FACES as usize],
    pub face_signatures: [u64; VSM_LOCAL_FACES as usize],
}

pub(crate) fn local_shadow_face_vps(
    request: LocalShadowRequest,
) -> [[[f32; 4]; 4]; VSM_LOCAL_FACES as usize] {
    local_lights::face_vps(request)
}

#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VirtualShadowPage {
    pub light: u16,
    pub level: u8,
    pub x: u16,
    pub y: u16,
}

impl VirtualShadowPage {
    pub fn new(light: u16, level: u8, x: u16, y: u16) -> Option<Self> {
        (level < VSM_CLIP_LEVELS
            && x < VSM_VIRTUAL_PAGES_PER_AXIS
            && y < VSM_VIRTUAL_PAGES_PER_AXIS)
            .then_some(Self { light, level, x, y })
    }

    pub fn new_local(point_light_index: u16, face: u8) -> Option<Self> {
        (usize::from(point_light_index) < VSM_MAX_LOCAL_SHADOW_REQUESTS && face < VSM_LOCAL_FACES)
            .then_some(Self {
                light: point_light_index + 1,
                level: face,
                x: 0,
                y: 0,
            })
    }

    pub fn local_light_index(self) -> Option<u16> {
        (self.light > 0 && self.level < VSM_LOCAL_FACES && self.x == 0 && self.y == 0)
            .then_some(self.light - 1)
    }

    fn table_index(self) -> usize {
        let axis = VSM_VIRTUAL_PAGES_PER_AXIS as usize;
        self.level as usize * axis * axis + self.y as usize * axis + self.x as usize
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PageRequest {
    pub page: VirtualShadowPage,
    pub physical_page: u16,
    pub needs_render: bool,
    pub evicted: Option<VirtualShadowPage>,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct VirtualShadowCacheStats {
    pub capacity: u16,
    pub resident: u16,
    pub requested: u32,
    pub hits: u32,
    pub misses: u32,
    pub evictions: u32,
    pub denied: u32,
    pub dirty: u16,
    pub rendered: u32,
    pub invalidated: u32,
    pub clipmap_level_rebases: u32,
    pub clipmap_pages_preserved: u32,
    pub clipmap_pages_dropped: u32,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct LocalShadowPageStats {
    requested: u32,
    hits: u32,
    misses: u32,
    denied: u32,
    invalidated: u32,
    rendered: u32,
}

#[derive(Copy, Clone, Debug)]
struct PhysicalPage {
    owner: Option<VirtualShadowPage>,
    last_used_frame: u64,
    rendered_frame: u64,
    rendered_signature: u64,
    dirty: bool,
}

impl Default for PhysicalPage {
    fn default() -> Self {
        Self {
            owner: None,
            last_used_frame: 0,
            rendered_frame: 0,
            rendered_signature: 0,
            dirty: true,
        }
    }
}

/// Fixed-budget, deterministic LRU cache.
///
/// Pages requested earlier in the current frame are protected from eviction.
/// If a frame requests more unique pages than the pool can hold, later
/// requests are denied and sample CSM instead of churning already selected
/// pages or exceeding the configured memory budget.
pub struct VirtualShadowPageCache {
    physical: Vec<PhysicalPage>,
    mapping: HashMap<VirtualShadowPage, u16>,
    frame: u64,
    stats: VirtualShadowCacheStats,
}

impl VirtualShadowPageCache {
    pub fn new(capacity: u16) -> Self {
        assert!(capacity > 0, "VSM page cache requires at least one page");
        Self {
            physical: vec![PhysicalPage::default(); capacity as usize],
            mapping: HashMap::with_capacity(capacity as usize),
            frame: 0,
            stats: VirtualShadowCacheStats {
                capacity,
                ..Default::default()
            },
        }
    }

    pub fn begin_frame(&mut self, frame: u64) {
        self.frame = frame.max(1);
        self.stats.requested = 0;
        self.stats.hits = 0;
        self.stats.misses = 0;
        self.stats.evictions = 0;
        self.stats.denied = 0;
        self.stats.rendered = 0;
        self.stats.invalidated = 0;
        self.stats.clipmap_level_rebases = 0;
        self.stats.clipmap_pages_preserved = 0;
        self.stats.clipmap_pages_dropped = 0;
    }

    pub fn request(
        &mut self,
        page: VirtualShadowPage,
        content_signature: u64,
    ) -> Option<PageRequest> {
        self.stats.requested = self.stats.requested.saturating_add(1);
        if let Some(&physical_page) = self.mapping.get(&page) {
            self.stats.hits = self.stats.hits.saturating_add(1);
            let slot = &mut self.physical[physical_page as usize];
            slot.last_used_frame = self.frame;
            if slot.rendered_signature != content_signature && !slot.dirty {
                slot.dirty = true;
                self.stats.invalidated = self.stats.invalidated.saturating_add(1);
            }
            let result = PageRequest {
                page,
                physical_page,
                needs_render: slot.dirty,
                evicted: None,
            };
            return Some(result);
        }

        self.stats.misses = self.stats.misses.saturating_add(1);
        let candidate = self
            .physical
            .iter()
            .position(|slot| slot.owner.is_none())
            .or_else(|| {
                self.physical
                    .iter()
                    .enumerate()
                    .filter(|(_, slot)| slot.last_used_frame < self.frame)
                    .min_by_key(|(physical_page, slot)| (slot.last_used_frame, *physical_page))
                    .map(|(physical_page, _)| physical_page)
            });
        let Some(physical_page) = candidate else {
            self.stats.denied = self.stats.denied.saturating_add(1);
            return None;
        };

        let evicted = self.physical[physical_page].owner;
        if let Some(old_page) = evicted {
            self.mapping.remove(&old_page);
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
        let physical_page = physical_page as u16;
        self.physical[physical_page as usize] = PhysicalPage {
            owner: Some(page),
            last_used_frame: self.frame,
            rendered_frame: 0,
            rendered_signature: content_signature,
            dirty: true,
        };
        self.mapping.insert(page, physical_page);
        Some(PageRequest {
            page,
            physical_page,
            needs_render: true,
            evicted,
        })
    }

    pub fn mark_rendered(&mut self, page: VirtualShadowPage, content_signature: u64) -> bool {
        let Some(&physical_page) = self.mapping.get(&page) else {
            return false;
        };
        let slot = &mut self.physical[physical_page as usize];
        if slot.owner != Some(page) {
            return false;
        }
        slot.rendered_frame = self.frame;
        slot.rendered_signature = content_signature;
        slot.dirty = false;
        self.stats.rendered = self.stats.rendered.saturating_add(1);
        true
    }

    pub fn finish_requests(&mut self) {
        self.refresh_counts();
    }

    pub fn record_stable_requests(&mut self, requested: usize) {
        let requested = requested.min(u32::MAX as usize) as u32;
        let directional_resident = self
            .physical
            .iter()
            .filter(|slot| slot.owner.is_some_and(|owner| owner.light == 0))
            .count()
            .min(u32::MAX as usize) as u32;
        let hits = requested.min(directional_resident);
        self.stats.requested = self.stats.requested.saturating_add(requested);
        self.stats.hits = self.stats.hits.saturating_add(hits);
        self.stats.misses = self
            .stats
            .misses
            .saturating_add(requested.saturating_sub(hits));
        self.stats.denied = self
            .stats
            .denied
            .saturating_add(requested.saturating_sub(hits));
    }

    pub fn invalidate_light(&mut self, light: u16) {
        for slot in &mut self.physical {
            if slot.owner.is_some_and(|owner| owner.light == light) && !slot.dirty {
                slot.dirty = true;
                self.stats.invalidated = self.stats.invalidated.saturating_add(1);
            }
        }
        self.refresh_counts();
    }

    pub fn invalidate_level(&mut self, light: u16, level: u8) {
        for slot in &mut self.physical {
            if slot
                .owner
                .is_some_and(|owner| owner.light == light && owner.level == level)
                && !slot.dirty
            {
                slot.dirty = true;
                self.stats.invalidated = self.stats.invalidated.saturating_add(1);
            }
        }
        self.refresh_counts();
    }

    pub fn scroll_level(&mut self, light: u16, level: u8, delta: [i32; 2]) {
        if delta == [0, 0] {
            return;
        }
        let axis = i32::from(VSM_VIRTUAL_PAGES_PER_AXIS);
        let mut preserved = 0u32;
        let mut dropped = 0u32;
        for slot in &mut self.physical {
            let Some(owner) = slot.owner else {
                continue;
            };
            if owner.light != light || owner.level != level {
                continue;
            }
            let x = i32::from(owner.x).saturating_add(delta[0]);
            let y = i32::from(owner.y).saturating_add(delta[1]);
            if (0..axis).contains(&x) && (0..axis).contains(&y) {
                slot.owner = Some(VirtualShadowPage {
                    x: x as u16,
                    y: y as u16,
                    ..owner
                });
                preserved = preserved.saturating_add(1);
            } else {
                *slot = PhysicalPage::default();
                dropped = dropped.saturating_add(1);
            }
        }
        self.mapping.clear();
        for (physical_page, slot) in self.physical.iter().enumerate() {
            if let Some(owner) = slot.owner {
                let previous = self.mapping.insert(owner, physical_page as u16);
                debug_assert!(
                    previous.is_none(),
                    "clipmap scroll produced duplicate pages"
                );
            }
        }
        self.stats.clipmap_level_rebases = self.stats.clipmap_level_rebases.saturating_add(1);
        self.stats.clipmap_pages_preserved =
            self.stats.clipmap_pages_preserved.saturating_add(preserved);
        self.stats.clipmap_pages_dropped = self.stats.clipmap_pages_dropped.saturating_add(dropped);
        self.refresh_counts();
    }

    pub fn invalidate_all(&mut self) {
        for slot in &mut self.physical {
            if slot.owner.is_some() && !slot.dirty {
                slot.dirty = true;
                self.stats.invalidated = self.stats.invalidated.saturating_add(1);
            }
        }
        self.refresh_counts();
    }

    pub fn invalidate_pages(&mut self, pages: &[VirtualShadowPage]) {
        for page in pages {
            let Some(&physical_page) = self.mapping.get(page) else {
                continue;
            };
            let slot = &mut self.physical[physical_page as usize];
            if !slot.dirty {
                slot.dirty = true;
                self.stats.invalidated = self.stats.invalidated.saturating_add(1);
            }
        }
        self.refresh_counts();
    }

    pub fn page_table(&self, light: u16) -> Vec<u32> {
        let axis = VSM_VIRTUAL_PAGES_PER_AXIS as usize;
        let mut table = vec![VSM_PAGE_TABLE_MISSING; VSM_CLIP_LEVELS as usize * axis * axis];
        for (&page, &physical_page) in &self.mapping {
            if page.light != light {
                continue;
            }
            let slot = &self.physical[physical_page as usize];
            if slot.dirty || slot.rendered_frame == 0 {
                continue;
            }
            let age = self
                .frame
                .saturating_sub(slot.rendered_frame)
                .saturating_add(1)
                // Only the first eight frames are meaningful: the shader
                // reaches 100% VSM at age 8. Saturating here makes the page
                // table byte-stable afterward, so it needs no steady upload.
                .min(8) as u16;
            table[page.table_index()] = (physical_page as u32 + 1) | ((age as u32) << 16);
        }
        table
    }

    fn encoded_page(&self, page: VirtualShadowPage) -> u32 {
        let Some(&physical_page) = self.mapping.get(&page) else {
            return VSM_PAGE_TABLE_MISSING;
        };
        let slot = &self.physical[physical_page as usize];
        if slot.dirty || slot.rendered_frame == 0 {
            return VSM_PAGE_TABLE_MISSING;
        }
        let age = self
            .frame
            .saturating_sub(slot.rendered_frame)
            .saturating_add(1)
            .min(8) as u32;
        u32::from(physical_page) + 1 | (age << 16)
    }

    fn request_state(&self, page: VirtualShadowPage) -> Option<(bool, u64)> {
        let physical_page = *self.mapping.get(&page)?;
        let slot = &self.physical[physical_page as usize];
        Some((slot.dirty, slot.rendered_signature))
    }

    fn light_counts(&self, light: u16) -> (u16, u16) {
        let mut resident = 0u16;
        let mut dirty = 0u16;
        for slot in &self.physical {
            if !slot.owner.is_some_and(|owner| owner.light == light) {
                continue;
            }
            resident = resident.saturating_add(1);
            dirty = dirty.saturating_add(u16::from(slot.dirty));
        }
        (resident, dirty)
    }

    pub fn stats(&self) -> VirtualShadowCacheStats {
        self.stats
    }

    pub fn level_counts(&self, light: u16) -> [(u16, u16); VSM_CLIP_LEVELS as usize] {
        let mut counts = [(0u16, 0u16); VSM_CLIP_LEVELS as usize];
        for slot in &self.physical {
            let Some(owner) = slot.owner else {
                continue;
            };
            if owner.light != light {
                continue;
            }
            counts[owner.level as usize].0 += 1;
            if slot.dirty {
                counts[owner.level as usize].1 += 1;
            }
        }
        counts
    }

    pub fn debug_virtual_rgb(&self, light: u16, scale: u32) -> (u32, u32, Vec<u8>) {
        debug::virtual_rgb(self, light, scale)
    }

    pub fn debug_physical_rgb(&self, scale: u32) -> (u32, u32, Vec<u8>) {
        debug::physical_rgb(self, scale)
    }

    pub fn debug_legend_rgb(scale: u32) -> (u32, u32, Vec<u8>) {
        debug::legend_rgb(scale)
    }

    pub fn memory_bytes(&self) -> u64 {
        let edge = VSM_PHYSICAL_PAGE_SIZE as u64;
        edge * edge * std::mem::size_of::<f32>() as u64 * self.physical.len() as u64
    }

    fn refresh_counts(&mut self) {
        self.stats.resident = self
            .physical
            .iter()
            .filter(|slot| slot.owner.is_some())
            .count() as u16;
        self.stats.dirty = self
            .physical
            .iter()
            .filter(|slot| slot.owner.is_some() && slot.dirty)
            .count() as u16;
    }
}

/// Runtime policy wrapper for the directional prototype.
///
/// The cache foundation is intentionally opt-in until physical page rendering
/// and sampling are connected and qualified. With the default environment it
/// performs no demand walk and allocates no GPU memory, so landing the
/// foundation cannot change images, frame time, or residency.
pub struct DirectionalVirtualShadowMap {
    requested_by_user: bool,
    capability_eligible: bool,
    enabled: bool,
    selection_reason: &'static str,
    sampling_active: bool,
    dynamic_global_fallback: bool,
    dynamic_overlay_pages: Vec<VirtualShadowPage>,
    dynamic_overlay_rendered_pages: usize,
    dynamic_overlay_draws: usize,
    dynamic_overlay_deferred_pages: usize,
    page_cutout_draws: usize,
    page_skinned_draws: usize,
    cache: VirtualShadowPageCache,
    gpu: Option<GpuVirtualShadowResources>,
    frame: u64,
    previous_level_vps: Option<[[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize]>,
    previous_clipmap_keys: Option<[DirectionalClipmapCacheKey; VSM_CLIP_LEVELS as usize]>,
    previous_content_signatures: Option<[u64; VSM_CLIP_LEVELS as usize]>,
    previous_demand_signature: u64,
    fallback_demand: Vec<VirtualShadowPage>,
    receiver_demand: Vec<VirtualShadowPage>,
    receiver_bounds_signature: u64,
    receiver_demand_level_vps: Option<[[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize]>,
    receiver_bounds_count: usize,
    receiver_marking_backend: &'static str,
    last_demand_count: usize,
    receiver_demand_active: bool,
    pending: Vec<PageRequest>,
    uploaded_page_table: Vec<u32>,
    page_table_may_age_until: u64,
    uploaded_sampling_params: Option<DirectionalVsmSamplingParams>,
    render_budget: usize,
    local_requests: Vec<local_lights::LocalShadowRequest>,
    local_selected: Vec<PreparedLocalShadowLight>,
    local_admission_stats: LocalShadowAdmissionStats,
    local_page_stats: [LocalShadowPageStats; VSM_MAX_LOCAL_SHADOW_REQUESTS],
}

impl DirectionalVirtualShadowMap {
    pub fn new(device: &wgpu::Device, shadow_uniform_layout: &wgpu::BindGroupLayout) -> Self {
        let selection = virtual_shadow_selection();
        let requested_capacity = if selection.enabled {
            env_u16(
                "BLOOM_VSM_PHYSICAL_PAGES",
                VSM_DEFAULT_PHYSICAL_PAGES,
                1,
                4096,
            )
        } else {
            1
        };
        let capacity = requested_capacity.min(
            device
                .limits()
                .max_texture_array_layers
                .min(u16::MAX as u32) as u16,
        );
        let page_uniform_bytes =
            crate::shadows::SHADOW_UNIFORM_STRIDE as u64 * crate::shadows::SHADOW_MAX_NODES as u64;
        let buffer_limited_budget =
            (device.limits().max_buffer_size / page_uniform_bytes).min(u16::MAX as u64) as u16;
        let max_render_budget = capacity
            .min(VSM_MAX_PAGE_RENDER_BUDGET)
            .min(buffer_limited_budget.max(1));
        let render_budget = env_u16("BLOOM_VSM_PAGE_BUDGET", 8, 1, max_render_budget).into();
        let gpu = selection.enabled.then(|| {
            GpuVirtualShadowResources::new(device, shadow_uniform_layout, capacity, render_budget)
        });
        let fallback_demand = if selection.enabled {
            centered_directional_demand(0)
        } else {
            Vec::new()
        };
        Self {
            requested_by_user: selection.requested_by_user,
            capability_eligible: selection.capability_eligible,
            enabled: selection.enabled,
            selection_reason: selection.reason,
            sampling_active: false,
            dynamic_global_fallback: false,
            dynamic_overlay_pages: Vec::new(),
            dynamic_overlay_rendered_pages: 0,
            dynamic_overlay_draws: 0,
            dynamic_overlay_deferred_pages: 0,
            page_cutout_draws: 0,
            page_skinned_draws: 0,
            cache: VirtualShadowPageCache::new(capacity),
            gpu,
            frame: 0,
            previous_level_vps: None,
            previous_clipmap_keys: None,
            previous_content_signatures: None,
            previous_demand_signature: 0,
            fallback_demand,
            receiver_demand: Vec::new(),
            receiver_bounds_signature: 0,
            receiver_demand_level_vps: None,
            receiver_bounds_count: 0,
            receiver_marking_backend: "disabled",
            last_demand_count: 0,
            receiver_demand_active: false,
            pending: Vec::with_capacity(render_budget),
            uploaded_page_table: Vec::new(),
            page_table_may_age_until: 0,
            uploaded_sampling_params: None,
            render_budget,
            local_requests: Vec::with_capacity(if selection.enabled {
                VSM_MAX_LOCAL_SHADOW_REQUESTS
            } else {
                0
            }),
            local_selected: Vec::with_capacity(VSM_MAX_LOCAL_SHADOW_LIGHTS),
            local_admission_stats: LocalShadowAdmissionStats::default(),
            local_page_stats: [LocalShadowPageStats::default(); VSM_MAX_LOCAL_SHADOW_REQUESTS],
        }
    }

    pub fn clear_local_requests(&mut self) {
        self.local_requests.clear();
        if self.enabled {
            self.local_selected.clear();
            self.local_admission_stats = LocalShadowAdmissionStats::default();
            self.local_page_stats =
                [LocalShadowPageStats::default(); VSM_MAX_LOCAL_SHADOW_REQUESTS];
        }
    }

    pub fn submit_local_request(
        &mut self,
        light_index: u16,
        position: [f32; 3],
        range: f32,
        intensity: f32,
    ) -> bool {
        if !self.enabled
            || self.local_requests.len() >= VSM_MAX_LOCAL_SHADOW_REQUESTS
            || usize::from(light_index) >= VSM_MAX_LOCAL_SHADOW_REQUESTS
            || position.iter().any(|value| !value.is_finite())
            || !range.is_finite()
            || range <= 0.0
            || !intensity.is_finite()
            || intensity <= 0.0
        {
            return false;
        }
        self.local_requests.push(local_lights::LocalShadowRequest {
            light_index,
            position,
            range,
            intensity,
        });
        true
    }

    pub(crate) fn admit_local_requests(
        &mut self,
        camera: [f32; 3],
        camera_planes: &[[f32; 4]; 6],
    ) -> Vec<LocalShadowRequest> {
        let (admitted, stats) = local_lights::admit(&self.local_requests, camera, camera_planes);
        self.local_admission_stats = stats;
        admitted
    }

    pub(crate) fn local_requests(&self) -> &[LocalShadowRequest] {
        &self.local_requests
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        level_vps: [[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize],
        clipmap_keys: Option<[DirectionalClipmapCacheKey; VSM_CLIP_LEVELS as usize]>,
        content_signatures: [u64; VSM_CLIP_LEVELS as usize],
        receiver_bounds: Option<&[([f32; 3], [f32; 3])]>,
        dynamic_bounds: &[([f32; 3], [f32; 3])],
        local_lights: &[PreparedLocalShadowLight],
    ) {
        self.frame = self.frame.wrapping_add(1).max(1);
        self.cache.begin_frame(self.frame);
        self.pending.clear();
        if !self.enabled {
            return;
        }
        self.dynamic_overlay_rendered_pages = 0;
        self.dynamic_overlay_draws = 0;
        self.dynamic_overlay_deferred_pages = 0;
        self.page_cutout_draws = 0;
        self.page_skinned_draws = 0;
        self.local_page_stats = [LocalShadowPageStats::default(); VSM_MAX_LOCAL_SHADOW_REQUESTS];
        let level_vps_unchanged = self.previous_level_vps == Some(level_vps);
        let content_unchanged = self.previous_content_signatures == Some(content_signatures);
        if !level_vps_unchanged || !content_unchanged {
            if let (Some(previous_keys), Some(current_keys)) =
                (self.previous_clipmap_keys, clipmap_keys)
            {
                let scrolls = std::array::from_fn::<_, { VSM_CLIP_LEVELS as usize }, _>(|level| {
                    current_keys[level].scroll_from(previous_keys[level])
                });
                if scrolls.iter().flatten().any(|delta| *delta != [0, 0]) {
                    // Dynamic depth is frame-specific. Dirty its old address
                    // before the physical owner moves to the new coordinate.
                    self.cache.invalidate_pages(&self.dynamic_overlay_pages);
                }
                for level in 0..VSM_CLIP_LEVELS as usize {
                    match scrolls[level] {
                        Some(delta) if delta != [0, 0] => {
                            self.cache.scroll_level(0, level as u8, delta);
                        }
                        Some(_)
                            if self
                                .previous_level_vps
                                .is_some_and(|vps| vps[level] != level_vps[level]) =>
                        {
                            // Same origin and stable matrix fields should
                            // produce the exact same VP. Treat any discrepancy
                            // as unsafe.
                            self.cache.invalidate_level(0, level as u8);
                        }
                        Some(_) => {}
                        None => self.cache.invalidate_level(0, level as u8),
                    }
                    if self
                        .previous_content_signatures
                        .is_some_and(|signatures| signatures[level] != content_signatures[level])
                    {
                        self.cache.invalidate_level(0, level as u8);
                    }
                }
            } else if let Some(previous) = self.previous_level_vps {
                for level in 0..VSM_CLIP_LEVELS as usize {
                    if previous[level] != level_vps[level]
                        || self.previous_content_signatures.is_some_and(|signatures| {
                            signatures[level] != content_signatures[level]
                        })
                    {
                        self.cache.invalidate_level(0, level as u8);
                    }
                }
            } else {
                self.cache.invalidate_light(0);
            }
        }
        self.previous_level_vps = Some(level_vps);
        self.previous_clipmap_keys = clipmap_keys;
        self.previous_content_signatures = Some(content_signatures);

        let completed_gpu_demand = self
            .gpu
            .as_mut()
            .and_then(|gpu| gpu.receiver_demand.poll(device));
        let receiver_bounds_signature = receiver_bounds
            .filter(|bounds| !bounds.is_empty())
            .map(receiver_bounds_signature);
        self.receiver_bounds_count = receiver_bounds.map_or(0, <[_]>::len);
        if let Some(signature) = receiver_bounds_signature {
            let bounds = receiver_bounds.expect("non-empty receiver bounds have a signature");
            let gpu_wanted = self
                .gpu
                .as_ref()
                .is_some_and(|gpu| gpu.receiver_demand.wants_gpu(bounds.len()));
            let gpu_validated = self
                .gpu
                .as_ref()
                .is_some_and(|gpu| gpu.receiver_demand.validated());
            let current_demand_exact = self.receiver_bounds_signature == signature
                && self.receiver_demand_level_vps == Some(level_vps);
            if !gpu_wanted {
                if !current_demand_exact {
                    self.receiver_demand = directional_receiver_demand(level_vps, bounds, 0);
                    self.receiver_bounds_signature = signature;
                    self.receiver_demand_level_vps = Some(level_vps);
                }
                self.receiver_marking_backend = "fixed-cpu";
            } else {
                let projection_changed = self.receiver_demand_level_vps != Some(level_vps);
                if !gpu_validated || self.receiver_demand.is_empty() || projection_changed {
                    if !current_demand_exact {
                        self.receiver_demand = directional_receiver_demand(level_vps, bounds, 0);
                        self.receiver_bounds_signature = signature;
                        self.receiver_demand_level_vps = Some(level_vps);
                    }
                    self.receiver_marking_backend = if gpu_validated {
                        "fixed-cpu-transition"
                    } else {
                        "fixed-cpu-validation"
                    };
                } else if let Some(completed) =
                    completed_gpu_demand.filter(|completed| completed.level_vps == level_vps)
                {
                    let completed_is_current = completed.bounds_signature == signature;
                    self.receiver_demand = completed.demand;
                    self.receiver_bounds_signature = completed.bounds_signature;
                    self.receiver_demand_level_vps = Some(level_vps);
                    self.receiver_marking_backend = if completed_is_current {
                        "gpu-async"
                    } else {
                        "gpu-async-lagged"
                    };
                } else if current_demand_exact {
                    self.receiver_marking_backend = "gpu-validated-cache";
                } else {
                    // Retain the previous exact result while the next marker
                    // runs. Pages newly touched by receiver motion are absent
                    // and therefore sample current CSM until readback lands.
                    self.receiver_marking_backend = "gpu-async-pending";
                }

                let demand_is_current = self.receiver_bounds_signature == signature
                    && self.receiver_demand_level_vps == Some(level_vps);
                if !gpu_validated || !demand_is_current {
                    let expected = (!gpu_validated).then(|| self.receiver_demand.clone());
                    if let Some(gpu) = self.gpu.as_mut() {
                        gpu.receiver_demand.record(
                            device, queue, encoder, level_vps, bounds, signature, expected,
                        );
                    }
                }
            }
        } else {
            self.receiver_demand.clear();
            self.receiver_bounds_signature = 0;
            self.receiver_demand_level_vps = None;
            self.receiver_marking_backend = if self.enabled {
                "center-fallback"
            } else {
                "disabled"
            };
        }
        let receiver_demand_active = !self.receiver_demand.is_empty();
        let demand_len = if receiver_demand_active {
            self.receiver_demand.len()
        } else {
            self.fallback_demand.len()
        };
        self.update_dynamic_policy(level_vps, dynamic_bounds, demand_len);
        self.local_selected.clear();
        self.local_selected.extend_from_slice(local_lights);
        for local in local_lights {
            let page_stats = &mut self.local_page_stats[local.request.light_index as usize];
            for face in 0..VSM_LOCAL_FACES {
                let page = VirtualShadowPage::new_local(local.request.light_index, face)
                    .expect("admitted local light has a valid face address");
                page_stats.requested = page_stats.requested.saturating_add(1);
                match self.cache.request_state(page) {
                    Some((was_dirty, signature)) => {
                        page_stats.hits = page_stats.hits.saturating_add(1);
                        if signature != local.face_signatures[face as usize] && !was_dirty {
                            page_stats.invalidated = page_stats.invalidated.saturating_add(1);
                        }
                    }
                    None => page_stats.misses = page_stats.misses.saturating_add(1),
                }
                let Some(request) = self
                    .cache
                    .request(page, local.face_signatures[face as usize])
                else {
                    page_stats.denied = page_stats.denied.saturating_add(1);
                    continue;
                };
                if request.needs_render && self.pending.len() < self.render_budget {
                    self.pending.push(request);
                }
            }
        }
        let demand = if receiver_demand_active {
            &self.receiver_demand[..]
        } else {
            &self.fallback_demand[..]
        };
        let demand_signature = demand_signature(demand);
        self.last_demand_count = demand.len();
        self.receiver_demand_active = receiver_demand_active;
        if self.dynamic_global_fallback {
            self.cache.finish_requests();
            self.sync_sampling_params(queue);
            return;
        }
        let demand_unchanged = level_vps_unchanged
            && content_unchanged
            && self.previous_demand_signature == demand_signature;
        if demand_unchanged && self.cache.stats().dirty == 0 {
            self.cache.record_stable_requests(demand.len());
            if self.frame <= self.page_table_may_age_until {
                self.upload_page_table_if_changed(queue);
            }
            self.cache.finish_requests();
            self.sync_sampling_params(queue);
            return;
        }
        self.previous_demand_signature = demand_signature;

        for &page in &self.dynamic_overlay_pages {
            if !demand.contains(&page) {
                continue;
            }
            let Some(request) = self
                .cache
                .request(page, content_signatures[page.level as usize])
            else {
                continue;
            };
            if request.needs_render {
                if self.pending.len() < VSM_DYNAMIC_OVERLAY_PAGE_BUDGET {
                    self.pending.push(request);
                } else {
                    self.dynamic_overlay_deferred_pages += 1;
                }
            }
        }
        for &page in demand {
            if self.dynamic_overlay_contains(page) {
                continue;
            }
            if let Some(request) = self
                .cache
                .request(page, content_signatures[page.level as usize])
            {
                if request.needs_render && self.pending.len() < self.render_budget {
                    self.pending.push(request);
                }
            }
        }
        self.cache.finish_requests();
        self.upload_page_table_if_changed(queue);
        self.sync_sampling_params(queue);
    }

    pub fn pending(&self) -> &[PageRequest] {
        &self.pending
    }

    pub fn finish_rendered_pages(
        &mut self,
        queue: &wgpu::Queue,
        rendered: &[(VirtualShadowPage, u64)],
    ) {
        for &(page, signature) in rendered {
            if self.cache.mark_rendered(page, signature) {
                if let Some(light_index) = page.local_light_index() {
                    let stats = &mut self.local_page_stats[light_index as usize];
                    stats.rendered = stats.rendered.saturating_add(1);
                }
            }
        }
        if !rendered.is_empty() {
            self.page_table_may_age_until = self
                .page_table_may_age_until
                .max(self.frame.saturating_add(7));
        }
        self.cache.finish_requests();
        self.upload_page_table_if_changed(queue);
        self.sync_sampling_params(queue);
    }

    fn update_dynamic_policy(
        &mut self,
        level_vps: [[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize],
        dynamic_bounds: &[([f32; 3], [f32; 3])],
        demand_count: usize,
    ) {
        let mut next = directional_dynamic_fallback_pages(level_vps, dynamic_bounds, 0);
        let full_address_space = VSM_VIRTUAL_PAGES_PER_AXIS as usize
            * VSM_VIRTUAL_PAGES_PER_AXIS as usize
            * VSM_CLIP_LEVELS as usize;
        let global_fallback =
            !next.is_empty() && (demand_count < 128 || next.len() == full_address_space);
        if global_fallback {
            next.clear();
        }

        let mut dirty_pages = std::mem::replace(&mut self.dynamic_overlay_pages, next);
        dirty_pages.extend_from_slice(&self.dynamic_overlay_pages);
        dirty_pages.sort_unstable();
        dirty_pages.dedup();
        self.cache.invalidate_pages(&dirty_pages);
        self.dynamic_global_fallback = global_fallback;
        if global_fallback {
            self.cache.invalidate_light(0);
        }
        let sampling_active = self.enabled && self.gpu.is_some() && !global_fallback;
        self.sampling_active = sampling_active;
    }

    pub fn dynamic_overlay_contains(&self, page: VirtualShadowPage) -> bool {
        self.dynamic_overlay_pages.contains(&page)
    }

    pub fn record_dynamic_overlay_work(
        &mut self,
        rendered_pages: usize,
        draws: usize,
        deferred_pages: usize,
        cutout_draws: usize,
        skinned_draws: usize,
    ) {
        self.dynamic_overlay_rendered_pages = rendered_pages;
        self.dynamic_overlay_draws = draws;
        self.dynamic_overlay_deferred_pages += deferred_pages;
        self.page_cutout_draws = cutout_draws;
        self.page_skinned_draws = skinned_draws;
    }

    pub fn requested(&self) -> bool {
        self.enabled
    }

    pub fn physical_page_view(&self, physical_page: u16) -> Option<&wgpu::TextureView> {
        self.gpu
            .as_ref()?
            .physical_page_views
            .get(physical_page as usize)
    }

    pub fn render_uniform_buffer(&self) -> Option<&wgpu::Buffer> {
        self.gpu.as_ref().map(|gpu| &gpu.render_uniform_buffer)
    }

    pub fn render_uniform_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.gpu.as_ref().map(|gpu| &gpu.render_uniform_bind_group)
    }

    pub fn physical_array_view(&self) -> Option<&wgpu::TextureView> {
        self.gpu.as_ref().map(|gpu| &gpu.physical_array_view)
    }

    pub fn page_table_view(&self) -> Option<&wgpu::TextureView> {
        self.gpu.as_ref().map(|gpu| &gpu.page_table_view)
    }

    pub fn sampling_params_buffer(&self) -> Option<&wgpu::Buffer> {
        self.gpu.as_ref().map(|gpu| &gpu.sampling_params_buffer)
    }

    /// Start mapping receiver-demand readback only after the encoder that
    /// produced it has been submitted. The callback is collected by the next
    /// non-blocking `prepare` poll.
    pub fn after_submit_gpu_receiver(&mut self) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.receiver_demand.after_submit();
        }
    }

    fn upload_page_table_if_changed(&mut self, queue: &wgpu::Queue) {
        if !self.sampling_active {
            return;
        }
        let mut table = self.cache.page_table(0);
        force_dynamic_overlay_age(&mut table, &self.dynamic_overlay_pages);
        if table == self.uploaded_page_table {
            return;
        }
        if let Some(gpu) = self.gpu.as_ref() {
            gpu.upload_page_table(queue, &table);
        }
        self.uploaded_page_table = table;
    }

    fn sampling_params(&self) -> DirectionalVsmSamplingParams {
        let mut local_light_meta = [[0u32; 4]; VSM_MAX_LOCAL_SHADOW_REQUESTS];
        let mut local_slots = [LocalVsmSamplingSlot {
            face_vps: [crate::renderer::IDENTITY_MAT4; VSM_LOCAL_FACES as usize],
            face_pages_0_3: [0; 4],
            face_pages_4_5: [0; 4],
        }; VSM_MAX_LOCAL_SHADOW_LIGHTS];
        for (slot_index, local) in self.local_selected.iter().enumerate() {
            local_light_meta[local.shading_index as usize][0] = 1;
            let mut encoded = [0u32; VSM_LOCAL_FACES as usize];
            for face in 0..VSM_LOCAL_FACES {
                let page = VirtualShadowPage::new_local(local.request.light_index, face)
                    .expect("selected local light has a valid face address");
                encoded[face as usize] = self.cache.encoded_page(page);
            }
            local_slots[slot_index] = LocalVsmSamplingSlot {
                face_vps: local.face_vps,
                face_pages_0_3: encoded[..4].try_into().expect("four local faces"),
                face_pages_4_5: [encoded[4], encoded[5], 0, 0],
            };
            if encoded.iter().all(|entry| *entry != VSM_PAGE_TABLE_MISSING) {
                local_light_meta[local.shading_index as usize][0] = slot_index as u32 + 2;
            }
        }
        DirectionalVsmSamplingParams {
            level_vps: self
                .previous_level_vps
                .unwrap_or([crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize]),
            words: [
                u32::from(self.sampling_active),
                VSM_VIRTUAL_PAGES_PER_AXIS as u32,
                VSM_PAGE_INTERIOR as u32,
                VSM_PAGE_BORDER as u32,
            ],
            local_light_meta,
            local_slots,
        }
    }

    fn sync_sampling_params(&mut self, queue: &wgpu::Queue) {
        let params = self.sampling_params();
        if self.uploaded_sampling_params.as_ref() == Some(&params) {
            return;
        }
        if let Some(gpu) = self.gpu.as_ref() {
            gpu.upload_sampling_params(queue, &params);
        }
        self.uploaded_sampling_params = Some(params);
    }

    pub fn invalidate(&mut self) {
        self.cache.invalidate_all();
        self.previous_level_vps = None;
        self.previous_clipmap_keys = None;
        self.previous_content_signatures = None;
        self.previous_demand_signature = 0;
        self.dynamic_global_fallback = false;
        self.dynamic_overlay_pages.clear();
        self.pending.clear();
        self.sampling_active = false;
        self.uploaded_sampling_params = None;
    }

    pub fn debug_images(&self) -> Vec<(&'static str, u32, u32, Vec<u8>)> {
        if !self.enabled {
            return Vec::new();
        }
        let (virtual_width, virtual_height, virtual_rgb) = self.cache.debug_virtual_rgb(0, 4);
        let (physical_width, physical_height, physical_rgb) = self.cache.debug_physical_rgb(4);
        let (legend_width, legend_height, legend_rgb) =
            VirtualShadowPageCache::debug_legend_rgb(12);
        vec![
            (
                "virtual-shadow-pages",
                virtual_width,
                virtual_height,
                virtual_rgb,
            ),
            (
                "virtual-shadow-physical",
                physical_width,
                physical_height,
                physical_rgb,
            ),
            (
                "virtual-shadow-legend",
                legend_width,
                legend_height,
                legend_rgb,
            ),
        ]
    }

    pub fn report_json(&self) -> String {
        report::json(self)
    }
}

struct GpuVirtualShadowResources {
    _physical_texture: wgpu::Texture,
    physical_array_view: wgpu::TextureView,
    physical_page_views: Vec<wgpu::TextureView>,
    page_table_texture: wgpu::Texture,
    page_table_view: wgpu::TextureView,
    render_uniform_buffer: wgpu::Buffer,
    render_uniform_bind_group: wgpu::BindGroup,
    sampling_params_buffer: wgpu::Buffer,
    receiver_demand: gpu_receiver::GpuReceiverDemand,
}

impl GpuVirtualShadowResources {
    fn new(
        device: &wgpu::Device,
        shadow_uniform_layout: &wgpu::BindGroupLayout,
        physical_pages: u16,
        render_budget: usize,
    ) -> Self {
        let physical_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vsm_physical_depth_pages"),
            size: wgpu::Extent3d {
                width: VSM_PHYSICAL_PAGE_SIZE as u32,
                height: VSM_PHYSICAL_PAGE_SIZE as u32,
                depth_or_array_layers: physical_pages as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let physical_array_view = physical_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("vsm_physical_depth_array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let physical_page_views = (0..physical_pages as u32)
            .map(|physical_page| {
                physical_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("vsm_physical_depth_page"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: physical_page,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let page_table_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vsm_directional_page_table"),
            size: wgpu::Extent3d {
                width: VSM_VIRTUAL_PAGES_PER_AXIS as u32,
                height: VSM_VIRTUAL_PAGES_PER_AXIS as u32,
                depth_or_array_layers: VSM_CLIP_LEVELS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let page_table_view = page_table_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("vsm_directional_page_table_array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        // Queue writes become visible at submit, not at encode time. VSM page
        // matrices therefore cannot share the CSM uniform buffer: doing so
        // would replace matrices referenced by already-encoded cascade draws.
        let render_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vsm_render_uniforms"),
            size: crate::shadows::SHADOW_UNIFORM_STRIDE as u64
                * crate::shadows::SHADOW_MAX_NODES as u64
                * render_budget as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let render_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vsm_render_uniform_bg"),
            layout: shadow_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &render_uniform_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<
                        crate::shadows::ShadowUniforms,
                    >() as u64),
                }),
            }],
        });
        let sampling_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vsm_sampling_params"),
            size: VSM_SAMPLING_PARAMS_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            _physical_texture: physical_texture,
            physical_array_view,
            physical_page_views,
            page_table_texture,
            page_table_view,
            render_uniform_buffer,
            render_uniform_bind_group,
            sampling_params_buffer,
            receiver_demand: gpu_receiver::GpuReceiverDemand::new(device),
        }
    }

    fn upload_page_table(&self, queue: &wgpu::Queue, table: &[u32]) {
        let axis = VSM_VIRTUAL_PAGES_PER_AXIS as usize;
        let layers = VSM_CLIP_LEVELS as usize;
        debug_assert_eq!(table.len(), axis * axis * layers);
        // WebGPU texture copies require 256-byte row alignment. A 32-wide
        // R32Uint row is 128 bytes, so stage each logical row into a padded
        // 256-byte row before queue upload.
        const PADDED_ROW_BYTES: usize = 256;
        let row_words = PADDED_ROW_BYTES / std::mem::size_of::<u32>();
        let mut padded = vec![0u32; row_words * axis * layers];
        for layer in 0..layers {
            for y in 0..axis {
                let source = (layer * axis + y) * axis;
                let destination = (layer * axis + y) * row_words;
                padded[destination..destination + axis]
                    .copy_from_slice(&table[source..source + axis]);
            }
        }
        queue.write_texture(
            self.page_table_texture.as_image_copy(),
            bytemuck::cast_slice(&padded),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(PADDED_ROW_BYTES as u32),
                rows_per_image: Some(axis as u32),
            },
            wgpu::Extent3d {
                width: axis as u32,
                height: axis as u32,
                depth_or_array_layers: layers as u32,
            },
        );
    }

    fn upload_sampling_params(&self, queue: &wgpu::Queue, params: &DirectionalVsmSamplingParams) {
        queue.write_buffer(&self.sampling_params_buffer, 0, bytemuck::bytes_of(params));
    }
}

/// Crop one full cascade VP to a single virtual page, including the physical
/// page's guard texels. Virtual page Y is texture-space (zero at the top), so
/// it is intentionally inverted when converted to WebGPU NDC.
pub fn directional_page_vp(level_vp: [[f32; 4]; 4], page: VirtualShadowPage) -> [[f32; 4]; 4] {
    let axis = VSM_VIRTUAL_PAGES_PER_AXIS as f32;
    let physical_over_interior = VSM_PHYSICAL_PAGE_SIZE as f32 / VSM_PAGE_INTERIOR as f32;
    let half_ndc = physical_over_interior / axis;
    let scale = half_ndc.recip();
    let center_x = (f32::from(page.x) + 0.5) * (2.0 / axis) - 1.0;
    let center_y = 1.0 - (f32::from(page.y) + 0.5) * (2.0 / axis);
    let crop = [
        [scale, 0.0, 0.0, 0.0],
        [0.0, scale, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-center_x * scale, -center_y * scale, 0.0, 1.0],
    ];
    crate::renderer::mat4_multiply(crop, level_vp)
}

const DIRECTIONAL_VSM_SCENE_BINDINGS: &str =
    include_str!("../shaders/virtual_shadows/scene_bindings.wgsl");
const DIRECTIONAL_VSM_SCENE_HELPER: &str =
    include_str!("../shaders/virtual_shadows/scene_helper.wgsl");

/// Build the opt-in scene-shader variant. The canonical source remains
/// byte-for-byte unchanged when VSM is disabled, avoiding an extra branch or
/// binding in the default renderer.
pub(crate) fn directional_scene_shader(source: &str) -> String {
    let helper_marker = "fn sample_shadow(world_pos: vec3<f32>, geo_n: vec3<f32>) -> f32 {";
    let helper_offset = source
        .find(helper_marker)
        .expect("scene shader missing sample_shadow marker");
    let mut output = String::with_capacity(
        source.len() + DIRECTIONAL_VSM_SCENE_BINDINGS.len() + DIRECTIONAL_VSM_SCENE_HELPER.len(),
    );
    output.push_str(DIRECTIONAL_VSM_SCENE_BINDINGS);
    output.push_str(&source[..helper_offset]);
    output.push_str(DIRECTIONAL_VSM_SCENE_HELPER);
    output.push_str(&source[helper_offset..]);
    output = output.replace(
        "let shadow_val = sample_cascade(cascade, shadow_uv, depth_ref);",
        "let shadow_val = sample_virtual_shadow(cascade, recv_pos, shadow_uv, depth_ref);",
    );
    output = output.replace(
        "let next_val = sample_cascade(next_cascade, next_uv, next_depth_ref);",
        "let next_val = sample_virtual_shadow(next_cascade, next_pos, next_uv, next_depth_ref);",
    );
    let light_index = if source.contains("let light_index = cluster_indices[") {
        "light_index"
    } else {
        "i"
    };
    let point_intensity = "pl.color.w * atten2,\n                             base_color";
    let shadowed_point_intensity = format!(
        "pl.color.w * atten2 * sample_local_shadow({light_index}, in.world_pos),\n                             base_color"
    );
    let output = output.replace(point_intensity, &shadowed_point_intensity);
    debug_assert!(
        !source.contains("point_light_count") || output.contains(&shadowed_point_intensity),
        "VSM scene variant must shade its point-light path"
    );
    output
}

/// Add fail-closed local VSM sampling to the legacy/immediate 3D path.
/// The canonical shader remains byte-identical when VSM is not selected.
pub(crate) fn local_immediate_shader(source: &str) -> String {
    let local_helper_offset = DIRECTIONAL_VSM_SCENE_HELPER
        .find("fn local_shadow_face(")
        .expect("VSM scene helper missing local-shadow section");
    let local_helper = &DIRECTIONAL_VSM_SCENE_HELPER[local_helper_offset..];
    let point_term = "pl.color.rgb * pl.color.w * diff * atten2;";
    let shadowed_point_term =
        "pl.color.rgb * pl.color.w * diff * atten2 * sample_local_shadow(i, in.world_pos);";
    let shaded = source.replace(point_term, shadowed_point_term);
    debug_assert!(
        shaded.contains(shadowed_point_term),
        "VSM immediate variant must shade its point-light path"
    );
    format!(
        "{DIRECTIONAL_VSM_SCENE_BINDINGS}\n@group(1) @binding(8) var shadow_samp: sampler_comparison;\n{local_helper}\n{shaded}"
    )
}

const DIRECTIONAL_VSM_MATERIAL_BINDINGS: &str =
    include_str!("../shaders/virtual_shadows/material_bindings.wgsl");
const DIRECTIONAL_VSM_MATERIAL_HELPER: &str =
    include_str!("../shaders/virtual_shadows/material_helper.wgsl");

/// Add VSM sampling to ABI materials that include the engine shadow helper.
/// Materials that do not receive sun shadows only gain unused declarations
/// in this opt-in shader variant.
pub(crate) fn directional_material_shader(source: String) -> String {
    let mut output = String::with_capacity(
        source.len()
            + DIRECTIONAL_VSM_MATERIAL_BINDINGS.len()
            + DIRECTIONAL_VSM_MATERIAL_HELPER.len(),
    );
    output.push_str(DIRECTIONAL_VSM_MATERIAL_BINDINGS);
    if !source.contains("fn sample_shadow_cascade(") {
        output.push_str(&source);
        return output;
    }
    let renamed = source.replacen(
        "fn sample_shadow_cascade(",
        "fn sample_shadow_cascade_csm(",
        1,
    );
    let marker = "fn sample_sun_shadow(world_pos: vec3<f32>) -> f32 {";
    let offset = renamed
        .find(marker)
        .expect("material shadow helper missing sample_sun_shadow");
    output.push_str(&renamed[..offset]);
    output.push_str(DIRECTIONAL_VSM_MATERIAL_HELPER);
    output.push_str(&renamed[offset..]);
    output
}

fn env_u16(name: &str, default: u16, min: u16, max: u16) -> u16 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(default)
        .clamp(min, max)
}

fn demand_signature(demand: &[VirtualShadowPage]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for byte in (demand.len() as u64).to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    for page in demand {
        for byte in page.light.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        for byte in [page.level] {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        for byte in page.x.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        for byte in page.y.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

fn receiver_bounds_signature(receiver_bounds: &[([f32; 3], [f32; 3])]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for byte in (receiver_bounds.len() as u64).to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    for (bmin, bmax) in receiver_bounds {
        for value in bmin.iter().chain(bmax.iter()) {
            for byte in value.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
    }
    hash
}

fn projected_directional_page_rect(
    level_vp: &[[f32; 4]; 4],
    planes: &[[f32; 4]; 6],
    bmin: [f32; 3],
    bmax: [f32; 3],
    guard_pages: i32,
) -> Option<(u16, u16, u16, u16)> {
    if bmin[0] > bmax[0]
        || bmin
            .iter()
            .chain(bmax.iter())
            .any(|value| !value.is_finite())
        || crate::scene::aabb_outside_frustum(planes, bmin, bmax)
    {
        return None;
    }

    let mut ndc_min = [f32::INFINITY; 2];
    let mut ndc_max = [f32::NEG_INFINITY; 2];
    for corner in 0..8 {
        let world = [
            if corner & 1 == 0 { bmin[0] } else { bmax[0] },
            if corner & 2 == 0 { bmin[1] } else { bmax[1] },
            if corner & 4 == 0 { bmin[2] } else { bmax[2] },
            1.0,
        ];
        let clip = crate::renderer::mat4_mul_vec4(level_vp, &world);
        if !clip[3].is_finite() || clip[3].abs() <= f32::EPSILON {
            continue;
        }
        let x = clip[0] / clip[3];
        let y = clip[1] / clip[3];
        if x.is_finite() && y.is_finite() {
            ndc_min[0] = ndc_min[0].min(x);
            ndc_min[1] = ndc_min[1].min(y);
            ndc_max[0] = ndc_max[0].max(x);
            ndc_max[1] = ndc_max[1].max(y);
        }
    }
    if !ndc_min[0].is_finite() {
        return None;
    }

    let uv_min = [
        (ndc_min[0] * 0.5 + 0.5).clamp(0.0, 1.0),
        (1.0 - (ndc_max[1] * 0.5 + 0.5)).clamp(0.0, 1.0),
    ];
    let uv_max = [
        (ndc_max[0] * 0.5 + 0.5).clamp(0.0, 1.0),
        (1.0 - (ndc_min[1] * 0.5 + 0.5)).clamp(0.0, 1.0),
    ];
    let axis = i32::from(VSM_VIRTUAL_PAGES_PER_AXIS);
    let page_min_x = ((uv_min[0] * axis as f32).floor() as i32 - guard_pages).clamp(0, axis - 1);
    let page_min_y = ((uv_min[1] * axis as f32).floor() as i32 - guard_pages).clamp(0, axis - 1);
    let page_max_x = ((uv_max[0] * axis as f32).floor() as i32 + guard_pages).clamp(0, axis - 1);
    let page_max_y = ((uv_max[1] * axis as f32).floor() as i32 + guard_pages).clamp(0, axis - 1);
    Some((
        page_min_x as u16,
        page_min_y as u16,
        page_max_x as u16,
        page_max_y as u16,
    ))
}

fn force_dynamic_overlay_age(table: &mut [u32], pages: &[VirtualShadowPage]) {
    for &page in pages {
        if page.light == 0
            && page.level < VSM_CLIP_LEVELS
            && page.x < VSM_VIRTUAL_PAGES_PER_AXIS
            && page.y < VSM_VIRTUAL_PAGES_PER_AXIS
        {
            if let Some(entry) = table.get_mut(page.table_index()) {
                if *entry != VSM_PAGE_TABLE_MISSING {
                    *entry = (*entry & 0xffff) | (8 << 16);
                }
            }
        }
    }
}

/// Pages intersecting a dynamic caster, with a two-page PCF/jitter guard.
/// Unbounded casters return the full address space to request global CSM.
pub fn directional_dynamic_fallback_pages(
    level_vps: [[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize],
    dynamic_bounds: &[([f32; 3], [f32; 3])],
    light: u16,
) -> Vec<VirtualShadowPage> {
    if dynamic_bounds.is_empty() {
        return Vec::new();
    }
    let page_count = VSM_VIRTUAL_PAGES_PER_AXIS as usize
        * VSM_VIRTUAL_PAGES_PER_AXIS as usize
        * VSM_CLIP_LEVELS as usize;
    let unbounded = dynamic_bounds.iter().any(|(bmin, bmax)| {
        bmin[0] > bmax[0]
            || bmin
                .iter()
                .chain(bmax.iter())
                .any(|value| !value.is_finite())
    });
    let mut marked = vec![unbounded; page_count];
    let mut priority = vec![(u16::MAX, u16::MAX); page_count];
    if !unbounded {
        for level in 0..VSM_CLIP_LEVELS as usize {
            let planes = crate::scene::extract_frustum_planes(&level_vps[level]);
            for &(bmin, bmax) in dynamic_bounds {
                let Some((min_x, min_y, max_x, max_y)) =
                    projected_directional_page_rect(&level_vps[level], &planes, bmin, bmax, 2)
                else {
                    continue;
                };
                let center_x = i32::from(min_x + max_x);
                let center_y = i32::from(min_y + max_y);
                for y in min_y..=max_y {
                    for x in min_x..=max_x {
                        let page = VirtualShadowPage {
                            light,
                            level: level as u8,
                            x,
                            y,
                        };
                        let index = page.table_index();
                        let dx = (i32::from(x) * 2 - center_x).unsigned_abs() as u16;
                        let dy = (i32::from(y) * 2 - center_y).unsigned_abs() as u16;
                        priority[index] = priority[index].min((dx.max(dy), dx + dy));
                        marked[index] = true;
                    }
                }
            }
        }
    }

    let axis = VSM_VIRTUAL_PAGES_PER_AXIS as usize;
    let pages: Vec<_> = marked
        .into_iter()
        .enumerate()
        .filter_map(|(index, is_marked)| {
            is_marked.then(|| {
                let level = index / (axis * axis);
                let within_level = index % (axis * axis);
                VirtualShadowPage {
                    light,
                    level: level as u8,
                    x: (within_level % axis) as u16,
                    y: (within_level / axis) as u16,
                }
            })
        })
        .collect();
    page_priority::center_first(pages, &priority)
}

/// Deterministic center-first demand used by the directional prototype.
///
/// The returned footprint is intentionally bounded. Missing outer pages use
/// CSM, and later receiver-driven marking can replace this policy without
/// changing the cache or page-table ABI.
pub fn centered_directional_demand(light: u16) -> Vec<VirtualShadowPage> {
    const HALF_WIDTHS: [u16; VSM_CLIP_LEVELS as usize] = [6, 4, 2];
    let center = VSM_VIRTUAL_PAGES_PER_AXIS / 2;
    let mut per_level: [Vec<VirtualShadowPage>; VSM_CLIP_LEVELS as usize] =
        std::array::from_fn(|_| Vec::new());
    for (level, half_width) in HALF_WIDTHS.into_iter().enumerate() {
        let min = center - half_width;
        let max = center + half_width;
        for y in min..max {
            for x in min..max {
                per_level[level].push(
                    VirtualShadowPage::new(light, level as u8, x, y)
                        .expect("centered demand stays in the virtual address space"),
                );
            }
        }
        // Prioritize the pages closest to the clipmap center. Use doubled
        // page-center coordinates so the even-sized footprint has four
        // equally-near center pages without floating-point ordering.
        per_level[level].sort_by_key(|page| {
            let dx = (i32::from(page.x) * 2 + 1 - i32::from(center) * 2).unsigned_abs();
            let dy = (i32::from(page.y) * 2 + 1 - i32::from(center) * 2).unsigned_abs();
            (dx.max(dy), dx + dy, page.y, page.x)
        });
    }

    // Interleave levels so a frequently-invalidated near clipmap cannot
    // consume the whole per-frame render budget and starve mid/far coverage.
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
            break;
        }
    }
    pages
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform_ndc(matrix: &[[f32; 4]; 4], point: [f32; 4]) -> [f32; 3] {
        let clip = crate::renderer::mat4_mul_vec4(matrix, &point);
        [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]]
    }

    fn page(x: u16) -> VirtualShadowPage {
        VirtualShadowPage::new(0, 0, x, 0).unwrap()
    }

    fn rgb_at(rgb: &[u8], width: u32, x: u32, y: u32) -> [u8; 3] {
        let offset = ((y * width + x) * 3) as usize;
        rgb[offset..offset + 3].try_into().unwrap()
    }

    #[test]
    fn invalid_virtual_coordinates_are_rejected() {
        assert!(VirtualShadowPage::new(0, VSM_CLIP_LEVELS, 0, 0).is_none());
        assert!(VirtualShadowPage::new(0, 0, VSM_VIRTUAL_PAGES_PER_AXIS, 0).is_none());
    }

    #[test]
    fn reuse_is_stable_and_signature_changes_dirty_the_page() {
        let mut cache = VirtualShadowPageCache::new(2);
        cache.begin_frame(1);
        let first = cache.request(page(0), 7).unwrap();
        assert!(first.needs_render);
        assert!(cache.mark_rendered(page(0), 7));
        let hit = cache.request(page(0), 7).unwrap();
        assert_eq!(first.physical_page, hit.physical_page);
        assert!(!hit.needs_render);
        assert!(cache.request(page(0), 8).unwrap().needs_render);
    }

    #[test]
    fn current_frame_pages_are_never_evicted() {
        let mut cache = VirtualShadowPageCache::new(2);
        cache.begin_frame(1);
        cache.request(page(0), 1).unwrap();
        cache.request(page(1), 1).unwrap();
        assert!(cache.request(page(2), 1).is_none());
        assert_eq!(cache.stats().denied, 1);
    }

    #[test]
    fn stable_request_accounting_skips_cache_walk_without_hiding_fallbacks() {
        let mut cache = VirtualShadowPageCache::new(2);
        cache.begin_frame(1);
        cache.request(page(0), 1).unwrap();
        cache.request(page(1), 1).unwrap();
        cache.finish_requests();
        cache.begin_frame(2);
        cache.record_stable_requests(3);
        assert_eq!(cache.stats().requested, 3);
        assert_eq!(cache.stats().hits, 2);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().denied, 1);
    }

    #[test]
    fn debug_images_distinguish_misses_invalidations_levels_and_free_pages() {
        let mut cache = VirtualShadowPageCache::new(4);
        cache.begin_frame(1);
        for level in 0..VSM_CLIP_LEVELS {
            let page = VirtualShadowPage::new(0, level, 0, 0).unwrap();
            cache.request(page, 1).unwrap();
            if level > 0 {
                cache.mark_rendered(page, 1);
            }
        }
        cache.finish_requests();

        let (virtual_width, virtual_height, virtual_rgb) = cache.debug_virtual_rgb(0, 2);
        assert_eq!(virtual_width, u32::from(VSM_VIRTUAL_PAGES_PER_AXIS) * 2);
        assert_eq!(
            virtual_height,
            u32::from(VSM_VIRTUAL_PAGES_PER_AXIS) * u32::from(VSM_CLIP_LEVELS) * 2,
        );
        assert_eq!(
            virtual_rgb.len(),
            (virtual_width * virtual_height * 3) as usize
        );
        assert_eq!(
            rgb_at(&virtual_rgb, virtual_width, 0, 0),
            debug::MISS_UNRENDERED
        );
        assert_eq!(
            rgb_at(
                &virtual_rgb,
                virtual_width,
                0,
                u32::from(VSM_VIRTUAL_PAGES_PER_AXIS) * 2,
            ),
            debug::LEVELS[1],
        );
        assert_eq!(
            rgb_at(
                &virtual_rgb,
                virtual_width,
                0,
                u32::from(VSM_VIRTUAL_PAGES_PER_AXIS) * 4,
            ),
            debug::LEVELS[2],
        );

        let (physical_width, physical_height, physical_rgb) = cache.debug_physical_rgb(2);
        assert_eq!(
            physical_rgb.len(),
            (physical_width * physical_height * 3) as usize
        );
        assert_eq!(
            rgb_at(&physical_rgb, physical_width, 0, 0),
            debug::MISS_UNRENDERED
        );
        assert_eq!(
            rgb_at(&physical_rgb, physical_width, 2, 0),
            debug::LEVELS[1]
        );
        assert_eq!(
            rgb_at(&physical_rgb, physical_width, 4, 0),
            debug::LEVELS[2]
        );
        assert_eq!(rgb_at(&physical_rgb, physical_width, 6, 0), debug::FREE);

        cache.mark_rendered(page(0), 1);
        cache.begin_frame(2);
        assert!(cache.request(page(0), 2).unwrap().needs_render);
        let (virtual_width, _, virtual_rgb) = cache.debug_virtual_rgb(0, 1);
        assert_eq!(
            rgb_at(&virtual_rgb, virtual_width, 0, 0),
            debug::INVALIDATED
        );
        let (physical_width, _, physical_rgb) = cache.debug_physical_rgb(1);
        assert_eq!(
            rgb_at(&physical_rgb, physical_width, 0, 0),
            debug::INVALIDATED
        );
    }

    #[test]
    fn debug_legend_is_a_stable_machine_readable_palette() {
        let (width, height, rgb) = VirtualShadowPageCache::debug_legend_rgb(2);
        assert_eq!((width, height), (12, 2));
        let expected = [
            debug::FREE,
            debug::MISS_UNRENDERED,
            debug::INVALIDATED,
            debug::LEVELS[0],
            debug::LEVELS[1],
            debug::LEVELS[2],
        ];
        for (index, color) in expected.into_iter().enumerate() {
            assert_eq!(rgb_at(&rgb, width, index as u32 * 2, 0), color);
        }
    }

    #[test]
    fn lru_eviction_is_deterministic() {
        let mut cache = VirtualShadowPageCache::new(2);
        cache.begin_frame(1);
        cache.request(page(0), 1).unwrap();
        cache.request(page(1), 1).unwrap();
        cache.begin_frame(2);
        cache.request(page(1), 1).unwrap();
        let request = cache.request(page(2), 1).unwrap();
        assert_eq!(request.evicted, Some(page(0)));
        assert_eq!(request.physical_page, 0);
    }

    #[test]
    fn clipmap_scroll_preserves_overlap_and_drops_only_the_boundary() {
        let mut cache = VirtualShadowPageCache::new(3);
        cache.begin_frame(1);
        for x in [0, 1, VSM_VIRTUAL_PAGES_PER_AXIS - 1] {
            let page = page(x);
            cache.request(page, 7).unwrap();
            cache.mark_rendered(page, 7);
        }
        cache.finish_requests();

        cache.begin_frame(2);
        cache.scroll_level(0, 0, [-1, 0]);
        let table = cache.page_table(0);
        assert_ne!(table[page(0).table_index()], VSM_PAGE_TABLE_MISSING);
        assert_ne!(
            table[page(VSM_VIRTUAL_PAGES_PER_AXIS - 2).table_index()],
            VSM_PAGE_TABLE_MISSING,
        );
        assert_eq!(
            table[page(VSM_VIRTUAL_PAGES_PER_AXIS - 1).table_index()],
            VSM_PAGE_TABLE_MISSING,
        );
        assert_eq!(cache.stats().resident, 2);
        assert_eq!(cache.stats().dirty, 0);
        assert_eq!(cache.stats().clipmap_level_rebases, 1);
        assert_eq!(cache.stats().clipmap_pages_preserved, 2);
        assert_eq!(cache.stats().clipmap_pages_dropped, 1);
    }

    #[test]
    fn dirty_pages_are_missing_until_rendered_and_then_age() {
        let mut cache = VirtualShadowPageCache::new(1);
        cache.begin_frame(1);
        cache.request(page(0), 42).unwrap();
        assert_eq!(cache.page_table(0)[page(0).table_index()], 0);
        cache.mark_rendered(page(0), 42);
        let first = cache.page_table(0)[page(0).table_index()];
        assert_eq!(first & 0xffff, 1);
        assert_eq!(first >> 16, 1);
        cache.begin_frame(4);
        let aged = cache.page_table(0)[page(0).table_index()];
        assert_eq!(aged >> 16, 4);
        cache.begin_frame(100);
        let saturated = cache.page_table(0)[page(0).table_index()];
        assert_eq!(saturated >> 16, 8);
        cache.invalidate_light(0);
        assert_eq!(cache.page_table(0)[page(0).table_index()], 0);
    }

    #[test]
    fn configured_capacity_is_a_hard_memory_bound() {
        let cache = VirtualShadowPageCache::new(8);
        let edge = VSM_PHYSICAL_PAGE_SIZE as u64;
        assert_eq!(cache.memory_bytes(), edge * edge * 4 * 8);
        assert_eq!(cache.stats().capacity, 8);
    }

    #[test]
    fn centered_demand_fits_default_pool() {
        let demand = centered_directional_demand(0);
        assert_eq!(demand.len(), 224);
        assert!(demand.len() <= VSM_DEFAULT_PHYSICAL_PAGES as usize);
        let mut sorted = demand.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), demand.len());
        assert_eq!(
            demand[..3]
                .iter()
                .map(|page| page.level)
                .collect::<Vec<_>>(),
            vec![0, 1, 2],
        );
        assert!(demand[..3]
            .iter()
            .all(|page| (15..=16).contains(&page.x) && (15..=16).contains(&page.y)));
    }

    #[test]
    fn receiver_demand_is_bounded_unique_and_fair_across_levels() {
        let demand = directional_receiver_demand(
            [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize],
            &[([-1.0; 3], [1.0; 3])],
            7,
        );
        assert_eq!(
            demand.len(),
            VSM_DIRECTIONAL_LEVEL_PAGE_CAPS
                .iter()
                .copied()
                .sum::<usize>()
        );
        let mut sorted = demand.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), demand.len());
        assert_eq!(
            demand[..3]
                .iter()
                .map(|page| page.level)
                .collect::<Vec<_>>(),
            vec![0, 1, 2],
        );
        assert!(demand.iter().all(|page| page.light == 7));
    }

    #[test]
    fn receiver_demand_marks_local_footprint_with_guard_pages() {
        let demand = directional_receiver_demand(
            [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize],
            &[([-0.02, -0.02, 0.4], [0.02, 0.02, 0.6])],
            0,
        );
        assert_eq!(demand.iter().filter(|page| page.level == 0).count(), 16);
        assert!(demand
            .iter()
            .all(|page| { (14..=17).contains(&page.x) && (14..=17).contains(&page.y) }));
    }

    #[test]
    fn receiver_demand_rejects_bounds_outside_light_volume() {
        let demand = directional_receiver_demand(
            [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize],
            &[([2.0, 2.0, 2.0], [3.0, 3.0, 3.0])],
            0,
        );
        assert!(demand.is_empty());
    }

    #[test]
    fn dynamic_overlays_cover_only_guarded_caster_pages() {
        let pages = directional_dynamic_fallback_pages(
            [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize],
            &[([-0.02, -0.02, 0.4], [0.02, 0.02, 0.6])],
            0,
        );
        assert_eq!(pages.len(), 36 * VSM_CLIP_LEVELS as usize);
        assert!(pages
            .iter()
            .all(|page| { (13..=18).contains(&page.x) && (13..=18).contains(&page.y) }));
        assert!(pages[..4].iter().all(|page| page.level == 0
            && (15..=16).contains(&page.x)
            && (15..=16).contains(&page.y)));

        let mut table = vec![
            99;
            VSM_VIRTUAL_PAGES_PER_AXIS as usize
                * VSM_VIRTUAL_PAGES_PER_AXIS as usize
                * VSM_CLIP_LEVELS as usize
        ];
        force_dynamic_overlay_age(&mut table, &pages);
        assert_eq!(
            table
                .iter()
                .filter(|entry| **entry == 99 | (8 << 16))
                .count(),
            pages.len()
        );
        assert_eq!(
            table.iter().filter(|entry| **entry == 99).count(),
            table.len() - pages.len()
        );
    }

    #[test]
    fn targeted_invalidation_never_exposes_stale_dynamic_depth() {
        let mut cache = VirtualShadowPageCache::new(2);
        cache.begin_frame(1);
        for x in 0..2 {
            cache.request(page(x), 7).unwrap();
            cache.mark_rendered(page(x), 7);
        }
        cache.invalidate_pages(&[page(1)]);
        let table = cache.page_table(0);
        assert_ne!(table[page(0).table_index()], VSM_PAGE_TABLE_MISSING);
        assert_eq!(table[page(1).table_index()], VSM_PAGE_TABLE_MISSING);
        assert_eq!(cache.stats().dirty, 1);
    }

    #[test]
    fn unbounded_dynamic_caster_preserves_whole_frame_fallback() {
        let pages = directional_dynamic_fallback_pages(
            [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize],
            &[([1.0, 1.0, 1.0], [-1.0, -1.0, -1.0])],
            0,
        );
        assert_eq!(
            pages.len(),
            VSM_VIRTUAL_PAGES_PER_AXIS as usize
                * VSM_VIRTUAL_PAGES_PER_AXIS as usize
                * VSM_CLIP_LEVELS as usize
        );
    }

    #[test]
    fn offscreen_dynamic_caster_does_not_mask_resident_pages() {
        let pages = directional_dynamic_fallback_pages(
            [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize],
            &[([2.0, 2.0, 2.0], [3.0, 3.0, 3.0])],
            0,
        );
        assert!(pages.is_empty());
    }

    #[test]
    fn receiver_demand_and_signature_are_deterministic() {
        let bounds = [
            ([-0.8, -0.4, 0.2], [-0.2, 0.1, 0.8]),
            ([0.1, -0.2, 0.1], [0.7, 0.6, 0.9]),
        ];
        let vps = [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize];
        let first = directional_receiver_demand(vps, &bounds, 0);
        let second = directional_receiver_demand(vps, &bounds, 0);
        assert_eq!(first, second);
        assert_eq!(demand_signature(&first), demand_signature(&second));
        assert_ne!(
            demand_signature(&first),
            demand_signature(&centered_directional_demand(0))
        );
    }

    #[test]
    fn coordinator_is_inert_without_explicit_request() {
        if virtual_shadows_requested() {
            return;
        }
        // Runtime construction requires a device; the non-GPU cache already
        // proves inert behavior above. Keep the environment contract here.
        let cache = VirtualShadowPageCache::new(1);
        assert_eq!(cache.stats().resident, 0);
        assert_eq!(
            cache
                .page_table(0)
                .iter()
                .filter(|entry| **entry != 0)
                .count(),
            0
        );
    }

    #[test]
    fn page_crop_maps_interior_edges_to_guard_texels() {
        let page = VirtualShadowPage::new(0, 0, 9, 12).unwrap();
        let crop = directional_page_vp(crate::renderer::IDENTITY_MAT4, page);
        let axis = VSM_VIRTUAL_PAGES_PER_AXIS as f32;
        let left = f32::from(page.x) * (2.0 / axis) - 1.0;
        let right = f32::from(page.x + 1) * (2.0 / axis) - 1.0;
        let top = 1.0 - f32::from(page.y) * (2.0 / axis);
        let bottom = 1.0 - f32::from(page.y + 1) * (2.0 / axis);
        let expected = VSM_PAGE_INTERIOR as f32 / VSM_PHYSICAL_PAGE_SIZE as f32;

        let left_ndc = transform_ndc(&crop, [left, 0.5 * (top + bottom), 0.5, 1.0]);
        let right_ndc = transform_ndc(&crop, [right, 0.5 * (top + bottom), 0.5, 1.0]);
        let top_ndc = transform_ndc(&crop, [0.5 * (left + right), top, 0.5, 1.0]);
        let bottom_ndc = transform_ndc(&crop, [0.5 * (left + right), bottom, 0.5, 1.0]);

        assert!((left_ndc[0] + expected).abs() < 1.0e-5);
        assert!((right_ndc[0] - expected).abs() < 1.0e-5);
        assert!((top_ndc[1] - expected).abs() < 1.0e-5);
        assert!((bottom_ndc[1] + expected).abs() < 1.0e-5);
    }

    #[test]
    fn page_crop_preserves_depth() {
        let page = VirtualShadowPage::new(0, 2, 16, 16).unwrap();
        let crop = directional_page_vp(crate::renderer::IDENTITY_MAT4, page);
        let transformed = transform_ndc(&crop, [0.03125, -0.03125, 0.37, 1.0]);
        assert!((transformed[2] - 0.37).abs() < 1.0e-6);
    }

    #[test]
    fn scene_shader_variant_injects_bindings_and_both_cascade_samples() {
        let source = r#"
let shadow_val = sample_cascade(cascade, shadow_uv, depth_ref);
fn sample_shadow(world_pos: vec3<f32>, geo_n: vec3<f32>) -> f32 {
let next_val = sample_cascade(next_cascade, next_uv, next_depth_ref);
}
"#;
        let variant = directional_scene_shader(source);
        assert!(variant.contains("@binding(13) var vsm_page_table"));
        assert!(variant.contains("@binding(14) var vsm_physical_pages"));
        assert!(variant.contains("@binding(15) var<uniform> vsm_params"));
        assert_eq!(variant.matches("sample_virtual_shadow(").count(), 3);
        assert!(variant.contains("sample_virtual_shadow(cascade, recv_pos,"));
        assert!(variant.contains("sample_virtual_shadow(next_cascade, next_pos,"));
        assert!(variant.contains("level_vps: array<mat4x4<f32>, 3>"));
        assert!(!source.contains("vsm_page_table"));
    }

    #[test]
    fn material_shader_variant_wraps_the_canonical_cascade_sampler() {
        let source = r#"
fn sample_shadow_cascade(
  cascade_idx: u32, world_pos: vec3<f32>,
) -> f32 {
  return 1.0;
}
fn sample_sun_shadow(world_pos: vec3<f32>) -> f32 {
  return sample_shadow_cascade(0u, world_pos);
}
"#;
        let variant = directional_material_shader(source.to_owned());
        assert!(variant.contains("@binding(10) var vsm_page_table"));
        assert!(variant.contains("fn sample_shadow_cascade_csm("));
        assert_eq!(variant.matches("fn sample_shadow_cascade(").count(), 1);
        assert!(variant.contains("sample_shadow_cascade_csm(cascade_idx, world_pos)"));
        assert!(variant.contains("level_vps: array<mat4x4<f32>, 3>"));
    }

    #[test]
    fn sampling_uniform_matches_wgsl_layout() {
        assert_eq!(
            VSM_SAMPLING_PARAMS_BYTES,
            3 * 64
                + 16
                + VSM_MAX_LOCAL_SHADOW_REQUESTS as u64 * 16
                + VSM_MAX_LOCAL_SHADOW_LIGHTS as u64 * (VSM_LOCAL_FACES as u64 * 64 + 32)
        );
        assert_eq!(std::mem::align_of::<DirectionalVsmSamplingParams>(), 16);
    }

    #[test]
    fn immediate_shader_variant_injects_local_shadow_sampling() {
        let source = r#"
struct PointLight { position: vec4<f32>, color: vec4<f32> };
struct Lighting { point_lights: array<PointLight, 256> };
struct VertexOutput3D { world_pos: vec3<f32> };
@group(1) @binding(0) var<uniform> lighting: Lighting;
fn local_lighting(in: VertexOutput3D, i: u32, diff: f32, atten2: f32) -> vec3<f32> {
    let pl = lighting.point_lights[i];
    return pl.color.rgb * pl.color.w * diff * atten2;
}
"#;
        let variant = local_immediate_shader(source);
        assert!(variant.contains("@binding(14) var vsm_physical_pages"));
        assert!(variant.contains("@binding(8) var shadow_samp"));
        assert!(variant.contains("diff * atten2 * sample_local_shadow(i, in.world_pos)"));
        let result = wgpu::naga::front::wgsl::parse_str(&variant);
        if let Err(error) = result.as_ref() {
            panic!(
                "VSM immediate variant failed WGSL parsing:\n{}",
                error.emit_to_string(&variant),
            );
        }
    }

    #[test]
    fn material_shadow_variant_parses_through_naga() {
        let source = format!(
            "{}\n{}",
            include_str!("../shaders/material_abi.wgsl"),
            include_str!("../shaders/common/shadows.wgsl"),
        );
        let variant = directional_material_shader(source);
        let result = wgpu::naga::front::wgsl::parse_str(&variant);
        if let Err(error) = result.as_ref() {
            panic!(
                "VSM material shadow variant failed WGSL parsing:\n{}",
                error.emit_to_string(&variant),
            );
        }
    }
}
