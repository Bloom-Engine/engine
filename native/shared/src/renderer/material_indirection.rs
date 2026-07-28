//! Capability-tiered material and texture indirection.
//!
//! This module is deliberately independent from the legacy material bind-group
//! path. Tier C can therefore continue to produce the exact same commands and
//! pixels while Tier A/B resource tables are populated for GPU-driven draws.
//! The future visibility-buffer and GPU-culling passes consume the typed IDs
//! exposed here rather than owning backend bind groups.

use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::capabilities::{forced_renderer_tier, RendererCapabilities, RendererCapabilityTier};
use super::layered_pbr::{
    global_material_lobe_mask, global_material_version, pack_global_material_metadata,
    MaterialLobeMask,
};

const ID_SLOT_BITS: u32 = 20;
const ID_SLOT_MASK: u32 = (1 << ID_SLOT_BITS) - 1;
const ID_GENERATION_MASK: u32 = (1 << (32 - ID_SLOT_BITS)) - 1;
const DEFAULT_RESOURCE_CAPACITY: usize = 8_192;
const TIER_A_TARGET_TEXTURES: u32 = 4_096;
const TIER_A_TARGET_SAMPLERS: u32 = 64;
const INITIAL_MATERIAL_CAPACITY: usize = 64;

pub const TIER_A_FEATURES: wgpu::Features = wgpu::Features::TEXTURE_BINDING_ARRAY
    .union(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);

/// Add Tier A's optional features and exact adapter limits to a device request.
///
/// Platform bring-up paths call this after choosing their mandatory limits.
/// Unsupported adapters are unchanged and naturally select Tier B/C later.
pub fn request_tier_a_if_supported(
    supported: wgpu::Features,
    adapter_limits: &wgpu::Limits,
    required_features: &mut wgpu::Features,
    required_limits: &mut wgpu::Limits,
) {
    if !supported.contains(TIER_A_FEATURES) {
        return;
    }
    *required_features |= TIER_A_FEATURES;
    required_limits.max_binding_array_elements_per_shader_stage =
        adapter_limits.max_binding_array_elements_per_shader_stage;
    required_limits.max_binding_array_sampler_elements_per_shader_stage =
        adapter_limits.max_binding_array_sampler_elements_per_shader_stage;
}

/// Common behavior for the typed 32-bit IDs stored in scene and GPU records.
pub trait StableResourceId:
    Copy + Clone + Eq + Ord + std::hash::Hash + std::fmt::Debug + Send + Sync + 'static
{
    const FALLBACK: Self;
    fn from_parts(slot: usize, generation: u32) -> Option<Self>;
    fn raw(self) -> u32;

    fn descriptor_index(self) -> usize {
        (self.raw() & ID_SLOT_MASK) as usize
    }

    fn generation(self) -> u32 {
        self.raw() >> ID_SLOT_BITS
    }

    fn is_fallback(self) -> bool {
        self.raw() == 0
    }
}

macro_rules! stable_id {
    ($name:ident) => {
        #[repr(transparent)]
        #[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name(u32);

        impl $name {
            pub const FALLBACK: Self = Self(0);

            pub const fn raw(self) -> u32 {
                self.0
            }

            pub const fn descriptor_index(self) -> u32 {
                self.0 & ID_SLOT_MASK
            }

            pub const fn generation(self) -> u32 {
                self.0 >> ID_SLOT_BITS
            }
        }

        impl StableResourceId for $name {
            const FALLBACK: Self = Self::FALLBACK;

            fn from_parts(slot: usize, generation: u32) -> Option<Self> {
                let one_based = slot.checked_add(1)?;
                if one_based > ID_SLOT_MASK as usize || generation > ID_GENERATION_MASK {
                    return None;
                }
                Some(Self((generation << ID_SLOT_BITS) | one_based as u32))
            }

            fn raw(self) -> u32 {
                self.0
            }
        }
    };
}

stable_id!(MaterialId);
stable_id!(TextureId);
stable_id!(SamplerId);
stable_id!(MeshId);
stable_id!(BufferViewId);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResolveStatus {
    Resident,
    Fallback,
    Stale,
    Retiring,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResidencyUpdate<I> {
    Resident(I),
    Retired(I),
    Reclaimed(I),
}

struct ResidencySlot<T> {
    generation: u32,
    live: bool,
    retire_epoch: Option<u64>,
    value: Option<T>,
}

/// Generation-safe resource storage with GPU-completion-delayed reclamation.
///
/// Retirement makes an ID resolve to the diagnostic fallback immediately, but
/// the owned resource is retained until `collect(completed_epoch)` proves the
/// last referencing submission finished. Only then is the slot reusable and
/// its generation incremented.
pub struct ResidencyTable<I: StableResourceId, T> {
    slots: Vec<ResidencySlot<T>>,
    free: Vec<usize>,
    updates: Vec<ResidencyUpdate<I>>,
    max_live: usize,
    live_count: usize,
    _id: PhantomData<I>,
}

impl<I: StableResourceId, T> ResidencyTable<I, T> {
    pub fn new(max_live: usize) -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            updates: Vec::new(),
            max_live: max_live.min(ID_SLOT_MASK as usize),
            live_count: 0,
            _id: PhantomData,
        }
    }

    pub fn insert(&mut self, value: T) -> Result<I, &'static str> {
        if self.live_count >= self.max_live {
            return Err("residency table limit reached");
        }
        let slot = if let Some(slot) = self.free.pop() {
            self.slots[slot].value = Some(value);
            self.slots[slot].live = true;
            self.slots[slot].retire_epoch = None;
            slot
        } else {
            if self.slots.len() >= self.max_live {
                return Err("residency table limit reached");
            }
            let slot = self.slots.len();
            self.slots.push(ResidencySlot {
                generation: 0,
                live: true,
                retire_epoch: None,
                value: Some(value),
            });
            slot
        };
        self.live_count += 1;
        let id = I::from_parts(slot, self.slots[slot].generation)
            .ok_or("resource ID space exhausted")?;
        self.updates.push(ResidencyUpdate::Resident(id));
        Ok(id)
    }

    pub fn resolve(&self, id: I) -> (Option<&T>, ResolveStatus) {
        if id.is_fallback() {
            return (None, ResolveStatus::Fallback);
        }
        let descriptor_index = id.descriptor_index();
        let Some(slot_index) = descriptor_index.checked_sub(1) else {
            return (None, ResolveStatus::Fallback);
        };
        let Some(slot) = self.slots.get(slot_index) else {
            return (None, ResolveStatus::Stale);
        };
        if slot.generation != id.generation() {
            return (None, ResolveStatus::Stale);
        }
        if !slot.live {
            return (None, ResolveStatus::Retiring);
        }
        match slot.value.as_ref() {
            Some(value) => (Some(value), ResolveStatus::Resident),
            None => (None, ResolveStatus::Stale),
        }
    }

    pub fn resolve_mut(&mut self, id: I) -> (Option<&mut T>, ResolveStatus) {
        if id.is_fallback() {
            return (None, ResolveStatus::Fallback);
        }
        let Some(slot_index) = id.descriptor_index().checked_sub(1) else {
            return (None, ResolveStatus::Fallback);
        };
        let Some(slot) = self.slots.get_mut(slot_index) else {
            return (None, ResolveStatus::Stale);
        };
        if slot.generation != id.generation() {
            return (None, ResolveStatus::Stale);
        }
        if !slot.live {
            return (None, ResolveStatus::Retiring);
        }
        match slot.value.as_mut() {
            Some(value) => (Some(value), ResolveStatus::Resident),
            None => (None, ResolveStatus::Stale),
        }
    }

    pub fn retire(&mut self, id: I, completion_epoch: u64) -> bool {
        if id.is_fallback() {
            return false;
        }
        let Some(slot_index) = id.descriptor_index().checked_sub(1) else {
            return false;
        };
        let Some(slot) = self.slots.get_mut(slot_index) else {
            return false;
        };
        if slot.generation != id.generation() || !slot.live || slot.value.is_none() {
            return false;
        }
        slot.live = false;
        slot.retire_epoch = Some(completion_epoch);
        self.live_count = self.live_count.saturating_sub(1);
        self.updates.push(ResidencyUpdate::Retired(id));
        true
    }

    pub fn collect(&mut self, completed_epoch: u64) -> usize {
        let mut reclaimed = 0;
        for (slot_index, slot) in self.slots.iter_mut().enumerate() {
            if slot.live
                || slot.value.is_none()
                || slot
                    .retire_epoch
                    .is_none_or(|epoch| epoch > completed_epoch)
            {
                continue;
            }
            let old_id = I::from_parts(slot_index, slot.generation)
                .expect("existing residency slot must have an encodable ID");
            slot.value.take();
            slot.retire_epoch = None;
            slot.generation = (slot.generation + 1) & ID_GENERATION_MASK;
            self.free.push(slot_index);
            self.updates.push(ResidencyUpdate::Reclaimed(old_id));
            reclaimed += 1;
        }
        reclaimed
    }

    pub fn drain_updates(&mut self) -> Vec<ResidencyUpdate<I>> {
        std::mem::take(&mut self.updates)
    }

    pub fn live_count(&self) -> usize {
        self.live_count
    }

    pub fn max_live(&self) -> usize {
        self.max_live
    }

    fn slots(&self) -> &[ResidencySlot<T>] {
        &self.slots
    }
}

/// Tracks resource retirement against actual queue-completion callbacks.
#[derive(Clone)]
pub struct GpuCompletionTracker {
    next_epoch: Arc<AtomicU64>,
    completed_epoch: Arc<AtomicU64>,
}

impl Default for GpuCompletionTracker {
    fn default() -> Self {
        Self {
            next_epoch: Arc::new(AtomicU64::new(0)),
            completed_epoch: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl GpuCompletionTracker {
    /// Return an epoch that completes after all work submitted before this call.
    pub fn track_submitted_work(&self, queue: &wgpu::Queue) -> u64 {
        let epoch = self.next_epoch.fetch_add(1, Ordering::Relaxed) + 1;
        let completed = Arc::clone(&self.completed_epoch);
        queue.on_submitted_work_done(move || {
            completed.fetch_max(epoch, Ordering::Release);
        });
        epoch
    }

    pub fn completed_epoch(&self) -> u64 {
        self.completed_epoch.load(Ordering::Acquire)
    }

    #[cfg(test)]
    fn mark_complete_for_test(&self, epoch: u64) {
        self.completed_epoch.fetch_max(epoch, Ordering::Release);
    }
}

/// Runtime-selected backend for global material and texture lookup.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum MaterialBindingTier {
    /// Existing per-material bind groups. Compatibility and oracle path.
    C = 1,
    /// Paged texture arrays/atlases with deterministic page grouping.
    B = 2,
    /// Descriptor-indexed global texture and sampler tables.
    A = 3,
}

impl MaterialBindingTier {
    pub const fn name(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
        }
    }

    pub fn from_override(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "a" | "tier-a" | "3" => Some(Self::A),
            "b" | "tier-b" | "2" => Some(Self::B),
            "c" | "tier-c" | "1" => Some(Self::C),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MaterialBindingCapabilities {
    pub detected_tier: MaterialBindingTier,
    pub selected_tier: MaterialBindingTier,
    pub override_tier: Option<MaterialBindingTier>,
    pub texture_binding_array: bool,
    pub non_uniform_indexing: bool,
    pub max_binding_array_elements: u32,
    pub max_binding_array_samplers: u32,
    pub max_texture_array_layers: u32,
    pub max_sampled_textures: u32,
    pub max_samplers: u32,
    pub max_material_records: u32,
    pub tier_a_texture_capacity: u32,
    pub tier_a_sampler_capacity: u32,
    pub tier_b_page_capacity: u32,
    pub diagnostic: Option<String>,
}

impl MaterialBindingCapabilities {
    pub fn detect(features: wgpu::Features, limits: &wgpu::Limits) -> Self {
        let detected_tier = detected_material_tier(features, limits);
        let material_override = std::env::var("BLOOM_MATERIAL_TIER")
            .ok()
            .and_then(|value| MaterialBindingTier::from_override(&value));
        let requested_override =
            lower_material_override(material_override, forced_renderer_material_tier());
        Self::with_override(features, limits, detected_tier, requested_override)
    }

    pub fn detect_with_override(
        features: wgpu::Features,
        limits: &wgpu::Limits,
        override_tier: Option<MaterialBindingTier>,
    ) -> Self {
        let detected_tier = detected_material_tier(features, limits);
        let requested_override =
            lower_material_override(override_tier, forced_renderer_material_tier());
        Self::with_override(features, limits, detected_tier, requested_override)
    }

    fn with_override(
        features: wgpu::Features,
        limits: &wgpu::Limits,
        detected_tier: MaterialBindingTier,
        requested_override: Option<MaterialBindingTier>,
    ) -> Self {
        let (selected_tier, override_tier, diagnostic) = match requested_override {
            Some(requested) if requested <= detected_tier => (requested, Some(requested), None),
            Some(requested) => (
                detected_tier,
                None,
                Some(format!(
                    "requested Tier {} exceeds adapter Tier {}; using Tier {}",
                    requested.name(),
                    detected_tier.name(),
                    detected_tier.name()
                )),
            ),
            None => (detected_tier, None, None),
        };
        let record_size = std::mem::size_of::<GpuMaterialRecord>() as u64;
        let max_material_records = (limits.max_storage_buffer_binding_size / record_size)
            .min(ID_SLOT_MASK as u64)
            .max(1) as u32;
        let tier_a_texture_capacity = limits
            .max_binding_array_elements_per_shader_stage
            .saturating_sub(1)
            .min(TIER_A_TARGET_TEXTURES);
        let tier_a_sampler_capacity = limits
            .max_binding_array_sampler_elements_per_shader_stage
            .saturating_sub(1)
            .min(TIER_A_TARGET_SAMPLERS);
        Self {
            detected_tier,
            selected_tier,
            override_tier,
            texture_binding_array: features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY),
            non_uniform_indexing: features.contains(
                wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            ),
            max_binding_array_elements: limits.max_binding_array_elements_per_shader_stage,
            max_binding_array_samplers: limits.max_binding_array_sampler_elements_per_shader_stage,
            max_texture_array_layers: limits.max_texture_array_layers,
            max_sampled_textures: limits.max_sampled_textures_per_shader_stage,
            max_samplers: limits.max_samplers_per_shader_stage,
            max_material_records,
            tier_a_texture_capacity,
            tier_a_sampler_capacity,
            tier_b_page_capacity: limits.max_texture_array_layers.clamp(1, 256),
            diagnostic,
        }
    }

    pub fn report_json(&self) -> String {
        let mut out = String::from("{\"version\":1,\"detected_tier\":\"");
        out.push_str(self.detected_tier.name());
        out.push_str("\",\"selected_tier\":\"");
        out.push_str(self.selected_tier.name());
        out.push_str("\",\"override_tier\":");
        match self.override_tier {
            Some(tier) => {
                out.push('"');
                out.push_str(tier.name());
                out.push('"');
            }
            None => out.push_str("null"),
        }
        out.push_str(",\"features\":{\"texture_binding_array\":");
        out.push_str(if self.texture_binding_array {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"non_uniform_indexing\":");
        out.push_str(if self.non_uniform_indexing {
            "true"
        } else {
            "false"
        });
        out.push_str("},\"limits\":{\"max_binding_array_elements\":");
        out.push_str(&self.max_binding_array_elements.to_string());
        out.push_str(",\"max_binding_array_samplers\":");
        out.push_str(&self.max_binding_array_samplers.to_string());
        out.push_str(",\"max_texture_array_layers\":");
        out.push_str(&self.max_texture_array_layers.to_string());
        out.push_str(",\"max_sampled_textures\":");
        out.push_str(&self.max_sampled_textures.to_string());
        out.push_str(",\"max_samplers\":");
        out.push_str(&self.max_samplers.to_string());
        out.push_str(",\"max_material_records\":");
        out.push_str(&self.max_material_records.to_string());
        out.push_str("},\"capacities\":{\"tier_a_textures\":");
        out.push_str(&self.tier_a_texture_capacity.to_string());
        out.push_str(",\"tier_a_samplers\":");
        out.push_str(&self.tier_a_sampler_capacity.to_string());
        out.push_str(",\"tier_b_page_layers\":");
        out.push_str(&self.tier_b_page_capacity.to_string());
        out.push_str("},\"diagnostic\":");
        match &self.diagnostic {
            Some(message) => json_string(&mut out, message),
            None => out.push_str("null"),
        }
        out.push('}');
        out
    }
}

fn detected_material_tier(features: wgpu::Features, limits: &wgpu::Limits) -> MaterialBindingTier {
    match RendererCapabilities::detect_with_override(features, limits, None).detected_tier {
        RendererCapabilityTier::Baseline => MaterialBindingTier::C,
        RendererCapabilityTier::Modern => MaterialBindingTier::B,
        RendererCapabilityTier::HighEnd => MaterialBindingTier::A,
    }
}

fn forced_renderer_material_tier() -> Option<MaterialBindingTier> {
    forced_renderer_tier().map(|tier| match tier {
        RendererCapabilityTier::Baseline => MaterialBindingTier::C,
        RendererCapabilityTier::Modern => MaterialBindingTier::B,
        RendererCapabilityTier::HighEnd => MaterialBindingTier::A,
    })
}

fn lower_material_override(
    material: Option<MaterialBindingTier>,
    renderer: Option<MaterialBindingTier>,
) -> Option<MaterialBindingTier> {
    match (material, renderer) {
        (Some(material), Some(renderer)) => Some(material.min(renderer)),
        (material, renderer) => material.or(renderer),
    }
}

/// Storage-buffer record shared by every material tier.
///
/// Texture/sampler fields contain typed-ID raw values. The shader validates a
/// material generation before using the record and falls back to record zero
/// on stale/non-resident IDs.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterialRecord {
    /// x=generation, y=layered-PBR version/mask, z=user-param byte offset,
    /// w=user-param size. `header.y` packs version in its high 8 bits and the
    /// lobe mask in its low 24 bits.
    pub header: [u32; 4],
    pub base_color: [f32; 4],
    pub metal_rough: [f32; 4],
    pub emissive: [f32; 4],
    pub shading_model: [f32; 4],
    pub foliage_params: [f32; 4],
    /// base-color, normal, metallic-roughness, emissive.
    pub texture_ids_0: [u32; 4],
    /// occlusion, planar reflection, albedo array/page, normal array/page.
    pub texture_ids_1: [u32; 4],
    /// MR array/page, reserved.
    pub texture_ids_2: [u32; 4],
    /// base, normal, MR, emissive sampler IDs.
    pub sampler_ids_0: [u32; 4],
    /// occlusion, reflection, array/page, reserved sampler IDs.
    pub sampler_ids_1: [u32; 4],
}

impl Default for GpuMaterialRecord {
    fn default() -> Self {
        Self {
            header: [
                0,
                pack_global_material_metadata(MaterialLobeMask::NONE),
                0,
                0,
            ],
            base_color: [1.0, 1.0, 1.0, 1.0],
            metal_rough: [0.0, 1.0, 0.0, 0.0],
            emissive: [0.0; 4],
            shading_model: [0.0, 1.0, 1.0, 1.0],
            foliage_params: [0.5, 0.5, 0.0, 0.0],
            texture_ids_0: [0; 4],
            texture_ids_1: [0; 4],
            texture_ids_2: [0; 4],
            sampler_ids_0: [0; 4],
            sampler_ids_1: [0; 4],
        }
    }
}

impl GpuMaterialRecord {
    pub(crate) fn layered_pbr_version(&self) -> u32 {
        global_material_version(self.header[1])
    }

    pub(crate) fn layered_pbr_lobe_mask(&self) -> MaterialLobeMask {
        global_material_lobe_mask(self.header[1])
    }

    fn normalize_layered_pbr_metadata(&mut self) {
        let mask = if self.layered_pbr_version()
            == crate::renderer::layered_pbr::MATERIAL_RECORD_VERSION
        {
            self.layered_pbr_lobe_mask()
        } else {
            // Version zero is the pre-layered record. Its old flags lane was
            // unused, so no bit may silently acquire a lobe meaning.
            MaterialLobeMask::NONE
        };
        self.header[1] = pack_global_material_metadata(mask);
    }
}

struct GpuMaterialTable {
    records: ResidencyTable<MaterialId, GpuMaterialRecord>,
    fallback: GpuMaterialRecord,
    buffer: wgpu::Buffer,
    buffer_capacity: usize,
    dirty: bool,
    buffer_recreated: bool,
}

impl GpuMaterialTable {
    fn new(device: &wgpu::Device, max_records: usize) -> Self {
        let capacity = INITIAL_MATERIAL_CAPACITY.min(max_records.max(1));
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global_material_records"),
            size: (capacity * std::mem::size_of::<GpuMaterialRecord>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            records: ResidencyTable::new(max_records),
            fallback: GpuMaterialRecord::default(),
            buffer,
            buffer_capacity: capacity,
            dirty: true,
            buffer_recreated: true,
        }
    }

    fn allocate(
        &mut self,
        device: &wgpu::Device,
        mut record: GpuMaterialRecord,
    ) -> Result<MaterialId, &'static str> {
        record.normalize_layered_pbr_metadata();
        let id = self.records.insert(record)?;
        self.ensure_capacity(device, id.descriptor_index() as usize + 1);
        self.dirty = true;
        Ok(id)
    }

    fn update(&mut self, id: MaterialId, mut record: GpuMaterialRecord) -> ResolveStatus {
        let (slot, status) = self.records.resolve_mut(id);
        if let Some(slot) = slot {
            record.normalize_layered_pbr_metadata();
            record.header[0] = id.generation();
            *slot = record;
            self.dirty = true;
        }
        status
    }

    fn retire(&mut self, id: MaterialId, completion_epoch: u64) -> bool {
        let retired = self.records.retire(id, completion_epoch);
        self.dirty |= retired;
        retired
    }

    fn collect(&mut self, completed_epoch: u64) -> usize {
        let reclaimed = self.records.collect(completed_epoch);
        self.dirty |= reclaimed > 0;
        reclaimed
    }

    fn ensure_capacity(&mut self, device: &wgpu::Device, required: usize) {
        if required <= self.buffer_capacity {
            return;
        }
        let max_capacity = self.records.max_live().saturating_add(1);
        let capacity = required.next_power_of_two().min(max_capacity).max(required);
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global_material_records"),
            size: (capacity * std::mem::size_of::<GpuMaterialRecord>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.buffer_capacity = capacity;
        self.buffer_recreated = true;
        self.dirty = true;
    }

    fn flush(&mut self, queue: &wgpu::Queue) -> bool {
        if !self.dirty {
            return false;
        }
        let mut upload = vec![self.fallback; self.buffer_capacity];
        for (slot_index, slot) in self.records.slots().iter().enumerate() {
            if !slot.live {
                continue;
            }
            let Some(mut record) = slot.value else {
                continue;
            };
            record.header[0] = slot.generation;
            upload[slot_index + 1] = record;
        }
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&upload));
        self.dirty = false;
        true
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TextureColorSpace {
    Srgb,
    Linear,
    HdrLinear,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TextureSemantic {
    BaseColor,
    Normal,
    MetallicRoughness,
    Emissive,
    Occlusion,
    General,
}

pub struct ResidentTexture {
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub color_space: TextureColorSpace,
    pub semantic: TextureSemantic,
    /// True when the view format performs the sRGB transfer in hardware.
    /// False + `Srgb` asks the shared WGSL helper to decode explicitly.
    pub hardware_srgb_decode: bool,
    /// Only D2 float views can enter Tier A's `texture_2d` table. D2Array
    /// resources still receive stable IDs for Tier B page records, but their
    /// descriptor entry remains the safe fallback.
    pub global_2d: bool,
}

pub struct ResidentSampler {
    pub sampler: wgpu::Sampler,
}

#[derive(Clone, Debug, Default)]
pub struct ResidentMesh {
    pub vertex_count: u32,
    pub index_count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ResidentBufferView {
    pub byte_offset: u64,
    pub byte_size: u64,
}

#[derive(Clone, Debug)]
pub struct TierBMaterialTextures {
    pub material: MaterialId,
    pub textures: Vec<TextureId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TierBPage {
    pub textures: Vec<TextureId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TierBDraw {
    pub original_draw_index: usize,
    pub material: MaterialId,
    pub page: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TierBDispatchPlan {
    pub pages: Vec<TierBPage>,
    pub draws: Vec<TierBDraw>,
    pub page_switches: u32,
    pub fallback_materials: Vec<MaterialId>,
}

/// Deterministic first-fit paging and stable grouping for Tier B.
///
/// A material whose unique texture set exceeds the adapter page limit is sent
/// to the Tier C fallback. All other draws are stable-sorted by page, bounding
/// bind-group switches to the number of populated pages.
pub fn build_tier_b_dispatch_plan(
    draws: &[TierBMaterialTextures],
    page_capacity: u32,
) -> TierBDispatchPlan {
    let capacity = page_capacity.max(1) as usize;
    let mut pages: Vec<BTreeSet<TextureId>> = Vec::new();
    let mut draw_pages = Vec::with_capacity(draws.len());
    let mut fallback_materials = Vec::new();

    for (draw_index, draw) in draws.iter().enumerate() {
        let needed: BTreeSet<TextureId> = draw
            .textures
            .iter()
            .copied()
            .filter(|id| !id.is_fallback())
            .collect();
        if needed.len() > capacity {
            fallback_materials.push(draw.material);
            draw_pages.push(TierBDraw {
                original_draw_index: draw_index,
                material: draw.material,
                page: None,
            });
            continue;
        }
        let page = pages
            .iter()
            .position(|resident| resident.union(&needed).count() <= capacity)
            .unwrap_or_else(|| {
                pages.push(BTreeSet::new());
                pages.len() - 1
            });
        pages[page].extend(needed);
        draw_pages.push(TierBDraw {
            original_draw_index: draw_index,
            material: draw.material,
            page: Some(page as u32),
        });
    }

    draw_pages.sort_by_key(|draw| (draw.page.unwrap_or(u32::MAX), draw.original_draw_index));
    let page_switches = draw_pages
        .iter()
        .filter_map(|draw| draw.page)
        .fold((None, 0u32), |(last, switches), page| {
            if last == Some(page) {
                (last, switches)
            } else {
                (Some(page), switches + 1)
            }
        })
        .1;
    TierBDispatchPlan {
        pages: pages
            .into_iter()
            .map(|textures| TierBPage {
                textures: textures.into_iter().collect(),
            })
            .collect(),
        draws: draw_pages,
        page_switches,
        fallback_materials,
    }
}

/// All typed resource tables and optional Tier A global binding state.
pub struct MaterialIndirection {
    pub capabilities: MaterialBindingCapabilities,
    materials: GpuMaterialTable,
    pub textures: ResidencyTable<TextureId, ResidentTexture>,
    pub samplers: ResidencyTable<SamplerId, ResidentSampler>,
    pub meshes: ResidencyTable<MeshId, ResidentMesh>,
    pub buffer_views: ResidencyTable<BufferViewId, ResidentBufferView>,
    completion: GpuCompletionTracker,
    fallback_texture: Option<wgpu::TextureView>,
    fallback_sampler: Option<wgpu::Sampler>,
    resource_generations: wgpu::Buffer,
    resource_generation_capacity: usize,
    pub global_layout: Option<wgpu::BindGroupLayout>,
    pub global_bind_group: Option<wgpu::BindGroup>,
    resources_dirty: bool,
    stale_fallbacks: u64,
    limit_fallbacks: u64,
    last_tier_b_pages: u32,
    last_tier_b_switches: u32,
    last_tier_b_fallbacks: u32,
}

impl MaterialIndirection {
    pub fn new(device: &wgpu::Device) -> Self {
        let capabilities = MaterialBindingCapabilities::detect(device.features(), &device.limits());
        let texture_capacity = match capabilities.detected_tier {
            MaterialBindingTier::A => capabilities.tier_a_texture_capacity as usize,
            _ => DEFAULT_RESOURCE_CAPACITY,
        }
        .max(1);
        let sampler_capacity = match capabilities.detected_tier {
            MaterialBindingTier::A => capabilities.tier_a_sampler_capacity as usize,
            _ => 256,
        }
        .max(1);
        let resource_generation_capacity = texture_capacity.max(sampler_capacity) + 1;
        let resource_generations = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global_resource_generations"),
            size: (resource_generation_capacity * std::mem::size_of::<[u32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            materials: GpuMaterialTable::new(device, capabilities.max_material_records as usize),
            textures: ResidencyTable::new(texture_capacity),
            samplers: ResidencyTable::new(sampler_capacity),
            meshes: ResidencyTable::new(DEFAULT_RESOURCE_CAPACITY),
            buffer_views: ResidencyTable::new(DEFAULT_RESOURCE_CAPACITY * 2),
            capabilities,
            completion: GpuCompletionTracker::default(),
            fallback_texture: None,
            fallback_sampler: None,
            resource_generations,
            resource_generation_capacity,
            global_layout: None,
            global_bind_group: None,
            resources_dirty: true,
            stale_fallbacks: 0,
            limit_fallbacks: 0,
            last_tier_b_pages: 0,
            last_tier_b_switches: 0,
            last_tier_b_fallbacks: 0,
        }
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn active_layered_material_count(&self) -> usize {
        self.materials
            .records
            .slots()
            .iter()
            .filter_map(|slot| if slot.live { slot.value.as_ref() } else { None })
            .filter(|record| !record.layered_pbr_lobe_mask().is_empty())
            .count()
    }

    pub fn initialize_fallbacks(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) {
        self.fallback_texture = Some(texture.clone());
        self.fallback_sampler = Some(sampler.clone());
        self.rebuild_layout(device);
        self.resources_dirty = true;
    }

    pub fn allocate_material(
        &mut self,
        device: &wgpu::Device,
        record: GpuMaterialRecord,
    ) -> MaterialId {
        match self.materials.allocate(device, record) {
            Ok(id) => id,
            Err(error) => {
                self.limit_fallbacks += 1;
                crate::ffi::log_error(&format!(
                    "bloom: material indirection allocation failed ({error}); using diagnostic fallback"
                ));
                MaterialId::FALLBACK
            }
        }
    }

    pub fn update_material(&mut self, id: MaterialId, record: GpuMaterialRecord) -> bool {
        let status = self.materials.update(id, record);
        if status != ResolveStatus::Resident {
            self.stale_fallbacks += 1;
            return false;
        }
        true
    }

    pub fn register_texture(&mut self, texture: ResidentTexture) -> TextureId {
        match self.textures.insert(texture) {
            Ok(id) => {
                self.resources_dirty = true;
                id
            }
            Err(error) => {
                self.limit_fallbacks += 1;
                crate::ffi::log_error(&format!(
                    "bloom: global texture table allocation failed ({error}); using diagnostic fallback"
                ));
                TextureId::FALLBACK
            }
        }
    }

    pub fn register_sampler(&mut self, sampler: wgpu::Sampler) -> SamplerId {
        match self.samplers.insert(ResidentSampler { sampler }) {
            Ok(id) => {
                self.resources_dirty = true;
                id
            }
            Err(error) => {
                self.limit_fallbacks += 1;
                crate::ffi::log_error(&format!(
                    "bloom: global sampler table allocation failed ({error}); using diagnostic fallback"
                ));
                SamplerId::FALLBACK
            }
        }
    }

    pub fn register_mesh(&mut self, mesh: ResidentMesh) -> MeshId {
        self.meshes.insert(mesh).unwrap_or_else(|_| {
            self.limit_fallbacks += 1;
            MeshId::FALLBACK
        })
    }

    pub fn register_buffer_view(&mut self, view: ResidentBufferView) -> BufferViewId {
        self.buffer_views.insert(view).unwrap_or_else(|_| {
            self.limit_fallbacks += 1;
            BufferViewId::FALLBACK
        })
    }

    pub fn retire_texture(&mut self, queue: &wgpu::Queue, id: TextureId) -> bool {
        let epoch = self.completion.track_submitted_work(queue);
        let retired = self.textures.retire(id, epoch);
        self.resources_dirty |= retired;
        retired
    }

    pub fn retire_sampler(&mut self, queue: &wgpu::Queue, id: SamplerId) -> bool {
        let epoch = self.completion.track_submitted_work(queue);
        let retired = self.samplers.retire(id, epoch);
        self.resources_dirty |= retired;
        retired
    }

    /// Retire a standalone material record. Scene-graph materials may be
    /// shared by thousands of nodes, so their cache owns only this ID rather
    /// than the mesh/buffer-view IDs retired by `retire_cached_mesh`.
    pub fn retire_material(&mut self, queue: &wgpu::Queue, id: MaterialId) -> bool {
        let epoch = self.completion.track_submitted_work(queue);
        self.materials.retire(id, epoch)
    }

    pub fn retire_materials(
        &mut self,
        queue: &wgpu::Queue,
        ids: impl IntoIterator<Item = MaterialId>,
    ) -> usize {
        let ids: Vec<_> = ids
            .into_iter()
            .filter(|id| *id != MaterialId::FALLBACK)
            .collect();
        if ids.is_empty() {
            return 0;
        }
        let epoch = self.completion.track_submitted_work(queue);
        ids.into_iter()
            .filter(|id| self.materials.retire(*id, epoch))
            .count()
    }

    /// Retire the IDs owned by one cached mesh with a single queue callback.
    pub fn retire_cached_mesh(
        &mut self,
        queue: &wgpu::Queue,
        material: MaterialId,
        mesh: MeshId,
        buffer_views: &[BufferViewId],
    ) {
        let epoch = self.completion.track_submitted_work(queue);
        self.materials.retire(material, epoch);
        self.meshes.retire(mesh, epoch);
        for &buffer_view in buffer_views {
            self.buffer_views.retire(buffer_view, epoch);
        }
    }

    pub fn flush(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let completed = self.completion.completed_epoch();
        self.materials.collect(completed);
        let reclaimed = self.textures.collect(completed)
            + self.samplers.collect(completed)
            + self.meshes.collect(completed)
            + self.buffer_views.collect(completed);
        self.resources_dirty |= reclaimed > 0;
        let material_buffer_changed = self.materials.buffer_recreated;
        self.materials.flush(queue);
        if self.resources_dirty {
            self.flush_resource_generations(queue);
        }
        if self.capabilities.selected_tier == MaterialBindingTier::A
            && (self.resources_dirty || material_buffer_changed)
        {
            self.rebuild_global_bind_group(device);
        }
        self.materials.buffer_recreated = false;
        self.resources_dirty = false;
    }

    /// Apply a debug override. Codes: 0=auto, 1=C, 2=B, 3=A.
    /// Requests above the adapter's detected tier are rejected.
    pub fn set_tier_override(&mut self, device: &wgpu::Device, code: u32) -> bool {
        let requested = match code {
            0 => None,
            1 => Some(MaterialBindingTier::C),
            2 => Some(MaterialBindingTier::B),
            3 => Some(MaterialBindingTier::A),
            _ => return false,
        };
        let next = MaterialBindingCapabilities::detect_with_override(
            device.features(),
            &device.limits(),
            requested,
        );
        let accepted = requested.is_none() || next.override_tier == requested;
        self.capabilities = next;
        self.rebuild_layout(device);
        self.resources_dirty = true;
        accepted
    }

    /// Plan and retain Tier B paging telemetry for a GPU-driven dispatch.
    pub fn plan_tier_b_dispatch(&mut self, draws: &[TierBMaterialTextures]) -> TierBDispatchPlan {
        let plan = build_tier_b_dispatch_plan(draws, self.capabilities.tier_b_page_capacity);
        self.last_tier_b_pages = plan.pages.len() as u32;
        self.last_tier_b_switches = plan.page_switches;
        self.last_tier_b_fallbacks = plan.fallback_materials.len() as u32;
        plan
    }

    pub fn report_json(&self) -> String {
        let mut out = self.capabilities.report_json();
        out.pop();
        out.push_str(",\"residency\":{\"materials\":");
        out.push_str(&self.materials.records.live_count().to_string());
        out.push_str(",\"textures\":");
        out.push_str(&self.textures.live_count().to_string());
        out.push_str(",\"samplers\":");
        out.push_str(&self.samplers.live_count().to_string());
        out.push_str(",\"meshes\":");
        out.push_str(&self.meshes.live_count().to_string());
        out.push_str(",\"buffer_views\":");
        out.push_str(&self.buffer_views.live_count().to_string());
        out.push_str(",\"stale_fallbacks\":");
        out.push_str(&self.stale_fallbacks.to_string());
        out.push_str(",\"limit_fallbacks\":");
        out.push_str(&self.limit_fallbacks.to_string());
        out.push_str("},\"dispatch\":{\"tier_a_per_material_bind_group_switches\":0");
        out.push_str(",\"tier_b_last_page_count\":");
        out.push_str(&self.last_tier_b_pages.to_string());
        out.push_str(",\"tier_b_last_page_switches\":");
        out.push_str(&self.last_tier_b_switches.to_string());
        out.push_str(",\"tier_b_last_fallback_materials\":");
        out.push_str(&self.last_tier_b_fallbacks.to_string());
        out.push_str("}}");
        out
    }

    pub fn material_buffer(&self) -> &wgpu::Buffer {
        &self.materials.buffer
    }

    fn rebuild_layout(&mut self, device: &wgpu::Device) {
        if self.capabilities.selected_tier != MaterialBindingTier::A {
            self.global_layout = None;
            self.global_bind_group = None;
            return;
        }
        let texture_count = self.capabilities.tier_a_texture_capacity.saturating_add(1);
        let sampler_count = self.capabilities.tier_a_sampler_capacity.saturating_add(1);
        let Some(texture_count) = NonZeroU32::new(texture_count) else {
            return;
        };
        let Some(sampler_count) = NonZeroU32::new(sampler_count) else {
            return;
        };
        self.global_layout = Some(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("global_material_indirection_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT
                            | wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: Some(texture_count),
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: Some(sampler_count),
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT
                            | wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        ));
        self.global_bind_group = None;
    }

    fn rebuild_global_bind_group(&mut self, device: &wgpu::Device) {
        let (Some(layout), Some(fallback_texture), Some(fallback_sampler)) = (
            self.global_layout.as_ref(),
            self.fallback_texture.as_ref(),
            self.fallback_sampler.as_ref(),
        ) else {
            return;
        };
        let texture_count = self.capabilities.tier_a_texture_capacity as usize + 1;
        let sampler_count = self.capabilities.tier_a_sampler_capacity as usize + 1;
        let mut texture_refs = vec![fallback_texture; texture_count];
        for (slot_index, slot) in self.textures.slots().iter().enumerate() {
            let descriptor_index = slot_index + 1;
            if descriptor_index >= texture_count || !slot.live {
                continue;
            }
            if let Some(texture) = slot.value.as_ref().filter(|texture| texture.global_2d) {
                texture_refs[descriptor_index] = &texture.view;
            }
        }
        let mut sampler_refs = vec![fallback_sampler; sampler_count];
        for (slot_index, slot) in self.samplers.slots().iter().enumerate() {
            let descriptor_index = slot_index + 1;
            if descriptor_index >= sampler_count || !slot.live {
                continue;
            }
            if let Some(sampler) = slot.value.as_ref() {
                sampler_refs[descriptor_index] = &sampler.sampler;
            }
        }
        self.global_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("global_material_indirection_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.materials.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureViewArray(&texture_refs),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::SamplerArray(&sampler_refs),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.resource_generations.as_entire_binding(),
                },
            ],
        }));
    }

    fn flush_resource_generations(&self, queue: &wgpu::Queue) {
        let mut generations = vec![[0u32; 4]; self.resource_generation_capacity];
        for (slot_index, slot) in self.textures.slots().iter().enumerate() {
            let descriptor_index = slot_index + 1;
            if descriptor_index >= generations.len() {
                break;
            }
            generations[descriptor_index][0] = if slot.live {
                slot.generation
            } else {
                (slot.generation + 1) & ID_GENERATION_MASK
            };
            if let Some(texture) = slot.value.as_ref().filter(|_| slot.live) {
                generations[descriptor_index][2] = match texture.color_space {
                    TextureColorSpace::Srgb => 1 | if texture.hardware_srgb_decode { 2 } else { 0 },
                    TextureColorSpace::Linear => 0,
                    TextureColorSpace::HdrLinear => 4,
                };
                generations[descriptor_index][3] = match texture.semantic {
                    TextureSemantic::BaseColor => 1,
                    TextureSemantic::Normal => 2,
                    TextureSemantic::MetallicRoughness => 3,
                    TextureSemantic::Emissive => 4,
                    TextureSemantic::Occlusion => 5,
                    TextureSemantic::General => 0,
                };
            }
        }
        for (slot_index, slot) in self.samplers.slots().iter().enumerate() {
            let descriptor_index = slot_index + 1;
            if descriptor_index >= generations.len() {
                break;
            }
            generations[descriptor_index][1] = if slot.live {
                slot.generation
            } else {
                (slot.generation + 1) & ID_GENERATION_MASK
            };
        }
        queue.write_buffer(
            &self.resource_generations,
            0,
            bytemuck::cast_slice(&generations),
        );
    }
}

fn json_string(out: &mut String, value: &str) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => {
                use std::fmt::Write;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::util::DeviceExt;

    #[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
    struct TestId(u32);

    impl StableResourceId for TestId {
        const FALLBACK: Self = Self(0);

        fn from_parts(slot: usize, generation: u32) -> Option<Self> {
            let one_based = slot.checked_add(1)?;
            (one_based <= ID_SLOT_MASK as usize && generation <= ID_GENERATION_MASK)
                .then_some(Self((generation << ID_SLOT_BITS) | one_based as u32))
        }

        fn raw(self) -> u32 {
            self.0
        }
    }

    #[test]
    fn typed_ids_are_generation_safe_and_zero_is_fallback() {
        let mut table = ResidencyTable::<TestId, &'static str>::new(2);
        let first = table.insert("first").unwrap();
        assert_eq!(first.raw(), 1);
        assert_eq!(
            table.resolve(first),
            (Some(&"first"), ResolveStatus::Resident)
        );
        assert!(table.retire(first, 7));
        assert_eq!(table.resolve(first), (None, ResolveStatus::Retiring));
        assert_eq!(table.collect(6), 0);
        assert_eq!(table.insert("blocked"), Ok(TestId(2)));
        assert_eq!(table.collect(7), 1);
        let reused = table.insert("reused").unwrap();
        assert_eq!(reused.descriptor_index(), first.descriptor_index());
        assert_ne!(reused.generation(), first.generation());
        assert_eq!(table.resolve(first), (None, ResolveStatus::Stale));
        assert_eq!(
            table.resolve(reused),
            (Some(&"reused"), ResolveStatus::Resident)
        );
        assert_eq!(
            table.resolve(TestId::FALLBACK),
            (None, ResolveStatus::Fallback)
        );
    }

    #[test]
    fn queue_tracker_never_completes_before_callback() {
        let tracker = GpuCompletionTracker::default();
        assert_eq!(tracker.completed_epoch(), 0);
        tracker.mark_complete_for_test(3);
        assert_eq!(tracker.completed_epoch(), 3);
        tracker.mark_complete_for_test(2);
        assert_eq!(tracker.completed_epoch(), 3);
    }

    #[test]
    fn capability_selection_uses_features_and_limits_not_platform() {
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_texture_array_layers = 256;
        limits.max_sampled_textures_per_shader_stage = 16;
        let tier_b = MaterialBindingCapabilities::detect_with_override(
            wgpu::Features::empty(),
            &limits,
            None,
        );
        assert_eq!(tier_b.detected_tier, MaterialBindingTier::B);

        limits.max_binding_array_elements_per_shader_stage = 500_000;
        limits.max_binding_array_sampler_elements_per_shader_stage = 1_000;
        let features = wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        let tier_a = MaterialBindingCapabilities::detect_with_override(features, &limits, None);
        assert_eq!(tier_a.detected_tier, MaterialBindingTier::A);
        assert_eq!(tier_a.tier_a_texture_capacity, TIER_A_TARGET_TEXTURES);

        let forced_c = MaterialBindingCapabilities::detect_with_override(
            features,
            &limits,
            Some(MaterialBindingTier::C),
        );
        assert_eq!(forced_c.selected_tier, MaterialBindingTier::C);

        let rejected_a = MaterialBindingCapabilities::detect_with_override(
            wgpu::Features::empty(),
            &wgpu::Limits::downlevel_defaults(),
            Some(MaterialBindingTier::A),
        );
        assert_ne!(rejected_a.selected_tier, MaterialBindingTier::A);
        assert!(rejected_a.diagnostic.is_some());
    }

    #[test]
    fn tier_b_paging_is_deterministic_bounded_and_falls_back_safely() {
        let material = |slot| MaterialId::from_parts(slot, 0).unwrap();
        let texture = |slot| TextureId::from_parts(slot, 0).unwrap();
        let draws = vec![
            TierBMaterialTextures {
                material: material(0),
                textures: vec![texture(0), texture(1)],
            },
            TierBMaterialTextures {
                material: material(1),
                textures: vec![texture(1), texture(2)],
            },
            TierBMaterialTextures {
                material: material(2),
                textures: vec![texture(3), texture(4)],
            },
            TierBMaterialTextures {
                material: material(3),
                textures: vec![texture(5), texture(6), texture(7), texture(8)],
            },
        ];
        let a = build_tier_b_dispatch_plan(&draws, 3);
        let b = build_tier_b_dispatch_plan(&draws, 3);
        assert_eq!(a, b);
        assert_eq!(a.pages.len(), 2);
        assert!(a.page_switches <= a.pages.len() as u32);
        assert_eq!(a.fallback_materials, vec![material(3)]);
        assert_eq!(a.draws.last().unwrap().page, None);
    }

    #[test]
    fn stress_4096_textures_and_10k_draws_remains_bounded() {
        let textures: Vec<_> = (0..4_096)
            .map(|slot| TextureId::from_parts(slot, 0).unwrap())
            .collect();
        let draws: Vec<_> = (0..10_000)
            .map(|draw| {
                let base = (draw * 4) % textures.len();
                TierBMaterialTextures {
                    material: MaterialId::from_parts(draw % 128, 0).unwrap(),
                    textures: vec![
                        textures[base],
                        textures[(base + 1) % textures.len()],
                        textures[(base + 2) % textures.len()],
                        textures[(base + 3) % textures.len()],
                    ],
                }
            })
            .collect();
        let plan = build_tier_b_dispatch_plan(&draws, 256);
        assert_eq!(plan.draws.len(), 10_000);
        assert_eq!(plan.pages.len(), 16);
        assert_eq!(plan.page_switches, 16);
        assert!(plan.fallback_materials.is_empty());
        assert!(plan.pages.iter().all(|page| page.textures.len() <= 256));
    }

    #[test]
    fn gpu_record_layout_is_storage_buffer_safe() {
        assert_eq!(std::mem::align_of::<GpuMaterialRecord>(), 16);
        assert_eq!(std::mem::size_of::<GpuMaterialRecord>(), 176);
        let record = GpuMaterialRecord::default();
        assert_eq!(record.base_color, [1.0; 4]);
        assert_eq!(
            record.layered_pbr_version(),
            crate::renderer::layered_pbr::MATERIAL_RECORD_VERSION
        );
        assert!(record.layered_pbr_lobe_mask().is_empty());
        let mut legacy = GpuMaterialRecord::default();
        legacy.header[1] = u32::MAX;
        legacy.normalize_layered_pbr_metadata();
        assert_eq!(
            legacy.layered_pbr_version(),
            crate::renderer::layered_pbr::MATERIAL_RECORD_VERSION
        );
        assert!(legacy.layered_pbr_lobe_mask().is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn try_tier_a_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok()?;
        let supported = adapter.features();
        if !supported.contains(TIER_A_FEATURES) {
            return None;
        }
        let adapter_limits = adapter.limits();
        let mut features = wgpu::Features::empty();
        let mut limits = wgpu::Limits::downlevel_defaults();
        request_tier_a_if_supported(supported, &adapter_limits, &mut features, &mut limits);
        limits.max_storage_buffer_binding_size = adapter_limits
            .max_storage_buffer_binding_size
            .min(wgpu::Limits::default().max_storage_buffer_binding_size);
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("material_indirection_tier_a_test"),
            required_features: features,
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            ..Default::default()
        }))
        .ok()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn render_global_material_pixel(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        indirection: &MaterialIndirection,
        material_id: MaterialId,
    ) -> [u8; 4] {
        let source = format!(
            "{}\n\
             struct TestVertexOut {{ @builtin(position) position: vec4<f32>, }};\n\
             @vertex fn vs_main(@builtin(vertex_index) index: u32) -> TestVertexOut {{\n\
               var positions = array<vec2<f32>, 3>(\n\
                 vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));\n\
               var out: TestVertexOut;\n\
               out.position = vec4<f32>(positions[index], 0.0, 1.0);\n\
               return out;\n\
             }}\n\
             @fragment fn fs_main() -> @location(0) vec4<f32> {{\n\
               return bloom_sample_base_color(\n\
                 bloom_material_record({}u), vec2<f32>(0.5, 0.5));\n\
             }}\n",
            include_str!("../../shaders/material_indirection.wgsl"),
            material_id.raw(),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("material_indirection_tier_a_shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("material_indirection_tier_a_pipeline_layout"),
            bind_group_layouts: &[None, None, indirection.global_layout.as_ref()],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("material_indirection_tier_a_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("material_indirection_tier_a_target"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material_indirection_tier_a_readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("material_indirection_tier_a_encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("material_indirection_tier_a_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(
                2,
                indirection
                    .global_bind_group
                    .as_ref()
                    .expect("Tier A bind group"),
                &[],
            );
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        [mapped[0], mapped[1], mapped[2], mapped[3]]
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn tier_a_binds_4096_textures_decodes_srgb_and_rejects_reused_stale_id() {
        let Some((device, queue)) = try_tier_a_device() else {
            eprintln!("Tier A adapter unavailable; device-backed descriptor test skipped");
            return;
        };
        let fallback_texture = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("material_indirection_fallback"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255, 255, 255, 255],
        );
        let fallback_view = fallback_texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let source_texture = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("material_indirection_source"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[128, 0, 0, 255],
        );
        let source_view = source_texture.create_view(&Default::default());
        let mut indirection = MaterialIndirection::new(&device);
        assert_eq!(
            indirection.capabilities.selected_tier,
            MaterialBindingTier::A
        );
        indirection.initialize_fallbacks(&device, &fallback_view, &sampler);
        let sampler_id = indirection.register_sampler(sampler.clone());
        let mut first_texture_id = TextureId::FALLBACK;
        for texture_index in 0..TIER_A_TARGET_TEXTURES {
            let id = indirection.register_texture(ResidentTexture {
                view: source_view.clone(),
                width: 1,
                height: 1,
                mip_count: 1,
                color_space: TextureColorSpace::Srgb,
                semantic: TextureSemantic::BaseColor,
                hardware_srgb_decode: false,
                global_2d: true,
            });
            assert!(!id.is_fallback(), "texture {texture_index} fell back");
            if texture_index == 0 {
                first_texture_id = id;
            }
        }
        let mut record = GpuMaterialRecord::default();
        record.texture_ids_0[0] = first_texture_id.raw();
        record.sampler_ids_0[0] = sampler_id.raw();
        let material_id = indirection.allocate_material(&device, record);
        indirection.flush(&device, &queue);
        assert_eq!(indirection.textures.live_count(), 4_096);
        let srgb_pixel = render_global_material_pixel(&device, &queue, &indirection, material_id);
        assert!(
            (53..=57).contains(&srgb_pixel[0]),
            "manual sRGB decode drifted: {srgb_pixel:?}"
        );
        assert_eq!(&srgb_pixel[1..], &[0, 0, 255]);

        assert!(indirection.retire_texture(&queue, first_texture_id));
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        indirection.flush(&device, &queue);
        let replacement = indirection.register_texture(ResidentTexture {
            view: source_view,
            width: 1,
            height: 1,
            mip_count: 1,
            color_space: TextureColorSpace::Linear,
            semantic: TextureSemantic::BaseColor,
            hardware_srgb_decode: false,
            global_2d: true,
        });
        assert_eq!(
            replacement.descriptor_index(),
            first_texture_id.descriptor_index()
        );
        assert_ne!(replacement.generation(), first_texture_id.generation());
        indirection.flush(&device, &queue);
        let stale_pixel = render_global_material_pixel(&device, &queue, &indirection, material_id);
        assert_eq!(stale_pixel, [255, 255, 255, 255]);
    }
}
