//! Fixed-capacity GPU storage for cooked virtual-geometry pages.
//!
//! The compatibility geometry arena expands vertices and may grow. Virtual
//! geometry instead keeps validated cooked bytes in fixed-stride physical
//! slots. Stable generational mesh IDs plus a GPU-visible page table prevent
//! stale scene records from aliasing a newly registered asset.

use super::residency::{group_containing, group_lod, group_pages, parent_group};
use super::{ClusterGroup, ResidencyError, ResolvedClusterGroup, VirtualGeometryAsset};
use crate::renderer::material_indirection::{GpuCompletionTracker, StableResourceId};
use bloom_geometry_format::{sha256, MAX_PAGE_BYTES, MIN_PAGE_BYTES};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const ID_SLOT_BITS: u32 = 20;
const ID_SLOT_MASK: u32 = (1 << ID_SLOT_BITS) - 1;
const ID_GENERATION_MASK: u32 = (1 << (32 - ID_SLOT_BITS)) - 1;
const GPU_MESH_ENTRY_WORDS: u64 = 12;
const GPU_PAGE_ENTRY_WORDS: u64 = 4;
const GPU_CLUSTER_ENTRY_WORDS: u64 = 32;
const GPU_WORD_BYTES: u64 = std::mem::size_of::<u32>() as u64;
pub(super) const MAX_GPU_HIERARCHY_LEVELS: u32 = 32;
static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

pub const GPU_VIRTUAL_PAGE_RESIDENT: u32 = 1 << 0;
pub const GPU_VIRTUAL_PAGE_PINNED: u32 = 1 << 1;
pub const GPU_VIRTUAL_MESH_VALID: u32 = 1 << 0;
pub const GPU_VIRTUAL_MESH_MATERIALS_BOUND: u32 = 1 << 1;

/// Explicit translation from one archive material slot to the renderer's
/// generation-safe global material ID. `None` is glTF's default material.
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct VirtualMaterialBinding {
    pub source_material_index: Option<u32>,
    pub material_id: u32,
}

/// A generation-safe runtime mesh address. Zero is the shader fallback.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VirtualMeshId(u32);

impl VirtualMeshId {
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

    pub(super) const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }
}

impl StableResourceId for VirtualMeshId {
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

/// Stable page address: the local page is protected by its mesh generation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VirtualPageId {
    pub mesh: VirtualMeshId,
    pub page_index: u32,
}

/// GPU ABI for one registered virtual mesh (48 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVirtualMeshEntry {
    pub mesh_id: u32,
    pub page_table_base: u32,
    pub page_count: u32,
    pub cluster_table_base: u32,
    pub cluster_count: u32,
    pub root_cluster_count: u32,
    pub page_stride_bytes: u32,
    pub vertex_encoding: u32,
    pub format_version: u32,
    pub flags: u32,
    pub reserved: [u32; 2],
}

/// GPU ABI for one logical page (16 bytes). A zero `slot_plus_one` is missing.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVirtualPageEntry {
    pub slot_plus_one: u32,
    pub payload_bytes: u32,
    pub mesh_id: u32,
    pub flags: u32,
}

/// GPU traversal and decode metadata for one cooked cluster (128 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVirtualClusterEntry {
    /// xyz = object AABB minimum, w = accumulated object-space error.
    pub aabb_min_error: [f32; 4],
    /// xyz = object AABB maximum, w = conservative sphere radius.
    pub aabb_max_radius: [f32; 4],
    /// xyz = object-space sphere center, w reserved.
    pub sphere: [f32; 4],
    /// xyz = object-space normal-cone axis, w = minimum axis/normal dot.
    pub normal_cone: [f32; 4],
    /// mesh index, primitive index, bound GPU material ID/zero fallback, flags.
    pub identity: [u32; 4],
    /// logical page, LOD level, vertex count, triangle count.
    pub page_lod_counts: [u32; 4],
    /// Page-local vertex/index byte offsets, vertex stride, owning mesh ID.
    /// The owner sentinel prevents an absolute selected-record address from
    /// aliasing a cluster belonging to another mesh generation.
    pub payload: [u32; 4],
    /// parent start/count, child start/count; indices are mesh-local.
    pub relations: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuVirtualMeshEntry>() == 48);
const _: () = assert!(std::mem::size_of::<GpuVirtualPageEntry>() == 16);
const _: () = assert!(std::mem::size_of::<GpuVirtualClusterEntry>() == 128);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuVirtualGeometryConfig {
    pub capacity_bytes: u64,
    pub page_stride_bytes: u32,
    pub max_meshes: u32,
    pub max_page_records: u32,
    pub max_cluster_records: u32,
    pub max_clusters_per_group: u32,
    pub max_hierarchy_levels: u32,
    pub max_upload_bytes_per_frame: u64,
    pub max_upload_pages_per_frame: u32,
    pub max_evictions_per_frame: u32,
}

impl Default for GpuVirtualGeometryConfig {
    fn default() -> Self {
        Self {
            capacity_bytes: 64 * 1024 * 1024,
            page_stride_bytes: 64 * 1024,
            max_meshes: 4_096,
            max_page_records: 65_536,
            max_cluster_records: 262_144,
            max_clusters_per_group: 256,
            max_hierarchy_levels: 32,
            max_upload_bytes_per_frame: 8 * 1024 * 1024,
            max_upload_pages_per_frame: 128,
            max_evictions_per_frame: 128,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuPageTransition {
    pub group: ClusterGroup,
    pub uploaded: Vec<(VirtualPageId, u32)>,
    pub evicted: Vec<VirtualPageId>,
    pub resident_slot_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuVirtualGeometryTelemetry {
    pub frame: u64,
    pub capacity_bytes: u64,
    pub page_table_bytes: u64,
    pub mesh_table_bytes: u64,
    pub cluster_table_bytes: u64,
    pub total_gpu_bytes: u64,
    pub slot_count: u32,
    pub active_meshes: u32,
    pub live_page_records: u32,
    pub live_cluster_records: u32,
    pub resident_pages: u32,
    pub resident_slot_bytes: u64,
    pub resident_payload_bytes: u64,
    pub pinned_pages: u32,
    pub pinned_slot_bytes: u64,
    pub retiring_slots: u32,
    pub frame_upload_pages: u32,
    pub frame_upload_bytes: u64,
    pub frame_evictions: u32,
    pub uploads: u64,
    pub evictions: u64,
    pub denied_uploads: u64,
    pub exact_resolutions: u64,
    pub fallback_resolutions: u64,
    pub unresolved_requests: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct LogicalPageState {
    slot: Option<u32>,
    pinned: bool,
    last_use: u64,
}

struct LiveMesh {
    asset: Arc<VirtualGeometryAsset>,
    page_table_base: u32,
    cluster_table_base: u32,
    pages: Vec<LogicalPageState>,
    material_bindings: Option<BTreeMap<Option<u32>, u32>>,
}

struct RetiringMesh {
    completion_epoch: u64,
    page_table_base: u32,
    page_count: u32,
    cluster_table_base: u32,
    cluster_count: u32,
    physical_slots: Vec<u32>,
}

enum MeshLifecycle {
    Free,
    Live(LiveMesh),
    Retiring(RetiringMesh),
}

struct MeshSlot {
    generation: u32,
    lifecycle: MeshLifecycle,
}

#[derive(Clone, Copy, Debug, Default)]
struct PhysicalSlot {
    owner: Option<VirtualPageId>,
    pinned: bool,
    last_use: u64,
    retiring_until: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FreeRange {
    start: u32,
    count: u32,
}

#[derive(Default)]
struct Counters {
    frame_upload_pages: u32,
    frame_upload_bytes: u64,
    frame_evictions: u32,
    uploads: u64,
    evictions: u64,
    denied_uploads: u64,
    exact_resolutions: u64,
    fallback_resolutions: u64,
    unresolved_requests: u64,
}

/// Explicit, lazily constructed virtual-geometry GPU residency pool.
///
/// Construction allocates exactly the configured physical, mesh-table, and
/// page-table buffers. `Renderer` owns it only after explicit virtual-geometry
/// enablement, so the established compatibility path incurs no allocations,
/// bindings, branches, or pixels.
pub struct GpuVirtualGeometryPool {
    id: u64,
    config: GpuVirtualGeometryConfig,
    physical_buffer: wgpu::Buffer,
    mesh_table_buffer: wgpu::Buffer,
    page_table_buffer: wgpu::Buffer,
    cluster_table_buffer: wgpu::Buffer,
    mesh_entries: Vec<GpuVirtualMeshEntry>,
    page_entries: Vec<GpuVirtualPageEntry>,
    cluster_entries: Vec<GpuVirtualClusterEntry>,
    mesh_slots: Vec<MeshSlot>,
    free_mesh_slots: Vec<usize>,
    free_page_ranges: Vec<FreeRange>,
    free_cluster_ranges: Vec<FreeRange>,
    physical_slots: Vec<PhysicalSlot>,
    completion: GpuCompletionTracker,
    clock: u64,
    frame: u64,
    counters: Counters,
}

impl GpuVirtualGeometryPool {
    pub fn new(
        device: &wgpu::Device,
        config: GpuVirtualGeometryConfig,
    ) -> Result<Self, VirtualGeometryGpuError> {
        validate_config(device, config)?;
        let slot_count = config.capacity_bytes / u64::from(config.page_stride_bytes);
        let mesh_table_bytes = u64::from(config.max_meshes) * GPU_MESH_ENTRY_WORDS * GPU_WORD_BYTES;
        let page_table_bytes =
            u64::from(config.max_page_records) * GPU_PAGE_ENTRY_WORDS * GPU_WORD_BYTES;
        let cluster_table_bytes =
            u64::from(config.max_cluster_records) * GPU_CLUSTER_ENTRY_WORDS * GPU_WORD_BYTES;
        Ok(Self {
            id: NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed),
            config,
            physical_buffer: create_zeroed_buffer(
                device,
                "virtual_geometry_physical_pages",
                config.capacity_bytes,
            ),
            mesh_table_buffer: create_zeroed_buffer(
                device,
                "virtual_geometry_mesh_table",
                mesh_table_bytes,
            ),
            page_table_buffer: create_zeroed_buffer(
                device,
                "virtual_geometry_page_table",
                page_table_bytes,
            ),
            cluster_table_buffer: create_zeroed_buffer(
                device,
                "virtual_geometry_cluster_table",
                cluster_table_bytes,
            ),
            mesh_entries: vec![GpuVirtualMeshEntry::default(); config.max_meshes as usize],
            page_entries: vec![GpuVirtualPageEntry::default(); config.max_page_records as usize],
            cluster_entries: vec![
                GpuVirtualClusterEntry::default();
                config.max_cluster_records as usize
            ],
            mesh_slots: Vec::new(),
            free_mesh_slots: Vec::new(),
            free_page_ranges: vec![FreeRange {
                start: 0,
                count: config.max_page_records,
            }],
            free_cluster_ranges: vec![FreeRange {
                start: 0,
                count: config.max_cluster_records,
            }],
            physical_slots: vec![PhysicalSlot::default(); slot_count as usize],
            completion: GpuCompletionTracker::default(),
            clock: 0,
            frame: 1,
            counters: Counters::default(),
        })
    }

    pub fn begin_frame(&mut self, frame: u64) {
        self.frame = frame.max(1);
        self.counters.frame_upload_pages = 0;
        self.counters.frame_upload_bytes = 0;
        self.counters.frame_evictions = 0;
    }

    /// Register an immutable archive and upload every coarse-root page before
    /// returning its ID. The operation is planned completely before mutation.
    pub fn register_mesh(
        &mut self,
        queue: &wgpu::Queue,
        asset: Arc<VirtualGeometryAsset>,
    ) -> Result<VirtualMeshId, VirtualGeometryGpuError> {
        let archive = asset.archive();
        if archive.pages.is_empty() || archive.coarse_root_page_count() == 0 {
            return Err(VirtualGeometryGpuError::MissingCoarseFallback);
        }
        if archive.page_budget_bytes > self.config.page_stride_bytes
            || archive.maximum_page_bytes() > self.config.page_stride_bytes
        {
            return Err(VirtualGeometryGpuError::PageStrideExceeded {
                archive_bytes: archive.page_budget_bytes.max(archive.maximum_page_bytes()),
                pool_bytes: self.config.page_stride_bytes,
            });
        }
        if let Some((cluster_index, cluster)) =
            archive.clusters.iter().enumerate().find(|(_, c)| {
                c.parent_count > self.config.max_clusters_per_group
                    || c.child_count > self.config.max_clusters_per_group
                    || c.lod_level >= self.config.max_hierarchy_levels
            })
        {
            return Err(VirtualGeometryGpuError::TraversalLimitExceeded {
                cluster: cluster_index as u32,
                parent_count: cluster.parent_count,
                child_count: cluster.child_count,
                lod_level: cluster.lod_level,
            });
        }
        let page_count = u32::try_from(archive.pages.len())
            .map_err(|_| VirtualGeometryGpuError::PageTableExhausted)?;
        let cluster_count = u32::try_from(archive.clusters.len())
            .map_err(|_| VirtualGeometryGpuError::ClusterTableExhausted)?;
        let mut cluster_entries = encode_cluster_entries(archive)?;
        let root_page_count = archive.coarse_root_page_count() as u32;
        let root_cluster_count = archive.pages[..root_page_count as usize]
            .iter()
            .map(|page| page.cluster_count)
            .sum();
        let upload_bytes = archive.coarse_root_page_bytes();
        self.check_frame_budget(upload_bytes, root_page_count)?;
        let root_evictions = self.evictions_needed(root_page_count, &[])?;
        if self.counters.frame_evictions + root_evictions > self.config.max_evictions_per_frame {
            self.counters.denied_uploads += 1;
            return Err(VirtualGeometryGpuError::EvictionBudgetExceeded);
        }
        if self.live_pinned_pages().saturating_add(root_page_count)
            > self.physical_slots.len() as u32
        {
            return Err(VirtualGeometryGpuError::PinnedCapacityExceeded {
                required_pages: self.live_pinned_pages().saturating_add(root_page_count),
                capacity_pages: self.physical_slots.len() as u32,
            });
        }
        if self.find_page_range(page_count).is_none() {
            return Err(VirtualGeometryGpuError::PageTableExhausted);
        }
        if self.find_cluster_range(cluster_count).is_none() {
            return Err(VirtualGeometryGpuError::ClusterTableExhausted);
        }
        let mesh_slot_index = self
            .available_mesh_slot()
            .ok_or(VirtualGeometryGpuError::MeshTableExhausted)?;
        let target_slots = self.plan_physical_slots(root_page_count, &[])?;

        let table_base = self
            .allocate_page_range(page_count)
            .expect("preflight page-table range disappeared");
        let cluster_table_base = self
            .allocate_cluster_range(cluster_count)
            .expect("preflight cluster-table range disappeared");
        let mesh_slot_index = self.reserve_mesh_slot(mesh_slot_index);
        let mesh_id = self.mesh_id(mesh_slot_index);
        for cluster in &mut cluster_entries {
            cluster.payload[3] = mesh_id.raw();
        }
        let mut pages = vec![LogicalPageState::default(); archive.pages.len()];
        for page in pages.iter_mut().take(root_page_count as usize) {
            page.pinned = true;
        }
        self.mesh_slots[mesh_slot_index].lifecycle = MeshLifecycle::Live(LiveMesh {
            asset,
            page_table_base: table_base,
            cluster_table_base,
            pages,
            material_bindings: None,
        });
        let archive = self.live_mesh(mesh_id)?.asset.archive();
        let mesh_entry = GpuVirtualMeshEntry {
            mesh_id: mesh_id.raw(),
            page_table_base: table_base,
            page_count,
            cluster_table_base,
            cluster_count,
            root_cluster_count,
            page_stride_bytes: self.config.page_stride_bytes,
            vertex_encoding: match archive.vertex_encoding {
                bloom_geometry_format::VertexEncoding::Float32 => 1,
                bloom_geometry_format::VertexEncoding::Quantized => 2,
            },
            format_version: archive.format_version,
            flags: GPU_VIRTUAL_MESH_VALID,
            reserved: [0; 2],
        };
        self.write_cluster_entries(queue, cluster_table_base, &cluster_entries);

        for (page_index, physical_slot) in (0..root_page_count).zip(target_slots) {
            self.replace_physical_page(
                queue,
                physical_slot,
                VirtualPageId {
                    mesh: mesh_id,
                    page_index,
                },
                true,
            )?;
        }
        // Publish the generation only after every table/page write has been
        // queued. A subsequent submission can never observe a half mesh.
        self.write_mesh_entry(queue, mesh_slot_index, mesh_entry);
        self.counters.frame_upload_pages += root_page_count;
        self.counters.frame_upload_bytes += upload_bytes;
        self.counters.frame_evictions += root_evictions;
        self.counters.uploads += u64::from(root_page_count);
        self.counters.evictions += u64::from(root_evictions);
        Ok(mesh_id)
    }

    /// Atomically replace every archive material slot with the renderer's
    /// stable global material ID. Validation completes before either CPU or
    /// GPU metadata changes; queue ordering keeps a rebind behind older draws.
    pub fn bind_mesh_materials(
        &mut self,
        queue: &wgpu::Queue,
        mesh_id: VirtualMeshId,
        bindings: &[VirtualMaterialBinding],
    ) -> Result<(), VirtualGeometryGpuError> {
        let mesh_slot_index = self.live_mesh_slot_index(mesh_id)?;
        let (cluster_table_base, required, updated_entries, material_bindings) = {
            let mesh = self.live_mesh(mesh_id)?;
            let required = mesh
                .asset
                .archive()
                .clusters
                .iter()
                .map(|cluster| cluster.material_index)
                .collect::<BTreeSet<_>>();
            let mut supplied = BTreeMap::new();
            for binding in bindings {
                if binding.material_id == 0 {
                    return Err(VirtualGeometryGpuError::InvalidMaterialBinding(
                        binding.source_material_index,
                    ));
                }
                if supplied
                    .insert(binding.source_material_index, binding.material_id)
                    .is_some()
                {
                    return Err(VirtualGeometryGpuError::DuplicateMaterialBinding(
                        binding.source_material_index,
                    ));
                }
            }
            for source in &required {
                if !supplied.contains_key(source) {
                    return Err(VirtualGeometryGpuError::MissingMaterialBinding(*source));
                }
            }
            for source in supplied.keys() {
                if !required.contains(source) {
                    return Err(VirtualGeometryGpuError::UnusedMaterialBinding(*source));
                }
            }
            let updated_entries = mesh
                .asset
                .archive()
                .clusters
                .iter()
                .enumerate()
                .map(|(local_index, cluster)| {
                    let mut entry =
                        self.cluster_entries[mesh.cluster_table_base as usize + local_index];
                    entry.identity[2] = supplied[&cluster.material_index];
                    entry
                })
                .collect::<Vec<_>>();
            (mesh.cluster_table_base, required, updated_entries, supplied)
        };
        debug_assert_eq!(required.len(), material_bindings.len());
        self.write_cluster_entries(queue, cluster_table_base, &updated_entries);
        let MeshLifecycle::Live(mesh) = &mut self.mesh_slots[mesh_slot_index].lifecycle else {
            unreachable!();
        };
        mesh.material_bindings = Some(material_bindings);
        let mut mesh_entry = self.mesh_entries[mesh_slot_index];
        mesh_entry.flags |= GPU_VIRTUAL_MESH_MATERIALS_BOUND;
        self.write_mesh_entry(queue, mesh_slot_index, mesh_entry);
        Ok(())
    }

    pub fn make_group_resident(
        &mut self,
        queue: &wgpu::Queue,
        mesh_id: VirtualMeshId,
        cluster_index: u32,
    ) -> Result<GpuPageTransition, VirtualGeometryGpuError> {
        let missing_pages = self.missing_group_pages(mesh_id, cluster_index)?;
        let payloads = {
            let mesh = self.live_mesh(mesh_id)?;
            missing_pages
                .iter()
                .map(|page_index| {
                    let page_id = VirtualPageId {
                        mesh: mesh_id,
                        page_index: *page_index,
                    };
                    mesh.asset
                        .page_bytes(*page_index as usize)
                        .map(|bytes| (*page_index, bytes.to_vec()))
                        .ok_or(VirtualGeometryGpuError::MissingPage(page_id))
                })
                .collect::<Result<BTreeMap<_, _>, _>>()?
        };
        self.make_group_resident_with_pages(queue, mesh_id, cluster_index, &payloads)
    }

    pub(crate) fn missing_group_pages(
        &self,
        mesh_id: VirtualMeshId,
        cluster_index: u32,
    ) -> Result<Vec<u32>, VirtualGeometryGpuError> {
        let mesh = self.live_mesh(mesh_id)?;
        let group = group_containing(mesh.asset.archive(), cluster_index)?;
        Ok(group_pages(mesh.asset.archive(), group)
            .into_iter()
            .filter(|page| mesh.pages[*page as usize].slot.is_none())
            .collect())
    }

    /// Upload a completely materialized atomic group. The streamer validates
    /// pages again here before any eviction or table mutation, so an I/O race
    /// or corrupt worker result cannot partially replace resident geometry.
    pub(crate) fn make_group_resident_with_pages(
        &mut self,
        queue: &wgpu::Queue,
        mesh_id: VirtualMeshId,
        cluster_index: u32,
        payloads: &BTreeMap<u32, Vec<u8>>,
    ) -> Result<GpuPageTransition, VirtualGeometryGpuError> {
        let (group, required_pages, missing_pages) = {
            let mesh = self.live_mesh(mesh_id)?;
            let group = group_containing(mesh.asset.archive(), cluster_index)?;
            let required_pages = group_pages(mesh.asset.archive(), group);
            let missing_pages = required_pages
                .iter()
                .copied()
                .filter(|page| mesh.pages[*page as usize].slot.is_none())
                .collect::<Vec<_>>();
            (group, required_pages, missing_pages)
        };
        if missing_pages.is_empty() {
            self.touch_pages(mesh_id, &required_pages)?;
            return Ok(GpuPageTransition {
                group,
                uploaded: Vec::new(),
                evicted: Vec::new(),
                resident_slot_bytes: self.resident_slot_bytes(),
            });
        }
        {
            let mesh = self.live_mesh(mesh_id)?;
            for page_index in &missing_pages {
                let page_id = VirtualPageId {
                    mesh: mesh_id,
                    page_index: *page_index,
                };
                let expected = &mesh.asset.archive().pages[*page_index as usize];
                let Some(payload) = payloads.get(page_index) else {
                    return Err(VirtualGeometryGpuError::MissingPagePayload(page_id));
                };
                if payload.len() != expected.payload_bytes as usize
                    || sha256(payload) != expected.sha256
                {
                    return Err(VirtualGeometryGpuError::InvalidPagePayload(page_id));
                }
            }
        }
        let upload_bytes = {
            let mesh = self.live_mesh(mesh_id)?;
            missing_pages
                .iter()
                .map(|page| mesh.asset.archive().pages[*page as usize].payload_bytes as u64)
                .sum()
        };
        self.check_frame_budget(upload_bytes, missing_pages.len() as u32)?;
        let protected = required_pages
            .iter()
            .map(|page_index| VirtualPageId {
                mesh: mesh_id,
                page_index: *page_index,
            })
            .collect::<Vec<_>>();
        let target_slots = self.plan_physical_slots(missing_pages.len() as u32, &protected)?;
        let evicted = target_slots
            .iter()
            .filter_map(|slot| self.physical_slots[*slot as usize].owner)
            .collect::<Vec<_>>();
        if self.counters.frame_evictions + evicted.len() as u32
            > self.config.max_evictions_per_frame
        {
            self.counters.denied_uploads += 1;
            return Err(VirtualGeometryGpuError::EvictionBudgetExceeded);
        }

        let uploaded = missing_pages
            .into_iter()
            .zip(target_slots)
            .map(|(page_index, physical_slot)| {
                let page_id = VirtualPageId {
                    mesh: mesh_id,
                    page_index,
                };
                self.replace_physical_page_with_payload(
                    queue,
                    physical_slot,
                    page_id,
                    false,
                    &payloads[&page_index],
                )?;
                Ok((page_id, physical_slot))
            })
            .collect::<Result<Vec<_>, VirtualGeometryGpuError>>()?;
        self.touch_pages(mesh_id, &required_pages)?;
        self.counters.frame_upload_pages += uploaded.len() as u32;
        self.counters.frame_upload_bytes += upload_bytes;
        self.counters.frame_evictions += evicted.len() as u32;
        self.counters.uploads += uploaded.len() as u64;
        self.counters.evictions += evicted.len() as u64;
        Ok(GpuPageTransition {
            group,
            uploaded,
            evicted,
            resident_slot_bytes: self.resident_slot_bytes(),
        })
    }

    pub fn resolve_cluster(
        &mut self,
        mesh_id: VirtualMeshId,
        cluster_index: u32,
    ) -> Result<Option<ResolvedClusterGroup>, VirtualGeometryGpuError> {
        let (requested_group, requested_lod) = {
            let mesh = self.live_mesh(mesh_id)?;
            let group = group_containing(mesh.asset.archive(), cluster_index)?;
            (group, group_lod(mesh.asset.archive(), group))
        };
        let mut group = requested_group;
        loop {
            let (resident, parent, lod, pages) = {
                let mesh = self.live_mesh(mesh_id)?;
                let pages = group_pages(mesh.asset.archive(), group);
                let resident = pages
                    .iter()
                    .all(|page| mesh.pages[*page as usize].slot.is_some());
                (
                    resident,
                    parent_group(mesh.asset.archive(), group),
                    group_lod(mesh.asset.archive(), group),
                    pages,
                )
            };
            if resident {
                self.touch_pages(mesh_id, &pages)?;
                let fallback_levels = lod.saturating_sub(requested_lod);
                if fallback_levels == 0 {
                    self.counters.exact_resolutions += 1;
                } else {
                    self.counters.fallback_resolutions += 1;
                }
                return Ok(Some(ResolvedClusterGroup {
                    group,
                    lod_level: lod,
                    requested_lod_level: requested_lod,
                    fallback_levels,
                }));
            }
            let Some(parent) = parent else {
                self.counters.unresolved_requests += 1;
                return Ok(None);
            };
            group = parent;
        }
    }

    /// Invalidate a mesh immediately, but delay ID, slot, and table-range reuse
    /// until queue completion proves no older GPU command can reference them.
    pub fn retire_mesh(
        &mut self,
        queue: &wgpu::Queue,
        mesh_id: VirtualMeshId,
    ) -> Result<u64, VirtualGeometryGpuError> {
        let mesh_slot_index = self.live_mesh_slot_index(mesh_id)?;
        let (page_table_base, page_count, cluster_table_base, cluster_count, physical_slots) = {
            let MeshLifecycle::Live(mesh) = &self.mesh_slots[mesh_slot_index].lifecycle else {
                unreachable!();
            };
            (
                mesh.page_table_base,
                mesh.pages.len() as u32,
                mesh.cluster_table_base,
                mesh.asset.archive().clusters.len() as u32,
                mesh.pages
                    .iter()
                    .filter_map(|page| page.slot)
                    .collect::<Vec<_>>(),
            )
        };
        self.write_mesh_entry(queue, mesh_slot_index, GpuVirtualMeshEntry::default());
        for page_index in 0..page_count {
            self.write_page_entry(
                queue,
                page_table_base + page_index,
                GpuVirtualPageEntry::default(),
            );
        }
        // Flush the metadata invalidation before establishing the retirement
        // fence. This also orders it after every previously submitted draw.
        let _ = queue.submit(std::iter::empty());
        let completion_epoch = self.completion.track_submitted_work(queue);
        for slot in &physical_slots {
            let physical = &mut self.physical_slots[*slot as usize];
            physical.pinned = false;
            physical.retiring_until = Some(completion_epoch);
        }
        self.mesh_slots[mesh_slot_index].lifecycle = MeshLifecycle::Retiring(RetiringMesh {
            completion_epoch,
            page_table_base,
            page_count,
            cluster_table_base,
            cluster_count,
            physical_slots,
        });
        Ok(completion_epoch)
    }

    pub fn collect_completed(&mut self) -> usize {
        let completed = self.completion.completed_epoch();
        let mut reclaimed = Vec::new();
        for (index, slot) in self.mesh_slots.iter().enumerate() {
            if matches!(
                &slot.lifecycle,
                MeshLifecycle::Retiring(retired) if retired.completion_epoch <= completed
            ) {
                reclaimed.push(index);
            }
        }
        for index in &reclaimed {
            let retired = match std::mem::replace(
                &mut self.mesh_slots[*index].lifecycle,
                MeshLifecycle::Free,
            ) {
                MeshLifecycle::Retiring(retired) => retired,
                _ => unreachable!(),
            };
            for physical_slot in retired.physical_slots {
                self.physical_slots[physical_slot as usize] = PhysicalSlot::default();
            }
            self.release_page_range(retired.page_table_base, retired.page_count);
            self.release_cluster_range(retired.cluster_table_base, retired.cluster_count);
            self.mesh_slots[*index].generation =
                (self.mesh_slots[*index].generation + 1) & ID_GENERATION_MASK;
            self.free_mesh_slots.push(*index);
        }
        reclaimed.len()
    }

    pub fn is_page_resident(&self, page: VirtualPageId) -> Result<bool, VirtualGeometryGpuError> {
        let mesh = self.live_mesh(page.mesh)?;
        mesh.pages
            .get(page.page_index as usize)
            .map(|state| state.slot.is_some())
            .ok_or(VirtualGeometryGpuError::MissingPage(page))
    }

    pub fn mesh_entry(
        &self,
        mesh_id: VirtualMeshId,
    ) -> Result<GpuVirtualMeshEntry, VirtualGeometryGpuError> {
        let index = self.live_mesh_slot_index(mesh_id)?;
        Ok(self.mesh_entries[index])
    }

    pub fn page_entry(
        &self,
        page: VirtualPageId,
    ) -> Result<GpuVirtualPageEntry, VirtualGeometryGpuError> {
        let mesh = self.live_mesh(page.mesh)?;
        if page.page_index as usize >= mesh.pages.len() {
            return Err(VirtualGeometryGpuError::MissingPage(page));
        }
        Ok(self.page_entries[(mesh.page_table_base + page.page_index) as usize])
    }

    pub fn telemetry(&self) -> GpuVirtualGeometryTelemetry {
        let mut telemetry = GpuVirtualGeometryTelemetry {
            frame: self.frame,
            capacity_bytes: self.config.capacity_bytes,
            page_table_bytes: self.page_table_bytes(),
            mesh_table_bytes: self.mesh_table_bytes(),
            cluster_table_bytes: self.cluster_table_bytes(),
            total_gpu_bytes: self
                .config
                .capacity_bytes
                .saturating_add(self.page_table_bytes())
                .saturating_add(self.mesh_table_bytes())
                .saturating_add(self.cluster_table_bytes()),
            slot_count: self.physical_slots.len() as u32,
            frame_upload_pages: self.counters.frame_upload_pages,
            frame_upload_bytes: self.counters.frame_upload_bytes,
            frame_evictions: self.counters.frame_evictions,
            uploads: self.counters.uploads,
            evictions: self.counters.evictions,
            denied_uploads: self.counters.denied_uploads,
            exact_resolutions: self.counters.exact_resolutions,
            fallback_resolutions: self.counters.fallback_resolutions,
            unresolved_requests: self.counters.unresolved_requests,
            ..GpuVirtualGeometryTelemetry::default()
        };
        for mesh_slot in &self.mesh_slots {
            match &mesh_slot.lifecycle {
                MeshLifecycle::Live(mesh) => {
                    telemetry.active_meshes += 1;
                    telemetry.live_page_records += mesh.pages.len() as u32;
                    telemetry.live_cluster_records += mesh.asset.archive().clusters.len() as u32;
                    for (page_index, page) in mesh.pages.iter().enumerate() {
                        if page.slot.is_none() {
                            continue;
                        }
                        telemetry.resident_pages += 1;
                        telemetry.resident_slot_bytes += u64::from(self.config.page_stride_bytes);
                        telemetry.resident_payload_bytes +=
                            u64::from(mesh.asset.archive().pages[page_index].payload_bytes);
                        if page.pinned {
                            telemetry.pinned_pages += 1;
                            telemetry.pinned_slot_bytes += u64::from(self.config.page_stride_bytes);
                        }
                    }
                }
                MeshLifecycle::Retiring(mesh) => {
                    telemetry.retiring_slots += mesh.physical_slots.len() as u32;
                }
                MeshLifecycle::Free => {}
            }
        }
        telemetry
    }

    pub fn physical_buffer(&self) -> &wgpu::Buffer {
        &self.physical_buffer
    }

    pub fn mesh_table_buffer(&self) -> &wgpu::Buffer {
        &self.mesh_table_buffer
    }

    pub fn page_table_buffer(&self) -> &wgpu::Buffer {
        &self.page_table_buffer
    }

    pub fn cluster_table_buffer(&self) -> &wgpu::Buffer {
        &self.cluster_table_buffer
    }

    pub fn cluster_entry(
        &self,
        mesh_id: VirtualMeshId,
        cluster_index: u32,
    ) -> Result<GpuVirtualClusterEntry, VirtualGeometryGpuError> {
        let mesh = self.live_mesh(mesh_id)?;
        if cluster_index as usize >= mesh.asset.archive().clusters.len() {
            return Err(VirtualGeometryGpuError::Residency(
                ResidencyError::MissingCluster(cluster_index),
            ));
        }
        Ok(self.cluster_entries[(mesh.cluster_table_base + cluster_index) as usize])
    }

    pub(crate) fn asset(
        &self,
        mesh_id: VirtualMeshId,
    ) -> Result<&Arc<VirtualGeometryAsset>, VirtualGeometryGpuError> {
        Ok(&self.live_mesh(mesh_id)?.asset)
    }

    #[cfg(test)]
    pub(super) fn bound_material_id(
        &self,
        mesh_id: VirtualMeshId,
        source_material_index: Option<u32>,
    ) -> Result<u32, VirtualGeometryGpuError> {
        Ok(self
            .live_mesh(mesh_id)?
            .material_bindings
            .as_ref()
            .and_then(|bindings| bindings.get(&source_material_index))
            .copied()
            .unwrap_or(0))
    }

    pub const fn config(&self) -> GpuVirtualGeometryConfig {
        self.config
    }

    pub(super) const fn id(&self) -> u64 {
        self.id
    }

    fn check_frame_budget(
        &mut self,
        bytes: u64,
        pages: u32,
    ) -> Result<(), VirtualGeometryGpuError> {
        if self.counters.frame_upload_bytes.saturating_add(bytes)
            > self.config.max_upload_bytes_per_frame
            || self.counters.frame_upload_pages.saturating_add(pages)
                > self.config.max_upload_pages_per_frame
        {
            self.counters.denied_uploads += 1;
            return Err(VirtualGeometryGpuError::UploadBudgetExceeded {
                requested_bytes: bytes,
                remaining_bytes: self
                    .config
                    .max_upload_bytes_per_frame
                    .saturating_sub(self.counters.frame_upload_bytes),
                requested_pages: pages,
                remaining_pages: self
                    .config
                    .max_upload_pages_per_frame
                    .saturating_sub(self.counters.frame_upload_pages),
            });
        }
        Ok(())
    }

    fn replace_physical_page(
        &mut self,
        queue: &wgpu::Queue,
        physical_slot: u32,
        page_id: VirtualPageId,
        pinned: bool,
    ) -> Result<(), VirtualGeometryGpuError> {
        let payload = {
            let mesh = self.live_mesh(page_id.mesh)?;
            mesh.asset
                .page_bytes(page_id.page_index as usize)
                .ok_or(VirtualGeometryGpuError::MissingPage(page_id))?
                .to_vec()
        };
        self.replace_physical_page_with_payload(queue, physical_slot, page_id, pinned, &payload)
    }

    fn replace_physical_page_with_payload(
        &mut self,
        queue: &wgpu::Queue,
        physical_slot: u32,
        page_id: VirtualPageId,
        pinned: bool,
        payload: &[u8],
    ) -> Result<(), VirtualGeometryGpuError> {
        let previous = self.physical_slots[physical_slot as usize].owner;
        if let Some(previous) = previous {
            let (table_index, previous_state) = self.page_state_location(previous)?;
            previous_state.slot = None;
            self.write_page_entry(queue, table_index, GpuVirtualPageEntry::default());
        }
        let table_index = self.live_mesh(page_id.mesh)?.page_table_base + page_id.page_index;
        let offset = u64::from(physical_slot) * u64::from(self.config.page_stride_bytes);
        write_buffer_padded(queue, &self.physical_buffer, offset, payload);
        self.clock = self.clock.saturating_add(1);
        let clock = self.clock;
        let (_, page_state) = self.page_state_location(page_id)?;
        page_state.slot = Some(physical_slot);
        page_state.pinned = pinned;
        page_state.last_use = clock;
        self.physical_slots[physical_slot as usize] = PhysicalSlot {
            owner: Some(page_id),
            pinned,
            last_use: clock,
            retiring_until: None,
        };
        self.write_page_entry(
            queue,
            table_index,
            GpuVirtualPageEntry {
                slot_plus_one: physical_slot + 1,
                payload_bytes: payload.len() as u32,
                mesh_id: page_id.mesh.raw(),
                flags: GPU_VIRTUAL_PAGE_RESIDENT | if pinned { GPU_VIRTUAL_PAGE_PINNED } else { 0 },
            },
        );
        Ok(())
    }

    fn touch_pages(
        &mut self,
        mesh_id: VirtualMeshId,
        page_indices: &[u32],
    ) -> Result<(), VirtualGeometryGpuError> {
        self.clock = self.clock.saturating_add(1);
        let clock = self.clock;
        let mesh_slot_index = self.live_mesh_slot_index(mesh_id)?;
        let slots = {
            let MeshLifecycle::Live(mesh) = &mut self.mesh_slots[mesh_slot_index].lifecycle else {
                unreachable!();
            };
            page_indices
                .iter()
                .filter_map(|page_index| {
                    mesh.pages.get_mut(*page_index as usize).and_then(|page| {
                        page.last_use = clock;
                        page.slot
                    })
                })
                .collect::<Vec<_>>()
        };
        for slot in slots {
            self.physical_slots[slot as usize].last_use = clock;
        }
        Ok(())
    }

    fn plan_physical_slots(
        &self,
        count: u32,
        protected: &[VirtualPageId],
    ) -> Result<Vec<u32>, VirtualGeometryGpuError> {
        let mut selected = self
            .physical_slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.owner.is_none() && slot.retiring_until.is_none())
            .map(|(index, _)| index as u32)
            .take(count as usize)
            .collect::<Vec<_>>();
        if selected.len() < count as usize {
            let mut evictable = self
                .physical_slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| {
                    !slot.pinned
                        && slot.retiring_until.is_none()
                        && slot.owner.is_some_and(|owner| !protected.contains(&owner))
                })
                .map(|(index, slot)| (slot.last_use, index as u32))
                .collect::<Vec<_>>();
            evictable.sort_unstable();
            selected.extend(
                evictable
                    .into_iter()
                    .map(|(_, index)| index)
                    .take(count as usize - selected.len()),
            );
        }
        if selected.len() != count as usize {
            return Err(VirtualGeometryGpuError::PhysicalPoolExhausted {
                requested_pages: count,
                available_pages: selected.len() as u32,
            });
        }
        Ok(selected)
    }

    fn evictions_needed(
        &self,
        count: u32,
        protected: &[VirtualPageId],
    ) -> Result<u32, VirtualGeometryGpuError> {
        Ok(self
            .plan_physical_slots(count, protected)?
            .iter()
            .filter(|slot| self.physical_slots[**slot as usize].owner.is_some())
            .count() as u32)
    }

    fn live_mesh(&self, id: VirtualMeshId) -> Result<&LiveMesh, VirtualGeometryGpuError> {
        let index = self.live_mesh_slot_index(id)?;
        let MeshLifecycle::Live(mesh) = &self.mesh_slots[index].lifecycle else {
            unreachable!();
        };
        Ok(mesh)
    }

    fn live_mesh_slot_index(&self, id: VirtualMeshId) -> Result<usize, VirtualGeometryGpuError> {
        if id.is_fallback() {
            return Err(VirtualGeometryGpuError::FallbackMesh);
        }
        let index = id.descriptor_index() as usize - 1;
        let Some(slot) = self.mesh_slots.get(index) else {
            return Err(VirtualGeometryGpuError::StaleMesh(id));
        };
        if slot.generation != id.generation() {
            return Err(VirtualGeometryGpuError::StaleMesh(id));
        }
        match slot.lifecycle {
            MeshLifecycle::Live(_) => Ok(index),
            MeshLifecycle::Retiring(_) => Err(VirtualGeometryGpuError::RetiringMesh(id)),
            MeshLifecycle::Free => Err(VirtualGeometryGpuError::StaleMesh(id)),
        }
    }

    fn page_state_location(
        &mut self,
        page: VirtualPageId,
    ) -> Result<(u32, &mut LogicalPageState), VirtualGeometryGpuError> {
        let index = self.live_mesh_slot_index(page.mesh)?;
        let MeshLifecycle::Live(mesh) = &mut self.mesh_slots[index].lifecycle else {
            unreachable!();
        };
        let state = mesh
            .pages
            .get_mut(page.page_index as usize)
            .ok_or(VirtualGeometryGpuError::MissingPage(page))?;
        Ok((mesh.page_table_base + page.page_index, state))
    }

    fn available_mesh_slot(&self) -> Option<usize> {
        self.free_mesh_slots.last().copied().or_else(|| {
            (self.mesh_slots.len() < self.config.max_meshes as usize)
                .then_some(self.mesh_slots.len())
        })
    }

    fn reserve_mesh_slot(&mut self, expected: usize) -> usize {
        let index = if let Some(index) = self.free_mesh_slots.pop() {
            index
        } else {
            let index = self.mesh_slots.len();
            self.mesh_slots.push(MeshSlot {
                generation: 0,
                lifecycle: MeshLifecycle::Free,
            });
            index
        };
        debug_assert_eq!(index, expected);
        index
    }

    fn mesh_id(&self, index: usize) -> VirtualMeshId {
        VirtualMeshId::from_parts(index, self.mesh_slots[index].generation)
            .expect("configured mesh table fits stable ID space")
    }

    fn find_page_range(&self, count: u32) -> Option<u32> {
        self.free_page_ranges
            .iter()
            .find(|range| range.count >= count)
            .map(|range| range.start)
    }

    fn find_cluster_range(&self, count: u32) -> Option<u32> {
        self.free_cluster_ranges
            .iter()
            .find(|range| range.count >= count)
            .map(|range| range.start)
    }

    fn allocate_page_range(&mut self, count: u32) -> Option<u32> {
        let index = self
            .free_page_ranges
            .iter()
            .position(|range| range.count >= count)?;
        let start = self.free_page_ranges[index].start;
        self.free_page_ranges[index].start += count;
        self.free_page_ranges[index].count -= count;
        if self.free_page_ranges[index].count == 0 {
            self.free_page_ranges.remove(index);
        }
        Some(start)
    }

    fn allocate_cluster_range(&mut self, count: u32) -> Option<u32> {
        allocate_table_range(&mut self.free_cluster_ranges, count)
    }

    fn release_page_range(&mut self, start: u32, count: u32) {
        release_table_range(&mut self.free_page_ranges, start, count);
    }

    fn release_cluster_range(&mut self, start: u32, count: u32) {
        release_table_range(&mut self.free_cluster_ranges, start, count);
    }

    fn write_mesh_entry(&mut self, queue: &wgpu::Queue, index: usize, entry: GpuVirtualMeshEntry) {
        self.mesh_entries[index] = entry;
        queue.write_buffer(
            &self.mesh_table_buffer,
            index as u64 * std::mem::size_of::<GpuVirtualMeshEntry>() as u64,
            bytemuck::bytes_of(&entry),
        );
    }

    fn write_page_entry(&mut self, queue: &wgpu::Queue, index: u32, entry: GpuVirtualPageEntry) {
        self.page_entries[index as usize] = entry;
        queue.write_buffer(
            &self.page_table_buffer,
            u64::from(index) * std::mem::size_of::<GpuVirtualPageEntry>() as u64,
            bytemuck::bytes_of(&entry),
        );
    }

    fn write_cluster_entries(
        &mut self,
        queue: &wgpu::Queue,
        first: u32,
        entries: &[GpuVirtualClusterEntry],
    ) {
        let start = first as usize;
        self.cluster_entries[start..start + entries.len()].copy_from_slice(entries);
        queue.write_buffer(
            &self.cluster_table_buffer,
            u64::from(first) * std::mem::size_of::<GpuVirtualClusterEntry>() as u64,
            bytemuck::cast_slice(entries),
        );
    }

    fn live_pinned_pages(&self) -> u32 {
        self.mesh_slots
            .iter()
            .filter_map(|slot| match &slot.lifecycle {
                MeshLifecycle::Live(mesh) => Some(mesh),
                _ => None,
            })
            .flat_map(|mesh| mesh.pages.iter())
            .filter(|page| page.pinned && page.slot.is_some())
            .count() as u32
    }

    fn resident_slot_bytes(&self) -> u64 {
        self.physical_slots
            .iter()
            .filter(|slot| slot.owner.is_some() && slot.retiring_until.is_none())
            .count() as u64
            * u64::from(self.config.page_stride_bytes)
    }

    fn mesh_table_bytes(&self) -> u64 {
        u64::from(self.config.max_meshes) * GPU_MESH_ENTRY_WORDS * GPU_WORD_BYTES
    }

    fn page_table_bytes(&self) -> u64 {
        u64::from(self.config.max_page_records) * GPU_PAGE_ENTRY_WORDS * GPU_WORD_BYTES
    }

    fn cluster_table_bytes(&self) -> u64 {
        u64::from(self.config.max_cluster_records) * GPU_CLUSTER_ENTRY_WORDS * GPU_WORD_BYTES
    }
}

fn validate_config(
    device: &wgpu::Device,
    config: GpuVirtualGeometryConfig,
) -> Result<(), VirtualGeometryGpuError> {
    let stride = config.page_stride_bytes;
    if !(MIN_PAGE_BYTES..=MAX_PAGE_BYTES).contains(&stride)
        || !stride.is_power_of_two()
        || config.capacity_bytes == 0
        || config.capacity_bytes > u64::from(u32::MAX)
        || !config.capacity_bytes.is_multiple_of(u64::from(stride))
        || config.max_meshes == 0
        || config.max_meshes > ID_SLOT_MASK
        || config.max_page_records == 0
        || config.max_cluster_records == 0
        || config.max_clusters_per_group == 0
        || config.max_hierarchy_levels == 0
        || config.max_hierarchy_levels > MAX_GPU_HIERARCHY_LEVELS
        || config.max_upload_bytes_per_frame < u64::from(stride)
        || config.max_upload_pages_per_frame == 0
    {
        return Err(VirtualGeometryGpuError::InvalidConfig);
    }
    let mesh_bytes = u64::from(config.max_meshes) * GPU_MESH_ENTRY_WORDS * GPU_WORD_BYTES;
    let page_bytes = u64::from(config.max_page_records) * GPU_PAGE_ENTRY_WORDS * GPU_WORD_BYTES;
    let cluster_bytes =
        u64::from(config.max_cluster_records) * GPU_CLUSTER_ENTRY_WORDS * GPU_WORD_BYTES;
    let limits = device.limits();
    for (label, bytes) in [
        ("physical page pool", config.capacity_bytes),
        ("mesh table", mesh_bytes),
        ("page table", page_bytes),
        ("cluster table", cluster_bytes),
    ] {
        if bytes > limits.max_buffer_size || bytes > limits.max_storage_buffer_binding_size {
            return Err(VirtualGeometryGpuError::DeviceLimitExceeded {
                resource: label,
                requested_bytes: bytes,
                max_buffer_bytes: limits
                    .max_buffer_size
                    .min(limits.max_storage_buffer_binding_size),
            });
        }
    }
    Ok(())
}

fn encode_cluster_entries(
    archive: &bloom_geometry_format::GeometryArchive,
) -> Result<Vec<GpuVirtualClusterEntry>, VirtualGeometryGpuError> {
    archive
        .clusters
        .iter()
        .enumerate()
        .map(|(cluster_index, cluster)| {
            let page = &archive.pages[cluster.page_index as usize];
            let vertex_offset = cluster
                .vertex_offset
                .checked_sub(page.payload_offset)
                .and_then(|offset| u32::try_from(offset).ok())
                .ok_or(VirtualGeometryGpuError::InvalidClusterAddress(
                    cluster_index as u32,
                ))?;
            let index_offset = cluster
                .index_offset
                .checked_sub(page.payload_offset)
                .and_then(|offset| u32::try_from(offset).ok())
                .ok_or(VirtualGeometryGpuError::InvalidClusterAddress(
                    cluster_index as u32,
                ))?;
            Ok(GpuVirtualClusterEntry {
                aabb_min_error: [
                    cluster.aabb_min[0],
                    cluster.aabb_min[1],
                    cluster.aabb_min[2],
                    cluster.geometric_error,
                ],
                aabb_max_radius: [
                    cluster.aabb_max[0],
                    cluster.aabb_max[1],
                    cluster.aabb_max[2],
                    cluster.sphere_radius,
                ],
                sphere: [
                    cluster.sphere_center[0],
                    cluster.sphere_center[1],
                    cluster.sphere_center[2],
                    0.0,
                ],
                normal_cone: [
                    cluster.normal_cone_axis[0],
                    cluster.normal_cone_axis[1],
                    cluster.normal_cone_axis[2],
                    cluster.normal_cone_cutoff,
                ],
                identity: [
                    cluster.mesh_index,
                    cluster.primitive_index,
                    0,
                    cluster.flags,
                ],
                page_lod_counts: [
                    cluster.page_index,
                    cluster.lod_level,
                    cluster.vertex_count,
                    cluster.triangle_count,
                ],
                payload: [vertex_offset, index_offset, cluster.vertex_stride, 0],
                relations: [
                    cluster.parent,
                    cluster.parent_count,
                    cluster.first_child,
                    cluster.child_count,
                ],
            })
        })
        .collect()
}

fn allocate_table_range(ranges: &mut Vec<FreeRange>, count: u32) -> Option<u32> {
    let index = ranges.iter().position(|range| range.count >= count)?;
    let start = ranges[index].start;
    ranges[index].start += count;
    ranges[index].count -= count;
    if ranges[index].count == 0 {
        ranges.remove(index);
    }
    Some(start)
}

fn release_table_range(ranges: &mut Vec<FreeRange>, start: u32, count: u32) {
    ranges.push(FreeRange { start, count });
    ranges.sort_unstable_by_key(|range| range.start);
    let mut merged: Vec<FreeRange> = Vec::with_capacity(ranges.len());
    for range in ranges.drain(..) {
        if let Some(previous) = merged.last_mut() {
            if previous.start + previous.count == range.start {
                previous.count += range.count;
                continue;
            }
        }
        merged.push(range);
    }
    *ranges = merged;
}

fn create_zeroed_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn write_buffer_padded(queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, bytes: &[u8]) {
    if bytes.len().is_multiple_of(4) {
        queue.write_buffer(buffer, offset, bytes);
    } else {
        let mut padded = Vec::with_capacity(bytes.len().next_multiple_of(4));
        padded.extend_from_slice(bytes);
        padded.resize(bytes.len().next_multiple_of(4), 0);
        queue.write_buffer(buffer, offset, &padded);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VirtualGeometryGpuError {
    InvalidConfig,
    DeviceLimitExceeded {
        resource: &'static str,
        requested_bytes: u64,
        max_buffer_bytes: u64,
    },
    MissingCoarseFallback,
    PageStrideExceeded {
        archive_bytes: u32,
        pool_bytes: u32,
    },
    MeshTableExhausted,
    PageTableExhausted,
    ClusterTableExhausted,
    DuplicateMaterialBinding(Option<u32>),
    InvalidMaterialBinding(Option<u32>),
    MissingMaterialBinding(Option<u32>),
    UnusedMaterialBinding(Option<u32>),
    InvalidClusterAddress(u32),
    TraversalLimitExceeded {
        cluster: u32,
        parent_count: u32,
        child_count: u32,
        lod_level: u32,
    },
    PinnedCapacityExceeded {
        required_pages: u32,
        capacity_pages: u32,
    },
    PhysicalPoolExhausted {
        requested_pages: u32,
        available_pages: u32,
    },
    UploadBudgetExceeded {
        requested_bytes: u64,
        remaining_bytes: u64,
        requested_pages: u32,
        remaining_pages: u32,
    },
    EvictionBudgetExceeded,
    FallbackMesh,
    StaleMesh(VirtualMeshId),
    RetiringMesh(VirtualMeshId),
    MissingPage(VirtualPageId),
    MissingPagePayload(VirtualPageId),
    InvalidPagePayload(VirtualPageId),
    Residency(ResidencyError),
}

impl From<ResidencyError> for VirtualGeometryGpuError {
    fn from(value: ResidencyError) -> Self {
        Self::Residency(value)
    }
}

impl fmt::Display for VirtualGeometryGpuError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(formatter, "invalid virtual-geometry GPU pool config"),
            Self::DeviceLimitExceeded {
                resource,
                requested_bytes,
                max_buffer_bytes,
            } => write!(
                formatter,
                "virtual-geometry {resource} requires {requested_bytes} bytes but the device limit is {max_buffer_bytes}"
            ),
            Self::MissingCoarseFallback => {
                write!(formatter, "virtual mesh has no always-resident coarse fallback pages")
            }
            Self::PageStrideExceeded {
                archive_bytes,
                pool_bytes,
            } => write!(
                formatter,
                "archive page budget {archive_bytes} exceeds pool stride {pool_bytes}"
            ),
            Self::MeshTableExhausted => write!(formatter, "virtual mesh table is full"),
            Self::PageTableExhausted => write!(formatter, "virtual page table is full"),
            Self::ClusterTableExhausted => write!(formatter, "virtual cluster table is full"),
            Self::DuplicateMaterialBinding(source) => write!(
                formatter,
                "virtual mesh material source {source:?} was bound more than once"
            ),
            Self::InvalidMaterialBinding(source) => write!(
                formatter,
                "virtual mesh material source {source:?} was bound to fallback ID zero"
            ),
            Self::MissingMaterialBinding(source) => write!(
                formatter,
                "virtual mesh material source {source:?} has no GPU material binding"
            ),
            Self::UnusedMaterialBinding(source) => write!(
                formatter,
                "GPU material binding for virtual source {source:?} is unused"
            ),
            Self::InvalidClusterAddress(cluster) => write!(
                formatter,
                "virtual cluster {cluster} has no page-local GPU payload address"
            ),
            Self::TraversalLimitExceeded {
                cluster,
                parent_count,
                child_count,
                lod_level,
            } => write!(
                formatter,
                "virtual cluster {cluster} exceeds traversal limits: parents={parent_count}, children={child_count}, lod={lod_level}"
            ),
            Self::PinnedCapacityExceeded {
                required_pages,
                capacity_pages,
            } => write!(
                formatter,
                "pinned roots require {required_pages} slots but the pool has {capacity_pages}"
            ),
            Self::PhysicalPoolExhausted {
                requested_pages,
                available_pages,
            } => write!(
                formatter,
                "virtual page request needs {requested_pages} slots but only {available_pages} are replaceable"
            ),
            Self::UploadBudgetExceeded {
                requested_bytes,
                remaining_bytes,
                requested_pages,
                remaining_pages,
            } => write!(
                formatter,
                "virtual page upload requests {requested_pages} pages/{requested_bytes} bytes but this frame has {remaining_pages} pages/{remaining_bytes} bytes left"
            ),
            Self::EvictionBudgetExceeded => {
                write!(formatter, "virtual page eviction budget exhausted for this frame")
            }
            Self::FallbackMesh => write!(formatter, "fallback virtual mesh has no archive"),
            Self::StaleMesh(id) => write!(formatter, "virtual mesh ID {} is stale", id.raw()),
            Self::RetiringMesh(id) => {
                write!(formatter, "virtual mesh ID {} is retiring", id.raw())
            }
            Self::MissingPage(page) => write!(
                formatter,
                "virtual mesh {} has no page {}",
                page.mesh.raw(),
                page.page_index
            ),
            Self::MissingPagePayload(page) => write!(
                formatter,
                "virtual mesh {} page {} was not supplied for an atomic upload",
                page.mesh.raw(),
                page.page_index
            ),
            Self::InvalidPagePayload(page) => write!(
                formatter,
                "virtual mesh {} page {} failed its size or SHA-256 contract",
                page.mesh.raw(),
                page.page_index
            ),
            Self::Residency(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for VirtualGeometryGpuError {}
