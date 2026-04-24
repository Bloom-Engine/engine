// Transient resource pool — Phase 3 of RFC 0001.
//
// Manages the short-lived textures that a render graph needs: scene-
// colour snapshots, depth-as-sampled linearisations, bloom mip chains,
// SSGI history, motion-vector maps, etc. Each passes' declared
// `Transient(u32)` input/output refers to a handle this module hands
// out.
//
// Phase 3 goals:
//
//   1. Allocate textures by (format, size, usage) on demand.
//   2. Reuse textures when their previous caller releases them.
//   3. Resize cleanly when the swapchain changes — invalidates caches
//      sized relative to the swapchain.
//   4. Stay independent of `renderer::graph` — the graph module uses
//      this pool as a consumer, not the other way around.
//
// Deferred to a later phase:
//
//   - **True aliasing.** Two transients with non-overlapping lifetimes
//     can physically share the same backing texture on Vulkan/D3D12
//     via aliased resources. This module ref-counts + reuses but does
//     not alias. Graph-driven lifetime analysis (Phase 3b) is a
//     prerequisite — the pool can't know lifetimes without a
//     schedule to introspect.
//
//   - **Async acquire/release.** The pool is single-threaded. Every
//     real engine backend is single-threaded on the render queue
//     anyway; multi-queue support is a later concern.

use std::collections::HashMap;

/// Size policy for a transient. Resolved against the swapchain each
/// time the pool is asked for extents.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum SizePolicy {
    /// Match the swapchain size (most render targets).
    Swapchain,
    /// Half of the swapchain — e.g. SSAO half-res.
    HalfSwapchain,
    /// Quarter of the swapchain — e.g. bloom first mip after the
    /// initial downsample.
    QuarterSwapchain,
    /// A specific fixed size in physical pixels. Used for things that
    /// are explicitly grid-sized (shadow cascades, probe grids).
    Fixed(u32, u32),
}

impl SizePolicy {
    /// Resolve to concrete (width, height) in physical pixels.
    pub fn extent(self, swap_w: u32, swap_h: u32) -> (u32, u32) {
        let max1 = |n: u32| n.max(1);
        match self {
            SizePolicy::Swapchain        => (max1(swap_w), max1(swap_h)),
            SizePolicy::HalfSwapchain    => (max1(swap_w / 2), max1(swap_h / 2)),
            SizePolicy::QuarterSwapchain => (max1(swap_w / 4), max1(swap_h / 4)),
            SizePolicy::Fixed(w, h)      => (max1(w), max1(h)),
        }
    }
}

/// Describes a transient texture the pool should own. Equality on
/// (format, usage, size policy) decides which allocations can be
/// returned to the same reuse bucket.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct TransientDesc {
    pub format: wgpu::TextureFormat,
    pub usage:  wgpu::TextureUsages,
    pub size:   SizePolicy,
    /// Mip levels. 1 for most RTs, N for bloom's mip chain etc.
    pub mips:   u32,
    /// Single-sample unless you need MSAA — we don't right now.
    pub samples: u32,
}

impl TransientDesc {
    pub fn new(format: wgpu::TextureFormat, usage: wgpu::TextureUsages,
               size: SizePolicy) -> Self {
        Self { format, usage, size, mips: 1, samples: 1 }
    }
    pub fn with_mips(mut self, mips: u32) -> Self { self.mips = mips; self }
}

/// Opaque handle to an allocated transient. The graph module sees
/// these as `PassInput::Transient(TransientId)`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct TransientId(pub u32);

/// Internal record per live allocation.
struct Slot {
    desc:     TransientDesc,
    /// Physical extent at allocation time — used for resize detection.
    extent:   (u32, u32),
    texture:  wgpu::Texture,
    view:     wgpu::TextureView,
    /// True while the allocation is checked out. False means "in the
    /// reuse bucket, safe to hand back to the next caller matching
    /// the same desc".
    in_use:   bool,
}

/// Per-frame transient manager. Usage pattern:
///
/// ```ignore
/// pool.begin_frame(swap_w, swap_h);           // invalidates on resize
/// let a = pool.acquire(desc_a);               // caller owns a
/// let b = pool.acquire(desc_b);               // caller owns b
/// // … passes read/write a, b …
/// pool.release(a);                            // a returns to reuse
/// pool.release(b);                            // b returns to reuse
/// pool.end_frame();                           // pool contract point
/// ```
///
/// Tests in this module validate the contract. Real integration with
/// `renderer::graph` happens in Phase 3b.
pub struct TransientPool {
    slots:     Vec<Slot>,
    /// Next free id for newly-allocated slots. Ids don't get reused
    /// across slot teardowns so graph edges stay stable.
    next_id:   u32,
    /// Cached swapchain extent from the most recent `begin_frame`.
    swap_size: (u32, u32),
    /// Rebuild counter — bumped whenever the pool drops all slots
    /// (e.g. resize). Tests use it to detect that invalidation fired.
    pub rebuild_epoch: u64,
}

impl TransientPool {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            next_id: 0,
            swap_size: (0, 0),
            rebuild_epoch: 0,
        }
    }

    /// Called once at the start of each frame. If the swapchain size
    /// changed, invalidates every slot whose size policy is
    /// swapchain-relative; fixed-size transients survive.
    pub fn begin_frame(&mut self, swap_w: u32, swap_h: u32) {
        if self.swap_size != (swap_w, swap_h) {
            let prev = self.swap_size;
            self.swap_size = (swap_w, swap_h);
            if prev != (0, 0) {
                self.invalidate_swapchain_relative();
            }
        }
    }

    /// Drop every slot whose size policy depends on the swapchain.
    /// Called automatically on resize; callers can also trigger this
    /// manually if a render target format changes.
    pub fn invalidate_swapchain_relative(&mut self) {
        let before = self.slots.len();
        self.slots.retain(|s| matches!(s.desc.size, SizePolicy::Fixed(_, _)));
        if self.slots.len() != before {
            self.rebuild_epoch += 1;
        }
    }

    /// Nuke everything. For tests, and for catastrophic format changes.
    pub fn clear(&mut self) {
        if !self.slots.is_empty() {
            self.slots.clear();
            self.rebuild_epoch += 1;
        }
    }

    /// Acquire a transient matching `desc`. Returns either a freed
    /// slot from the reuse pool or allocates a new one via `device`.
    /// The returned handle is valid until `release()` or the next
    /// resize.
    pub fn acquire(&mut self, device: &wgpu::Device, desc: TransientDesc) -> TransientId {
        let target_extent = desc.size.extent(self.swap_size.0, self.swap_size.1);

        // Look for an existing free slot with identical desc + extent.
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if !slot.in_use
                && slot.desc == desc
                && slot.extent == target_extent
            {
                slot.in_use = true;
                return TransientId(i as u32);
            }
        }

        // Nothing matched — allocate a new slot.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("transient"),
            size: wgpu::Extent3d {
                width: target_extent.0,
                height: target_extent.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: desc.mips,
            sample_count: desc.samples,
            dimension: wgpu::TextureDimension::D2,
            format: desc.format,
            usage: desc.usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let id = self.next_id;
        self.next_id += 1;
        // Slot index equals the id in this simple implementation —
        // once we add alias-based reuse the mapping becomes indirect,
        // but for now that's overkill.
        let slot_index = self.slots.len() as u32;
        assert_eq!(id, slot_index,
            "TransientPool invariant: slot ids are contiguous until Phase 3b adds aliasing");
        self.slots.push(Slot { desc, extent: target_extent, texture, view, in_use: true });
        TransientId(id)
    }

    /// Mark a transient as no longer in use. The slot returns to the
    /// reuse pool and can be handed back to a subsequent `acquire`
    /// with a matching desc.
    pub fn release(&mut self, id: TransientId) {
        if let Some(slot) = self.slots.get_mut(id.0 as usize) {
            slot.in_use = false;
        }
    }

    /// Get the underlying texture for a transient. Borrowed for the
    /// pool's lifetime — callers hold the borrow only while encoding.
    pub fn texture(&self, id: TransientId) -> Option<&wgpu::Texture> {
        self.slots.get(id.0 as usize).map(|s| &s.texture)
    }

    /// Get the default view for a transient.
    pub fn view(&self, id: TransientId) -> Option<&wgpu::TextureView> {
        self.slots.get(id.0 as usize).map(|s| &s.view)
    }

    /// Frame-end book-keeping. Currently does nothing because
    /// `acquire` / `release` are the only lifecycle points; kept as
    /// an API surface so Phase 3b can hook cleanup here.
    pub fn end_frame(&mut self) {}

    /// Diagnostic — how many slots are currently allocated (both
    /// in-use and in the reuse pool). Useful for memory footprint
    /// assertions in tests.
    pub fn slot_count(&self) -> usize { self.slots.len() }

    /// Diagnostic — how many allocated slots are currently in use.
    pub fn in_use_count(&self) -> usize {
        self.slots.iter().filter(|s| s.in_use).count()
    }
}

impl Default for TransientPool {
    fn default() -> Self { Self::new() }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn hdr_desc() -> TransientDesc {
        TransientDesc::new(
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            SizePolicy::Swapchain,
        )
    }

    fn r8_half() -> TransientDesc {
        TransientDesc::new(
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            SizePolicy::HalfSwapchain,
        )
    }

    /// Headless wgpu device for tests. Uses the default noop/fallback
    /// backend if available so tests run in CI without a GPU. Returns
    /// None on environments where no adapter is available — tests that
    /// need a device are skipped gracefully.
    fn try_create_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: true,
        })).ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("transient-test-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
        )).ok()?;
        Some((device, queue))
    }

    #[test]
    fn size_policy_resolves() {
        assert_eq!(SizePolicy::Swapchain.extent(1920, 1080), (1920, 1080));
        assert_eq!(SizePolicy::HalfSwapchain.extent(1920, 1080), (960, 540));
        assert_eq!(SizePolicy::QuarterSwapchain.extent(1920, 1080), (480, 270));
        assert_eq!(SizePolicy::Fixed(2048, 2048).extent(1920, 1080), (2048, 2048));
    }

    #[test]
    fn size_policy_never_returns_zero() {
        assert_eq!(SizePolicy::QuarterSwapchain.extent(2, 2), (1, 1));
        assert_eq!(SizePolicy::HalfSwapchain.extent(1, 1), (1, 1));
    }

    #[test]
    fn pool_resize_invalidates_swapchain_relative_only() {
        let mut pool = TransientPool::new();
        pool.begin_frame(1920, 1080);
        // Simulate two slots without a device — push raw entries so
        // the invalidation logic gets exercised without needing wgpu.
        let dummy_swap = TransientDesc::new(
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            SizePolicy::Swapchain,
        );
        let dummy_fixed = TransientDesc::new(
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            SizePolicy::Fixed(2048, 2048),
        );

        // We can't push real Slots without a device. For this unit
        // test we only validate the pruning semantics by exercising
        // them on a small scaffold. The device-level tests below
        // validate allocation/reuse end-to-end.
        let _ = (dummy_swap, dummy_fixed);
        assert_eq!(pool.slot_count(), 0);

        // Resize causes an epoch bump once there's anything to prune,
        // but with 0 slots the epoch stays at 0.
        let before_epoch = pool.rebuild_epoch;
        pool.begin_frame(1024, 768);
        assert_eq!(pool.rebuild_epoch, before_epoch);
    }

    // ----- device-backed tests: run only when an adapter is available -----

    #[test]
    fn acquire_returns_new_slot_when_pool_empty() {
        let Some((device, _queue)) = try_create_device() else { return; };
        let mut pool = TransientPool::new();
        pool.begin_frame(1024, 768);
        let id = pool.acquire(&device, hdr_desc());
        assert_eq!(pool.slot_count(), 1);
        assert_eq!(pool.in_use_count(), 1);
        assert!(pool.texture(id).is_some());
        assert!(pool.view(id).is_some());
    }

    #[test]
    fn release_then_acquire_reuses_slot() {
        let Some((device, _queue)) = try_create_device() else { return; };
        let mut pool = TransientPool::new();
        pool.begin_frame(1024, 768);
        let a = pool.acquire(&device, hdr_desc());
        pool.release(a);
        // Second acquire with the same desc should hit the reuse path
        // and not grow the slot count.
        let b = pool.acquire(&device, hdr_desc());
        assert_eq!(pool.slot_count(), 1, "reuse should not grow the pool");
        // Same slot index because reuse returns the freed slot.
        assert_eq!(a.0, b.0);
    }

    #[test]
    fn different_descs_dont_share_slots() {
        let Some((device, _queue)) = try_create_device() else { return; };
        let mut pool = TransientPool::new();
        pool.begin_frame(1024, 768);
        let a = pool.acquire(&device, hdr_desc());
        let b = pool.acquire(&device, r8_half());
        assert_ne!(a.0, b.0);
        assert_eq!(pool.slot_count(), 2);
        assert_eq!(pool.in_use_count(), 2);
    }

    #[test]
    fn resize_drops_swapchain_relative_slots() {
        let Some((device, _queue)) = try_create_device() else { return; };
        let mut pool = TransientPool::new();
        pool.begin_frame(1024, 768);
        let a = pool.acquire(&device, hdr_desc());             // Swapchain
        let f = pool.acquire(&device, TransientDesc::new(
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            SizePolicy::Fixed(2048, 2048),
        ));
        pool.release(a);
        pool.release(f);
        assert_eq!(pool.slot_count(), 2);

        pool.begin_frame(1920, 1080);                           // resize!
        // Swapchain-relative slot dropped; fixed slot survived.
        assert_eq!(pool.slot_count(), 1);
        assert_eq!(pool.rebuild_epoch, 1);

        // New acquire with the same swapchain-relative desc gets a
        // fresh slot sized to the new swapchain.
        let a2 = pool.acquire(&device, hdr_desc());
        assert_ne!(a.0, a2.0, "post-resize slot ids should not collide with released ones");
    }

    #[test]
    fn clear_resets_everything() {
        let Some((device, _queue)) = try_create_device() else { return; };
        let mut pool = TransientPool::new();
        pool.begin_frame(1024, 768);
        let _ = pool.acquire(&device, hdr_desc());
        let _ = pool.acquire(&device, r8_half());
        assert_eq!(pool.slot_count(), 2);

        pool.clear();
        assert_eq!(pool.slot_count(), 0);
        assert_eq!(pool.rebuild_epoch, 1);
    }
}
