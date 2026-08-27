use super::{
    GpuVirtualGeometryPool, GpuVirtualHierarchySelector, GpuVirtualPageRequest,
    GpuVirtualTraversalCounters, VirtualGeometryGpuError, VirtualMeshId, VirtualPageId,
};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

const READBACK_SLOTS: usize = 2;
const COUNTER_BYTES: u64 = std::mem::size_of::<GpuVirtualTraversalCounters>() as u64;
const REQUEST_BYTES: u64 = std::mem::size_of::<GpuVirtualPageRequest>() as u64;

/// Fixed CPU/GPU feedback budgets for virtual page streaming. The renderer's
/// simple enable method uses this default; advanced callers can opt into an
/// explicit configuration without changing the ordinary rendering path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuVirtualStreamingConfig {
    /// Maximum request records copied from one completed traversal.
    pub max_readback_requests: u32,
    /// Maximum unique mesh/group requests retained across camera frames.
    pub max_pending_groups: u32,
    /// Maximum atomic residency attempts made during one renderer frame.
    pub max_group_attempts_per_frame: u32,
}

impl Default for GpuVirtualStreamingConfig {
    fn default() -> Self {
        Self {
            max_readback_requests: 4_096,
            max_pending_groups: 8_192,
            max_group_attempts_per_frame: 256,
        }
    }
}

impl GpuVirtualStreamingConfig {
    /// Preserve the simple renderer enable API while clamping feedback memory
    /// to the selector's actual request capacity.
    pub fn bounded_default(max_page_requests: u32) -> Self {
        let max_readback_requests = max_page_requests.min(4_096).max(1);
        Self {
            max_readback_requests,
            max_pending_groups: max_readback_requests.saturating_mul(2).max(1),
            max_group_attempts_per_frame: max_readback_requests.min(256).max(1),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuVirtualStreamingTelemetry {
    pub readback_capacity: u32,
    pub readback_bytes: u64,
    pub pending_groups: u32,
    pub in_flight_readbacks: u32,
    pub captures_recorded: u64,
    pub captures_skipped_busy: u64,
    pub captures_completed: u64,
    pub stale_completions: u64,
    pub map_failures: u64,
    pub attempted_requests: u64,
    pub copied_requests: u64,
    pub truncated_requests: u64,
    pub unique_group_requests: u64,
    pub dropped_pending_groups: u64,
    pub group_attempts: u64,
    pub groups_resolved: u64,
    pub groups_rejected: u64,
    pub budget_stalls: u64,
    pub uploaded_pages: u64,
    pub uploaded_bytes: u64,
    pub last_visible_groups: u32,
    pub last_frustum_culled_groups: u32,
    pub last_cone_culled_clusters: u32,
    pub last_refined_groups: u32,
    pub last_fallback_groups: u32,
    pub last_missing_current_pages: u32,
    pub last_selected_count: u32,
    pub last_selected_overflow: u32,
    pub last_request_overflow: u32,
    pub last_invalid_records: u32,
    pub last_depth_limit_fallbacks: u32,
    pub last_occlusion_culled_groups: u32,
    pub last_occlusion_uncertain_groups: u32,
}

struct FeedbackReadback {
    buffer: wgpu::Buffer,
    in_flight: bool,
    status: Arc<AtomicU8>,
    sequence: u64,
}

#[derive(Clone, Copy)]
struct PendingGroup {
    request: GpuVirtualPageRequest,
    last_seen_sequence: u64,
}

struct CompletedFeedback {
    sequence: u64,
    counters: GpuVirtualTraversalCounters,
    requests: Vec<GpuVirtualPageRequest>,
}

/// Renderer-owned asynchronous bridge from traversal requests to the fixed
/// page pool. Copies and mappings are double-buffered; polling is non-blocking,
/// and all uploads remain constrained by the pool's per-frame budgets.
pub struct GpuVirtualPageStreamer {
    config: GpuVirtualStreamingConfig,
    selector_request_capacity: u32,
    readbacks: [FeedbackReadback; READBACK_SLOTS],
    parity: usize,
    recorded_slot: Option<usize>,
    next_sequence: u64,
    latest_consumed_sequence: u64,
    pending: BTreeMap<(u32, u32), PendingGroup>,
    telemetry: GpuVirtualStreamingTelemetry,
}

impl GpuVirtualPageStreamer {
    pub fn new(
        device: &wgpu::Device,
        selector: &GpuVirtualHierarchySelector,
        config: GpuVirtualStreamingConfig,
    ) -> Result<Self, GpuVirtualStreamingError> {
        validate_config(device, selector, config)?;
        let readback_bytes = readback_bytes(config.max_readback_requests);
        let readbacks = std::array::from_fn(|slot| FeedbackReadback {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(if slot == 0 {
                    "virtual_geometry_feedback_readback_0"
                } else {
                    "virtual_geometry_feedback_readback_1"
                }),
                size: readback_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            in_flight: false,
            status: Arc::new(AtomicU8::new(0)),
            sequence: 0,
        });
        Ok(Self {
            config,
            selector_request_capacity: selector.config().max_page_requests,
            readbacks,
            parity: 0,
            recorded_slot: None,
            next_sequence: 1,
            latest_consumed_sequence: 0,
            pending: BTreeMap::new(),
            telemetry: GpuVirtualStreamingTelemetry {
                readback_capacity: config.max_readback_requests,
                readback_bytes: readback_bytes.saturating_mul(READBACK_SLOTS as u64),
                ..GpuVirtualStreamingTelemetry::default()
            },
        })
    }

    /// Poll completed mappings without waiting and retain only the newest
    /// completion. Older camera feedback is safe to discard because visible
    /// missing pages are requested again while their resident ancestor draws.
    pub fn poll(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        let mut completed = Vec::new();
        for readback in &mut self.readbacks {
            if !readback.in_flight {
                continue;
            }
            match readback.status.load(Ordering::Acquire) {
                0 => continue,
                1 => {
                    let feedback = decode_readback(
                        readback,
                        self.config.max_readback_requests,
                        self.selector_request_capacity,
                    );
                    readback.buffer.unmap();
                    completed.push(feedback);
                    self.telemetry.captures_completed =
                        self.telemetry.captures_completed.saturating_add(1);
                }
                _ => {
                    self.telemetry.map_failures = self.telemetry.map_failures.saturating_add(1);
                    log::error!("bloom: virtual-geometry page feedback mapping failed");
                }
            }
            readback.in_flight = false;
            readback.status.store(0, Ordering::Release);
        }

        completed.sort_unstable_by_key(|feedback| feedback.sequence);
        let Some(newest) = completed.pop() else {
            self.refresh_live_telemetry();
            return;
        };
        self.telemetry.stale_completions = self
            .telemetry
            .stale_completions
            .saturating_add(completed.len() as u64);
        if newest.sequence <= self.latest_consumed_sequence {
            self.telemetry.stale_completions = self.telemetry.stale_completions.saturating_add(1);
            self.refresh_live_telemetry();
            return;
        }
        self.latest_consumed_sequence = newest.sequence;
        self.ingest(newest);
        self.refresh_live_telemetry();
    }

    /// Encode one bounded copy after hierarchy traversal. If both readback
    /// slots are still mapped, this frame simply keeps drawing resident
    /// ancestors and records no feedback work.
    pub fn record(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        selector: &GpuVirtualHierarchySelector,
    ) -> bool {
        if self.recorded_slot.is_some() {
            return false;
        }
        let slot = [self.parity, 1 - self.parity]
            .into_iter()
            .find(|&slot| !self.readbacks[slot].in_flight);
        let Some(slot) = slot else {
            self.telemetry.captures_skipped_busy =
                self.telemetry.captures_skipped_busy.saturating_add(1);
            self.refresh_live_telemetry();
            return false;
        };
        encoder.copy_buffer_to_buffer(
            selector.counter_buffer(),
            0,
            &self.readbacks[slot].buffer,
            0,
            COUNTER_BYTES,
        );
        encoder.copy_buffer_to_buffer(
            selector.page_request_buffer(),
            0,
            &self.readbacks[slot].buffer,
            COUNTER_BYTES,
            REQUEST_BYTES * u64::from(self.config.max_readback_requests),
        );
        self.readbacks[slot].sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1).max(1);
        self.readbacks[slot].status.store(0, Ordering::Release);
        self.recorded_slot = Some(slot);
        self.parity = 1 - slot;
        self.telemetry.captures_recorded = self.telemetry.captures_recorded.saturating_add(1);
        true
    }

    /// Start mapping only after the encoder containing the copies was
    /// submitted. The callback records completion state and never blocks.
    pub fn after_submit(&mut self) {
        let Some(slot) = self.recorded_slot.take() else {
            return;
        };
        let readback = &mut self.readbacks[slot];
        readback.in_flight = true;
        let status = readback.status.clone();
        readback
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                status.store(u8::from(result.is_err()) + 1, Ordering::Release);
            });
        self.refresh_live_telemetry();
    }

    /// Apply newest requests in deterministic priority order. A failed group
    /// caused only by this frame's residency budgets is retained for a later
    /// frame; malformed or generation-stale requests are removed.
    pub fn service(&mut self, pool: &mut GpuVirtualGeometryPool, queue: &wgpu::Queue) {
        let mut blocked = BTreeSet::new();
        for _ in 0..self.config.max_group_attempts_per_frame {
            let Some(key) = self.next_pending_key(&blocked) else {
                break;
            };
            let request = self.pending[&key].request;
            self.telemetry.group_attempts = self.telemetry.group_attempts.saturating_add(1);

            let mesh = VirtualMeshId::from_raw(request.mesh_id);
            if pool
                .page_entry(VirtualPageId {
                    mesh,
                    page_index: request.page_index,
                })
                .is_err()
            {
                self.pending.remove(&key);
                self.telemetry.groups_rejected = self.telemetry.groups_rejected.saturating_add(1);
                continue;
            }
            match pool.make_group_resident(queue, mesh, request.source_cluster) {
                Ok(transition) => {
                    self.pending.remove(&key);
                    self.telemetry.groups_resolved =
                        self.telemetry.groups_resolved.saturating_add(1);
                    self.telemetry.uploaded_pages = self
                        .telemetry
                        .uploaded_pages
                        .saturating_add(transition.uploaded.len() as u64);
                    self.telemetry.uploaded_bytes = self.telemetry.uploaded_bytes.saturating_add(
                        transition
                            .uploaded
                            .iter()
                            .filter_map(|(page, _)| {
                                pool.asset(page.mesh).ok().map(|asset| {
                                    asset.archive().pages[page.page_index as usize].payload_bytes
                                        as u64
                                })
                            })
                            .sum::<u64>(),
                    );
                }
                Err(
                    VirtualGeometryGpuError::UploadBudgetExceeded { .. }
                    | VirtualGeometryGpuError::EvictionBudgetExceeded
                    | VirtualGeometryGpuError::PhysicalPoolExhausted { .. },
                ) => {
                    blocked.insert(key);
                    self.telemetry.budget_stalls = self.telemetry.budget_stalls.saturating_add(1);
                }
                Err(_) => {
                    self.pending.remove(&key);
                    self.telemetry.groups_rejected =
                        self.telemetry.groups_rejected.saturating_add(1);
                }
            }
        }
        self.refresh_live_telemetry();
    }

    pub fn telemetry(&self) -> GpuVirtualStreamingTelemetry {
        let mut telemetry = self.telemetry;
        telemetry.pending_groups = self.pending.len() as u32;
        telemetry.in_flight_readbacks = self
            .readbacks
            .iter()
            .filter(|readback| readback.in_flight)
            .count() as u32;
        telemetry
    }

    fn ingest(&mut self, feedback: CompletedFeedback) {
        self.telemetry.last_visible_groups = feedback.counters.visible_groups;
        self.telemetry.last_frustum_culled_groups = feedback.counters.frustum_culled_groups;
        self.telemetry.last_cone_culled_clusters = feedback.counters.cone_culled_clusters;
        self.telemetry.last_refined_groups = feedback.counters.refined_groups;
        self.telemetry.last_fallback_groups = feedback.counters.fallback_groups;
        self.telemetry.last_missing_current_pages = feedback.counters.missing_current_pages;
        self.telemetry.last_selected_count = feedback.counters.selected_count;
        self.telemetry.last_selected_overflow = feedback.counters.selected_overflow;
        self.telemetry.last_request_overflow = feedback.counters.request_overflow;
        self.telemetry.last_invalid_records = feedback.counters.invalid_records;
        self.telemetry.last_depth_limit_fallbacks = feedback.counters.depth_limit_fallbacks;
        self.telemetry.last_occlusion_culled_groups = feedback.counters.occlusion_culled_groups;
        self.telemetry.last_occlusion_uncertain_groups =
            feedback.counters.occlusion_uncertain_groups;
        let attempted = u64::from(feedback.counters.page_request_count);
        let copied = feedback.requests.len() as u64;
        self.telemetry.attempted_requests =
            self.telemetry.attempted_requests.saturating_add(attempted);
        self.telemetry.copied_requests = self.telemetry.copied_requests.saturating_add(copied);
        self.telemetry.truncated_requests = self
            .telemetry
            .truncated_requests
            .saturating_add(attempted.saturating_sub(copied));

        for request in feedback.requests {
            if request.mesh_id == 0 {
                continue;
            }
            let key = (request.mesh_id, request.source_cluster);
            self.pending
                .entry(key)
                .and_modify(|pending| {
                    pending.last_seen_sequence = feedback.sequence;
                    if (request.page_index, request.instance_id)
                        < (pending.request.page_index, pending.request.instance_id)
                    {
                        pending.request = request;
                    }
                })
                .or_insert_with(|| {
                    self.telemetry.unique_group_requests =
                        self.telemetry.unique_group_requests.saturating_add(1);
                    PendingGroup {
                        request,
                        last_seen_sequence: feedback.sequence,
                    }
                });
        }
        self.enforce_pending_capacity();
    }

    fn enforce_pending_capacity(&mut self) {
        let capacity = self.config.max_pending_groups as usize;
        if self.pending.len() <= capacity {
            return;
        }
        let mut priority = self
            .pending
            .iter()
            .map(|(&key, pending)| (Reverse(pending.last_seen_sequence), key))
            .collect::<Vec<_>>();
        priority.sort_unstable();
        for (_, key) in priority.into_iter().skip(capacity) {
            self.pending.remove(&key);
            self.telemetry.dropped_pending_groups =
                self.telemetry.dropped_pending_groups.saturating_add(1);
        }
    }

    fn next_pending_key(&self, blocked: &BTreeSet<(u32, u32)>) -> Option<(u32, u32)> {
        self.pending
            .iter()
            .filter(|(key, _)| !blocked.contains(key))
            .min_by_key(|(key, pending)| (Reverse(pending.last_seen_sequence), **key))
            .map(|(&key, _)| key)
    }

    fn refresh_live_telemetry(&mut self) {
        self.telemetry.pending_groups = self.pending.len() as u32;
        self.telemetry.in_flight_readbacks = self
            .readbacks
            .iter()
            .filter(|readback| readback.in_flight)
            .count() as u32;
    }
}

fn readback_bytes(requests: u32) -> u64 {
    COUNTER_BYTES + REQUEST_BYTES * u64::from(requests)
}

fn validate_config(
    device: &wgpu::Device,
    selector: &GpuVirtualHierarchySelector,
    config: GpuVirtualStreamingConfig,
) -> Result<(), GpuVirtualStreamingError> {
    if config.max_readback_requests == 0
        || config.max_readback_requests > selector.config().max_page_requests
        || config.max_pending_groups < config.max_readback_requests
        || config.max_group_attempts_per_frame == 0
    {
        return Err(GpuVirtualStreamingError::InvalidConfig);
    }
    let bytes = readback_bytes(config.max_readback_requests);
    if bytes > device.limits().max_buffer_size {
        return Err(GpuVirtualStreamingError::DeviceLimitExceeded {
            requested_bytes: bytes,
            maximum_bytes: device.limits().max_buffer_size,
        });
    }
    Ok(())
}

fn decode_readback(
    readback: &FeedbackReadback,
    readback_capacity: u32,
    selector_capacity: u32,
) -> CompletedFeedback {
    let mapped = readback.buffer.slice(..).get_mapped_range();
    let counters = bytemuck::pod_read_unaligned::<GpuVirtualTraversalCounters>(
        &mapped[..COUNTER_BYTES as usize],
    );
    let count = counters
        .page_request_count
        .min(selector_capacity)
        .min(readback_capacity) as usize;
    let request_bytes = std::mem::size_of::<GpuVirtualPageRequest>();
    let requests = mapped[COUNTER_BYTES as usize..]
        .chunks_exact(request_bytes)
        .take(count)
        .map(bytemuck::pod_read_unaligned::<GpuVirtualPageRequest>)
        .collect();
    drop(mapped);
    CompletedFeedback {
        sequence: readback.sequence,
        counters,
        requests,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuVirtualStreamingError {
    InvalidConfig,
    DeviceLimitExceeded {
        requested_bytes: u64,
        maximum_bytes: u64,
    },
}

impl fmt::Display for GpuVirtualStreamingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(formatter, "invalid virtual-geometry streaming config"),
            Self::DeviceLimitExceeded {
                requested_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "virtual-geometry feedback needs {requested_bytes} bytes per readback but the device limit is {maximum_bytes}"
            ),
        }
    }
}

impl std::error::Error for GpuVirtualStreamingError {}
