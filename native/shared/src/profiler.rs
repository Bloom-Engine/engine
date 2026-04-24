//! Frame profiler: CPU phase timings + optional GPU timestamp queries.
//!
//! Disabled by default (zero overhead when `enabled == false`). Enable at
//! runtime via `Profiler::set_enabled(true)` — typically wired to a FFI
//! setter so games can toggle it from TS.
//!
//! CPU side: `begin("label")` / `end("label")` stacks around the phase.
//! GPU side: create pipeline using `timestamp_writes_for(&mut encoder, "label")`
//! and plug the returned `RenderPassTimestampWrites` into the pass descriptor.
//! GPU timestamps require the `TIMESTAMP_QUERY` device feature (opt-in at
//! device creation). When the feature is unavailable the GPU path no-ops
//! and only CPU numbers are reported.
//!
//! At `frame_end()` the profiler copies per-frame samples into a rolling
//! window (default 120 frames) and computes averages. `summary()` returns
//! a human-readable table.
//!
//! This module intentionally keeps the data model flat (Vec of named
//! samples, not a tree) — post-FX passes are already flat and a tree
//! would be overkill for a first pass.

#[cfg(feature = "web")]
use web_time::Instant;
#[cfg(not(feature = "web"))]
use std::time::Instant;

use std::collections::HashMap;

const ROLLING_FRAMES: usize = 120;
const MAX_GPU_PAIRS: u32 = 32;

#[derive(Clone, Copy)]
struct CpuSample {
    start: Instant,
}

struct FrameSample {
    label: &'static str,
    cpu_us: f64,
    gpu_us: Option<f64>,
}

pub struct Profiler {
    pub enabled: bool,
    pub gpu_enabled: bool,

    open_cpu: HashMap<&'static str, CpuSample>,
    frame: Vec<FrameSample>,
    rolling: HashMap<&'static str, RollingStats>,
    frame_count: u64,

    // GPU timestamp query state
    query_set: Option<wgpu::QuerySet>,
    resolve_buffer: Option<wgpu::Buffer>,
    readback_buffer: Option<wgpu::Buffer>,
    timestamp_period_ns: f32,
    next_query: u32,
    // label -> (begin_index, end_index)
    pending_gpu: Vec<(&'static str, u32, u32)>,
}

struct RollingStats {
    cpu: [f64; ROLLING_FRAMES],
    gpu: [f64; ROLLING_FRAMES],
    has_gpu: bool,
    idx: usize,
    filled: usize,
}

impl RollingStats {
    fn new() -> Self {
        Self { cpu: [0.0; ROLLING_FRAMES], gpu: [0.0; ROLLING_FRAMES], has_gpu: false, idx: 0, filled: 0 }
    }
    fn push(&mut self, cpu: f64, gpu: Option<f64>) {
        self.cpu[self.idx] = cpu;
        if let Some(g) = gpu { self.gpu[self.idx] = g; self.has_gpu = true; }
        self.idx = (self.idx + 1) % ROLLING_FRAMES;
        self.filled = (self.filled + 1).min(ROLLING_FRAMES);
    }
    fn avg_cpu(&self) -> f64 {
        if self.filled == 0 { return 0.0; }
        let sum: f64 = self.cpu.iter().take(self.filled).sum();
        sum / self.filled as f64
    }
    fn avg_gpu(&self) -> Option<f64> {
        if !self.has_gpu || self.filled == 0 { return None; }
        let sum: f64 = self.gpu.iter().take(self.filled).sum();
        Some(sum / self.filled as f64)
    }
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            enabled: false,
            gpu_enabled: false,
            open_cpu: HashMap::new(),
            frame: Vec::new(),
            rolling: HashMap::new(),
            frame_count: 0,
            query_set: None,
            resolve_buffer: None,
            readback_buffer: None,
            timestamp_period_ns: 1.0,
            next_query: 0,
            pending_gpu: Vec::new(),
        }
    }

    /// Call after device creation. If the device supports `TIMESTAMP_QUERY`,
    /// allocates the query set + readback buffer so GPU timings become available.
    pub fn init_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return;
        }
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("bloom_profiler_queryset"),
            ty: wgpu::QueryType::Timestamp,
            count: MAX_GPU_PAIRS * 2,
        });
        let resolve_size = (MAX_GPU_PAIRS as u64) * 2 * 8;
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom_profiler_resolve"),
            size: resolve_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom_profiler_readback"),
            size: resolve_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.query_set = Some(query_set);
        self.resolve_buffer = Some(resolve_buffer);
        self.readback_buffer = Some(readback_buffer);
        self.timestamp_period_ns = queue.get_timestamp_period();
        self.gpu_enabled = true;
    }

    pub fn set_enabled(&mut self, on: bool) { self.enabled = on; }
    pub fn is_enabled(&self) -> bool { self.enabled }
    pub fn has_gpu(&self) -> bool { self.gpu_enabled }

    pub fn begin(&mut self, label: &'static str) {
        if !self.enabled { return; }
        self.open_cpu.insert(label, CpuSample { start: Instant::now() });
    }

    pub fn end(&mut self, label: &'static str) {
        if !self.enabled { return; }
        if let Some(s) = self.open_cpu.remove(label) {
            let us = s.start.elapsed().as_secs_f64() * 1_000_000.0;
            self.frame.push(FrameSample { label, cpu_us: us, gpu_us: None });
        }
    }

    /// Reserve a pair of GPU timestamp slot indices for the given label.
    /// Callers combine the returned indices with `query_set()` to build a
    /// `RenderPassTimestampWrites` for the pass descriptor. Returns None
    /// when profiling or GPU queries are disabled, or when no slots remain.
    pub fn reserve_gpu_pair(&mut self, label: &'static str) -> Option<(u32, u32)> {
        if !self.enabled || !self.gpu_enabled { return None; }
        self.query_set.as_ref()?;
        if self.next_query + 2 > MAX_GPU_PAIRS * 2 { return None; }
        let begin = self.next_query;
        let end = self.next_query + 1;
        self.next_query += 2;
        self.pending_gpu.push((label, begin, end));
        Some((begin, end))
    }

    pub fn query_set(&self) -> Option<&wgpu::QuerySet> {
        self.query_set.as_ref()
    }

    /// Convenience: build a `RenderPassTimestampWrites` for a pass in one call.
    /// Returns None when GPU queries are disabled — the caller should plug
    /// the returned `Option` straight into `RenderPassDescriptor.timestamp_writes`.
    pub fn pass_timestamp_writes(&mut self, label: &'static str) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let (b, e) = self.reserve_gpu_pair(label)?;
        let qs = self.query_set.as_ref()?;
        Some(wgpu::RenderPassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(b),
            end_of_pass_write_index: Some(e),
        })
    }

    /// Same as `pass_timestamp_writes` but for a compute pass descriptor.
    pub fn compute_pass_timestamp_writes(&mut self, label: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let (b, e) = self.reserve_gpu_pair(label)?;
        let qs = self.query_set.as_ref()?;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(b),
            end_of_pass_write_index: Some(e),
        })
    }

    /// Resolve any pending GPU queries into the readback buffer. Call once
    /// per frame, after all passes are encoded and before submit.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled || !self.gpu_enabled || self.next_query == 0 { return; }
        let (Some(qs), Some(resolve)) = (&self.query_set, &self.resolve_buffer) else { return; };
        encoder.resolve_query_set(qs, 0..self.next_query, resolve, 0);
        if let Some(readback) = &self.readback_buffer {
            let byte_count = (self.next_query as u64) * 8;
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, byte_count);
        }
    }

    /// End-of-frame bookkeeping. Reads back GPU timestamps from the
    /// previous frame (non-blocking map), folds samples into rolling
    /// stats, and clears per-frame state for the next frame.
    pub fn frame_end(&mut self, device: &wgpu::Device) {
        if !self.enabled {
            self.frame.clear();
            self.open_cpu.clear();
            self.next_query = 0;
            self.pending_gpu.clear();
            return;
        }

        if self.gpu_enabled && self.next_query > 0 {
            if let Some(readback) = &self.readback_buffer {
                let byte_count = (self.next_query as u64) * 8;
                let slice = readback.slice(0..byte_count);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
                let data = slice.get_mapped_range().to_vec();
                readback.unmap();
                let period = self.timestamp_period_ns as f64;
                let mut by_label: HashMap<&'static str, f64> = HashMap::new();
                for (label, b, e) in &self.pending_gpu {
                    let bo = (*b as usize) * 8;
                    let eo = (*e as usize) * 8;
                    if eo + 8 > data.len() { continue; }
                    let bt = u64::from_le_bytes(data[bo..bo+8].try_into().unwrap());
                    let et = u64::from_le_bytes(data[eo..eo+8].try_into().unwrap());
                    if et <= bt { continue; }
                    let us = (et - bt) as f64 * period / 1000.0;
                    *by_label.entry(*label).or_insert(0.0) += us;
                }
                for s in self.frame.iter_mut() {
                    if let Some(us) = by_label.remove(s.label) { s.gpu_us = Some(us); }
                }
                // GPU samples without a CPU counterpart — record them too.
                for (label, us) in by_label {
                    self.frame.push(FrameSample { label, cpu_us: 0.0, gpu_us: Some(us) });
                }
            }
        }

        for s in self.frame.drain(..) {
            let entry = self.rolling.entry(s.label).or_insert_with(RollingStats::new);
            entry.push(s.cpu_us, s.gpu_us);
        }
        self.open_cpu.clear();
        self.next_query = 0;
        self.pending_gpu.clear();
        self.frame_count = self.frame_count.wrapping_add(1);
    }

    pub fn summary(&self) -> String {
        if !self.enabled {
            return String::from("profiler: disabled\n");
        }
        let mut entries: Vec<(&&str, &RollingStats)> = self.rolling.iter()
            .map(|(k, v)| (k, v))
            .collect();
        entries.sort_by(|a, b| b.1.avg_cpu().partial_cmp(&a.1.avg_cpu()).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = String::new();
        out.push_str(&format!(
            "profiler (avg over last {} frames, gpu={}):\n",
            ROLLING_FRAMES.min(entries.first().map(|(_,s)| s.filled).unwrap_or(0)),
            if self.gpu_enabled { "yes" } else { "no" },
        ));
        out.push_str("  phase                         cpu us     gpu us\n");
        for (label, stats) in entries {
            let gpu = stats.avg_gpu().map(|v| format!("{:>9.1}", v)).unwrap_or_else(|| "        -".to_string());
            out.push_str(&format!("  {:<28} {:>9.1}   {}\n", label, stats.avg_cpu(), gpu));
        }
        out
    }

    /// Average total CPU frame time across the rolling window (sum of
    /// all phases). Useful for a single headline number.
    pub fn avg_frame_cpu_us(&self) -> f64 {
        self.rolling.values().map(|s| s.avg_cpu()).sum()
    }

    /// Average total GPU frame time where available.
    pub fn avg_frame_gpu_us(&self) -> f64 {
        self.rolling.values().filter_map(|s| s.avg_gpu()).sum()
    }

    /// Snapshot the rolling averages in a stable, CPU-time-descending
    /// order. Games call this once per overlay-draw frame and pull
    /// label/cpu/gpu out via the accessors below — HashMap iteration
    /// order would jitter the overlay otherwise.
    pub fn snapshot(&mut self) -> Vec<(&'static str, f64, Option<f64>)> {
        let mut v: Vec<(&'static str, f64, Option<f64>)> = self.rolling.iter()
            .map(|(k, s)| (*k, s.avg_cpu(), s.avg_gpu()))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }
}
