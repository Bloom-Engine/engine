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
    /// One-shot warning guard for GPU timestamp-pair exhaustion.
    budget_warned: bool,

    /// Phase 8 — last `ROLLING_FRAMES` frame totals (sum of all
    /// samples in `frame` at frame_end), in microseconds. Ring
    /// buffer indexed by `histogram_idx`; consumers pass through
    /// `bloom_profiler_frame_history` and render a bar chart.
    frame_total_cpu_us: [f64; ROLLING_FRAMES],
    frame_total_gpu_us: [f64; ROLLING_FRAMES],
    histogram_idx: usize,
    histogram_filled: usize,
}

struct RollingStats {
    cpu: [f64; ROLLING_FRAMES],
    gpu: [f64; ROLLING_FRAMES],
    has_gpu: bool,
    idx: usize,
    filled: usize,
    /// Frame index of the most recent sample. A pass that stops running
    /// (feature toggled off) must drop out of the readouts instead of
    /// showing its frozen average forever.
    last_frame: u64,
}

impl RollingStats {
    fn new() -> Self {
        Self { cpu: [0.0; ROLLING_FRAMES], gpu: [0.0; ROLLING_FRAMES], has_gpu: false, idx: 0, filled: 0, last_frame: 0 }
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
            budget_warned: false,
            frame_total_cpu_us: [0.0; ROLLING_FRAMES],
            frame_total_gpu_us: [0.0; ROLLING_FRAMES],
            histogram_idx: 0,
            histogram_filled: 0,
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

    pub fn set_enabled(&mut self, on: bool) {
        if on && !self.enabled {
            // Fresh measuring session: without this, stats captured before
            // a disable (potentially under a different feature set) show
            // until the rolling window refills and skew the first seconds
            // of every new session.
            self.rolling.clear();
            self.frame_total_cpu_us = [0.0; ROLLING_FRAMES];
            self.frame_total_gpu_us = [0.0; ROLLING_FRAMES];
            self.histogram_idx = 0;
            self.histogram_filled = 0;
        }
        self.enabled = on;
    }
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
        if self.next_query + 2 > MAX_GPU_PAIRS * 2 {
            if !self.budget_warned {
                self.budget_warned = true;
                eprintln!(
                    "bloom profiler: GPU timestamp budget ({} pairs) exhausted — later passes report CPU time only",
                    MAX_GPU_PAIRS
                );
            }
            return None;
        }
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

    /// End-of-frame bookkeeping. Resolves this frame's GPU timestamps via
    /// a BLOCKING map (map_async + poll(Wait) — serialises CPU⇄GPU, so
    /// wall-clock fps is pessimistic while the profiler is enabled; see
    /// docs/crash-triage-windows.md), folds samples into rolling stats,
    /// and clears per-frame state for the next frame.
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

        self.frame_end_cpu();
    }

    /// CPU-only end-of-frame: histogram update + drain into rolling.
    /// Split out so tests don't need a wgpu::Device. Production
    /// callers go through `frame_end` which handles GPU readback
    /// first and then delegates here.
    fn frame_end_cpu(&mut self) {
        // Phase 8 — sum the per-pass samples into a per-frame total
        // for the histogram before draining. CPU sums every sample;
        // GPU sums only those with timestamps (rest have None).
        let mut frame_cpu = 0.0;
        let mut frame_gpu = 0.0;
        for s in &self.frame {
            frame_cpu += s.cpu_us;
            if let Some(g) = s.gpu_us { frame_gpu += g; }
        }
        self.frame_total_cpu_us[self.histogram_idx] = frame_cpu;
        self.frame_total_gpu_us[self.histogram_idx] = frame_gpu;
        self.histogram_idx = (self.histogram_idx + 1) % ROLLING_FRAMES;
        self.histogram_filled = (self.histogram_filled + 1).min(ROLLING_FRAMES);

        let fc = self.frame_count;
        for s in self.frame.drain(..) {
            let entry = self.rolling.entry(s.label).or_insert_with(RollingStats::new);
            entry.push(s.cpu_us, s.gpu_us);
            entry.last_frame = fc;
        }
        // Drop entries that have not reported for several windows so a
        // disabled feature's passes leave the map instead of lingering.
        self.rolling.retain(|_, s| fc.saturating_sub(s.last_frame) <= (4 * ROLLING_FRAMES) as u64);
        self.open_cpu.clear();
        self.next_query = 0;
        self.pending_gpu.clear();
        self.frame_count = self.frame_count.wrapping_add(1);
    }

    /// Phase 8 — frame-history snapshot for the overlay's bar chart.
    /// Returns `(cpu_us, gpu_us)` pairs in chronological order, oldest
    /// first, exactly `ROLLING_FRAMES.min(filled)` entries long.
    pub fn frame_history(&self) -> Vec<(f64, f64)> {
        let n = self.histogram_filled;
        if n == 0 { return Vec::new(); }
        let start = if n < ROLLING_FRAMES { 0 } else { self.histogram_idx };
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let idx = (start + i) % ROLLING_FRAMES;
            out.push((self.frame_total_cpu_us[idx], self.frame_total_gpu_us[idx]));
        }
        out
    }

    pub fn summary(&self) -> String {
        if !self.enabled {
            return String::from("profiler: disabled\n");
        }
        let fc = self.frame_count;
        let mut entries: Vec<(&&str, &RollingStats)> = self.rolling.iter()
            .filter(|(_, s)| fc.saturating_sub(s.last_frame) <= ROLLING_FRAMES as u64)
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
        let fc = self.frame_count;
        self.rolling.values()
            .filter(|s| fc.saturating_sub(s.last_frame) <= ROLLING_FRAMES as u64)
            .map(|s| s.avg_cpu()).sum()
    }

    /// Average total GPU frame time where available.
    pub fn avg_frame_gpu_us(&self) -> f64 {
        let fc = self.frame_count;
        self.rolling.values()
            .filter(|s| fc.saturating_sub(s.last_frame) <= ROLLING_FRAMES as u64)
            .filter_map(|s| s.avg_gpu()).sum()
    }

    /// Snapshot the rolling averages in a stable, CPU-time-descending
    /// order. Games call this once per overlay-draw frame and pull
    /// label/cpu/gpu out via the accessors below — HashMap iteration
    /// order would jitter the overlay otherwise.
    pub fn snapshot(&mut self) -> Vec<(&'static str, f64, Option<f64>)> {
        let fc = self.frame_count;
        let mut v: Vec<(&'static str, f64, Option<f64>)> = self.rolling.iter()
            .filter(|(_, s)| fc.saturating_sub(s.last_frame) <= ROLLING_FRAMES as u64)
            .map(|(k, s)| (*k, s.avg_cpu(), s.avg_gpu()))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the profiler through `frame_end` purely on the CPU
    /// side (gpu_enabled stays false). Each "frame" pushes a single
    /// sample with the given cpu_us so the histogram totals are
    /// known.
    fn fake_frame(p: &mut Profiler, cpu_us: f64) {
        p.frame.push(FrameSample { label: "fake", cpu_us, gpu_us: None });
        // Skip the gpu readback path (no device available) and go
        // straight to the CPU-only histogram + drain path.
        p.frame_end_cpu();
    }

    #[test]
    fn frame_history_empty_before_any_frames() {
        let mut p = Profiler::new();
        p.set_enabled(true);
        assert!(p.frame_history().is_empty());
    }

    #[test]
    fn frame_history_records_in_order_under_capacity() {
        let mut p = Profiler::new();
        p.set_enabled(true);
        for i in 1..=5 {
            fake_frame(&mut p, i as f64 * 100.0);
        }
        let h = p.frame_history();
        assert_eq!(h.len(), 5);
        assert_eq!(h[0].0, 100.0);
        assert_eq!(h[4].0, 500.0);
    }

    #[test]
    fn stale_labels_drop_out_of_snapshot_and_evict() {
        let mut p = Profiler::new();
        p.set_enabled(true);
        // One label reports once, another keeps reporting.
        p.frame.push(FrameSample { label: "once", cpu_us: 5.0, gpu_us: None });
        p.frame_end_cpu();
        for _ in 0..(ROLLING_FRAMES + 1) { fake_frame(&mut p, 1.0); }
        let snap = p.snapshot();
        assert!(snap.iter().any(|(l, _, _)| *l == "fake"));
        assert!(
            !snap.iter().any(|(l, _, _)| *l == "once"),
            "a pass that stopped reporting must leave the snapshot"
        );
        for _ in 0..(4 * ROLLING_FRAMES) { fake_frame(&mut p, 1.0); }
        assert!(!p.rolling.contains_key("once"), "stale label must eventually evict");
    }

    #[test]
    fn reenable_starts_fresh_session() {
        let mut p = Profiler::new();
        p.set_enabled(true);
        fake_frame(&mut p, 100.0);
        assert!(!p.frame_history().is_empty());
        p.set_enabled(false);
        p.set_enabled(true);
        assert!(p.frame_history().is_empty(), "histogram must clear on re-enable");
        assert!(p.snapshot().is_empty(), "rolling stats must clear on re-enable");
    }

    #[test]
    fn frame_history_wraps_oldest_first_at_capacity() {
        let mut p = Profiler::new();
        p.set_enabled(true);
        // Push more than ROLLING_FRAMES so the ring wraps.
        for i in 1..=(ROLLING_FRAMES + 30) {
            fake_frame(&mut p, i as f64);
        }
        let h = p.frame_history();
        assert_eq!(h.len(), ROLLING_FRAMES);
        // The newest entry is whatever the last push was.
        assert_eq!(h.last().unwrap().0, (ROLLING_FRAMES + 30) as f64);
        // The oldest must be `(ROLLING_FRAMES + 30) - ROLLING_FRAMES + 1 = 31`.
        assert_eq!(h[0].0, 31.0);
        // Strictly monotonic (oldest → newest).
        for w in h.windows(2) {
            assert!(w[0].0 < w[1].0, "history must be in chronological order");
        }
    }
}
