# 010 — Overlap post-FX on compute queue

**Effort:** ~3 days (optimistic — see Deferred section) · **Expected gain:** Hides ~20% of post-FX cost via latency overlap · **Status:** deferred

## Problem

Every pass in Bloom runs serially on the graphics queue. The GPU does:

```
shadow_pass → main_hdr → SSAO → SSR → SSGI → bloom → compose → TAA → ... → present
```

All on one queue, one after another. The GPU can't start the next frame's
shadow pass while this frame's post-FX is still in flight because the depth
buffer is still being read.

UE5 uses **async compute**: post-FX passes that *don't read* from next-frame
dependencies (SSAO at frame N reads depth at frame N, but those depths are
fixed once main_hdr finishes) can be dispatched on a separate compute queue
in parallel with the next frame's graphics work.

On Apple Silicon, Metal supports two queues (`MTLCommandQueue`) that execute
concurrently. Overlapping 20-30% of the post-FX work is realistic.

## Proposed approach

1. **Create a second `wgpu::Queue`** (wgpu-hal-level; wgpu core only exposes
   one queue per device). Options:
   - Use wgpu's experimental async-compute feature if available.
   - Drop to `wgpu-hal` / Metal directly for the compute path.
2. **Split the command encoder** into "graphics" and "compute" encoders.
   SSAO/SSR/SSGI/bloom/ssgi_temporal as compute-shader variants (prereq:
   ticket 002 compute GTAO, ticket 003 stochastic SSR, ticket 007 probe
   SSGI).
3. **Synchronize via fences**: compute queue waits for main_hdr to finish,
   then produces SSAO/SSR/SSGI/bloom. Graphics queue can start next frame's
   shadow pass while post-FX runs on compute.
4. **Present signal**: final composite (graphics queue) waits for compute
   queue's post-FX to finish before sampling.

This ticket depends on having compute-shader versions of SSAO/SSR/SSGI first
(tickets 002, 003, 007).

## References

- Apple WWDC 2019 — "Metal async compute"
- UE5 `FRDGBuilder` async compute implementation
- DX12 & Vulkan async compute patterns (the GPU concepts are universal)

## Acceptance

- Sponza frame time drops ~15-20% over the same workload run serially.
- No visual artifacts from queue-ordering mistakes.
- Works on Apple Silicon and at least one discrete GPU (adapter feature gate).

## Notes for the implementer

- wgpu's support for multi-queue is partial as of wgpu 24 — may need an
  upgrade to newer wgpu or to drop to hal.
- Fences / semaphores: wgpu uses internal submission ordering. Expose or
  bypass via hal.
- Test carefully on discrete NVIDIA / AMD — async-compute bugs there are
  notorious (serialization under the hood).
- Can prototype serial-equivalent ordering first (everything on one queue,
  split encoders) to shake out bugs before enabling true async.

## Files likely to change

- `native/shared/src/renderer/mod.rs` (the old single `renderer.rs` was
  split into the `renderer/` module) — encoder splitting, queue creation,
  fence/semaphore plumbing.
- Possibly `native/macos/src/lib.rs` — if wgpu-hal drop is needed for
  second queue.

## Deferred — blocker: wgpu has no multi-queue public API

Audited wgpu 29 (current pin): `Adapter::request_device` returns exactly
one `(Device, Queue)`. There is no `Device::get_compute_queue()` /
`Instance::request_multiple_queues` / queue-family API. The only paths
to actually achieve async compute on Bloom today:

1. **Drop to wgpu-hal directly** for the second queue. wgpu-hal has the
   per-backend `Queue` abstractions (Metal / DX12 / Vulkan each support
   multiple queues at the hal level), but mixing wgpu-core and wgpu-hal
   in the same renderer is fragile — lifetime / submission ordering
   guarantees differ, and we'd lose the safe wgpu-core API for every
   resource the compute queue touches. Effectively rewrites the post-FX
   layer on a different abstraction.
2. **Go native per platform.** Use `metal-rs` on macOS, `windows-rs` /
   DX12 on Windows, `ash` / Vulkan on Linux / Android. Three separate
   implementations, each with its own sync primitives, for a ~1.3 ms
   frame-time win on a benchmark that's already vsync-capped at 60 fps.
3. **Wait for wgpu upstream.** Multi-queue support has been discussed
   but is not on a near-term roadmap as of wgpu 29.

Ticket's own sub-suggestion — "prototype serial-equivalent ordering
first" (split encoders, same queue) — doesn't help: on a single queue,
one big encoder + one submit generally outperforms multiple smaller
submits because every submit introduces driver overhead without
enabling parallelism.

Estimated true effort at current wgpu:

- Drop to wgpu-hal: 2-3 weeks redesign + cross-platform testing.
- Native per platform: 3 weeks+ (three impls, each platform's async-
  compute quirks).

Expected gain at vsync target: ~1.3 ms of the 16.7 ms budget — 8% of
one frame, less visible than the 1.3 ms would suggest because we're
already at the vsync ceiling. Worth pursuing only when:

- A target scene is pushing past 16.7 ms and post-FX is the bottleneck,
  or
- wgpu lands a stable multi-queue API.

Parked. Reopen when either condition holds.
