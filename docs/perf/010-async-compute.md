# 010 — Overlap post-FX on compute queue

**Effort:** ~3 days · **Expected gain:** Hides ~20% of post-FX cost via latency overlap · **Status:** open

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

- `native/shared/src/renderer.rs` — encoder splitting, queue creation,
  fence/semaphore plumbing.
- Possibly `native/macos/src/lib.rs` — if wgpu-hal drop is needed for
  second queue.
