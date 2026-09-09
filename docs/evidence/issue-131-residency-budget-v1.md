# Issue #131 residency-budget evidence v1

This checkpoint qualifies virtual-geometry page residency and streaming
staging at revision `2fabbaf389f0dbefa25241fd1ef846f0d05b2d56`. It adds
telemetry and regression assertions only: no pass, draw, shader branch,
allocation, budget, or accepted pixel changes.

## Fixed page residency

The file-backed 10,000,000-source-triangle workload ran for 180 moving-camera
warmup frames and 120 measured moving-camera frames at 640x360 on Apple M1 Max
Metal. The archive contains 245,500 clusters and 8,496 pages; the renderer
submitted 100 placements while demand-loading detail into its configured
64 MiB physical page pool.

| Measurement | Bytes |
|---|---:|
| Configured physical page pool | 67,108,864 |
| Page stride | 65,536 |
| Peak resident slots (955 pages) | 62,586,880 |
| Useful resident payload | 61,639,088 |
| Slot padding | 947,792 |
| Unused pool headroom | 4,521,984 |

Peak page residency used 93.2617% of the pool and exceeded it by zero bytes,
which is stronger than the one-page acceptance tolerance. The gate checks the
reported peak, not merely final-frame residency, and requires exact equality
between resident pages multiplied by page stride and resident-slot bytes.

## Explicit allocation and staging breakdown

GPU metadata is allocated independently of the physical residency budget and
is reported rather than hidden inside it:

- page table: 135,936 bytes;
- mesh table: 48 bytes;
- cluster table: 31,424,000 bytes;
- metadata total: 31,559,984 bytes;
- physical pool plus metadata: 98,668,848 bytes.

Streaming staging is also explicit:

- two fixed GPU feedback readbacks: 131,184 bytes total for 4,096 requests;
- CPU page-I/O reservation budget: 33,554,432 bytes;
- observed lifetime peak CPU reservation: 12,707,616 bytes (37.8716%);
- reported combined peak staging: 12,838,800 bytes;
- final CPU reservation: zero bytes.

The stress oracle derives the expected readback allocation from the two public
GPU record sizes, asserts the exact metadata sum, requires physical pool plus
metadata to equal the reported GPU allocation, and requires combined staging
to equal fixed readback bytes plus the preserved CPU-reservation peak.

## Runtime result

The run completed all 370 file requests with zero failures and read 75,515,520
bytes. It settled at 10,220 selected clusters with zero fallback groups,
missing-current pages, selected/request overflow, invalid records,
depth-limit fallbacks, or pending streaming groups. Wall mean was 8.8808 ms;
GPU mean/p95 were 5.5223/7.9104 ms, hierarchy selection was 1.9872 ms, and draw
emission was 0.1316 ms. All remain within the established stress limits.

## Regression gates

- 45 focused virtual-geometry release tests passed on the real GPU;
- enabled and disabled capability-report JSON passed real-GPU integration
  tests;
- the file-backed 10M release stress passed with the new exact budget checks;
- strict renderer correctness, suspiciousness, and performance Clippy passed;
- formatting and whitespace checks passed.

Discrete Vulkan and Direct3D 12 qualification remains pending hardware; this
checkpoint closes only the peak-residency/staging acceptance item.
