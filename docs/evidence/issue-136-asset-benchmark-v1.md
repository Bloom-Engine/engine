# Issue #136 cooked asset benchmark evidence

This checkpoint qualifies cooked texture memory/quality and mesh load time at
revision `07e748bd2e0df508f358cf6b4712b1dcb20160c6`. Measurements ran on an
Apple M1 Max (`arm64`) with macOS 26.5. They are CPU and storage-container
measurements with a warm OS cache; no GPU creation or upload time is claimed.

## Acceptance thresholds

Thresholds were fixed before accepting the final measurements:

- BC7 color: at least 70% complete-mip-chain GPU-memory reduction, luminance
  SSIM at least 0.99, RGB PSNR at least 40 dB, and source-decode/DDS-parse
  speedup at least 10x.
- Normal map: exact mip-zero RGB, zero replacement alpha, mean angular error
  at most 0.01 degrees, and source-decode/DDS-parse speedup at least 10x.
- Geometry: cooked bytes no larger than the source GLB and source-import to
  cooked-read/validation speedup at least 2x.

The normal-map gate is intentionally a fidelity gate, not a compression
claim. Testing the 4096x4096 Sponza normal with a color-error-optimized BC7
path exposed unacceptable angular damage. Recipe v2 therefore stores normal
maps as RGBA8 DDS: exact authored RGB plus zero variance at mip zero, followed
by the renderer's established normalized vector and accumulated
LEADR/Toksvig-variance mips. Color and data textures remain BC7. The runtime
DDS upload path now accepts RGBA8 on adapters without BC support and preserves
the precomputed chain instead of regenerating scalar mips from level zero.

## Texture results

The color control was Intel Sponza
`curtain_fabric_red_BaseColor.png`, 4096x4096 with 13 mips and 15 timing
iterations:

- raw RGBA8 mip chain: 89,478,484 bytes;
- cooked BC7 mip chain: 22,369,648 bytes, a 74.99997% reduction;
- luminance SSIM: 0.9990341; RGB PSNR: 52.2254 dB;
- RGB RMSE: 0.00244755; maximum RGB byte error: 14;
- source decode mean: 156.7951 ms;
- DDS parse-for-upload mean: 0.458889 ms, a 341.68x speedup;
- CPU fallback parse/decode mean: 61.4309 ms;
- offline cook: 10,848.77 ms.

The normal control was Intel Sponza `curtain_fabric_Normal.png`, 4096x4096
with 13 mips and 5 timing iterations:

- source/raw and cooked mip chain: 89,478,484 bytes; the deliberate memory
  reduction is 0%;
- mip-zero RGB RMSE and maximum byte error: exactly zero;
- replacement alpha mean/maximum error: exactly zero;
- mean angular error: 0.000000331 degrees; maximum: 0.00000191 degrees;
- source decode mean: 191.3534 ms;
- DDS parse-for-upload mean: 1.939992 ms, a 98.64x speedup;
- CPU fallback parse/decode mean: 21.3539 ms;
- offline cook: 320.57 ms.

The fixed-block BC7 color artifact is larger than its entropy-compressed PNG,
and the RGBA8 normal DDS is 1.98x its PNG. This is expected and is recorded,
not hidden: the transport is selected for predictable GPU memory, precomputed
mips, fidelity, and load latency rather than minimum shipping bytes.

## Geometry result

The geometry control cooked `DamagedHelmet.glb` with the quantized32 vertex
format and eight requested hierarchy levels, then ran 25 iterations:

- source GLB: 3,773,916 bytes;
- cooked `.bgeo`: 1,446,496 bytes, 38.33% of the source size;
- archive: 632 clusters, 23 pages, 33,706 hierarchy triangles, and 1,363,968
  payload bytes;
- source glTF import mean: 97.2481 ms;
- complete cooked file read, structure validation, and hashes mean: 8.64054
  ms;
- measured speedup: 11.2549x.

The source timing includes glTF document parsing plus imported buffers and
images. The cooked timing includes reading every artifact byte and validating
the archive and hashes. Both exclude GPU resource creation/upload.

## Commands and qualification

The accepted reports came from:

```shell
tools/bloom-cook/target/release/bloom-cook texture-benchmark \
  examples/intel-sponza/assets/textures/curtain_fabric_red_BaseColor.png \
  --iterations 15

tools/bloom-cook/target/release/bloom-cook texture-benchmark \
  examples/intel-sponza/assets/textures/curtain_fabric_Normal.png \
  --normal --iterations 5

tools/bloom-cook/target/release/bloom-cook geometry \
  examples/renderer-test/assets/DamagedHelmet.glb /tmp/helmet.bgeo \
  --hierarchy-levels 8 --vertex-format quantized32

tools/bloom-cook/target/release/bloom-cook geometry-load-benchmark \
  examples/renderer-test/assets/DamagedHelmet.glb /tmp/helmet.bgeo \
  --iterations 25
```

Qualification results:

- `bloom-cook`: 43 release tests passed; strict Clippy and formatting passed;
- native shared library: 474 release tests passed, 1 intentional hot-reload
  ignore; strict correctness/suspicious/performance Clippy passed;
- `git diff --check`: passed;
- the file-line ratchet still reports the same five unrelated pre-existing
  oversized files. Every changed file is below 2,000 lines; the largest is
  879 lines.

This evidence qualifies only the cooked texture memory/quality and mesh load
time acceptance item. Full shipping-scene cooking and automatic
adapter/platform variant selection remain open.
