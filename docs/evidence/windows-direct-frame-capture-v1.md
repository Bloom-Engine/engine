# Direct-frame PNG capture

An installed native game using `setDirect2DMode(true)` rendered and exited but
never produced its requested PNG. `Renderer::end_frame()` submitted the draws
without servicing the screenshot request; the normal scene path already had
framebuffer readback support.

The direct path now records the existing output-texture readback after drawing,
submits it and completes the capture before presentation. Frames with no capture
keep their existing submission behavior. Render-target-only frames leave the
output request pending. Requests for scene diagnostic attachments wait for an
eligible scene frame. The existing readback implementation is reused unchanged.

The direct frame method is extracted to `renderer/direct_frame.rs` to keep the
large renderer module shrinking. Removing the capture addition restores the
previous method text exactly; an extraction receipt is retained with the evidence.

## Validation

The installed-package checker now compiles its actual Jolt physics fixture in
both scene and direct-2D modes. Each starts on physical Radeon 760M DX12 and
Vulkan, and all four PNG files match all 16,384 expected pixels. They have the
same SHA-256:
`8a509d87d3aa3fab96e0a9e2c67228e187f1bc0853cb4726aa5844799440f30e`.
The first native compile takes 246.453 seconds and the second 3.125 seconds;
startup/capture takes 3.203 to 5.859 seconds. These are test durations, not FPS.

A renderer regression checks uncaptured frames, capture deferral across a
smaller render texture and two successive output captures with different exact
pixels. All five render-target tests pass on DX12 and Vulkan. Each backend also
passes all 93 image goldens with the original assertions and four existing
ignored cases. Contracts, formatting and strict Clippy pass. The first WASM
check caught a misplaced native-only module guard during extraction; restoring
it to `quality_capture` and keeping `direct_frame` available on both platforms
passes the WASM compile check. The failure and correction are retained.

The first checker result hashes the normalized UTF-8 entry text. The final
checker hashes the actual written entry file and also records installed renderer
source hashes. That metadata correction does not change the rendering fixture.

This is native headless capture acceptance with the qualified Perry 0.5.1220
runtime profile and local SDK DXC on PATH. Visible presentation, packaged DXC,
browser capture/startup, general long native paths and the complete starter
lifecycle remain separate requirements. No image baselines or thresholds change.

Raw results are retained under
`tools/quality/out/windows-engine-plan/direct-2d-capture/`, including the original
missing-capture failure from the installed-native diagnosis.
