#!/usr/bin/env python3
"""Qualify Bloom's presented 2D frame and temporal 3D fallback in real Chrome."""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import math
import os
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCHEMA = "bloom-web-browser-smoke-v2"
TSR_WARMUP_FRAMES = 12
TSR_SEQUENCE_FRAMES = 8
TSR_MAX_NATIVE_FRAME_RMSE = 0.03
TSR_MAX_NATIVE_MOTION_DERIVATIVE_RMSE = 0.027
TSR_MIN_MOTION_ACTIVITY = 0.05
TSR_MIN_SCENE_LUMA_STDDEV = 12.0

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.quality.khronos_materials import png_rgb
from tools.quality.tsr_motion_compare import reference_metrics

SMOKE_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html,body { margin:0; width:100%; height:100%; overflow:hidden; background:#000 }
    #bloom-canvas { position:fixed; inset:0; width:100%; height:100%; display:block }
    #loading { position:fixed; left:4px; top:4px; color:white }
  </style>
</head>
<body>
<canvas id="bloom-canvas"></canvas><div id="loading">BOOTING</div>
<script>
window.__bloomReady = new Promise((resolve, reject) => {
  window.__bloomReadyResolve = resolve;
  window.__bloomReadyReject = reject;
});
globalThis.__joltFactory = async () => { throw new Error("physics omitted in render smoke"); };
const requestAdapter = GPU.prototype.requestAdapter;
GPU.prototype.requestAdapter = async function(...args) {
  const adapter = await requestAdapter.apply(this, args);
  globalThis.__bloomSmokeAdapter = adapter;
  return adapter;
};
const requestDevice = GPUAdapter.prototype.requestDevice;
GPUAdapter.prototype.requestDevice = async function(...args) {
  const device = await requestDevice.apply(this, args);
  globalThis.__bloomSmokeDevice = device;
  return device;
};
</script>
<script type="module">
import { bootBloomGame } from "./bloom_glue.js";
try {
  await window.__bloomReady;
  const bloom = await bootBloomGame();
  const device = globalThis.__bloomSmokeDevice;
  if (!device) throw new Error("Bloom did not request a WebGPU device");
  const helmetResponse = await fetch("./DamagedHelmet.glb");
  if (!helmetResponse.ok) throw new Error("Damaged Helmet fixture is missing");
  const helmet = bloom.bloom_load_model_bytes(
    new Uint8Array(await helmetResponse.arrayBuffer()),
  );
  if (!(helmet > 0)) throw new Error("Damaged Helmet fixture failed to load");
  device.pushErrorScope("validation");
  bloom.bloom_set_direct_2d_mode(1);
  bloom.bloom_clear_background(32, 112, 224, 255);
  bloom.bloom_begin_drawing();
  bloom.bloom_clear_background(32, 112, 224, 255);
  bloom.bloom_end_drawing();
  await device.queue.onSubmittedWorkDone();
  const validationError = await device.popErrorScope();
  if (validationError) throw new Error("WebGPU validation: " + validationError.message);
  let tsrScopeActive = false;
  const renderTsrFrame = async (cameraX) => {
    bloom.bloom_begin_drawing();
    bloom.bloom_clear_background(23, 34, 55, 255);
    bloom.bloom_begin_mode_3d(
      cameraX, 2.35, 6.2,
      0.0, 0.15, -0.65,
      0.0, 1.0, 0.0,
      48.0, 0.0,
    );
    bloom.bloom_draw_plane(0.0, -1.0, -0.5, 12.0, 12.0, 92, 104, 124, 255);
    bloom.bloom_draw_cube(-1.75, -0.1, -0.75, 1.15, 1.8, 1.15, 208, 84, 58, 255);
    bloom.bloom_draw_model(helmet, 0.0, -0.05, -0.85, 1.15, 255, 255, 255, 255);
    bloom.bloom_draw_cylinder(1.7, -0.05, -0.75, 0.58, 0.78, 1.9, 232, 185, 72, 255);
    for (let i = -5; i <= 5; i += 1) {
      bloom.bloom_draw_cube(i * 0.31, 0.15, -2.35, 0.028, 1.75, 0.04, 220, 226, 234, 255);
    }
    bloom.bloom_draw_cube(0.0, -0.42, -2.35, 3.2, 0.035, 0.05, 220, 226, 234, 255);
    bloom.bloom_draw_cube(0.0, 0.68, -2.35, 3.2, 0.035, 0.05, 220, 226, 234, 255);
    bloom.bloom_end_mode_3d();
    bloom.bloom_end_drawing();
    await new Promise((resolve) => requestAnimationFrame(resolve));
    await device.queue.onSubmittedWorkDone();
  };
  globalThis.__bloomSmokePrepareTsr = async (scale, warmupFrames, taaEnabled) => {
    if (!tsrScopeActive) {
      device.pushErrorScope("validation");
      tsrScopeActive = true;
    }
    bloom.bloom_set_direct_2d_mode(0);
    bloom.bloom_set_quality_preset(3);
    bloom.bloom_set_render_scale(scale);
    bloom.bloom_set_taa_enabled(taaEnabled ? 1 : 0);
    bloom.bloom_set_ssgi_enabled(1);
    bloom.bloom_set_ssr_enabled(1);
    bloom.bloom_set_shadows_enabled(1);
    bloom.bloom_set_motion_blur_enabled(0);
    bloom.bloom_set_auto_exposure(0);
    bloom.bloom_set_manual_exposure(1.0);
    bloom.bloom_set_sharpen_strength(0.0);
    bloom.bloom_set_film_grain(0.0);
    bloom.bloom_set_chromatic_aberration(0.0);
    bloom.bloom_set_env_intensity(0.35);
    bloom.bloom_set_ambient_light(0.18, 0.21, 0.28, 0.85);
    bloom.bloom_set_directional_light(-0.55, -1.0, -0.4, 1.0, 0.92, 0.78, 3.5);
    bloom.bloom_set_procedural_sky(1, 1.0, 1.0, 0.25);
    bloom.bloom_reset_temporal_history();
    for (let frame = 0; frame < warmupFrames; frame += 1) {
      await renderTsrFrame(-0.042);
    }
    return bloom.bloom_get_render_scale();
  };
  globalThis.__bloomSmokeRenderTsrFrame = renderTsrFrame;
  globalThis.__bloomSmokeFinishTsr = async () => {
    await device.queue.onSubmittedWorkDone();
    const error = tsrScopeActive ? await device.popErrorScope() : null;
    tsrScopeActive = false;
    return {
      validationError: error ? error.message : null,
      capabilities: JSON.parse(bloom.bloom_get_renderer_capabilities()),
    };
  };
  document.documentElement.dataset.bloomFrame =
    "direct-2d-clear-rgba-32-112-224-255";
  document.documentElement.dataset.bloomSmoke = "pass";
  document.getElementById("loading").remove();
} catch (error) {
  document.documentElement.dataset.bloomSmoke = "fail";
  document.documentElement.dataset.bloomError = String(error?.message ?? error);
  document.getElementById("loading").textContent = "BLOOM_WEB_SMOKE_FAIL";
}
</script>
</body>
</html>
"""


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        pass


def browser_path(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for candidate in (
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def receive_exact(stream: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = stream.recv(length - len(chunks))
        if not chunk:
            raise ConnectionError("browser closed the DevTools connection")
        chunks.extend(chunk)
    return bytes(chunks)


class DevTools:
    def __init__(self, url: str):
        from urllib.parse import urlparse

        parsed = urlparse(url)
        self.stream = socket.create_connection((parsed.hostname, parsed.port), timeout=10)
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        request = (
            f"GET {target} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "Origin: http://127.0.0.1\r\n\r\n"
        )
        self.stream.sendall(request.encode("ascii"))
        response = bytearray()
        while b"\r\n\r\n" not in response:
            response.extend(self.stream.recv(4096))
        if not response.startswith(b"HTTP/1.1 101"):
            raise ConnectionError(f"DevTools WebSocket handshake failed: {response[:200]!r}")
        self.next_id = 1

    def close(self) -> None:
        self.stream.close()

    def send_json(self, value: dict[str, Any]) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        mask = secrets.token_bytes(4)
        length = len(payload)
        header = bytearray([0x81])
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack(">H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack(">Q", length))
        header.extend(mask)
        header.extend(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self.stream.sendall(header)

    def receive_json(self) -> dict[str, Any]:
        first, second = receive_exact(self.stream, 2)
        opcode = first & 0x0F
        length = second & 0x7F
        if length == 126:
            length = struct.unpack(">H", receive_exact(self.stream, 2))[0]
        elif length == 127:
            length = struct.unpack(">Q", receive_exact(self.stream, 8))[0]
        if second & 0x80:
            mask = receive_exact(self.stream, 4)
            payload = bytes(
                byte ^ mask[index % 4]
                for index, byte in enumerate(receive_exact(self.stream, length))
            )
        else:
            payload = receive_exact(self.stream, length)
        if opcode == 8:
            raise ConnectionError("browser closed the DevTools target")
        if opcode != 1:
            return self.receive_json()
        return json.loads(payload.decode("utf-8"))

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        call_id = self.next_id
        self.next_id += 1
        self.send_json({"id": call_id, "method": method, "params": params or {}})
        while True:
            message = self.receive_json()
            if message.get("id") == call_id:
                if "error" in message:
                    raise RuntimeError(f"{method} failed: {message['error']}")
                return message.get("result", {})


def paeth(a: int, b: int, c: int) -> int:
    estimate = a + b - c
    distances = (abs(estimate - a), abs(estimate - b), abs(estimate - c))
    return (a, b, c)[distances.index(min(distances))]


def png_channel_means(path: Path) -> tuple[float, float, float]:
    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("browser screenshot is not a PNG")
    offset = 8
    width = height = color_type = bit_depth = interlace = None
    compressed = bytearray()
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", data
            )
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            break
    if not width or not height or bit_depth != 8 or color_type not in (2, 6) or interlace:
        raise ValueError("unsupported browser screenshot PNG layout")
    channels = 3 if color_type == 2 else 4
    stride = width * channels
    raw = zlib.decompress(bytes(compressed))
    rows: list[bytearray] = []
    cursor = 0
    previous = bytearray(stride)
    for _ in range(height):
        filter_kind = raw[cursor]
        cursor += 1
        encoded = raw[cursor : cursor + stride]
        cursor += stride
        row = bytearray(stride)
        for index, value in enumerate(encoded):
            left = row[index - channels] if index >= channels else 0
            up = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_kind == 0:
                decoded = value
            elif filter_kind == 1:
                decoded = value + left
            elif filter_kind == 2:
                decoded = value + up
            elif filter_kind == 3:
                decoded = value + ((left + up) // 2)
            elif filter_kind == 4:
                decoded = value + paeth(left, up, upper_left)
            else:
                raise ValueError(f"unsupported PNG filter {filter_kind}")
            row[index] = decoded & 0xFF
        rows.append(row)
        previous = row
    totals = [0, 0, 0]
    samples = 0
    # Ignore a thin browser-edge band and average the presented canvas.
    for row in rows[max(1, height // 20) : height - max(1, height // 20)]:
        for index in range(max(1, width // 20) * channels, (width - max(1, width // 20)) * channels, channels):
            totals[0] += row[index]
            totals[1] += row[index + 1]
            totals[2] += row[index + 2]
            samples += 1
    return tuple(total / samples for total in totals)  # type: ignore[return-value]


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def free_local_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def devtools_target(port: int, url: str, deadline: float) -> str | None:
    endpoint = f"http://127.0.0.1:{port}/json/list"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(endpoint, timeout=1) as response:
                targets = json.load(response)
            for target in targets:
                if target.get("type") == "page" and target.get("url") == url:
                    return str(target["webSocketDebuggerUrl"])
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.1)
    return None


def evaluated_value(devtools: DevTools, expression: str) -> Any:
    result = devtools.call(
        "Runtime.evaluate",
        {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": True,
        },
    )
    return result.get("result", {}).get("value")


def capture_screenshot(devtools: DevTools, path: Path) -> None:
    capture = devtools.call(
        "Page.captureScreenshot",
        {"format": "png", "fromSurface": True},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(capture["data"]))


def load_rgb_sequence(
    paths: Sequence[Path],
) -> tuple[int, int, list[list[tuple[int, int, int]]]]:
    if len(paths) < 2:
        raise ValueError("TSR browser gate requires at least two frames")
    dimensions: tuple[int, int] | None = None
    frames: list[list[tuple[int, int, int]]] = []
    for path in paths:
        width, height, pixels = png_rgb(path)
        if dimensions is None:
            dimensions = (width, height)
        elif dimensions != (width, height):
            raise ValueError("TSR browser sequence dimensions changed")
        frames.append(pixels)
    assert dimensions is not None
    return dimensions[0], dimensions[1], frames


def sequence_activity(frames: Sequence[Sequence[tuple[int, int, int]]]) -> float:
    if len(frames) < 2:
        raise ValueError("motion activity requires at least two frames")
    absolute_difference = 0
    samples = 0
    for previous, current in zip(frames, frames[1:]):
        if len(previous) != len(current):
            raise ValueError("motion sequence pixel count changed")
        for prior_pixel, current_pixel in zip(previous, current):
            for channel in range(3):
                absolute_difference += abs(current_pixel[channel] - prior_pixel[channel])
                samples += 1
    return absolute_difference / samples


def luma_stddev(frame: Sequence[tuple[int, int, int]]) -> float:
    values = [
        0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
        for pixel in frame
    ]
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def enforce_tsr_limits(metrics: dict[str, float]) -> list[str]:
    failures = []
    if metrics["native_frame_rmse"] > TSR_MAX_NATIVE_FRAME_RMSE:
        failures.append(
            "Web 0.75 TSR frame RMSE "
            f"{metrics['native_frame_rmse']:.9f} exceeds "
            f"{TSR_MAX_NATIVE_FRAME_RMSE:.9f}"
        )
    if (
        metrics["native_motion_derivative_rmse"]
        > TSR_MAX_NATIVE_MOTION_DERIVATIVE_RMSE
    ):
        failures.append(
            "Web 0.75 TSR motion-derivative RMSE "
            f"{metrics['native_motion_derivative_rmse']:.9f} exceeds "
            f"{TSR_MAX_NATIVE_MOTION_DERIVATIVE_RMSE:.9f}"
        )
    if metrics["native_motion_activity_rgb_8bit"] < TSR_MIN_MOTION_ACTIVITY:
        failures.append("native WebGPU negative control did not move")
    if metrics["fractional_motion_activity_rgb_8bit"] < TSR_MIN_MOTION_ACTIVITY:
        failures.append("fractional WebGPU negative control did not move")
    if metrics["fractional_scene_luma_stddev_8bit"] < TSR_MIN_SCENE_LUMA_STDDEV:
        failures.append("WebGPU 3D TSR scene is flat or missing")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "target" / "ci" / "web-smoke"),
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for sequence_dir in ("native", "fractional", "fractional-no-taa"):
        shutil.rmtree(out_dir / sequence_dir, ignore_errors=True)
    package_dir = REPO_ROOT / "native" / "web" / "pkg"
    required = [package_dir / "bloom_web.js", package_dir / "bloom_web_bg.wasm"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        write_report(
            out_dir / "result.json",
            {"schema": REPORT_SCHEMA, "status": "fail", "failures": [f"missing {missing}"]},
        )
        print(f"FAIL: wasm-pack output missing: {missing}")
        return 1
    browser = browser_path(args.browser)
    if browser is None:
        print("FAIL: Chrome/Chromium is required for the web browser smoke")
        return 2

    site = out_dir / "site"
    if site.exists():
        shutil.rmtree(site)
    site.mkdir()
    shutil.copytree(package_dir, site / "pkg")
    shutil.copy2(REPO_ROOT / "native" / "web" / "bloom_glue.js", site)
    shutil.copy2(REPO_ROOT / "native" / "web" / "jolt_bridge.js", site)
    shutil.copy2(
        REPO_ROOT / "examples" / "renderer-test" / "assets" / "DamagedHelmet.glb",
        site,
    )
    (site / "index.html").write_text(SMOKE_HTML, encoding="utf-8")

    handler = lambda *handler_args, **handler_kwargs: QuietHandler(
        *handler_args, directory=str(site), **handler_kwargs
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/index.html"
    screenshot = out_dir / "frame.png"
    screenshot.unlink(missing_ok=True)
    profile = tempfile.mkdtemp(prefix="bloom-web-smoke-")
    debug_port = free_local_port()
    command = [browser, "--headless=new"]
    command.extend(
        [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu-sandbox",
            "--enable-unsafe-webgpu",
            "--ignore-gpu-blocklist",
        ]
    )
    if sys.platform.startswith("linux"):
        # Hosted Linux runners have no display GPU. These are Chromium's own
        # WebGPU SwiftShader test switches: explicitly select the software
        # adapter and initialize ANGLE for canvas/compositor interop.
        command.extend(
            [
                "--use-webgpu-adapter=swiftshader",
                "--use-gpu-in-tests",
                "--enable-accelerated-2d-canvas",
            ]
        )
    command.extend(
        [
            "--remote-allow-origins=*",
            f"--remote-debugging-port={debug_port}",
            "--window-size=320,240",
            f"--user-data-dir={profile}",
            url,
        ]
    )
    started = time.perf_counter()
    stdout_log = out_dir / "browser.stdout.log"
    stderr_log = out_dir / "browser.stderr.log"
    process: subprocess.Popen[bytes] | None = None
    devtools: DevTools | None = None
    marker = "pending"
    browser_error: str | None = None
    adapter_info: dict[str, Any] | None = None
    frame_signature: str | None = None
    renderer_capabilities: dict[str, Any] | None = None
    native_paths: list[Path] = []
    fractional_paths: list[Path] = []
    no_taa_paths: list[Path] = []
    tsr_metrics: dict[str, float] | None = None
    failures: list[str] = []
    try:
        with stdout_log.open("wb") as stdout, stderr_log.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
            )
        deadline = time.monotonic() + args.timeout
        target = devtools_target(debug_port, url, deadline)
        if target is None:
            failures.append("DevTools did not expose the Bloom smoke page")
        else:
            devtools = DevTools(target)
            devtools.call("Page.enable")
            devtools.call("Runtime.enable")
            while time.monotonic() < deadline:
                marker = str(
                    evaluated_value(
                        devtools,
                        'document.documentElement.dataset.bloomSmoke || "pending"',
                    )
                )
                if marker in ("pass", "fail"):
                    break
                time.sleep(0.1)
            raw_adapter_info = evaluated_value(
                devtools,
                """(() => {
                  const adapter = globalThis.__bloomSmokeAdapter;
                  if (!adapter) return null;
                  const info = adapter.info ?? {};
                  return {
                    vendor: info.vendor ?? "",
                    architecture: info.architecture ?? "",
                    device: info.device ?? "",
                    description: info.description ?? "",
                    backend: info.backend ?? "",
                    type: info.type ?? "",
                  };
                })()""",
            )
            if isinstance(raw_adapter_info, dict):
                adapter_info = raw_adapter_info
            if marker == "fail":
                browser_error = str(
                    evaluated_value(
                        devtools,
                        'document.documentElement.dataset.bloomError || "unknown error"',
                    )
                )
                failures.append(f"Bloom browser frame failed: {browser_error}")
            elif marker != "pass":
                failures.append("browser timed out before completing a Bloom frame")
            else:
                raw_frame_signature = evaluated_value(
                    devtools,
                    "document.documentElement.dataset.bloomFrame || null",
                )
                if isinstance(raw_frame_signature, str):
                    frame_signature = raw_frame_signature
                if frame_signature != "direct-2d-clear-rgba-32-112-224-255":
                    failures.append("browser did not report the submitted known frame")
            capture_screenshot(devtools, screenshot)
            if marker == "pass":
                motion_positions = [
                    -0.042 + 0.084 * frame / (TSR_SEQUENCE_FRAMES - 1)
                    for frame in range(TSR_SEQUENCE_FRAMES)
                ]
                for label, scale, taa_enabled, paths in (
                    ("native", 1.0, True, native_paths),
                    ("fractional", 0.75, True, fractional_paths),
                    ("fractional-no-taa", 0.75, False, no_taa_paths),
                ):
                    actual_scale = evaluated_value(
                        devtools,
                        "window.__bloomSmokePrepareTsr("
                        f"{scale}, {TSR_WARMUP_FRAMES}, "
                        f"{str(taa_enabled).lower()})",
                    )
                    if not isinstance(actual_scale, (int, float)) or not math.isclose(
                        float(actual_scale), scale, abs_tol=1e-6
                    ):
                        failures.append(
                            f"WebGPU {label} render scale did not apply: {actual_scale!r}"
                        )
                    for frame, camera_x in enumerate(motion_positions):
                        evaluated_value(
                            devtools,
                            f"window.__bloomSmokeRenderTsrFrame({camera_x!r})",
                        )
                        path = out_dir / label / f"sequence-{frame:03}.png"
                        capture_screenshot(devtools, path)
                        paths.append(path)
                finish = evaluated_value(
                    devtools,
                    "window.__bloomSmokeFinishTsr()",
                )
                if not isinstance(finish, dict):
                    failures.append("WebGPU TSR fixture did not return final diagnostics")
                else:
                    if finish.get("validationError"):
                        failures.append(
                            f"WebGPU TSR validation: {finish['validationError']}"
                        )
                    raw_capabilities = finish.get("capabilities")
                    if isinstance(raw_capabilities, dict):
                        renderer_capabilities = raw_capabilities
                        runtime = raw_capabilities.get("runtime_support", {})
                        if runtime.get("hardware_ray_query") is not False:
                            failures.append(
                                "browser fallback gate unexpectedly enabled hardware ray query"
                            )
                        if runtime.get("ssgi_trace_backend") not in (
                            "hiz-screen",
                            "sdf-clipmap",
                        ):
                            failures.append(
                                "browser did not report an active software SSGI fallback"
                            )
                    else:
                        failures.append("browser renderer capability report is missing")
    except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
        failures.append(f"browser automation failed: {exc}")
    finally:
        if devtools is not None:
            devtools.close()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        shutil.rmtree(profile, ignore_errors=True)
        shutil.rmtree(site, ignore_errors=True)

    if process is not None and process.returncode not in (0, -9, -15):
        failures.append(f"browser exited {process.returncode}")
    means: tuple[float, float, float] | None = None
    if marker == "pass" and not screenshot.is_file():
        failures.append("DevTools did not produce a screenshot")
    elif screenshot.is_file():
        try:
            means = png_channel_means(screenshot)
            red, green, blue = means
            if not (blue > green + 30 and green > red + 30 and blue > 180):
                failures.append(
                    f"browser screenshot is not the known blue frame: {means}"
                )
        except (OSError, ValueError, zlib.error) as exc:
            failures.append(f"cannot validate browser screenshot: {exc}")
    if (
        len(native_paths) == TSR_SEQUENCE_FRAMES
        and len(fractional_paths) == TSR_SEQUENCE_FRAMES
        and len(no_taa_paths) == TSR_SEQUENCE_FRAMES
    ):
        try:
            native_width, native_height, native_frames = load_rgb_sequence(native_paths)
            fractional_width, fractional_height, fractional_frames = load_rgb_sequence(
                fractional_paths
            )
            no_taa_width, no_taa_height, no_taa_frames = load_rgb_sequence(no_taa_paths)
            if len(
                {
                    (native_width, native_height),
                    (fractional_width, fractional_height),
                    (no_taa_width, no_taa_height),
                }
            ) != 1:
                raise ValueError("browser TSR control dimensions differ")
            tsr_metrics = reference_metrics(fractional_frames, native_frames)
            no_taa_metrics = reference_metrics(no_taa_frames, native_frames)
            tsr_metrics.update(
                {
                    "native_motion_activity_rgb_8bit": sequence_activity(native_frames),
                    "fractional_motion_activity_rgb_8bit": sequence_activity(
                        fractional_frames
                    ),
                    "fractional_scene_luma_stddev_8bit": luma_stddev(
                        fractional_frames[-1]
                    ),
                    "no_taa_native_frame_rmse": no_taa_metrics[
                        "native_frame_rmse"
                    ],
                    "no_taa_native_motion_derivative_rmse": no_taa_metrics[
                        "native_motion_derivative_rmse"
                    ],
                }
            )
            failures.extend(enforce_tsr_limits(tsr_metrics))
        except (OSError, ValueError, zlib.error) as exc:
            failures.append(f"cannot validate WebGPU TSR sequence: {exc}")
    elif marker == "pass":
        failures.append("WebGPU TSR sequence is incomplete")
    report = {
        "schema": REPORT_SCHEMA,
        "status": "fail" if failures else "pass",
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "browser": browser,
        "command": command,
        "url": url,
        "dom_marker": marker,
        "browser_error": browser_error,
        "adapter_info": adapter_info,
        "screenshot": "frame.png" if screenshot.is_file() else None,
        "frame_signature": frame_signature,
        "compositor_screenshot_rgb_means": means,
        "tsr_native_match": {
            "warmup_frames": TSR_WARMUP_FRAMES,
            "sequence_frames": TSR_SEQUENCE_FRAMES,
            "native_render_scale": 1.0,
            "fractional_render_scale": 0.75,
            "metrics": tsr_metrics,
            "limits": {
                "max_native_frame_rmse": TSR_MAX_NATIVE_FRAME_RMSE,
                "max_native_motion_derivative_rmse":
                    TSR_MAX_NATIVE_MOTION_DERIVATIVE_RMSE,
                "min_motion_activity_rgb_8bit": TSR_MIN_MOTION_ACTIVITY,
                "min_fractional_scene_luma_stddev_8bit":
                    TSR_MIN_SCENE_LUMA_STDDEV,
            },
            "native_directory": "native" if native_paths else None,
            "fractional_directory": "fractional" if fractional_paths else None,
            "no_taa_control_directory": "fractional-no-taa" if no_taa_paths else None,
        },
        "renderer_capabilities": renderer_capabilities,
        "stdout": stdout_log.name,
        "stderr": stderr_log.name,
        "failures": failures,
    }
    write_report(out_dir / "result.json", report)
    if failures:
        print("FAIL: web browser smoke")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"PASS: browser presented known frame; RGB means={means}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
