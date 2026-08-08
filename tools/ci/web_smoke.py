#!/usr/bin/env python3
"""Boot Bloom WebGPU in a real browser and verify a presented known-color frame."""

from __future__ import annotations

import argparse
import base64
import http.server
import json
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
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCHEMA = "bloom-web-browser-smoke-v1"

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
  device.pushErrorScope("validation");
  bloom.bloom_set_direct_2d_mode(1);
  bloom.bloom_clear_background(32, 112, 224, 255);
  bloom.bloom_begin_drawing();
  bloom.bloom_clear_background(32, 112, 224, 255);
  bloom.bloom_end_drawing();
  await device.queue.onSubmittedWorkDone();
  const validationError = await device.popErrorScope();
  if (validationError) throw new Error("WebGPU validation: " + validationError.message);
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
            capture = devtools.call(
                "Page.captureScreenshot",
                {"format": "png", "fromSurface": True},
            )
            screenshot.write_bytes(base64.b64decode(capture["data"]))
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
