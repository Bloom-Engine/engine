#!/usr/bin/env python3
"""CI acceptance for a real Perry-compiled game, its frame, cleanup and trap control."""

import argparse
import base64
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.ci.native_package_smoke import check_frame
from tools.ci.web_smoke import QuietHandler, DevTools, browser_path, devtools_target, free_local_port

MONITOR = """
(() => {
globalThis.__compiledGameErrors = [];
const describe = (value) => String(value?.stack || value).slice(0, 16384);
const record = (value) => { if (__compiledGameErrors.length < 32) __compiledGameErrors.push(describe(value)); };
const originalError = console.error;
console.error = (...args) => { record(args.map(describe).join(' ')); originalError.apply(console, args); };
addEventListener('error', (event) => record(event.error || event.message || 'script load error'));
addEventListener('unhandledrejection', (event) => record(event.reason));
globalThis.__joltFactory = async () => { throw new Error('physics omitted in compiled render smoke'); };
if (globalThis.GPU) {
  const request = GPU.prototype.requestAdapter;
  GPU.prototype.requestAdapter = async function(...args) {
    const adapter = await request.apply(this, args);
    globalThis.__compiledGameAdapter = adapter;
    return adapter;
  };
}
})();
"""


def evaluate(devtools, expression):
    result = devtools.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
    if result.get("exceptionDetails"):
        raise RuntimeError(f"browser evaluation failed: {result['exceptionDetails']}")
    return result.get("result", {}).get("value")


def validate_state(name, state):
    if name == "game":
        if state["errors"] or state["frames"] != "8" or state["cleanups"] != "1":
            raise RuntimeError(f"compiled game failed startup/frame/cleanup acceptance: {state}")
    elif state["expectedFault"] != "BLOOM_EXPECTED_STARTUP_FAILURE" or not state["errors"] or state["frames"] is not None or state["cleanups"] is not None:
        raise RuntimeError(f"intentional compiled startup failure was not rejected: {state}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser")
    parser.add_argument("--game", type=Path, default=ROOT / "target/ci/compiled-web-game")
    parser.add_argument("--out", type=Path, default=ROOT / "target/ci/web-smoke/compiled-game")
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "compiled-game.png").unlink(missing_ok=True)
    report = {"schema": "bloom-compiled-web-smoke-v1", "status": "running", "cases": [], "failures": []}
    def save():
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    process = devtools = server = thread = None
    temp_parent = Path(tempfile.gettempdir()).resolve()
    temporary = Path(tempfile.mkdtemp(prefix="bloom-compiled-web-", dir=temp_parent)).resolve()
    try:
        compiler_report = json.loads((args.game / "result.json").read_text(encoding="utf-8"))
        if compiler_report.get("status") != "pass" or compiler_report.get("compiler_version") != "perry 0.5.1220":
            raise RuntimeError("compiled-game artifact did not pass with the required Perry version")
        current_source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if compiler_report.get("source_commit") != current_source:
            raise RuntimeError("compiled game and browser engine checkout have different source commits")
        report["compiled_game"] = compiler_report
        engine_package = ROOT / "native/web/pkg"
        engine_wasm = engine_package / "bloom_web_bg.wasm"
        report["engine_wasm_sha256"] = hashlib.sha256(engine_wasm.read_bytes()).hexdigest()
        site = temporary / "site"
        site.mkdir()
        shutil.copytree(engine_package, site / "pkg")
        for name in ("bloom_glue.js", "game_loop.mjs", "jolt_bridge.js"):
            shutil.copyfile(ROOT / "native/web" / name, site / name)
        for name in ("game", "trap-control"):
            page = args.game / f"{name}.html"
            identity = next(p for p in compiler_report["pages"] if p["name"] == name)
            if hashlib.sha256(page.read_bytes()).hexdigest() != identity["html_sha256"]:
                raise RuntimeError(f"{name}: compiled page hash differs from its compiler receipt")
            shutil.copyfile(page, site / page.name)
        browser = browser_path(args.browser)
        if browser is None:
            raise RuntimeError("Chrome/Chromium is required for compiled-game acceptance")
        handler = lambda *a, **kw: QuietHandler(*a, directory=str(site), **kw)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        debug_port = free_local_port()
        command = [browser, "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
                   "--disable-gpu-sandbox", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist",
                   "--remote-allow-origins=*", f"--remote-debugging-port={debug_port}",
                   "--window-size=320,240", f"--user-data-dir={temporary / 'profile'}", "about:blank"]
        if sys.platform.startswith("linux"):
            command[1:1] = ["--use-webgpu-adapter=swiftshader", "--use-gpu-in-tests", "--enable-accelerated-2d-canvas"]
        report["browser_command"] = command
        with (out / "browser.stdout.log").open("wb") as stdout, (out / "browser.stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        target = devtools_target(debug_port, "about:blank", time.monotonic() + args.timeout)
        if target is None:
            raise RuntimeError("browser did not expose the owned test page")
        devtools = DevTools(target)
        devtools.call("Page.enable")
        devtools.call("Runtime.enable")
        devtools.call("Emulation.setDeviceMetricsOverride", {"width": 128, "height": 128, "deviceScaleFactor": 1, "mobile": False})
        devtools.call("Page.addScriptToEvaluateOnNewDocument", {"source": MONITOR})
        for name in ("game", "trap-control"):
            if name != "game":
                evaluate(devtools, "localStorage.clear()")
            url = f"http://127.0.0.1:{server.server_port}/{name}.html"
            devtools.call("Page.navigate", {"url": url})
            started = time.monotonic()
            state = None
            while time.monotonic() - started < args.timeout:
                state = evaluate(devtools, "(() => { if (location.href !== " + json.dumps(url) +
                                 " || !Array.isArray(globalThis.__compiledGameErrors)) return null; return ({" + """
                  frames: localStorage.getItem('bloom_fs:compiled-web-frames'),
                  cleanups: localStorage.getItem('bloom_fs:compiled-web-cleanups'),
                  expectedFault: localStorage.getItem('bloom_fs:compiled-web-expected-fault'),
                  errors: globalThis.__compiledGameErrors || [],
                }); })()""")
                if isinstance(state, dict) and (state["errors"] or state["cleanups"] is not None):
                    break
                time.sleep(0.1)
            if not isinstance(state, dict):
                raise RuntimeError(f"{name}: no game state before timeout")
            case = {"name": name, "state": state, "duration_seconds": round(time.monotonic() - started, 3)}
            report["cases"].append(case)
            save()
            validate_state(name, state)
            if name == "game":
                evaluate(devtools, "new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
                capture = devtools.call("Page.captureScreenshot", {"format": "png", "fromSurface": True})
                frame = out / "compiled-game.png"
                frame.write_bytes(base64.b64decode(capture["data"]))
                case["frame"] = check_frame(frame)
                case["adapter"] = evaluate(devtools, "(() => { const i = globalThis.__compiledGameAdapter?.info; return i ? { vendor:i.vendor, architecture:i.architecture, device:i.device, description:i.description } : null; })()")
            case["status"] = "pass"
            save()
        report["status"] = "pass"
        print("PASS: actual Perry game rendered its exact frame and cleaned up; startup trap control rejected")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, StopIteration, subprocess.SubprocessError) as exc:
        report["status"] = "fail"
        report["failures"].append(str(exc))
        print(f"FAIL: {exc}")
        return 1
    finally:
        if devtools is not None:
            devtools.close()
        if process is not None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)
        save()
        if temporary.parent != temp_parent or temporary.is_symlink() or temporary.is_junction():
            raise RuntimeError(f"refusing cleanup outside owned temporary directory: {temporary}")
        shutil.rmtree(temporary)


if __name__ == "__main__":
    raise SystemExit(main())
