#!/usr/bin/env python3
"""Render the complete, unchanged installed starter website in hosted Chrome."""

import argparse
import base64
import hashlib
import http.server
import json
import math
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
from tools.ci.compiled_web_smoke import evaluate
from tools.ci.web_smoke import QuietHandler, DevTools, browser_path, devtools_target, free_local_port
from tools.quality.khronos_materials import png_rgb


def validate_state(state):
    if not isinstance(state, dict) or state.get('errors') or state.get('stopped') is not True:
        raise RuntimeError(f'starter failed or did not stop normally: {state}')
    for name, expected in [('frames', 8), ('cleanups', 1), ('registrations', 1)]:
        if type(state.get(name)) is not int or state[name] != expected:
            raise RuntimeError(f'starter {name} must be {expected}')
    reads = state.get('reads', [])
    if len(reads) != 1 or reads[0].get('path') != 'assets/welcome.txt' or reads[0].get('value', '').strip() != 'Hello, Bloom!':
        raise RuntimeError('starter did not read the actual welcome asset')
    if not any(item.get('path', '').endswith('/assets/welcome.txt') and item.get('status') == 200 for item in state.get('fetches', [])):
        raise RuntimeError('starter did not fetch its welcome asset successfully')
    texts, rects = state.get('texts', []), state.get('rects', [])
    if state.get('clears') != [[0, 0, 0, 255] for _ in range(8)]:
        raise RuntimeError('each starter frame must clear to opaque black')
    if len(texts) != 8 or len(rects) != 8:
        raise RuntimeError('each starter frame must draw its text and square')
    for text in texts:
        if len(text) != 8 or text[0].strip() != 'Hello, Bloom!' or text[1:] != [24, 24, 24, 255, 255, 255, 255]:
            raise RuntimeError('starter text draw differs from the shipped template')
    for rect in rects:
        if len(rect) != 8 or not isinstance(rect[0], (int, float)) or not math.isfinite(rect[0]) or not 288 <= rect[0] <= 448 or rect[1:] != [193, 64, 64, 255, 255, 255, 255]:
            raise RuntimeError('starter square draw differs from the shipped template')
    return rects[-1]


def check_pixels(width, height, pixels, rect):
    if (width, height) != (800, 450) or len(pixels) != 800 * 450:
        raise RuntimeError('starter capture must contain the complete 800x450 viewport')
    x = rect[0]
    text_pixels = 0
    for index, pixel in enumerate(pixels):
        px, py = index % width, index // width
        text_region = 24 <= px < 250 and 24 <= py < 56
        square_region = math.floor(x) - 1 <= px <= math.ceil(x + 64) + 1 and 192 <= py <= 258
        if math.ceil(x) + 1 <= px < math.floor(x + 64) - 1 and 194 <= py < 256:
            if pixel != (255, 255, 255):
                raise RuntimeError('starter square interior is missing or incorrect')
        elif text_region:
            if max(pixel) >= 32: text_pixels += 1
        elif not square_region and pixel != (0, 0, 0):
            raise RuntimeError('starter background contains unexpected content')
    if not 100 <= text_pixels <= 3000:
        raise RuntimeError(f'starter text raster is missing or filled: {text_pixels} lit pixels')
    return dict(width=width, height=height, text_pixels=text_pixels, square_x=x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser')
    parser.add_argument('--game', type=Path, default=ROOT / 'target/ci/starter-web')
    parser.add_argument('--out', type=Path, default=ROOT / 'target/ci/web-smoke/starter')
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0: parser.error('timeout must be finite and positive')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = dict(schema='bloom-starter-browser-v1', status='running', failures=[])
    def save(): (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    process = devtools = server = thread = None
    parent = Path(tempfile.gettempdir()).resolve()
    temporary = Path(tempfile.mkdtemp(prefix='bloom-starter-browser-', dir=parent)).resolve()
    try:
        prepared = json.loads((args.game / 'result.json').read_text(encoding='utf-8'))
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        if prepared.get('status') != 'pass' or prepared.get('source_commit') != head or not prepared.get('web', {}).get('source_unchanged'):
            raise RuntimeError('starter artifact did not qualify the unchanged source at this checkout')
        if prepared.get('web_command', {}).get('status') != 'pass':
            raise RuntimeError('installed bloom run --web did not qualify its served website')
        site = (args.game / 'site').resolve()
        actual = {}
        for current, dirs, files in os.walk(site, followlinks=False):
            directory = Path(current)
            if any((directory / name).is_symlink() or (directory / name).is_junction() for name in dirs + files):
                raise RuntimeError('starter artifact contains an unexpected filesystem link')
            for name in files:
                file = directory / name
                actual[file.relative_to(site).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        if actual != prepared['web']['files']: raise RuntimeError('starter website differs from its build receipt')
        report['prepared'] = prepared
        browser = browser_path(args.browser)
        if browser is None: raise RuntimeError('Chrome/Chromium is required for full starter acceptance')
        handler = lambda *a, **kw: QuietHandler(*a, directory=str(site), **kw)
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        debug_port = free_local_port()
        command = [browser, '--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu-sandbox',
                   '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--remote-allow-origins=*',
                   f'--remote-debugging-port={debug_port}', '--window-size=900,600', f'--user-data-dir={temporary / "profile"}', 'about:blank']
        report['browser_command'] = command
        with (out / 'browser.stdout.log').open('wb') as stdout, (out / 'browser.stderr.log').open('wb') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        target = devtools_target(debug_port, 'about:blank', time.monotonic() + args.timeout)
        if target is None: raise RuntimeError('browser did not expose the owned starter page')
        devtools = DevTools(target)
        devtools.call('Page.enable')
        devtools.call('Runtime.enable')
        devtools.call('Emulation.setDeviceMetricsOverride', dict(width=800, height=450, deviceScaleFactor=1, mobile=False))
        devtools.call('Page.addScriptToEvaluateOnNewDocument', {'source': (ROOT / 'tools/ci/starter_monitor.js').read_text()})
        url = f'http://127.0.0.1:{server.server_port}/'
        devtools.call('Page.navigate', {'url': url})
        started = time.monotonic()
        state = None
        while time.monotonic() - started < args.timeout:
            state = evaluate(devtools, f'location.href === {json.dumps(url)} ? globalThis.__starterProbe || null : null')
            if isinstance(state, dict) and (state.get('errors') or state.get('cleanups')): break
            time.sleep(0.1)
        report['state'] = state
        report['duration_seconds'] = round(time.monotonic() - started, 3)
        save()
        rect = validate_state(state)
        evaluate(devtools, 'new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
        final_state = evaluate(devtools, 'globalThis.__starterProbe')
        validate_state(final_state)
        report['state'] = final_state
        capture = devtools.call('Page.captureScreenshot', {'format': 'png', 'fromSurface': True})
        frame = out / 'starter.png'
        frame.write_bytes(base64.b64decode(capture['data']))
        report['frame'] = {**check_pixels(*png_rgb(frame), rect), 'sha256': hashlib.sha256(frame.read_bytes()).hexdigest()}
        report['adapter'] = evaluate(devtools, '(() => { const i = globalThis.__starterAdapter?.info; return i ? {vendor:i.vendor,architecture:i.architecture,device:i.device,description:i.description} : null; })()')
        report['status'] = 'pass'
        print('PASS: unchanged installed starter reads its asset, renders text/square, and cleans up once in Chrome')
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report['status'] = 'fail'
        report['failures'].append(str(error))
        if devtools is not None:
            try:
                capture = devtools.call('Page.captureScreenshot', {'format': 'png', 'fromSurface': True})
                (out / 'failure.png').write_bytes(base64.b64decode(capture['data']))
            except (OSError, ValueError, RuntimeError, KeyError) as capture_error:
                report['failure_capture_error'] = str(capture_error)
        print(f'FAIL: {error}')
        return 1
    finally:
        if devtools is not None: devtools.close()
        if process is not None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None: thread.join(timeout=2)
        save()
        if temporary.parent != parent or temporary.is_symlink() or temporary.is_junction():
            raise RuntimeError('refusing cleanup outside owned starter browser profile')
        shutil.rmtree(temporary)


if __name__ == '__main__':
    raise SystemExit(main())
