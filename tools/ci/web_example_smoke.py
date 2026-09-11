#!/usr/bin/env python3
"""Require real browser render/content/cleanup for six original game examples."""

import argparse
import base64
import hashlib
import http.server
import json
import math
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
from tools.ci.example_runtime import EXAMPLES, check_frame
from tools.ci.web_smoke import QuietHandler, DevTools, browser_path, devtools_target, free_local_port


def validate_state(name, state):
    if not isinstance(state, dict) or state.get('errors') != [] or state.get('badArguments') != [] or state.get('stopped') is not True:
        raise RuntimeError(f'{name}: runtime errors, invalid drawing arguments or abnormal stop: {state}')
    if any(type(state.get(key)) is not int or state[key] != value for key, value in [('frames', 8), ('cleanups', 1), ('registrations', 1)]):
        raise RuntimeError(f'{name}: require eight frames, one loop and one completed cleanup')
    if state.get('windows') != [list(EXAMPLES[name])]: raise RuntimeError(f'{name}: incorrect requested viewport')
    draws = state.get('frameDraws', [])
    if len(draws) != 8 or any(type(n) is not int or n <= 0 for n in draws):
        raise RuntimeError(f'{name}: each frame must draw game content')
    calls = state.get('calls', {})
    if calls.get('bloom_close_window', 0) < 1: raise RuntimeError(f'{name}: no normal engine stop')
    if name == 'space-blaster' and (calls.get('bloom_init_audio') != 1 or calls.get('bloom_close_audio') != 1):
        raise RuntimeError('space-blaster: audio must initialize and clean up once')
    if name == 'voxel-sandbox' and (calls.get('bloom_disable_cursor', 0) < 1 or calls.get('bloom_enable_cursor') != 1):
        raise RuntimeError('voxel-sandbox: cursor intent must be restored during cleanup')


def validate_prepared(prepared, head):
    if prepared.get('status') != 'pass' or prepared.get('compiler_version') != 'perry 0.5.1220' or prepared.get('source_commit') != head:
        raise RuntimeError('web example artifact did not qualify this checkout and compiler')
    cases = prepared.get('examples', [])
    if len(cases) != len(EXAMPLES) or {c['name'] for c in cases} != set(EXAMPLES):
        raise RuntimeError('web example artifact must contain each required game exactly once')
    if any(c.get('status') != 'pass' or c.get('source_unchanged') is not True for c in cases):
        raise RuntimeError('web example artifact must preserve the original game sources')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser')
    parser.add_argument('--game', type=Path, default=ROOT / 'target/ci/web-examples')
    parser.add_argument('--out', type=Path, default=ROOT / 'target/ci/web-smoke/examples')
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0: parser.error('timeout must be finite and positive')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = dict(schema='bloom-web-examples-v1', status='running', examples=[], failures=[],
                  scope='Original compiled examples in real Chrome with production renderer and bootstrap. Startup content and cleanup checks do not qualify complete gameplay, quality goldens or performance.')
    def save(): (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    parent = Path(tempfile.gettempdir()).resolve()
    temporary = Path(tempfile.mkdtemp(prefix='bloom-web-examples-', dir=parent)).resolve()
    process = devtools = server = thread = None
    try:
        prepared = json.loads((args.game / 'result.json').read_text(encoding='utf-8'))
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        validate_prepared(prepared, head)
        report['compiled_examples'] = prepared
        site = temporary / 'site'
        site.mkdir()
        shutil.copytree(ROOT / 'native/web/pkg', site / 'pkg')
        for name in ['bloom_glue.js', 'game_loop.mjs', 'jolt_bridge.js']:
            shutil.copyfile(ROOT / 'native/web' / name, site / name)
        (site / 'assets_manifest.json').write_text('{"files":[]}\n')
        report['engine_wasm_sha256'] = hashlib.sha256((site / 'pkg/bloom_web_bg.wasm').read_bytes()).hexdigest()
        for case in prepared['examples']:
            name = case['name']
            source = args.game / (name + '.ts')
            page = args.game / (name + '.html')
            if hashlib.sha256(source.read_bytes()).hexdigest() != case['source_sha256'] or source.read_text(encoding='utf-8') != (ROOT / 'examples' / name / 'main.ts').read_text(encoding='utf-8'):
                raise RuntimeError(f'{name}: compiled source differs from its receipt or this checkout')
            if hashlib.sha256(page.read_bytes()).hexdigest() != case['html_sha256']:
                raise RuntimeError(f'{name}: compiled page differs from its receipt')
            shutil.copyfile(page, site / page.name)
        browser = browser_path(args.browser)
        if browser is None: raise RuntimeError('Chrome/Chromium is required for example browser acceptance')
        handler = lambda *a, **kw: QuietHandler(*a, directory=str(site), **kw)
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = free_local_port()
        command = [browser, '--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu-sandbox',
                   '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--remote-allow-origins=*',
                   f'--remote-debugging-port={port}', '--window-size=1100,800', f'--user-data-dir={temporary / "profile"}', 'about:blank']
        report['browser_command'] = command
        with (out / 'browser.stdout.log').open('wb') as stdout, (out / 'browser.stderr.log').open('wb') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        target = devtools_target(port, 'about:blank', time.monotonic() + args.timeout)
        if target is None: raise RuntimeError('browser did not expose the owned example page')
        devtools = DevTools(target)
        devtools.call('Page.enable')
        devtools.call('Runtime.enable')
        devtools.call('Page.addScriptToEvaluateOnNewDocument', {'source': (ROOT / 'tools/ci/example_monitor.js').read_text(encoding='utf-8')})
        for name, (width, height) in EXAMPLES.items():
            case = dict(name=name, status='running')
            report['examples'].append(case)
            started = time.monotonic()
            try:
                devtools.call('Emulation.setDeviceMetricsOverride', dict(width=width, height=height, deviceScaleFactor=1, mobile=False))
                url = f'http://127.0.0.1:{server.server_port}/{name}.html'
                devtools.call('Page.navigate', {'url': url})
                state = None
                while time.monotonic() - started < args.timeout:
                    state = evaluate(devtools, f'location.href === {json.dumps(url)} ? globalThis.__exampleProbe || null : null')
                    if isinstance(state, dict) and (state.get('errors') or state.get('badArguments') or state.get('cleanups')): break
                    time.sleep(0.1)
                case['state'] = state
                save()
                validate_state(name, state)
                evaluate(devtools, 'new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
                case['state'] = evaluate(devtools, 'globalThis.__exampleProbe')
                validate_state(name, case['state'])
                if evaluate(devtools, 'document.pointerLockElement !== null'):
                    raise RuntimeError(f'{name}: pointer lock survived game cleanup')
                capture = devtools.call('Page.captureScreenshot', {'format': 'png', 'fromSurface': True})
                frame = out / (name + '.png')
                frame.write_bytes(base64.b64decode(capture['data']))
                case['frame'] = check_frame(name, frame)
                case['adapter'] = evaluate(devtools, '(() => { const i = globalThis.__exampleAdapter?.info; return i ? {vendor:i.vendor,architecture:i.architecture,device:i.device,description:i.description} : null; })()')
                case['status'] = 'pass'
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                case.update(status='fail', error=str(error))
                report['failures'].append(name + ': ' + str(error))
                try:
                    capture = devtools.call('Page.captureScreenshot', {'format': 'png', 'fromSurface': True})
                    (out / (name + '.failure.png')).write_bytes(base64.b64decode(capture['data']))
                except (OSError, ValueError, RuntimeError, KeyError) as capture_error:
                    case['failure_capture_error'] = str(capture_error)
            finally:
                case['duration_seconds'] = round(time.monotonic() - started, 3)
                save()
        report['status'] = 'pass' if all(c['status'] == 'pass' for c in report['examples']) else 'fail'
        print(report['status'].upper() + ': six real browser example render/cleanup checks')
        return 0 if report['status'] == 'pass' else 1
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report.update(status='fail')
        report['failures'].append(str(error))
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
            raise RuntimeError('refusing cleanup outside the owned example browser profile')
        shutil.rmtree(temporary)


if __name__ == '__main__': raise SystemExit(main())
