#!/usr/bin/env python3
"""Compile unchanged portable examples for the hosted browser runtime gate."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.ci.compile_examples import load_inventory
from tools.ci.example_runtime import EXAMPLES
from tools.ci.native_package_smoke import npm_command


def validate_imports(log, imports):
    required = {'bloom_init_window', 'bloom_run_game_with_cleanup', 'bloom_close_window', 'bloom_clear_background'}
    actual = {item['name'] for item in imports if item['module'] == 'ffi'}
    if 'Could not resolve import' in log or required - actual:
        raise RuntimeError(f'example imports are unresolved or missing: {sorted(required - actual)}')
    if not any(name.startswith('bloom_draw_') for name in actual):
        raise RuntimeError('compiled example has no engine drawing imports')
    return sorted(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--perry', default=os.environ.get('BLOOM_PERRY') or shutil.which('perry'))
    parser.add_argument('--out', type=Path, default=ROOT / 'target/ci/web-examples')
    args = parser.parse_args()
    if not args.perry: parser.error('Perry 0.5.1220 is required')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists(): parser.error('choose a fresh output directory for this compilation')
    report = dict(schema='bloom-compiled-web-examples-v1', status='running', commands=[], examples=[])
    parent = Path(tempfile.gettempdir()).resolve()
    temporary = Path(tempfile.mkdtemp(prefix='bwe-', dir=parent)).resolve()

    def save(): (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    def run(name, command):
        record = dict(name=name, command=command, cwd=str(temporary))
        report['commands'].append(record)
        start = time.monotonic()
        save()
        try:
            with (out / (name + '.log')).open('wb') as log:
                process = subprocess.run(command, cwd=temporary, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            record['exit_code'] = process.returncode
            if process.returncode: raise RuntimeError(f'{name} exited {process.returncode}; see retained log')
            return (out / (name + '.log')).read_text(encoding='utf-8', errors='replace')
        finally:
            record['duration_seconds'] = round(time.monotonic() - start, 3)
            save()

    try:
        inventory, failures = load_inventory()
        if failures or any('examples/' + name not in inventory for name in EXAMPLES):
            raise RuntimeError(f'canonical example inventory is invalid: {failures}')
        report['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        report['source_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT))
        report['compiler_version'] = run('compiler-version', [args.perry, '--version']).strip()
        if report['compiler_version'] != 'perry 0.5.1220': raise RuntimeError('examples require Perry 0.5.1220')
        report['compiler_sha256'] = hashlib.sha256(Path(args.perry).read_bytes()).hexdigest()
        manifest = dict(name='bloom-web-example-smoke', private=True, dependencies={'bloom': 'file:' + ROOT.as_posix()},
                        perry={'allow': {'nativeLibrary': ['bloom', 'bloom/*']}})
        (temporary / 'package.json').write_text(json.dumps(manifest) + '\n', encoding='utf-8')
        run('install', npm_command() + ['install', '--ignore-scripts', '--no-audit', '--no-fund', '--package-lock=false', '--install-links=false'])
        if (temporary / 'node_modules/bloom').resolve() != ROOT.resolve():
            raise RuntimeError('example dependency must resolve to this exact checkout')
        for name in EXAMPLES:
            case = dict(name=name, status='running')
            report['examples'].append(case)
            try:
                original = ROOT / 'examples' / name / 'main.ts'
                source = original.read_bytes()
                entry = temporary / (name + '.ts')
                entry.write_bytes(source)
                (out / (name + '.ts')).write_bytes(source)
                case['source_sha256'] = hashlib.sha256(source).hexdigest()
                raw, page = out / (name + '.perry.html'), out / (name + '.html')
                log = run(name + '-compile', [args.perry, 'compile', str(entry), '--target', 'wasm', '-o', str(raw)])
                encoded = re.search(r'window\.__perryWasmB64\s*=\s*"([A-Za-z0-9+/=]+)"', raw.read_text(encoding='utf-8'))
                if encoded is None: raise RuntimeError('compiler output has no game WASM')
                wasm = base64.b64decode(encoded[1], validate=True)
                binary = out / (name + '.wasm')
                binary.write_bytes(wasm)
                inspect = "const fs=require('node:fs');process.stdout.write(JSON.stringify(WebAssembly.Module.imports(new WebAssembly.Module(fs.readFileSync(process.argv[1])))));"
                imports = json.loads(run(name + '-imports', ['node', '-e', inspect, str(binary)]))
                case['engine_ffi_imports'] = validate_imports(log, imports)
                run(name + '-splice', ['node', str(ROOT / 'native/web/splice_game.cjs'), str(raw), str(page)])
                case.update(game_wasm_sha256=hashlib.sha256(wasm).hexdigest(), html_sha256=hashlib.sha256(page.read_bytes()).hexdigest(),
                            source_unchanged=original.read_bytes() == source)
                if not case['source_unchanged']: raise RuntimeError('original example changed during compilation')
                case['status'] = 'pass'
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                case.update(status='fail', error=str(error))
            save()
        report['status'] = 'pass' if all(case['status'] == 'pass' for case in report['examples']) else 'fail'
        print(report['status'].upper() + ': six unchanged examples compiled and spliced for browser acceptance')
        return 0 if report['status'] == 'pass' else 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report.update(status='fail', error=str(error))
        print(f'FAIL: {error}')
        return 1
    finally:
        save()
        if temporary.parent != parent or temporary.is_symlink() or temporary.is_junction():
            raise RuntimeError('refusing cleanup outside the owned web example project')
        shutil.rmtree(temporary)


if __name__ == '__main__': raise SystemExit(main())
