#!/usr/bin/env python3
"""Compile and run bounded copies of portable examples on the real native engine."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.ci.compile_examples import load_inventory, reject_duplicate_module_globals
from tools.ci.example_runtime import EXAMPLES, bounded_native_source, check_frame, validate_native_state
from tools.ci.native_package_smoke import npm_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--perry', default=os.environ.get('BLOOM_PERRY') or shutil.which('perry'))
    parser.add_argument('--out', type=Path, default=ROOT / 'target/ci/native-examples')
    parser.add_argument('--backend', choices=['dx12', 'vulkan'], default='dx12')
    args = parser.parse_args()
    if os.name != 'nt' or not args.perry:
        parser.error('this native example gate requires Windows and Perry 0.5.1220')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = dict(schema='bloom-native-examples-v1', status='running', commands=[], examples=[],
                  scope='Six portable examples: unchanged update/draw and cleanup bodies with bounded native capture instrumentation. Content checks do not replace quality goldens, presentation or gameplay acceptance.')
    # Keep linked source and fixture on one drive so Perry's module prefixes
    # retain their full relative paths instead of colliding index.ts names.
    parent = (ROOT / 'target/ci').resolve()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='bne-', dir=parent)).resolve()
    env = os.environ.copy()
    env.pop('CARGO_TARGET_DIR', None)
    env.setdefault('CARGO_BUILD_JOBS', '2')

    def save():
        (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    def run(name, command, cwd, run_env, timeout):
        record = dict(name=name, command=command, cwd=str(cwd))
        report['commands'].append(record)
        save()
        print(name, flush=True)
        started = time.monotonic()
        try:
            with (out / f'{name}.stdout.log').open('wb') as stdout, (out / f'{name}.stderr.log').open('wb') as stderr:
                process = subprocess.run(command, cwd=cwd, env=run_env, stdout=stdout, stderr=stderr, timeout=timeout)
            record['exit_code'] = process.returncode
            text = (out / f'{name}.stdout.log').read_text(encoding='utf-8', errors='replace')
            error_text = (out / f'{name}.stderr.log').read_text(encoding='utf-8', errors='replace')
            reject_duplicate_module_globals(text + error_text)
            if process.returncode or 'Could not resolve import' in text + error_text:
                raise RuntimeError(f'{name}: process or import resolution failed; see retained logs')
            return text
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            record['error'] = str(error)
            raise
        finally:
            record['duration_seconds'] = round(time.monotonic() - started, 3)
            save()

    save()
    try:
        inventory, failures = load_inventory()
        if failures or any('examples/' + name not in inventory for name in EXAMPLES):
            raise RuntimeError(f'canonical example inventory is invalid: {failures}')
        report['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        report['source_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT))
        report['compiler_sha256'] = hashlib.sha256(Path(args.perry).read_bytes()).hexdigest()
        if run('compiler-version', [args.perry, '--version'], ROOT, env, 30).strip() != 'perry 0.5.1220':
            raise RuntimeError('native examples require Perry 0.5.1220')
        report['runtime_source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
            'src/index.ts', 'src/math/index.ts', 'src/core/index.ts', 'tools/ci/example_runtime.py',
            'tools/ci/compile_examples.py', 'tools/ci/native_example_smoke.py']}
        manifest = dict(name='bloom-native-example-smoke', private=True,
                        dependencies={'bloom': 'file:' + ROOT.as_posix()},
                        perry={'allow': {'nativeLibrary': ['bloom', 'bloom/*']}})
        (temporary / 'package.json').write_text(json.dumps(manifest) + '\n', encoding='utf-8')
        run('install', npm_command() + ['install', '--ignore-scripts', '--no-audit', '--no-fund', '--package-lock=false', '--install-links=false'], temporary, env, 180)
        if (temporary / 'node_modules/bloom').resolve() != ROOT.resolve():
            raise RuntimeError('example dependency must resolve to this exact checkout')
        ctypes.windll.kernel32.SetErrorMode(0x0002 | 0x8000)
        for name in EXAMPLES:
            case = dict(name=name, backend=args.backend, status='running')
            report['examples'].append(case)
            save()
            try:
                source = (ROOT / 'examples' / name / 'main.ts').read_bytes()
                case['source_sha256'] = hashlib.sha256(source).hexdigest()
                (out / (name + '.original.ts')).write_bytes(source)
                bounded = bounded_native_source(source.decode('utf-8'))
                entry = temporary / (name + '.ts')
                entry.write_text(bounded, encoding='utf-8')
                (out / (name + '.bounded.ts')).write_bytes(entry.read_bytes())
                binary = temporary / (name + '.exe')
                run(name + '-compile', [args.perry, 'compile', str(entry), '-o', str(binary)], temporary, env, 1800)
                with binary.open('rb') as stream:
                    if stream.read(2) != b'MZ':
                        raise RuntimeError('compiler produced no native Windows binary')
                case['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
                runtime_dir = temporary / name
                runtime_dir.mkdir()
                runtime = env.copy()
                runtime.update(BLOOM_HEADLESS='1', BLOOM_HEADLESS_PIXEL_EXACT='1', BLOOM_WGPU_BACKEND=args.backend)
                try:
                    run(name + '-run', [str(binary)], runtime_dir, runtime, 180)
                finally:
                    for file in ['state.txt', 'frame.png']:
                        if (runtime_dir / file).is_file():
                            shutil.copyfile(runtime_dir / file, out / (name + '.' + file))
                case['state'] = validate_native_state((runtime_dir / 'state.txt').read_text(encoding='utf-8'))
                case['frame'] = check_frame(name, runtime_dir / 'frame.png')
                case['source_unchanged'] = (ROOT / 'examples' / name / 'main.ts').read_bytes() == source
                if not case['source_unchanged']:
                    raise RuntimeError('example source changed during the native run')
                case['status'] = 'pass'
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                case.update(status='fail', error=str(error))
            save()
        report['status'] = 'pass' if all(case['status'] == 'pass' for case in report['examples']) else 'fail'
        print(report['status'].upper() + ': six native example render/cleanup checks')
        return 0 if report['status'] == 'pass' else 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report.update(status='fail', error=str(error))
        print(f'FAIL: {error}')
        return 1
    finally:
        save()
        if temporary.parent != parent or temporary.is_symlink() or temporary.is_junction():
            raise RuntimeError('refusing cleanup outside the owned example project')
        shutil.rmtree(temporary)


if __name__ == '__main__':
    raise SystemExit(main())
