#!/usr/bin/env python3
"""Qualify public constants and shared palette identity in actual native/WASM code."""

import argparse
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
from tools.ci.native_package_smoke import npm_command
from tools.ci.compile_examples import reject_duplicate_module_globals

PREFIX = 'BLOOM_PALETTE_RESULT:'
# Public palette compatibility reference; both aliases must retain all channels.
COLORS = dict(SNOW='f5f5f5ff', WHITE='ffffffff', BLACK='000000ff', RED='e62937ff', GREEN='00e430ff',
              BLUE='0079f1ff', YELLOW='fdf900ff', ORANGE='ffa100ff', PINK='ff6dc2ff', PURPLE='c87affff',
              DARKGRAY='505050ff', LIGHTGRAY='c8c8c8ff', GRAY='828282ff', DARKBLUE='0052acff', SKYBLUE='66bfffff',
              LIME='009e2fff', DARKGREEN='00752cff', GOLD='ffcb00ff', MAROON='be2137ff', BROWN='7f6a4fff',
              BEIGE='d3b083ff', MAGENTA='ff00ffff', VIOLET='873cbeff', BLANK='00000000')
INPUTS = (list(range(65, 91)) + list(range(48, 58)) + list(range(112, 124)) + list(range(256, 260))
          + [32, 265, 27, 9, 8, 127, 260, 261, 262, 263, 264] + list(range(280, 288))
          + [39, 44, 45, 46, 47, 59, 61, 91, 92, 93, 96] + [0, 1, 2] + list(range(7)) + list(range(10)))
ROOT_CONSTANTS = [0, 1, 0, 1, 2, 0, 1, 2, 3, 4]


def validate_observations(output):
    rows = [line[len(PREFIX):] for line in output.splitlines() if line.startswith(PREFIX)]
    if len(rows) != 1: raise RuntimeError('palette fixture must emit exactly one result')
    data = json.loads(rows[0])
    if set(data) != {'colors', 'aliasMutation', 'restored', 'inputs', 'rootInputs', 'sharedMaps', 'rootConstants'} or any(data[key] is not True for key in ['aliasMutation', 'restored', 'sharedMaps']):
        raise RuntimeError('palette aliases lost shared mutation/restoration behavior')
    for key, expected in [('inputs', INPUTS), ('rootInputs', INPUTS), ('rootConstants', ROOT_CONSTANTS)]:
        if data[key] != expected or any(type(n) not in (int, float) for n in data[key]):
            raise RuntimeError(f'public constant values changed: {key}')
    if len(data['colors']) != len(COLORS): raise RuntimeError('palette fixture omitted or duplicated a color')
    for row, (name, rgba) in zip(data['colors'], COLORS.items()):
        channels = list(bytes.fromhex(rgba))
        if len(row) != 20 or row[0] != name or any(v is not True for v in row[-3:]) or row[1:17] != channels * 4 or any(type(n) not in (int, float) for n in row[1:17]):
            raise RuntimeError(f'palette value or identity changed for {name}: {row}')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--perry', default=os.environ.get('BLOOM_PERRY') or shutil.which('perry'))
    parser.add_argument('--out', type=Path, default=ROOT / 'target/ci/palette')
    args = parser.parse_args()
    if not args.perry: parser.error('Perry 0.5.1220 is required')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists(): parser.error('choose a fresh palette evidence directory')
    report = dict(schema='bloom-palette-v1', status='running', commands=[], cases=[])
    # The linked engine and fixture must share a drive. Perry 0.5.1220 falls
    # back to colliding index.ts module names when no common ancestor exists.
    parent = (ROOT / 'target/ci').resolve()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='bpc-', dir=parent)).resolve()
    env = os.environ.copy()
    env.pop('CARGO_TARGET_DIR', None)
    env.setdefault('CARGO_BUILD_JOBS', '2')
    def save(): (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    def run(name, command, expected_error=None):
        record = dict(name=name, command=command, cwd=str(temporary))
        report['commands'].append(record)
        started = time.monotonic()
        save()
        try:
            with (out / (name + '.log')).open('wb') as log:
                process = subprocess.run(command, cwd=temporary, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            record['exit_code'] = process.returncode
            text = (out / (name + '.log')).read_text(encoding='utf-8', errors='replace')
            reject_duplicate_module_globals(text)
            if expected_error:
                if process.returncode == 0 or expected_error not in text: raise RuntimeError(f'{name}: expected guard failure was not observed')
            elif process.returncode or 'Could not resolve import' in text: raise RuntimeError(f'{name}: execution or import resolution failed')
            return text
        finally:
            record['duration_seconds'] = round(time.monotonic() - started, 3)
            save()
    try:
        report['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        report['source_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT))
        report['compiler_version'] = run('compiler-version', [args.perry, '--version']).strip()
        if report['compiler_version'] != 'perry 0.5.1220': raise RuntimeError('palette requires Perry 0.5.1220')
        report['compiler_sha256'] = hashlib.sha256(Path(args.perry).read_bytes()).hexdigest()
        report['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
            'src/index.ts', 'src/core/index.ts', 'src/core/colors.ts', 'src/core/keys.ts', 'tools/ci/fixtures/palette.ts', 'tools/ci/palette_smoke.py',
            'tools/ci/perry_wasm_console.cjs', 'tools/ci/compile_examples.py', 'native/web/splice_game.cjs']}
        manifest = dict(name='bloom-palette-contract', private=True, dependencies={'bloom': 'file:' + ROOT.as_posix()},
                        perry={'allow': {'nativeLibrary': ['bloom', 'bloom/*']}})
        (temporary / 'package.json').write_text(json.dumps(manifest) + '\n', encoding='utf-8')
        run('install', npm_command() + ['install', '--ignore-scripts', '--no-audit', '--no-fund', '--package-lock=false', '--install-links=false'])
        if (temporary / 'node_modules/bloom').resolve() != ROOT.resolve(): raise RuntimeError('palette dependency differs from this checkout')
        fixture = (ROOT / 'tools/ci/fixtures/palette.ts').read_bytes()
        entry = temporary / 'palette.ts'
        entry.write_bytes(fixture)
        (out / 'palette.ts').write_bytes(fixture)
        helper = ROOT / 'tools/ci/perry_wasm_console.cjs'
        for mode in ['native', 'wasm']:
            artifact = out / ('palette.html' if mode == 'wasm' else 'palette.exe' if os.name == 'nt' else 'palette')
            run(mode + '-compile', [args.perry, 'compile', str(entry)] + (['--target', 'wasm'] if mode == 'wasm' else []) + ['-o', str(artifact)])
            command = ['node', str(helper), str(artifact), PREFIX, '--guard-unused-ffi'] if mode == 'wasm' else [str(artifact)]
            result = validate_observations(run(mode + '-run', command))
            report['cases'].append(dict(mode=mode, status='pass', observations=result, artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()))
            save()
        entry.write_bytes(fixture + b'\nconsole.log(getPlatform());\n')
        control = out / 'guard-control.html'
        run('guard-compile', [args.perry, 'compile', str(entry), '--target', 'wasm', '-o', str(control)])
        run('guard-run', ['node', str(helper), str(control), PREFIX, '--guard-unused-ffi'], expected_error='FFI execution forbidden: bloom_get_platform')
        report.update(status='pass', unexpected_ffi_rejected=True)
        print('PASS: native/WASM public palette values and shared identity; attempted engine call rejected')
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report.update(status='fail', error=str(error))
        print(f'FAIL: {error}')
        return 1
    finally:
        save()
        if temporary.parent != parent or temporary.is_symlink() or temporary.is_junction(): raise RuntimeError('refusing cleanup outside owned palette project')
        shutil.rmtree(temporary)


if __name__ == '__main__': raise SystemExit(main())
