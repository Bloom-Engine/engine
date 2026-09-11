#!/usr/bin/env python3
"""Require fractional scalar math from actual native and WASM Perry execution."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'BLOOM_SCALAR_MATH_RESULT:'
INPUTS = [-0.25, 0, 0.1, 0.25, 0.5, 0.75, 1, 1.25]


def validate_observations(output):
    lines = [line[len(PREFIX):] for line in output.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise RuntimeError('scalar fixture must emit exactly one result')
    actual = json.loads(lines[0])
    if not isinstance(actual, dict) or set(actual) != {'samples', 'nanPreserved', 'infinityPreserved', 'negativeZeroPreserved'}:
        raise RuntimeError('scalar fixture fields differ from the required contract')
    if any(actual[key] is not True for key in ['nanPreserved', 'infinityPreserved', 'negativeZeroPreserved']):
        raise RuntimeError('scalar functions lost NaN, infinity or signed-zero behavior')
    rows = actual['samples']
    if not isinstance(rows, list) or len(rows) != len(INPUTS):
        raise RuntimeError('scalar fixture omitted required sample points')
    for t, row in zip(INPUTS, rows):
        expected = [t, -2.5 + 7.25 * t, t ** 2, 1 - (1 - t) ** 2,
                    2 * t ** 2 if t < 0.5 else 1 - 2 * (1 - t) ** 2,
                    t ** 3, 1 - (1 - t) ** 3,
                    4 * t ** 3 if t < 0.5 else 1 - 4 * (1 - t) ** 3,
                    min(1, max(0, t)), -2.5 + 7.25 * t]
        if not isinstance(row, list) or len(row) != len(expected):
            raise RuntimeError(f'malformed scalar sample at {t}')
        for column, (value, target) in enumerate(zip(row, expected)):
            if type(value) not in (int, float) or not math.isfinite(value) or not math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f'scalar sample {t}, column {column}: expected {target}, found {value}')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--perry', default=os.environ.get('BLOOM_PERRY') or shutil.which('perry'))
    parser.add_argument('--out', type=Path, default=ROOT / 'target/ci/scalar-math')
    args = parser.parse_args()
    if not args.perry:
        parser.error('Perry 0.5.1220 is required')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = dict(schema='bloom-scalar-math-v1', status='running', commands=[], cases=[])

    def save():
        (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    def run(name, command):
        record = {'name': name, 'command': command}
        report['commands'].append(record)
        started = time.monotonic()
        save()
        try:
            with (out / f'{name}.log').open('wb') as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            record['exit_code'] = process.returncode
            output = (out / f'{name}.log').read_text(encoding='utf-8', errors='replace')
            if process.returncode or 'Could not resolve import' in output:
                raise RuntimeError(f'{name}: process or import resolution failed; see retained log')
            return output
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            record['error'] = str(error)
            raise
        finally:
            record['duration_seconds'] = round(time.monotonic() - started, 3)
            save()

    save()
    try:
        if run('compiler-version', [args.perry, '--version']).strip() != 'perry 0.5.1220':
            raise RuntimeError('scalar contract requires Perry 0.5.1220')
        report['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        report['source_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT))
        report['compiler_sha256'] = hashlib.sha256(Path(args.perry).read_bytes()).hexdigest()
        report['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
            'src/math/index.ts', 'tools/ci/fixtures/scalar-math.ts', 'tools/ci/scalar_math_smoke.py',
            'tools/ci/perry_wasm_console.cjs', 'native/web/splice_game.cjs']}
        fixture = ROOT / 'tools/ci/fixtures/scalar-math.ts'
        for mode in ['native', 'wasm']:
            artifact = out / ('scalar-math.html' if mode == 'wasm' else 'scalar-math.exe' if os.name == 'nt' else 'scalar-math')
            artifact.unlink(missing_ok=True)
            command = [args.perry, 'compile', str(fixture)]
            if mode == 'wasm':
                command += ['--target', 'wasm']
            run(mode + '-compile', command + ['-o', str(artifact)])
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise RuntimeError(f'{mode}: compiler produced no fresh artifact')
            command = ['node', str(ROOT / 'tools/ci/perry_wasm_console.cjs'), str(artifact), PREFIX] if mode == 'wasm' else [str(artifact)]
            observations = validate_observations(run(mode + '-run', command))
            report['cases'].append(dict(mode=mode, status='pass', observations=observations,
                                        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()))
            save()
        report['status'] = 'pass'
        print('PASS: actual native and WASM scalar interpolation/easing preserve fractional and IEEE values')
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report.update(status='fail', error=str(error))
        print(f'FAIL: {error}')
        return 1
    finally:
        save()


if __name__ == '__main__':
    raise SystemExit(main())
