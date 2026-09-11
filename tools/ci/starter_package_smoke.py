#!/usr/bin/env python3
"""Require the packed starter CLI to create a project with the exact engine package."""

import argparse
import base64
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.ci.native_package_smoke import npm_command
from tools.ci.starter_web_run import run_web


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "target/ci/installed-starter")
    parser.add_argument("--web", action="store_true", help="Build and retain the full installed starter website")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "bloom-installed-starter-v1", "status": "running", "commands": []}
    def save():
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    def run(name, command, cwd, env=None, expected_error=None, timeout=240):
        record = {"name": name, "command": command, "cwd": str(cwd)}
        report["commands"].append(record)
        save()
        start = time.monotonic()
        with (out / f"{name}.stdout.log").open("wb") as stdout, (out / f"{name}.stderr.log").open("wb") as stderr:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=timeout)
        record.update(exit_code=result.returncode, duration_seconds=round(time.monotonic() - start, 3))
        save()
        stdout = (out / f"{name}.stdout.log").read_text(encoding="utf-8", errors="replace")
        stderr = (out / f"{name}.stderr.log").read_text(encoding="utf-8", errors="replace")
        if expected_error:
            if result.returncode == 0 or expected_error not in stdout + stderr:
                raise RuntimeError(f"{name}: missing expected nonzero failure: {expected_error}")
        elif result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}; see retained logs")
        return stdout
    parent = Path(tempfile.gettempdir()).resolve()
    temporary = Path(tempfile.mkdtemp(prefix="bsc-", dir=parent)).resolve()
    save()
    try:
        npm = npm_command()
        report["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        report["source_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
        packed = json.loads(run("pack", npm + ["pack", "--json", "--ignore-scripts", "--pack-destination", str(temporary)], ROOT))[0]
        archive = temporary / packed["filename"]
        report["package_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        bootstrap = temporary / "bootstrap"
        bootstrap.mkdir()
        (bootstrap / "package.json").write_text('{"name":"starter-cli-bootstrap","private":true}\n')
        run("install", npm + ["install", "--ignore-scripts", "--no-audit", "--no-fund", str(archive)], bootstrap)
        run("help", npm + ["exec", "--", "bloom", "--help"], bootstrap)
        project = temporary / "game"
        run("create", npm + ["exec", "--", "bloom", "new", str(project)], bootstrap)
        manifest = json.loads((project / "package.json").read_text())
        if manifest["dependencies"]["@bloomengine/engine"] != "file:.bloom/engine.tgz":
            raise RuntimeError("generated project must pin the exact engine archive")
        installed = project / "node_modules/@bloomengine/engine"
        source_files = ["tools/cli/bloom.cjs", "tools/cli/toolchain.cjs", "tools/cli/serve.cjs",
                        "tools/cli/templates/main.ts", "tools/cli/templates/README.md",
                        "tools/ci/setup_windows_perry.py"]
        if args.web:
            source_files += ['native/web/splice_game.cjs', 'native/web/game_loop.mjs',
                             'src/core/index.ts', 'src/core/game_lifecycle.ts', 'src/core/fixed_step.ts']
        report["installed_source_sha256"] = {}
        for name in source_files:
            # npm normalizes executable shebang line endings on Windows.
            if (installed / name).read_text(encoding="utf-8") != (ROOT / name).read_text(encoding="utf-8"):
                raise RuntimeError(f"installed starter source differs: {name}")
            report["installed_source_sha256"][name] = hashlib.sha256((installed / name).read_bytes()).hexdigest()
        if (project / "main.ts").read_text(encoding="utf-8") != (ROOT / "tools/cli/templates/main.ts").read_text(encoding="utf-8"):
            raise RuntimeError("generated game source differs from the shipped template")
        if (project / "assets/welcome.txt").read_text().strip() != "Hello, Bloom!":
            raise RuntimeError("starter asset is missing")
        report["project_engine_archive_sha256"] = hashlib.sha256((project / ".bloom/engine.tgz").read_bytes()).hexdigest()
        report["project_config"] = json.loads((project / "bloom.json").read_text())
        run("unsupported-target", npm + ["exec", "--", "bloom", "build", "--target", "android"], project, expected_error="Unsupported starter target")
        missing_env = {**os.environ, "BLOOM_PERRY": str(temporary / "missing-perry.exe")}
        run("missing-compiler", npm + ["exec", "--", "bloom", "build", "--target", "web"], project, env=missing_env, expected_error="not found")
        if args.web:
            compiler = os.environ.get('BLOOM_PERRY') or shutil.which('perry')
            wasm_pack = os.environ.get('BLOOM_WASM_PACK') or shutil.which('wasm-pack')
            if not compiler or not wasm_pack:
                raise RuntimeError('Perry and wasm-pack are required for installed starter web acceptance')
            report['compiler_version'] = run('perry-version', [compiler, '--version'], project).strip()
            report['wasm_pack_version'] = run('wasm-pack-version', [wasm_pack, '--version'], project).strip()
            if report['compiler_version'] != 'perry 0.5.1220' or report['wasm_pack_version'] != 'wasm-pack 0.15.0':
                raise RuntimeError('starter web gate requires Perry 0.5.1220 and wasm-pack 0.15.0')
            report['compiler_sha256'] = hashlib.sha256(Path(compiler).read_bytes()).hexdigest()
            web_env = os.environ.copy()
            web_env['CARGO_TARGET_DIR'] = str(ROOT / 'native/web/target')
            report['web_command'] = run_web(project, out, web_env)
            log = (out / 'installed-web-run.stdout.log').read_text(encoding='utf-8', errors='replace') + (out / 'installed-web-run.stderr.log').read_text(encoding='utf-8', errors='replace')
            if 'Could not resolve import' in log:
                raise RuntimeError('starter web build contains unresolved Perry imports')
            website = project / 'dist/web'
            html = (website / 'index.html').read_text(encoding='utf-8')
            encoded = re.search(r'window\.__perryWasmB64\s*=\s*"([A-Za-z0-9+/=]+)"', html)
            if encoded is None:
                raise RuntimeError('starter web build has no compiled game WASM')
            game_wasm = base64.b64decode(encoded[1], validate=True)
            (out / 'starter.game.wasm').write_bytes(game_wasm)
            inspect = "const fs=require('node:fs');process.stdout.write(JSON.stringify(WebAssembly.Module.imports(new WebAssembly.Module(fs.readFileSync(process.argv[1])))));"
            imports = json.loads(run('game-imports', ['node', '-e', inspect, str(out / 'starter.game.wasm')], project))
            required = {'bloom_init_window', 'bloom_run_game_with_cleanup', 'bloom_read_file', 'bloom_draw_text', 'bloom_draw_rect'}
            actual = {item['name'] for item in imports if item['module'] == 'ffi'}
            if required - actual:
                raise RuntimeError(f'starter is missing required compiled imports: {sorted(required - actual)}')
            if json.loads((website / 'assets_manifest.json').read_text()) != {'files': ['assets/welcome.txt']}:
                raise RuntimeError('starter asset manifest differs from the shipped asset inventory')
            if (website / 'assets/welcome.txt').read_text().strip() != 'Hello, Bloom!':
                raise RuntimeError('starter text asset changed during the build')
            if (project / 'main.ts').read_text(encoding='utf-8') != (ROOT / 'tools/cli/templates/main.ts').read_text(encoding='utf-8'):
                raise RuntimeError('starter source was changed for browser acceptance')
            # Retain the complete output of the actual installed build command.
            # Reusing an output directory must never retain stale website files.
            site = out / 'site'
            if site.exists():
                raise RuntimeError('starter web evidence output already contains a website; choose a fresh --out')
            shutil.copytree(website, site)
            (out / 'starter.ts').write_bytes((project / 'main.ts').read_bytes())
            report['web'] = {'source_unchanged': True, 'game_wasm_sha256': hashlib.sha256(game_wasm).hexdigest(),
                             'engine_ffi_imports': sorted(actual), 'files': {}}
            for file in sorted(site.rglob('*')):
                if file.is_symlink() or file.is_junction():
                    raise RuntimeError('starter website unexpectedly contains a filesystem link')
                if file.is_file():
                    report['web']['files'][file.relative_to(site).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        report["status"] = "pass"
        print("PASS: installed CLI creates an exact-package starter; setup failure controls rejected")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        report.update(status="fail", error=str(error))
        print(f"FAIL: {error}")
        return 1
    finally:
        save()
        if temporary.parent != parent or temporary.is_symlink() or temporary.is_junction():
            raise RuntimeError(f"refusing cleanup outside owned temporary directory: {temporary}")
        shutil.rmtree(temporary)


if __name__ == "__main__":
    raise SystemExit(main())
