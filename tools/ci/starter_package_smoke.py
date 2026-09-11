#!/usr/bin/env python3
"""Require the packed starter CLI to create a project with the exact engine package."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "target/ci/installed-starter")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "bloom-installed-starter-v1", "status": "running", "commands": []}
    def save():
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    def run(name, command, cwd, env=None, expected_error=None):
        record = {"name": name, "command": command, "cwd": str(cwd)}
        report["commands"].append(record)
        save()
        start = time.monotonic()
        with (out / f"{name}.stdout.log").open("wb") as stdout, (out / f"{name}.stderr.log").open("wb") as stderr:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=240)
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
