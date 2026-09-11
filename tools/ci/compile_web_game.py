#!/usr/bin/env python3
"""Compile actual Perry startup and failure-control pages for browser acceptance."""

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
import time

ROOT = Path(__file__).resolve().parents[2]
VERSION = "perry 0.5.1220"
sys.path.insert(0, str(ROOT))
from tools.ci.native_package_smoke import npm_command


def validate_compilation(log, imports):
    if "Could not resolve import" in log:
        raise RuntimeError("Perry could not resolve a game import; a zero compiler exit is insufficient")
    required = {"bloom_init_window", "bloom_set_target_fps", "bloom_set_direct_2d_mode",
                "bloom_run_game_with_cleanup", "bloom_write_file", "bloom_draw_rect"}
    actual = {item["name"] for item in imports if item["module"] == "ffi"}
    missing = required - actual
    if missing:
        raise RuntimeError(f"compiled game is missing required engine FFI imports: {sorted(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perry", default=os.environ.get("BLOOM_PERRY") or shutil.which("perry"))
    parser.add_argument("--out", type=Path, default=ROOT / "target/ci/compiled-web-game")
    args = parser.parse_args()
    if not args.perry:
        parser.error("Perry 0.5.1220 is required; prepare the pinned example toolchain")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries = ROOT / "target/ci/compiled-web-entries"
    entries.mkdir(parents=True, exist_ok=True)
    fixture = (ROOT / "tools/ci/fixtures/compiled-web.ts").read_text(encoding="utf-8")
    marker = "const BLOOM_SMOKE_FAIL_STARTUP = false;"
    assert fixture.count(marker) == 1
    report = {"schema": "bloom-compiled-web-game-v1", "status": "running", "commands": [], "pages": [],
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}

    def save():
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def run(name, command):
        start = time.monotonic()
        with (out / f"{name}.log").open("wb") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        report["commands"].append({"name": name, "command": command, "exit_code": result.returncode,
                                   "duration_seconds": round(time.monotonic() - start, 3)})
        save()
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}; see {out / (name + '.log')}")

    save()
    try:
        # Perry does not resolve package self-references from an arbitrary
        # source folder. Install an explicit local dependency for this project.
        manifest = {"name": "bloom-compiled-web-fixture", "private": True,
                    "dependencies": {"@bloomengine/engine": "file:" + os.path.relpath(ROOT, entries).replace(os.sep, "/")},
                    "perry": {"allow": {"nativeLibrary": ["@bloomengine/engine", "@bloomengine/engine/*"]}}}
        (entries / "package.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        run("fixture-install", npm_command() + ["install", "--prefix", str(entries), "--ignore-scripts", "--no-audit", "--no-fund", "--package-lock=false", "--install-links=false"])
        dependency = entries / "node_modules/@bloomengine/engine"
        if dependency.resolve() != ROOT.resolve():
            raise RuntimeError("compiled fixture dependency must resolve to this exact engine checkout")
        run("perry-version", [args.perry, "--version"])
        report["compiler_version"] = (out / "perry-version.log").read_text(encoding="utf-8").strip()
        if report["compiler_version"] != VERSION:
            raise RuntimeError(f"expected {VERSION}; found {report['compiler_version']}")
        report["compiler_sha256"] = hashlib.sha256(Path(args.perry).read_bytes()).hexdigest()
        for name, fails in [("game", False), ("trap-control", True)]:
            source = fixture.replace(marker, "const BLOOM_SMOKE_FAIL_STARTUP = true;") if fails else fixture
            entry = entries / f"{name}.ts"
            entry.write_text(source, encoding="utf-8", newline="\n")
            (out / f"{name}.ts").write_bytes(entry.read_bytes())
            raw, page = out / f"{name}.perry.html", out / f"{name}.html"
            raw.unlink(missing_ok=True)
            page.unlink(missing_ok=True)
            run(name + "-compile", [args.perry, "compile", str(entry), "--target", "wasm", "-o", str(raw)])
            html = raw.read_text(encoding="utf-8")
            match = re.search(r'window\.__perryWasmB64\s*=\s*"([A-Za-z0-9+/=]+)"', html)
            if match is None:
                raise RuntimeError(f"{name}: Perry output has no embedded game WASM")
            wasm = base64.b64decode(match[1], validate=True)
            if not wasm.startswith(b"\0asm"):
                raise RuntimeError(f"{name}: invalid game WASM header")
            wasm_path = out / f"{name}.game.wasm"
            wasm_path.write_bytes(wasm)
            inspect = "const fs=require('node:fs');process.stdout.write(JSON.stringify(WebAssembly.Module.imports(new WebAssembly.Module(fs.readFileSync(process.argv[1])))));"
            run(name + "-imports", ["node", "-e", inspect, str(wasm_path)])
            imports = json.loads((out / f"{name}-imports.log").read_text(encoding="utf-8"))
            validate_compilation((out / f"{name}-compile.log").read_text(encoding="utf-8", errors="replace"), imports)
            run(name + "-splice", ["node", str(ROOT / "native/web/splice_game.cjs"), str(raw), str(page)])
            report["pages"].append({"name": name, "entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
                                     "game_wasm_sha256": hashlib.sha256(wasm).hexdigest(), "game_wasm_bytes": len(wasm),
                                     "html_sha256": hashlib.sha256(page.read_bytes()).hexdigest(),
                                     "engine_ffi_imports": sorted(item["name"] for item in imports if item["module"] == "ffi"),
                                     "expected_startup_failure": fails})
        report["status"] = "pass"
        print("PASS: compiled and spliced actual Perry startup and failure-control pages")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        report.update(status="fail", error=str(exc))
        print(f"FAIL: {exc}")
        return 1
    finally:
        save()


if __name__ == "__main__":
    raise SystemExit(main())
