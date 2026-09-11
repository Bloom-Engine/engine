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
import time

ROOT = Path(__file__).resolve().parents[2]
VERSION = "perry 0.5.1220"


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
            run(name + "-splice", ["node", str(ROOT / "native/web/splice_game.cjs"), str(raw), str(page)])
            report["pages"].append({"name": name, "entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
                                     "game_wasm_sha256": hashlib.sha256(wasm).hexdigest(), "game_wasm_bytes": len(wasm),
                                     "html_sha256": hashlib.sha256(page.read_bytes()).hexdigest(),
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
