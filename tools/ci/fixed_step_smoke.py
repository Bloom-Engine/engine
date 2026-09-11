#!/usr/bin/env python3
"""Check actual native and WASM Perry timing/lifecycle observations."""

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
EXPECTED = {
    "uniformTicks": 50, "variedTicks": 50, "uniformAlpha": 0.0, "variedAlpha": 0.0,
    "longTicks": 2500, "longAlpha": 0.0, "cappedSteps": 3, "cappedAlpha": 0.5,
    "cappedDropped": 0.205, "resumedTicks": 4,
    "invalidConfig": True, "invalidCap": True, "rejectedNegative": True,
    "rejectedNaN": True, "rejectedInfinity": True, "preservedAlpha": 0.5,
    "events": "IUDFFUDC", "total": 0.04, "drawn": 0.03, "lastTick": 2,
    "beforeInitRejected": True, "duplicateInitRejected": True, "afterDisposeRejected": True,
    "stopReturned": True, "stopEvents": "FC",
    "overflowRejected": True, "hugeAlpha": 0.9, "tinyBounded": True,
    "finiteContract": True, "sparseStarted": True, "sparseDraws": 1,
    "updateStopEvents": "UC", "invalidDriverRejected": True, "invalidCalls": 0,
}


def validate_observations(output):
    lines = [line.removeprefix("BLOOM_FIXED_STEP_RESULT:") for line in output.splitlines()
             if line.startswith("BLOOM_FIXED_STEP_RESULT:")]
    if len(lines) != 1:
        raise RuntimeError("fixture must emit exactly one result; compiler/process success is insufficient")
    actual = json.loads(lines[0])
    if set(actual) != set(EXPECTED):
        raise RuntimeError("fixture result fields differ from the required contract")
    for key, expected in EXPECTED.items():
        value = actual[key]
        if isinstance(expected, float):
            good = type(value) in (int, float) and math.isfinite(value) and abs(value - expected) <= 1e-10
        else:
            good = type(value) is type(expected) and value == expected
        if not good:
            raise RuntimeError(f"{key}: expected {expected!r}, found {value!r}")
    return actual


def validate_game_lifecycle(value, expected_frames=None):
    """Validate observations written by the real engine fixture's hooks."""
    if not isinstance(value, str) or len(value.split(',')) != 6:
        raise RuntimeError('missing or malformed game lifecycle observations')
    try:
        init, updates, draws, fixed, tick = [int(v) for v in value.split(',')[:5]]
        alpha = float(value.split(',')[5])
    except ValueError as error:
        raise RuntimeError('malformed game lifecycle observations') from error
    if (init != 1 or updates != draws or not 8 <= draws <= 120 or fixed < 1 or tick != fixed
            or not math.isfinite(alpha) or not 0 <= alpha < 1
            or (expected_frames is not None and draws != expected_frames)):
        raise RuntimeError(f'game lifecycle hooks failed: {value}')
    return dict(init=init, updates=updates, draws=draws, fixed_updates=fixed, last_tick=tick, alpha=alpha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perry", default=os.environ.get("BLOOM_PERRY") or shutil.which("perry"))
    parser.add_argument("--out", type=Path, default=ROOT / "target/ci/fixed-step")
    args = parser.parse_args()
    if not args.perry:
        parser.error("Perry 0.5.1220 is required")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "bloom-fixed-game-lifecycle-v1", "status": "running", "commands": [], "cases": []}
    def save():
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    def run(name, command):
        start = time.monotonic()
        with (out / f"{name}.log").open("wb") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        report["commands"].append({"name": name, "command": command, "exit_code": result.returncode,
                                   "duration_seconds": round(time.monotonic() - start, 3)})
        save()
        output = (out / f"{name}.log").read_text(encoding="utf-8", errors="replace")
        if result.returncode or "Could not resolve import" in output:
            raise RuntimeError(f"{name}: process or import resolution failed; see retained log")
        return output
    try:
        if run("compiler-version", [args.perry, "--version"]).strip() != "perry 0.5.1220":
            raise RuntimeError("timing fixture requires Perry 0.5.1220")
        report["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        report["source_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
        report["compiler_sha256"] = hashlib.sha256(Path(args.perry).read_bytes()).hexdigest()
        fixture = ROOT / "tools/ci/fixtures/fixed-step.ts"
        report["source_sha256"] = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("src/core/fixed_step.ts", "src/core/game_lifecycle.ts", "src/core/numbers.ts",
                         "tools/ci/fixtures/fixed-step.ts", "tools/ci/perry_wasm_console.cjs")
        }
        for mode in ("native", "wasm"):
            artifact = out / ("fixed-step.html" if mode == "wasm" else "fixed-step.exe" if os.name == "nt" else "fixed-step")
            artifact.unlink(missing_ok=True)
            command = [args.perry, "compile", str(fixture)]
            if mode == "wasm":
                command += ["--target", "wasm"]
            run(mode + "-compile", command + ["-o", str(artifact)])
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise RuntimeError(f"{mode}: no fresh compiled artifact")
            execute = ["node", str(ROOT / "tools/ci/perry_wasm_console.cjs"), str(artifact)] if mode == "wasm" else [str(artifact)]
            observations = validate_observations(run(mode + "-run", execute))
            report["cases"].append({"mode": mode, "status": "pass", "observations": observations,
                                    "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()})
            save()
        report["status"] = "pass"
        print("PASS: actual native and WASM Perry fixed-step/lifecycle contracts")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report.update(status="fail", error=str(error))
        print(f"FAIL: {error}")
        return 1
    finally:
        save()


if __name__ == "__main__":
    raise SystemExit(main())
