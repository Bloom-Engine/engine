#!/usr/bin/env python3
"""Validate and compile every canonical Perry example."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).with_name("examples.json")
REPORT_SCHEMA = "bloom-example-compile-v1"


def load_inventory() -> tuple[list[str], list[str]]:
    failures: list[str] = []
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"cannot read {MANIFEST_PATH}: {exc}"]
    if manifest.get("schema") != "bloom-canonical-examples-v1":
        failures.append("example manifest has an unknown schema")
    examples = manifest.get("examples")
    if not isinstance(examples, list) or not all(
        isinstance(item, str) and item for item in examples
    ):
        return [], failures + ["examples must be a non-empty string array"]
    if examples != sorted(set(examples)):
        failures.append("examples must be sorted and unique")
    discovered = sorted(
        str(path.parent.relative_to(REPO_ROOT)).replace(os.sep, "/")
        for path in (REPO_ROOT / "examples").glob("*/package.json")
        if (path.parent / "main.ts").is_file()
    )
    missing = sorted(set(discovered) - set(examples))
    stale = sorted(set(examples) - set(discovered))
    if missing:
        failures.append(f"unlisted canonical examples: {missing}")
    if stale:
        failures.append(f"listed examples missing package.json/main.ts: {stale}")
    for relative in examples:
        directory = (REPO_ROOT / relative).resolve()
        try:
            directory.relative_to(REPO_ROOT / "examples")
        except ValueError:
            failures.append(f"example escapes examples/: {relative}")
    return examples, failures


def ensure_engine_dependency(directory: Path) -> None:
    dependency = directory / "node_modules" / "bloom"
    if dependency.exists():
        return
    dependency.parent.mkdir(parents=True, exist_ok=True)
    try:
        dependency.symlink_to(REPO_ROOT, target_is_directory=True)
    except OSError:
        npm = shutil.which("npm")
        if npm is None:
            raise RuntimeError("npm is required when directory symlinks are unavailable")
        subprocess.run(
            [
                npm,
                "install",
                "--ignore-scripts",
                "--no-audit",
                "--no-fund",
                "--package-lock=false",
            ],
            cwd=directory,
            check=True,
        )


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate inventory only")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "target" / "ci" / "examples"),
    )
    args = parser.parse_args()
    examples, failures = load_inventory()
    if failures:
        for failure in failures:
            print(f"FAIL  {failure}", file=sys.stderr)
        return 1
    print(f"canonical examples: {len(examples)}")
    if args.check:
        print("PASS: canonical example inventory is complete")
        return 0

    perry = shutil.which("perry")
    if perry is None:
        print("FAIL  perry is required to compile canonical examples", file=sys.stderr)
        return 2

    out_dir = Path(args.out).resolve()
    bin_dir = out_dir / "bin"
    log_dir = out_dir / "logs"
    bin_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for relative in examples:
        directory = REPO_ROOT / relative
        name = directory.name
        output = bin_dir / name
        print(f"[example] {relative}", flush=True)
        ensure_engine_dependency(directory)
        case_started = time.perf_counter()
        result = subprocess.run(
            [perry, "compile", "main.ts", "-o", str(output), "--no-link"],
            cwd=directory,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (log_dir / f"{name}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (log_dir / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
        record = {
            "example": relative,
            "status": "pass" if result.returncode == 0 else "fail",
            "mode": "codegen-no-link",
            "exit_code": result.returncode,
            "duration_ms": round((time.perf_counter() - case_started) * 1000, 3),
            "stdout": f"logs/{name}.stdout.log",
            "stderr": f"logs/{name}.stderr.log",
        }
        records.append(record)
        print(f"[example] {relative}: {record['status']}", flush=True)

    failed = [record["example"] for record in records if record["status"] != "pass"]
    report = {
        "schema": REPORT_SCHEMA,
        "status": "fail" if failed else "pass",
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "perry": subprocess.run(
            [perry, "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout.strip(),
        "examples": records,
        "failures": failed,
    }
    write_report(out_dir / "result.json", report)
    if failed:
        print(f"FAIL: {len(failed)} canonical example(s) failed: {failed}", file=sys.stderr)
        return 1
    print(f"PASS: all {len(records)} canonical examples compiled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
