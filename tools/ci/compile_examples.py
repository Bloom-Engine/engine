#!/usr/bin/env python3
"""Validate and compile every canonical Perry example."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).with_name("examples.json")
REPORT_SCHEMA = "bloom-example-compile-v2"


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


def is_native_binary(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 32:
        return False
    with path.open("rb") as binary:
        magic = binary.read(4)
    return magic[:2] == b"MZ" or magic in {
        b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf",
        b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate inventory only")
    parser.add_argument("--example", action="append", help="select an inventory entry; default: all")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per example, including native dependencies")
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

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.example:
        unknown = sorted(set(args.example) - set(examples))
        if unknown:
            parser.error(f"examples are not in the canonical inventory: {unknown}")
        examples = [name for name in examples if name in args.example]

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
    report = {
        "schema": REPORT_SCHEMA,
        "status": "running",
        "mode": "native-compile-link",
        "selected_examples": examples,
        "examples": records,
        "compiler": perry,
        "environment": {
            name: os.environ[name]
            for name in ("PERRY_WORKSPACE_ROOT", "PERRY_RUNTIME_DIR", "PERRY_NO_AUTO_OPTIMIZE")
            if name in os.environ
        },
    }
    if (REPO_ROOT / ".git").exists():
        report["source_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            stdout=subprocess.PIPE, encoding="utf-8",
        ).stdout.strip()
    write_report(out_dir / "result.json", report)
    for relative in examples:
        directory = REPO_ROOT / relative
        name = directory.name
        output = bin_dir / (name + (".exe" if os.name == "nt" else ""))
        # A successful command must produce a new executable. A previous run's
        # output must never turn a compiler that emitted nothing into a pass.
        fresh_output = output.with_name(f"{name}-{uuid.uuid4().hex}{output.suffix}")
        print(f"[example] {relative}", flush=True)
        case_started = time.perf_counter()
        command = [perry, "compile", "main.ts", "-o", str(fresh_output)]
        error = None
        exit_code = None
        with (log_dir / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout, \
             (log_dir / f"{name}.stderr.log").open("w", encoding="utf-8") as stderr:
            try:
                ensure_engine_dependency(directory)
                result = subprocess.run(
                    command, cwd=directory, stdout=stdout, stderr=stderr,
                    check=False, timeout=args.timeout,
                )
                exit_code = result.returncode
                if exit_code != 0:
                    error = f"compiler exited with code {exit_code}"
                elif not is_native_binary(fresh_output):
                    error = "compiler did not produce a new native executable"
                else:
                    fresh_output.replace(output)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                error = str(exc)
                stderr.write(f"\n{error}\n")
        record = {
            "example": relative,
            "status": "pass" if error is None else "fail",
            "mode": "native-compile-link",
            "command": command,
            "exit_code": exit_code,
            "error": error,
            "duration_ms": round((time.perf_counter() - case_started) * 1000, 3),
            "stdout": f"logs/{name}.stdout.log",
            "stderr": f"logs/{name}.stderr.log",
        }
        if error is None:
            with output.open("rb") as binary:
                digest = hashlib.file_digest(binary, "sha256").hexdigest()
            record.update(artifact=f"bin/{output.name}", bytes=output.stat().st_size, sha256=digest)
        records.append(record)
        write_report(out_dir / "result.json", report)
        print(f"[example] {relative}: {record['status']}", flush=True)
        if error:
            print(f"  {error}; see {log_dir / (name + '.stderr.log')}", file=sys.stderr)

    failed = [record["example"] for record in records if record["status"] != "pass"]
    report.update({
        "status": "fail" if failed else "pass",
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "perry": subprocess.run(
            [perry, "--version"],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout.strip(),
        "failures": failed,
    })
    write_report(out_dir / "result.json", report)
    if failed:
        print(f"FAIL: {len(failed)} canonical example(s) failed: {failed}", file=sys.stderr)
        return 1
    print(f"PASS: all {len(records)} selected canonical examples compiled and linked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
