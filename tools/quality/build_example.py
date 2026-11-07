#!/usr/bin/env python3
"""Prepare a Perry example and compile it reproducibly for qualification."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(argv: list[str], cwd: Path) -> None:
    print("+ " + " ".join(argv), flush=True)
    result = subprocess.run(argv, cwd=cwd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--source", default="main.ts")
    parser.add_argument("--output", default="main")
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    if not (directory / "package.json").is_file():
        print(f"missing package.json: {directory}", file=sys.stderr)
        return 2
    if not (directory / "node_modules/bloom").exists():
        npm = shutil.which("npm")
        if npm is None:
            print("npm is required to prepare the example", file=sys.stderr)
            return 2
        run(
            [
                npm,
                "install",
                "--ignore-scripts",
                "--no-audit",
                "--no-fund",
                "--package-lock=false",
            ],
            directory,
        )
    perry = shutil.which("perry")
    if perry is None:
        print("perry is required to compile qualification scenes", file=sys.stderr)
        return 2
    run([perry, "compile", args.source, "-o", args.output], directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
