#!/usr/bin/env python3
"""Compatibility entry point; the packaged Node implementation owns splicing."""
import pathlib
import subprocess
import sys

if __name__ == "__main__":
    raise SystemExit(subprocess.call([
        "node", str(pathlib.Path(__file__).with_suffix(".cjs")), *sys.argv[1:]
    ]))
