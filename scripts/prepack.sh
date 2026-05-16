#!/usr/bin/env bash
# Runs immediately before `npm pack` / `npm publish` assembles the
# tarball. Its job is to verify the working tree contains everything
# the published package needs — most importantly, the JoltPhysics
# submodule materialised on disk, since the tarball vendors its
# sources rather than relying on a postinstall `git clone`.
#
# We deliberately do NOT auto-init the submodule here: doing so would
# silently publish whatever ref the submodule happens to point at,
# even if it's stale or uncommitted. A loud failure forces a
# deliberate `git submodule update --init` before publishing.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOLT_DIR="$ROOT/native/third_party/JoltPhysics"
JOLT_SRC="$JOLT_DIR/Jolt"

if [ ! -d "$JOLT_SRC" ] || [ -z "$(ls -A "$JOLT_SRC" 2>/dev/null)" ]; then
  echo "prepack: JoltPhysics submodule is not initialised." >&2
  echo "         Expected sources at: $JOLT_SRC" >&2
  echo "         Run: git submodule update --init --recursive" >&2
  exit 1
fi

if [ ! -f "$JOLT_DIR/LICENSE" ]; then
  echo "prepack: JoltPhysics/LICENSE missing — refusing to publish without upstream license." >&2
  exit 1
fi

echo "prepack: JoltPhysics sources present ($(du -sh "$JOLT_SRC" | cut -f1)). OK."
