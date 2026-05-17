#!/usr/bin/env bash
# Merge a cargo-built libbloom_<platform>.a (or .lib) with the
# JoltPhysics + bloom_jolt static archives into one self-contained
# fat archive that Perry can point a single `prebuilt:` path at.
#
# Why a fat archive: Perry's `prebuilt:` field takes a single file
# (issue #860 design). The Rust staticlib output is NOT self-contained
# — it has unresolved JPH::* / bloom_jolt symbols that get resolved by
# the final consumer link. On a hub that doesn't run cargo, there's no
# way to feed those extra archives in. So we pre-merge them here.
#
# Per-platform tooling (each runner has the right one natively):
#   - macOS / iOS / tvOS / watchOS  -> libtool -static  (BSD)
#   - Linux / Android               -> ar x then ar crs (GNU/LLVM)
#   - Windows                       -> lib.exe /OUT     (MSVC)
#
# Usage: bundle-engine-prebuilt.sh <platform> <rust-target-triple> <out-dir>
#   platform: macos|ios|tvos|watchos|linux|android|windows
#   rust-target-triple: e.g. aarch64-apple-darwin
#   out-dir: directory to write the bundled archive into
#
# Locates the Rust output relative to native/<platform>/target/, the
# Jolt archives relative to native/third_party/bloom_jolt/build/ or
# (preferred) the env var BLOOM_JOLT_PREBUILT_DIR/<os>-<arch>/.

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "usage: $0 <platform> <rust-target-triple> <out-dir>" >&2
  exit 2
fi

PLATFORM="$1"
TARGET="$2"
OUT_DIR="$3"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Resolve input paths.
# ---------------------------------------------------------------------------
NATIVE_DIR="$REPO_ROOT/native/$PLATFORM"

is_windows=0
case "$PLATFORM" in
  windows) is_windows=1 ;;
esac

if [ "$is_windows" = "1" ]; then
  CARGO_LIB_NAME="bloom_${PLATFORM}.lib"
  OUT_LIB_NAME="bloom_${PLATFORM}_bundled.lib"
  JOLT_NAME="Jolt.lib"
  JOLT_SHIM_NAME="bloom_jolt.lib"
else
  CARGO_LIB_NAME="libbloom_${PLATFORM}.a"
  OUT_LIB_NAME="libbloom_${PLATFORM}_bundled.a"
  JOLT_NAME="libJolt.a"
  JOLT_SHIM_NAME="libbloom_jolt.a"
fi

# Per-target Rust output sits under either target/release/ (host build)
# or target/<triple>/release/ (cross-compile). Try both.
CARGO_LIB=""
for candidate in \
  "$NATIVE_DIR/target/release/$CARGO_LIB_NAME" \
  "$NATIVE_DIR/target/$TARGET/release/$CARGO_LIB_NAME"; do
  if [ -f "$candidate" ]; then
    CARGO_LIB="$candidate"
    break
  fi
done

if [ -z "$CARGO_LIB" ]; then
  echo "bundle: cargo output not found for $PLATFORM ($CARGO_LIB_NAME)" >&2
  echo "  tried:" >&2
  echo "    $NATIVE_DIR/target/release/$CARGO_LIB_NAME" >&2
  echo "    $NATIVE_DIR/target/$TARGET/release/$CARGO_LIB_NAME" >&2
  exit 1
fi

# Map Rust target triple to the npm/jolt-prebuilt arch-key convention
# used by build.rs (see native/shared/build.rs::find_prebuilt_dir).
arch_key_for_triple() {
  case "$1" in
    aarch64-apple-darwin)        echo "macos-arm64" ;;
    x86_64-apple-darwin)         echo "macos-x64" ;;
    aarch64-apple-ios)           echo "ios-arm64" ;;
    aarch64-apple-ios-sim)       echo "ios-arm64-sim" ;;
    x86_64-apple-ios)            echo "ios-x64-sim" ;;
    aarch64-apple-tvos)          echo "tvos-arm64" ;;
    aarch64-apple-tvos-sim)      echo "tvos-arm64-sim" ;;
    x86_64-apple-tvos)           echo "tvos-x64-sim" ;;
    aarch64-linux-android)       echo "android-arm64" ;;
    armv7-linux-androideabi)     echo "android-armv7" ;;
    x86_64-linux-android)        echo "android-x64" ;;
    x86_64-pc-windows-msvc)      echo "win32-x64" ;;
    x86_64-unknown-linux-gnu)    echo "linux-x64" ;;
    aarch64-unknown-linux-gnu)   echo "linux-arm64" ;;
    *)                           echo "" ;;
  esac
}

JOLT_ARCH_KEY="$(arch_key_for_triple "$TARGET")"

# Locate jolt-prebuilt archives. Priority:
#   1. BLOOM_JOLT_PREBUILT_DIR/<arch-key>/  (CI sets this to staged artifacts)
#   2. <repo>/npm/jolt-prebuilt/lib/<arch-key>/  (local dev built libs)
#   3. <repo>/native/third_party/bloom_jolt/build/<os>-<arch>/lib/ (cmake fallback)
JOLT_DIR=""
if [ -n "${BLOOM_JOLT_PREBUILT_DIR:-}" ] && [ -n "$JOLT_ARCH_KEY" ]; then
  candidate="$BLOOM_JOLT_PREBUILT_DIR/$JOLT_ARCH_KEY"
  if [ -f "$candidate/$JOLT_NAME" ]; then
    JOLT_DIR="$candidate"
  fi
fi
if [ -z "$JOLT_DIR" ] && [ -n "$JOLT_ARCH_KEY" ]; then
  candidate="$REPO_ROOT/npm/jolt-prebuilt/lib/$JOLT_ARCH_KEY"
  if [ -f "$candidate/$JOLT_NAME" ]; then
    JOLT_DIR="$candidate"
  fi
fi
if [ -z "$JOLT_DIR" ]; then
  # Match build.rs's per-target dir name (target_os-target_arch).
  case "$TARGET" in
    aarch64-apple-darwin)       cm_dir="macos-aarch64" ;;
    x86_64-apple-darwin)        cm_dir="macos-x86_64" ;;
    aarch64-apple-ios)          cm_dir="ios-aarch64" ;;
    aarch64-apple-tvos)         cm_dir="tvos-aarch64" ;;
    aarch64-linux-android)      cm_dir="android-aarch64" ;;
    x86_64-pc-windows-msvc)     cm_dir="windows-x86_64" ;;
    x86_64-unknown-linux-gnu)   cm_dir="linux-x86_64" ;;
    *)                          cm_dir="" ;;
  esac
  if [ -n "$cm_dir" ]; then
    candidate="$REPO_ROOT/native/third_party/bloom_jolt/build/$cm_dir/lib"
    if [ -f "$candidate/$JOLT_NAME" ]; then
      JOLT_DIR="$candidate"
    fi
  fi
fi

if [ -z "$JOLT_DIR" ] || [ ! -f "$JOLT_DIR/$JOLT_NAME" ] || [ ! -f "$JOLT_DIR/$JOLT_SHIM_NAME" ]; then
  echo "bundle: jolt archives not found for target $TARGET" >&2
  echo "  set BLOOM_JOLT_PREBUILT_DIR or build jolt-prebuilt first." >&2
  exit 1
fi

OUT_PATH="$OUT_DIR/$OUT_LIB_NAME"
echo "bundle:"
echo "  in     $CARGO_LIB"
echo "  + jolt $JOLT_DIR/$JOLT_NAME"
echo "  + shim $JOLT_DIR/$JOLT_SHIM_NAME"
echo "  out    $OUT_PATH"
rm -f "$OUT_PATH"

# ---------------------------------------------------------------------------
# Merge.
# ---------------------------------------------------------------------------
case "$PLATFORM" in
  macos|ios|tvos|watchos)
    # BSD libtool -static merges archives cleanly, handles duplicate
    # member names by suffixing. Warnings about Rust object files with
    # "no symbols" (compiler_builtins, rustc_std_workspace_core stubs)
    # are expected and harmless.
    libtool -static -o "$OUT_PATH" \
      "$CARGO_LIB" \
      "$JOLT_DIR/$JOLT_NAME" \
      "$JOLT_DIR/$JOLT_SHIM_NAME"
    ;;
  linux|android)
    # GNU ar can't merge archives directly; extract first, prefix the
    # member names so identical filenames from different archives
    # (e.g. each archive's own compiler_builtins.o) don't collide,
    # then re-archive.
    WORK="$(mktemp -d)"
    trap 'rm -rf "$WORK"' EXIT
    AR="${AR:-ar}"
    extract() {
      local archive="$1" prefix="$2"
      local subdir="$WORK/$prefix"
      mkdir -p "$subdir"
      ( cd "$subdir" && "$AR" x "$archive" )
      for f in "$subdir"/*.o; do
        [ -e "$f" ] || continue
        mv "$f" "$WORK/${prefix}_$(basename "$f")"
      done
      rmdir "$subdir"
    }
    extract "$(cd "$(dirname "$CARGO_LIB")" && pwd)/$(basename "$CARGO_LIB")"   bloom
    extract "$JOLT_DIR/$JOLT_NAME"                                              jolt
    extract "$JOLT_DIR/$JOLT_SHIM_NAME"                                         jshim
    "$AR" crs "$OUT_PATH" "$WORK"/*.o
    ;;
  windows)
    # MSVC lib.exe takes /OUT:foo.lib + input libs. PATH must contain
    # the Visual Studio dev tools; on GitHub windows-latest runners the
    # `windows-msvc` setup-action puts them there.
    lib.exe "/OUT:$OUT_PATH" \
      "$CARGO_LIB" \
      "$JOLT_DIR/$JOLT_NAME" \
      "$JOLT_DIR/$JOLT_SHIM_NAME"
    ;;
  *)
    echo "bundle: unknown platform $PLATFORM" >&2
    exit 2
    ;;
esac

echo "bundle: ok — $(ls -lh "$OUT_PATH" | awk '{print $5}')"
