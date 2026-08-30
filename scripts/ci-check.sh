#!/usr/bin/env bash
# Bloom's versioned local/CI qualification entry point.
#
# Usage:
#   ./scripts/ci-check.sh --quick
#   ./scripts/ci-check.sh --full
#   ./scripts/ci-check.sh --web
#   ./scripts/ci-check.sh --cross
#   ./scripts/ci-check.sh --hardware
#   ./scripts/ci-check.sh --quick --component lint
#   ./scripts/ci-check.sh --list
#
# With no lane argument, the historical full-local-suite behavior is retained.
# Every invocation writes a JSON summary under target/ci unless --summary is
# supplied. CI selects components so independent jobs can run in parallel; a
# lane without --component runs all of its components in order.

set -euo pipefail

LANE=""
COMPONENT=""
SUMMARY_PATH=""
LIST_ONLY=0

usage() {
  sed -n '2,14p' "$0"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --quick|--full|--web|--cross|--hardware)
      if [ -n "$LANE" ]; then
        echo "choose exactly one lane" >&2
        exit 2
      fi
      LANE="${1#--}"
      ;;
    --component=*) COMPONENT="${1#*=}" ;;
    --summary=*) SUMMARY_PATH="${1#*=}" ;;
    --component)
      if [ "$#" -lt 2 ]; then
        echo "--component requires a value" >&2
        exit 2
      fi
      COMPONENT="$2"
      shift
      ;;
    --summary)
      if [ "$#" -lt 2 ]; then
        echo "--summary requires a value" >&2
        exit 2
      fi
      SUMMARY_PATH="$2"
      shift
      ;;
    --list) LIST_ONLY=1 ;;
    --fast)
      echo "warning: --fast is deprecated; use --quick" >&2
      LANE="quick"
      ;;
    --wasm)
      echo "warning: --wasm is deprecated; use --full or --web" >&2
      LANE="full"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

lane_components() {
  case "$1" in
    quick) printf '%s\n' "contracts lint shared-tests wasm-check quality-contract example-inventory" ;;
    full) printf '%s\n' "contracts lint shared-tests wasm-check quality-contract example-inventory host-build wasm-build" ;;
    web) printf '%s\n' "wasm-check wasm-build browser-smoke" ;;
    cross) printf '%s\n' "target-check" ;;
    hardware) printf '%s\n' "example-compile quality-check quality-faults quality-run fractional-native-throughput virtual-geometry-stress" ;;
    *)
      echo "unknown lane: $1" >&2
      return 2
      ;;
  esac
}

if [ "$LIST_ONLY" -eq 1 ]; then
  printf 'quick\t%s\n' "$(lane_components quick)"
  printf 'full\t%s\n' "$(lane_components full)"
  printf 'web\t%s\n' "$(lane_components web)"
  printf 'cross\t%s\n' "$(lane_components cross)"
  printf 'hardware\t%s\n' "$(lane_components hardware)"
  exit 0
fi

if [ -z "$LANE" ]; then
  LANE="full"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

host_os="$(uname -s)"
case "$host_os" in
  Darwin) host_crate="macos" ;;
  Linux) host_crate="linux" ;;
  MINGW*|MSYS*|CYGWIN*) host_crate="windows" ;;
  *) host_crate="" ;;
esac

ALLOWED_COMPONENTS="$(lane_components "$LANE")"
if [ -n "$COMPONENT" ]; then
  case " $ALLOWED_COMPONENTS " in
    *" $COMPONENT "*) COMPONENTS="$COMPONENT" ;;
    *)
      echo "component '$COMPONENT' does not belong to the '$LANE' lane" >&2
      exit 2
      ;;
  esac
else
  COMPONENTS="$ALLOWED_COMPONENTS"
fi

if [ -z "$SUMMARY_PATH" ]; then
  summary_component="${COMPONENT:-all}"
  SUMMARY_PATH="$ROOT/target/ci/${LANE}-${summary_component}.json"
elif [ "${SUMMARY_PATH#/}" = "$SUMMARY_PATH" ]; then
  SUMMARY_PATH="$ROOT/$SUMMARY_PATH"
fi

STARTED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
START_SECONDS="$(date '+%s')"
CURRENT_COMPONENT=""
COMPLETED_COMPONENTS=""

hr() {
  printf '\n==> %s\n' "$*"
}

append_completed() {
  if [ -n "$COMPLETED_COMPONENTS" ]; then
    COMPLETED_COMPONENTS="$COMPLETED_COMPONENTS,$1"
  else
    COMPLETED_COMPONENTS="$1"
  fi
}

write_summary() {
  status="$1"
  exit_code="$2"
  finished_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  duration_seconds="$(( $(date '+%s') - START_SECONDS ))"
  mkdir -p "$(dirname "$SUMMARY_PATH")"
  BLOOM_CI_LANE="$LANE" \
  BLOOM_CI_REQUESTED_COMPONENT="${COMPONENT:-all}" \
  BLOOM_CI_CURRENT_COMPONENT="$CURRENT_COMPONENT" \
  BLOOM_CI_COMPLETED_COMPONENTS="$COMPLETED_COMPONENTS" \
  BLOOM_CI_STATUS="$status" \
  BLOOM_CI_EXIT_CODE="$exit_code" \
  BLOOM_CI_HOST_OS="$host_os" \
  BLOOM_CI_STARTED_AT="$STARTED_AT" \
  BLOOM_CI_FINISHED_AT="$finished_at" \
  BLOOM_CI_DURATION_SECONDS="$duration_seconds" \
  BLOOM_CI_SUMMARY_PATH="$SUMMARY_PATH" \
  node -e '
    const fs = require("fs");
    const env = process.env;
    const split = (value) => value ? value.split(",") : [];
    const summary = {
      schema: "bloom-ci-summary-v1",
      lane: env.BLOOM_CI_LANE,
      requested_component: env.BLOOM_CI_REQUESTED_COMPONENT,
      current_component: env.BLOOM_CI_CURRENT_COMPONENT || null,
      completed_components: split(env.BLOOM_CI_COMPLETED_COMPONENTS),
      status: env.BLOOM_CI_STATUS,
      exit_code: Number(env.BLOOM_CI_EXIT_CODE),
      host_os: env.BLOOM_CI_HOST_OS,
      started_at: env.BLOOM_CI_STARTED_AT,
      finished_at: env.BLOOM_CI_FINISHED_AT,
      duration_seconds: Number(env.BLOOM_CI_DURATION_SECONDS)
    };
    fs.writeFileSync(env.BLOOM_CI_SUMMARY_PATH, JSON.stringify(summary, null, 2) + "\n");
  '
}

on_exit() {
  exit_code=$?
  trap - EXIT
  if [ "$exit_code" -eq 0 ]; then
    write_summary "pass" "$exit_code"
  else
    write_summary "fail" "$exit_code" || true
  fi
  exit "$exit_code"
}
trap on_exit EXIT

run_component() {
  CURRENT_COMPONENT="$1"
  case "$CURRENT_COMPONENT" in
    contracts)
      hr "CI command inventory"
      node tools/check-ci-contract.js
      hr "FFI/schema parity"
      node tools/validate-ffi.js
      hr "documentation and package contracts"
      node tools/validate-docs.js
      hr "file-size ratchet"
      node tools/check-file-lines.js
      ;;
    lint)
      hr "bloom-shared: cargo fmt --check"
      ( cd native/shared && cargo fmt --check )
      hr "bloom-shared: strict clippy correctness/performance policy"
      (
        cd native/shared
        cargo clippy --release --no-deps -- \
          -A warnings \
          -D clippy::correctness \
          -D clippy::suspicious \
          -D clippy::perf \
          -A clippy::empty-line-after-doc-comments \
          -A clippy::manual-memcpy \
          -A clippy::not-unsafe-ptr-arg-deref \
          -A clippy::cloned-ref-to-slice-refs
      )
      ;;
    shared-tests)
      hr "bloom-shared: cargo test --release"
      ( cd native/shared && cargo test --release )
      ;;
    wasm-check)
      hr "bloom-shared: cargo check (wasm32, web feature)"
      (
        cd native/shared
        cargo check \
          --target wasm32-unknown-unknown \
          --no-default-features \
          --features web
      )
      ;;
    quality-contract)
      hr "quality orchestration syntax and governance tests"
      python3 -m py_compile \
        tools/quality/run.py \
        tools/quality/build_example.py \
        tools/quality/khronos_materials.py \
        tools/quality/shadow_detail.py \
        tools/quality/prepare_virtual_geometry_stress.py \
        tools/quality/virtual_geometry_stress.py \
        tools/quality/vsm_caster_coverage.py \
        tools/quality/vsm_debug_views.py \
        tools/quality/vsm_motion_corpus.py \
        tools/quality/prepare_bistro.py \
        tools/ci/web_smoke.py \
        tools/ci/test_web_smoke.py
      python3 -m unittest \
        tools/quality/test_run.py \
        tools/quality/test_khronos_materials.py \
        tools/quality/test_shadow_detail.py \
        tools/quality/test_prepare_virtual_geometry_stress.py \
        tools/quality/test_virtual_geometry_stress.py \
        tools/quality/test_vsm_caster_coverage.py \
        tools/quality/test_vsm_debug_views.py \
        tools/quality/test_vsm_motion_corpus.py \
        tools/ci/test_web_smoke.py \
        -v
      hr "visual metric and fault-engine tests"
      cargo test --release --manifest-path tools/bloom-diff/Cargo.toml
      hr "offline asset cooker format, corruption, and determinism tests"
      cargo fmt --manifest-path crates/bloom-geometry-format/Cargo.toml -- --check
      cargo clippy --release --manifest-path crates/bloom-geometry-format/Cargo.toml \
        --no-deps -- -D warnings
      cargo fmt --manifest-path tools/bloom-cook/Cargo.toml -- --check
      cargo clippy --release --manifest-path tools/bloom-cook/Cargo.toml \
        --no-deps -- -D warnings
      cargo test --release --manifest-path tools/bloom-cook/Cargo.toml
      ;;
    example-inventory)
      hr "canonical TypeScript example inventory"
      python3 tools/ci/compile_examples.py --check
      ;;
    host-build)
      if [ -z "$host_crate" ]; then
        echo "unsupported host for native build: $host_os" >&2
        return 2
      fi
      hr "bloom-$host_crate: cargo build --release"
      ( cd "native/$host_crate" && cargo build --release )
      ;;
    wasm-build)
      if ! command -v wasm-pack >/dev/null 2>&1; then
        echo "wasm-pack is required for the '$LANE' lane" >&2
        return 2
      fi
      hr "bloom-web: wasm-pack build --release --target web"
      ( cd native/web && wasm-pack build --release --target web )
      ;;
    browser-smoke)
      hr "Bloom WebGPU real-browser known-frame smoke"
      python3 tools/ci/web_smoke.py
      ;;
    target-check)
      cross_crate="${BLOOM_CROSS_CRATE:-}"
      cross_target="${BLOOM_CROSS_TARGET:-}"
      cross_features="${BLOOM_CROSS_FEATURES:-}"
      if [ -z "$cross_crate" ] || [ -z "$cross_target" ]; then
        echo "BLOOM_CROSS_CRATE and BLOOM_CROSS_TARGET are required for target-check" >&2
        return 2
      fi
      case "$cross_crate" in
        android|ios|tvos|visionos|watchos) ;;
        *)
          echo "unsupported cross-target crate: $cross_crate" >&2
          return 2
          ;;
      esac
      case "$cross_target" in
        *-linux-android*)
          android_ndk="${ANDROID_NDK_HOME:-${ANDROID_NDK_LATEST_HOME:-}}"
          if [ -z "$android_ndk" ]; then
            echo "ANDROID_NDK_HOME or ANDROID_NDK_LATEST_HOME is required for Android checks" >&2
            return 2
          fi
          case "$host_os" in
            Linux) android_host="linux-x86_64" ;;
            Darwin) android_host="darwin-x86_64" ;;
            *)
              echo "unsupported Android NDK host: $host_os" >&2
              return 2
              ;;
          esac
          android_api="${BLOOM_ANDROID_API:-24}"
          case "$cross_target" in
            aarch64-linux-android) android_clang="aarch64-linux-android${android_api}-clang" ;;
            armv7-linux-androideabi) android_clang="armv7a-linux-androideabi${android_api}-clang" ;;
            x86_64-linux-android) android_clang="x86_64-linux-android${android_api}-clang" ;;
            *)
              echo "unsupported Android Rust target: $cross_target" >&2
              return 2
              ;;
          esac
          android_bin="$android_ndk/toolchains/llvm/prebuilt/$android_host/bin"
          android_cc="$android_bin/$android_clang"
          if [ ! -x "$android_cc" ]; then
            echo "Android compiler not found: $android_cc" >&2
            return 2
          fi
          target_env="$(printf '%s' "$cross_target" | tr '-' '_')"
          target_env_upper="$(printf '%s' "$target_env" | tr '[:lower:]' '[:upper:]')"
          export "CC_${target_env}=$android_cc"
          export "CXX_${target_env}=${android_cc}++"
          export "AR_${target_env}=$android_bin/llvm-ar"
          export "CARGO_TARGET_${target_env_upper}_LINKER=$android_cc"
          export ANDROID_PLATFORM="android-$android_api"
          ;;
      esac
      hr "bloom-$cross_crate: cargo check ($cross_target)"
      cargo_args=(check --locked --target "$cross_target" --no-default-features)
      if [ -n "$cross_features" ]; then
        cargo_args+=(--features "$cross_features")
      fi
      ( cd "native/$cross_crate" && cargo "${cargo_args[@]}" )
      ;;
    example-compile)
      hr "compile every canonical TypeScript example"
      python3 tools/ci/compile_examples.py
      ;;
    quality-check)
      hr "validate quality manifest, assets, and approved baselines"
      python3 tools/quality/run.py check
      ;;
    quality-faults)
      hr "prove seeded quality regressions are detected"
      python3 tools/quality/run.py faults \
        --out "${BLOOM_QUALITY_FAULTS_OUT:-tools/quality/out/ci-faults}" \
        --timeout "${BLOOM_QUALITY_TIMEOUT:-900}"
      ;;
    quality-run)
      if [ -z "${BLOOM_QUALITY_MACHINE_CLASS:-}" ]; then
        echo "BLOOM_QUALITY_MACHINE_CLASS is required for hardware quality runs" >&2
        return 2
      fi
      quality_suite="${BLOOM_QUALITY_SUITE:-full}"
      quality_out="${BLOOM_QUALITY_OUT:-tools/quality/out/ci-hardware}"
      hr "run '$quality_suite' quality suite on $BLOOM_QUALITY_MACHINE_CLASS"
      if [ -n "${BLOOM_QUALITY_CASE:-}" ]; then
        python3 tools/quality/run.py run "$quality_suite" \
          --case "$BLOOM_QUALITY_CASE" \
          --machine-class "$BLOOM_QUALITY_MACHINE_CLASS" \
          --out "$quality_out" \
          --timeout "${BLOOM_QUALITY_TIMEOUT:-1800}"
      else
        python3 tools/quality/run.py run "$quality_suite" \
          --machine-class "$BLOOM_QUALITY_MACHINE_CLASS" \
          --out "$quality_out" \
          --timeout "${BLOOM_QUALITY_TIMEOUT:-1800}"
      fi
      ;;
    fractional-native-throughput)
      throughput_out="${BLOOM_PROFILE_FRACTIONAL_TAA_OUT:-tools/quality/out/ci-fractional-native-throughput}"
      if [ "${throughput_out#/}" = "$throughput_out" ]; then
        throughput_out="$ROOT/$throughput_out"
      fi
      hr "qualify fractional 0.75 throughput against native 1.0"
      (
        cd native/shared
        BLOOM_PROFILE_FRACTIONAL_TAA_OUT="$throughput_out" \
        BLOOM_PROFILE_FRACTIONAL_TAA_FRAMES="${BLOOM_PROFILE_FRACTIONAL_TAA_FRAMES:-600}" \
        BLOOM_PROFILE_FRACTIONAL_TAA_PAIRS="${BLOOM_PROFILE_FRACTIONAL_TAA_PAIRS:-3}" \
        BLOOM_PROFILE_FRACTIONAL_TAA_MIN_ADVANTAGE="${BLOOM_PROFILE_FRACTIONAL_TAA_MIN_ADVANTAGE:-0.05}" \
        BLOOM_PROFILE_FRACTIONAL_TAA_CAMERA_STEP="${BLOOM_PROFILE_FRACTIONAL_TAA_CAMERA_STEP:-0.002}" \
        cargo test --release --test golden_render \
          quality_presets::profile_fractional_taa_native_advantage \
          -- --exact --ignored --nocapture
      )
      ;;
    virtual-geometry-stress)
      if [ -z "${BLOOM_VIRTUAL_STRESS_PLATFORM:-}" ] || [ -z "${BLOOM_VIRTUAL_STRESS_BACKEND:-}" ]; then
        echo "BLOOM_VIRTUAL_STRESS_PLATFORM and BLOOM_VIRTUAL_STRESS_BACKEND are required" >&2
        return 2
      fi
      vg_stress_out="${BLOOM_VIRTUAL_STRESS_OUT:-tools/quality/out/ci-virtual-geometry}"
      vg_stress_work="${BLOOM_VIRTUAL_STRESS_WORK:-${RUNNER_TEMP:-/tmp}/bloom-virtual-geometry-stress}"
      hr "run 10M virtual-geometry stress on $BLOOM_VIRTUAL_STRESS_BACKEND"
      python3 tools/quality/virtual_geometry_stress.py \
        --platform "$BLOOM_VIRTUAL_STRESS_PLATFORM" \
        --backend "$BLOOM_VIRTUAL_STRESS_BACKEND" \
        --work "$vg_stress_work" \
        --out "$vg_stress_out"
      ;;
    *)
      echo "unknown component: $CURRENT_COMPONENT" >&2
      return 2
      ;;
  esac
  append_completed "$CURRENT_COMPONENT"
}

for component in $COMPONENTS; do
  run_component "$component"
done

CURRENT_COMPONENT=""
hr "OK — '$LANE' lane passed"
