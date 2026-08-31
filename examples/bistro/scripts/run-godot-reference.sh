#!/bin/sh
set -eu

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/../../.." && pwd)
source_root=${BLOOM_GODOT_BISTRO_SOURCE:-"$repo_root/../.benchmarks/Bistro-Demo-Tweaked-source"}
generated_scene="$source_root/.bloom/BistroReference.gltf"

if [ ! -f "$source_root/project.godot" ] || [ ! -f "$source_root/MainScene.tscn" ]; then
  echo "Bistro-Demo-Tweaked source not found at: $source_root" >&2
  echo "Set BLOOM_GODOT_BISTRO_SOURCE to its source checkout." >&2
  exit 1
fi

python3 "$script_dir/prepare-godot-reference.py" "$source_root"
python3 "$repo_root/tools/quality/build_example.py" "$repo_root/examples/bistro"

cd "$repo_root/examples/bistro"
exec ./main --godot-reference --fullscreen --scene "$generated_scene" "$@"
