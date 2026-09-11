#!/bin/bash
# Compatibility entry point for existing checkout-based build commands.
exec node "$(dirname "$0")/build.cjs" "$@"
