#!/bin/bash
# Build the "Bloom inside Perry UI" demo for Windows (issue #2395).
#
# Requires a Perry build that includes the `BloomView` widget (perry-ui-windows
# + the BloomView dispatch/manifest wiring). Point $PERRY at it, or have a
# matching `perry` on PATH.
set -e

PERRY="${PERRY:-perry}"

"$PERRY" compile --target windows src/main.ts -o BloomEmbed
mv -f BloomEmbed BloomEmbed.exe 2>/dev/null || true

# Cranelift doesn't emit __chkstk stack probes, so large functions can skip the
# guard page and crash. Pre-commit a 1MB stack (same fix the Bloom games use).
EDITBIN="/c/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/editbin.exe"
if [ -f "$EDITBIN" ]; then
  "$EDITBIN" /STACK:67108864,1048576 BloomEmbed.exe
else
  echo "Warning: editbin not found; skipping stack fix. App may crash on launch."
fi

echo "Built BloomEmbed.exe"
if [ "$1" = "--run" ]; then
  cmd //c "$(cygpath -w "$(pwd)/BloomEmbed.exe")"
fi
