#!/usr/bin/env python3
"""Splice the Bloom engine bootstrap into Perry's self-contained WASM HTML.

Perry emits a complete HTML page: a classic <script> with its ~280-function
`rt` runtime (NaN-boxing, closures, fetch, ...) that defines bootPerryWasm and
callWasmClosure, followed by a classic <script> that calls
`bootPerryWasm("<base64>")` to instantiate the game.

This script rewrites that page so the Bloom engine loads first and the game
WASM boots only once the engine + FFI bridge are live:

  1. Inject the Bloom canvas, loading indicator, the `window.__bloomReady`
     promise (created synchronously during parse), and the deferred
     `bloom_glue.js` module that resolves it.
  2. Gate the existing `bootPerryWasm(...)` call on `window.__bloomReady`.

Usage: splice_game.py <perry_html_in> <output_html_out>
"""
import sys

PERRY_ROOT = '<div id="perry-root"></div>'
BOOT_MARKER = 'window.__perryWasmB64'
BOOT_CALL = 'bootPerryWasm("'
BOOT_CATCH = '").catch('

BLOOM_SHELL = '''
  <canvas id="bloom-canvas"></canvas>
  <div id="loading">Loading Bloom Engine...</div>
  <style>
    #bloom-canvas { position: fixed; inset: 0; width: 100vw; height: 100vh; display: block; }
    #loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
               color: #fff; font-family: monospace; font-size: 14px; z-index: 1; }
  </style>
  <!-- Bloom: engine + FFI bootstrap. The promise is created synchronously here
       so the gated bootPerryWasm() call below can await it before the deferred
       module runs. -->
  <script>
    window.__bloomReady = new Promise((resolve, reject) => {
      window.__bloomReadyResolve = resolve;
      window.__bloomReadyReject = reject;
    });
  </script>
  <script type="module" src="./bloom_glue.js"></script>
'''


def splice(html: str) -> str:
    if PERRY_ROOT not in html:
        raise SystemExit(
            'splice_game.py: could not find perry-root div in Perry HTML — '
            'output format may have changed; aborting so the build fails loudly.'
        )
    if BOOT_MARKER not in html:
        raise SystemExit(
            'splice_game.py: could not find the bootPerryWasm boot block — '
            'output format may have changed; aborting.'
        )

    # 1. Inject the Bloom shell right after Perry's root div.
    html = html.replace(PERRY_ROOT, PERRY_ROOT + BLOOM_SHELL, 1)

    # 2. Gate the boot call. Operate only on the tail beginning at the boot
    #    marker so we never touch a coincidental `bootPerryWasm("` / `").catch(`
    #    inside the large runtime script (the definition reads
    #    `bootPerryWasm(wasmBase64`, not `bootPerryWasm("`).
    idx = html.rindex(BOOT_MARKER)
    head, tail = html[:idx], html[idx:]
    if BOOT_CALL not in tail or BOOT_CATCH not in tail:
        raise SystemExit('splice_game.py: unexpected boot block shape; aborting.')
    tail = tail.replace(BOOT_CALL, 'window.__bloomReady.then(() => bootPerryWasm("', 1)
    tail = tail.replace(BOOT_CATCH, '")).catch(', 1)
    return head + tail


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit('Usage: splice_game.py <perry_html_in> <output_html_out>')
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        html = f.read()
    with open(sys.argv[2], 'w', encoding='utf-8') as f:
        f.write(splice(html))


if __name__ == '__main__':
    main()
