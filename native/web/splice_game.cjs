#!/usr/bin/env node
"use strict";

// Gate Perry's game entry on the Bloom engine and FFI bridge being ready.
const fs = require("node:fs");
const PERRY_ROOT = "<div id=\"perry-root\"></div>";
const BOOT_MARKER = "window.__perryWasmB64";
const BOOT_CALL = "bootPerryWasm(\"";
const BOOT_CATCH = "\").catch(";
const BLOOM_SHELL = `
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
`;

function splice(html) {
  if (!html.includes(PERRY_ROOT)) {
    throw new Error("could not find perry-root in Perry HTML; compiler output format may have changed");
  }
  if (!html.includes(BOOT_MARKER)) {
    throw new Error("could not find the Perry boot block; compiler output format may have changed");
  }
  html = html.replace(PERRY_ROOT, PERRY_ROOT + BLOOM_SHELL);
  const index = html.lastIndexOf(BOOT_MARKER);
  const head = html.slice(0, index);
  let tail = html.slice(index);
  if (!tail.includes(BOOT_CALL) || !tail.includes(BOOT_CATCH)) {
    throw new Error("unexpected Perry boot block shape; compiler output format may have changed");
  }
  tail = tail.replace(BOOT_CALL, 'window.__bloomReady.then(() => bootPerryWasm("');
  tail = tail.replace(BOOT_CATCH, '\")).catch(');
  return head + tail;
}

module.exports = { splice };
if (require.main === module) {
  try {
    if (process.argv.length !== 4) throw new Error("Usage: splice_game.cjs <perry_html_in> <output_html_out>");
    fs.writeFileSync(process.argv[3], splice(fs.readFileSync(process.argv[2], "utf8")), "utf8");
  } catch (error) {
    console.error(`splice_game: ${error.message}`);
    process.exitCode = 1;
  }
}
