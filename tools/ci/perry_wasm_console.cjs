"use strict";

// Execute a pure TypeScript contract fixture's actual Perry WASM/runtime in
// Node. Minimal DOM objects satisfy runtime style setup; no browser, engine
// renderer, GPU or network participates in the contract result. The optional
// unused-FFI mode installs guards that reject every attempted engine call.
const fs = require("node:fs");
const vm = require("node:vm");
const { installPerryVoidReturnCompatibility } = require("../../native/web/splice_game.cjs");

async function main() {
  if (process.argv.length < 3 || process.argv.length > 5 || (process.argv.length === 5 && process.argv[4] !== '--guard-unused-ffi')) throw new Error("Usage: perry_wasm_console.cjs <compiled-fixture.html> [RESULT_PREFIX:] [--guard-unused-ffi]");
  const prefix = process.argv[3] || "BLOOM_FIXED_STEP_RESULT:";
  if (!/^[A-Z_]+:$/.test(prefix)) throw new Error("Contract result prefix must be an uppercase identifier followed by a colon");
  const html = fs.readFileSync(process.argv[2], "utf8");
  const encoded = html.match(/window\.__perryWasmB64\s*=\s*"([A-Za-z0-9+/=]+)"/);
  if (!encoded) throw new Error("Perry output has no embedded WASM");
  const imports = WebAssembly.Module.imports(new WebAssembly.Module(Buffer.from(encoded[1], "base64")));
  const ffi = imports.filter(item => item.module === 'ffi');
  if (ffi.length && process.argv[4] !== '--guard-unused-ffi') throw new Error("Pure contract fixture unexpectedly requires engine FFI");
  const guardedFfi = Object.fromEntries(ffi.map(item => [item.name, () => {
    throw new Error('FFI execution forbidden: ' + item.name);
  }]));
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
  if (scripts.length !== 2) throw new Error("Unexpected Perry runtime/boot script shape");
  let timer;
  try {
    const results = [], errors = [];
    const element = { style: {}, appendChild() {}, addEventListener() {}, textContent: "" };
    const context = vm.createContext({
      __ffiImports: guardedFfi, WebAssembly, TextEncoder, TextDecoder, atob, btoa, performance, setTimeout, clearTimeout,
      console: {
        log(...args) {
          const line = args.map(String).join(" ");
          if (line.startsWith(prefix)) results.push(line);
        },
        warn() {},
        error(...args) { errors.push(args.map(String).join(" ")); },
      },
      document: { getElementById() { return element; }, createElement() { return element; },
        addEventListener() {}, body: element, head: element },
      navigator: {}, addEventListener() {},
    });
    context.window = context;
    context.self = context;
    vm.runInContext(scripts[0][1], context, { timeout: 5000 });
    vm.runInContext(`(${installPerryVoidReturnCompatibility.toString()})(globalThis);`, context, { timeout: 5000 });
    const boot = context.bootPerryWasm;
    if (typeof boot !== "function") throw new Error("Perry runtime has no boot entry");
    let completion;
    context.bootPerryWasm = (...args) => {
      if (completion) throw new Error("Perry fixture booted more than once");
      completion = boot(...args);
      return completion;
    };
    vm.runInContext(scripts[1][1], context, { timeout: 5000 });
    if (!completion || typeof completion.then !== "function") throw new Error("Perry fixture did not boot");
    await Promise.race([completion, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error("Perry WASM fixture did not finish")), 10000);
    })]);
    if (errors.length) throw new Error(errors.join("\n"));
    if (results.length !== 1) throw new Error("Perry fixture must emit exactly one contract result");
    process.stdout.write(results[0] + "\n");
  } finally { clearTimeout(timer); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
