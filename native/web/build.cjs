#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { splice } = require("./splice_game.cjs");

const HELP = `Usage: bloom-web [game.ts] [--output dist/web]

Build the engine and optional Perry game for the browser.
Requires Rust, wasm-pack and (for a game) Perry on PATH.
Optional: wasm-opt for additional size optimization.
Set BLOOM_PERRY, BLOOM_WASM_PACK or BLOOM_WASM_OPT to an executable path.
`;

class BuildError extends Error {
  constructor(message, exitCode = 1) {
    super(message);
    this.exitCode = exitCode;
  }
}

function parseArgs(args, cwd = process.cwd()) {
  let game;
  let output = "dist/web";
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === "--help" || arg === "-h") return { help: true };
    if (arg === "--output") {
      if (!args[i + 1] || args[i + 1].startsWith("--")) {
        throw new BuildError("--output requires a directory", 2);
      }
      output = args[++i];
    } else if (arg.startsWith("--output=")) {
      output = arg.slice("--output=".length);
      if (!output) throw new BuildError("--output requires a directory", 2);
    } else if (arg.startsWith("-")) {
      throw new BuildError(`unknown option: ${arg}`, 2);
    } else if (game !== undefined) {
      throw new BuildError("only one game entry file may be supplied", 2);
    } else {
      game = arg;
    }
  }
  return {
    game: game === undefined ? undefined : path.resolve(cwd, game),
    output: path.resolve(cwd, output),
  };
}

function runTool(program, args, { execute = spawnSync, cwd, optional = false } = {}) {
  const result = execute(program, args, {
    cwd, stdio: "inherit", shell: false, windowsHide: true,
  });
  if (result.error) {
    if (optional && result.error.code === "ENOENT") return false;
    const detail = result.error.code === "ENOENT"
      ? "executable not found; install it or set its BLOOM_* executable path"
      : result.error.message;
    throw new BuildError(`${program}: ${detail}`);
  }
  if (result.status !== 0) {
    throw new BuildError(
      `${program} failed (${result.signal ? `signal ${result.signal}` : `exit ${result.status}`})`,
      Number.isInteger(result.status) && result.status > 0 ? result.status : 1,
    );
  }
  return true;
}

function requireFile(file, label) {
  if (!fs.existsSync(file) || !fs.statSync(file).isFile() || fs.statSync(file).size === 0) {
    throw new BuildError(`${label} did not produce a non-empty file: ${file}`);
  }
}

function build(options, { execute = spawnSync, webDir = __dirname, env = process.env } = {}) {
  if (options.game) requireFile(options.game, "game entry");
  const wasmPack = env.BLOOM_WASM_PACK || "wasm-pack";
  const perry = env.BLOOM_PERRY || "perry";
  const wasmOpt = env.BLOOM_WASM_OPT || "wasm-opt";
  const run = (program, args, extra = {}) => runTool(program, args, { execute, ...extra });

  // Check required tools before compiling either module. Help needs no toolchain.
  run(wasmPack, ["--version"]);
  if (options.game) run(perry, ["--version"]);

  const temporaryRoot = path.resolve(os.tmpdir());
  const temporary = fs.mkdtempSync(path.join(temporaryRoot, "bloom-web-"));
  try {
    const pkg = path.join(temporary, "pkg");
    const index = path.join(temporary, "index.html");
    if (options.game) {
      console.log("Compiling game with Perry...");
      const gameHtml = path.join(temporary, "game.html");
      run(perry, ["compile", options.game, "--target", "wasm", "-o", gameHtml], {
        cwd: path.dirname(options.game),
      });
      requireFile(gameHtml, "Perry");
      fs.writeFileSync(index, splice(fs.readFileSync(gameHtml, "utf8")), "utf8");
    } else {
      fs.copyFileSync(path.join(webDir, "index.html"), index);
    }

    console.log("Building engine with wasm-pack...");
    run(wasmPack, ["build", "--target", "web", "--out-dir", pkg, "--no-typescript"], {
      cwd: webDir,
    });
    const wasm = path.join(pkg, "bloom_web_bg.wasm");
    requireFile(wasm, "wasm-pack");
    requireFile(path.join(pkg, "bloom_web.js"), "wasm-pack");

    const optimized = path.join(temporary, "optimized.wasm");
    if (run(wasmOpt, ["-Oz", wasm, "-o", optimized], { optional: !env.BLOOM_WASM_OPT })) {
      requireFile(optimized, "wasm-opt");
      fs.renameSync(optimized, wasm);
    } else {
      console.log("Skipping optional wasm-opt (not installed).");
    }

    // Build in a unique staging directory. Failed commands cannot reuse an old
    // pkg/ or overwrite the caller's previous successful distribution.
    fs.mkdirSync(options.output, { recursive: true });
    fs.cpSync(pkg, path.join(options.output, "pkg"), { recursive: true });
    for (const file of ["bloom_glue.js", "jolt_bridge.js"]) {
      fs.copyFileSync(path.join(webDir, file), path.join(options.output, file));
    }
    if (options.game) {
      const assets = path.join(path.dirname(options.game), "assets");
      if (fs.existsSync(assets)) {
        fs.cpSync(assets, path.join(options.output, "assets"), { recursive: true });
      }
    }
    fs.copyFileSync(index, path.join(options.output, "index.html"));
    console.log(`Build complete: ${options.output}`);
    console.log(`Engine WASM: ${fs.statSync(wasm).size} bytes`);
    console.log("Serve this directory over HTTP; see docs/web-target.md.");
  } finally {
    // Only remove the directory this invocation created under the temp root.
    if (path.dirname(path.resolve(temporary)) !== temporaryRoot) {
      throw new BuildError("temporary build directory escaped its root");
    }
    fs.rmSync(temporary, { recursive: true, force: true });
  }
}

function main(args = process.argv.slice(2)) {
  try {
    const options = parseArgs(args);
    if (options.help) console.log(HELP);
    else build(options);
    return 0;
  } catch (error) {
    console.error(`bloom-web: ${error.message}`);
    return error.exitCode || 1;
  }
}

module.exports = { BuildError, parseArgs, runTool, build, main };
if (require.main === module) process.exitCode = main();
