"use strict";

const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const crypto = require("node:crypto");
const { spawnSync } = require("node:child_process");

const PERRY_VERSION = "0.5.1220";

function execute(program, args, options = {}) {
  const result = spawnSync(program, args, {
    stdio: "inherit", shell: false, windowsHide: true, ...options,
  });
  if (result.error) {
    const detail = result.error.code === "ENOENT" ? "not found; install the documented prerequisite or set its BLOOM_* executable path" : result.error.message;
    throw new Error(`${program}: ${detail}`);
  }
  if (result.status !== 0) throw new Error(`${program} failed (${result.signal || `exit ${result.status}`})`);
  return result.stdout;
}

function runNpm(args, cwd, options = {}) {
  let npm = process.env.BLOOM_NPM;
  let cli = process.env.npm_execpath;
  if (!cli || !fs.existsSync(cli)) {
    cli = path.join(path.dirname(process.execPath), "node_modules/npm/bin/npm-cli.js");
  }
  if (!npm && fs.existsSync(cli)) return execute(process.execPath, [cli, ...args], { cwd, ...options });
  if (!npm && process.platform === "win32") {
    for (const directory of (process.env.PATH || "").split(path.delimiter)) {
      if (fs.existsSync(path.join(directory, "npm.exe"))) { npm = path.join(directory, "npm.exe"); break; }
      const candidate = path.join(directory, "node_modules/npm/bin/npm-cli.js");
      if (fs.existsSync(candidate)) return execute(process.execPath, [candidate, ...args], { cwd, ...options });
    }
    if (!npm) throw new Error("npm CLI not found. Install Node.js with npm, or set BLOOM_NPM to npm.exe.");
  }
  return execute(npm || "npm", args, { cwd, ...options });
}

function installDependencies(cwd) { return runNpm(["install", "--no-audit", "--no-fund"], cwd); }

function hash(file) { return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex"); }

function prepareToolchain({ native, engineRoot, env = process.env }) {
  const buildEnv = { ...env };
  let compiler = env.BLOOM_PERRY || "perry";
  if (native && process.platform === "win32" && !(env.BLOOM_PERRY && env.PERRY_RUNTIME_DIR && env.PERRY_WORKSPACE_ROOT)) {
    const cache = path.resolve(env.BLOOM_TOOLCHAIN_DIR || path.join(env.LOCALAPPDATA || os.homedir(), "Bloom", "perry-" + PERRY_VERSION));
    const metadata = path.join(cache, "toolchain.json");
    if (!fs.existsSync(metadata)) {
      console.log("Preparing the matching Windows compiler and runtime (first build only)...");
      execute(env.BLOOM_PYTHON || "python", [path.join(engineRoot, "tools/ci/setup_windows_perry.py"), "--out", cache], { env: buildEnv });
    }
    const receipt = JSON.parse(fs.readFileSync(metadata, "utf8"));
    if (receipt.schema !== "bloom-windows-perry-toolchain-v1" || receipt.version !== PERRY_VERSION ||
        receipt.source_sha !== "06137858dc8c6f80975238377138f2f948d6ef88" ||
        receipt.runtime_rustflags !== "-C panic=unwind" ||
        hash(receipt.compiler) !== receipt.compiler_sha256) {
      throw new Error(`Windows toolchain mismatch at ${metadata}; use a fresh BLOOM_TOOLCHAIN_DIR.`);
    }
    for (const name of ["perry_runtime.lib", "perry_stdlib.lib"]) {
      if (hash(path.join(receipt.environment.PERRY_RUNTIME_DIR, name)) !== receipt.runtime_libraries_sha256[name]) {
        throw new Error(`Windows runtime mismatch: ${name}; use a fresh BLOOM_TOOLCHAIN_DIR.`);
      }
    }
    compiler = receipt.compiler;
    Object.assign(buildEnv, receipt.environment);
  }
  const version = execute(compiler, ["--version"], { env: buildEnv, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] }).trim();
  if (version !== `perry ${PERRY_VERSION}`) throw new Error(`Compiler mismatch: expected Perry ${PERRY_VERSION}, found ${version}. Set BLOOM_PERRY to the compatible executable.`);
  if (native && buildEnv.CARGO_TARGET_DIR) throw new Error("Perry native builds currently require crate-local Cargo output. Unset CARGO_TARGET_DIR for this command.");
  execute("cargo", ["--version"], { env: buildEnv });
  return { compiler, env: buildEnv };
}

module.exports = { PERRY_VERSION, execute, runNpm, installDependencies, prepareToolchain };
