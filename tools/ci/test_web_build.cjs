"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { test } = require("node:test");
const { build, parseArgs, runTool } = require("../../native/web/build.cjs");
const { splice } = require("../../native/web/splice_game.cjs");

const perryHtml = `<div id="perry-root"></div><script>
function bootPerryWasm(wasmBase64) { return Promise.resolve(); }
window.__perryWasmB64 = "AA==";
bootPerryWasm("AA==").catch(console.error);
</script>`;

function fixture(t, behavior = {}) {
  const temporaryRoot = path.resolve(os.tmpdir());
  const root = fs.mkdtempSync(path.join(temporaryRoot, "bloom-web-test-"));
  t.after(() => {
    assert.equal(path.dirname(path.resolve(root)), temporaryRoot);
    fs.rmSync(root, { recursive: true, force: true });
  });
  const gameDir = path.join(root, "game with spaces & Unicode-é");
  const webDir = path.join(root, "engine");
  const output = path.join(root, "output with spaces");
  fs.mkdirSync(path.join(gameDir, "assets"), { recursive: true });
  fs.mkdirSync(webDir);
  fs.writeFileSync(path.join(gameDir, "main.ts"), "// fixture game");
  fs.writeFileSync(path.join(gameDir, "assets", "hello.txt"), "fixture asset");
  for (const name of ["bloom_glue.js", "game_loop.mjs", "jolt_bridge.js", "index.html"]) {
    fs.writeFileSync(path.join(webDir, name), `fixture ${name}`);
  }
  const fakeTool = path.join(root, "fixture-tool.cjs");
  fs.writeFileSync(fakeTool, `
    const fs = require("node:fs");
    const path = require("node:path");
    const [tool, ...args] = process.argv.slice(2);
    const behavior = ${JSON.stringify(behavior)};
    if (args[0] === "--version") process.exit(0);
    if (tool === behavior.failTool) process.exit(23);
    if (tool === "perry") {
      fs.writeFileSync(args[args.indexOf("-o") + 1], ${JSON.stringify(perryHtml)});
    } else if (tool === "wasm-pack") {
      const pkg = args[args.indexOf("--out-dir") + 1];
      fs.mkdirSync(pkg);
      if (!behavior.omitWasm) fs.writeFileSync(path.join(pkg, "bloom_web_bg.wasm"), "fixture wasm");
      fs.writeFileSync(path.join(pkg, "bloom_web.js"), "fixture bindings");
    } else if (tool === "wasm-opt") {
      fs.copyFileSync(args[1], args[args.indexOf("-o") + 1]);
    }
  `);
  const calls = [];
  const execute = (tool, args, options) => {
    calls.push({ tool, args, options });
    if (tool === "wasm-opt" && behavior.missingOptimizer) {
      return { error: Object.assign(new Error("not installed"), { code: "ENOENT" }) };
    }
    // These are subprocess fixtures, not engine/compiler acceptance evidence.
    // Real process exits exercise the failure boundary on Windows and Unix.
    return spawnSync(process.execPath, [fakeTool, tool, ...args], options);
  };
  return {
    root, gameDir, webDir, output, calls,
    options: { game: path.join(gameDir, "main.ts"), output },
    dependencies: { execute, webDir, env: {} },
  };
}

test("CLI help works without tools; malformed arguments fail before a build", () => {
  const cli = path.resolve(__dirname, "../../native/web/build.cjs");
  const help = spawnSync(process.execPath, [cli, "--help"], {
    encoding: "utf8", env: { ...process.env, PATH: "" }, windowsHide: true,
  });
  assert.equal(help.status, 0, help.stderr);
  assert.match(help.stdout, /--output/);
  for (const args of [["--output"], ["--output="], ["--unknown"], ["a.ts", "b.ts"]]) {
    assert.throws(() => parseArgs(args), (error) => error.exitCode === 2);
  }
});

test("compiler exit code is retained", () => {
  assert.throws(
    () => runTool(process.execPath, ["-e", "process.exit(23)"]),
    (error) => error.exitCode === 23 && /exit 23/.test(error.message),
  );
});

test("failed wasm-pack cannot reuse old output or assemble a false success", async (t) => {
  const f = fixture(t, { failTool: "wasm-pack" });
  fs.mkdirSync(path.join(f.output, "pkg"), { recursive: true });
  const previous = path.join(f.output, "pkg", "bloom_web_bg.wasm");
  fs.writeFileSync(previous, "previous distribution");
  await assert.rejects(() => build(f.options, f.dependencies), (error) => error.exitCode === 23);
  assert.equal(fs.readFileSync(previous, "utf8"), "previous distribution");
  assert.equal(fs.existsSync(path.join(f.output, "index.html")), false);
  assert.equal(f.calls.some((call) => call.tool === "wasm-opt"), false);
});

test("a zero exit with missing artifacts still fails", async (t) => {
  const f = fixture(t, { omitWasm: true });
  await assert.rejects(() => build(f.options, f.dependencies), /did not produce a non-empty file/);
  assert.equal(fs.existsSync(f.output), false);
});

test("fresh assembly supports spaces, assets and repeated builds without pkg nesting", async (t) => {
  const f = fixture(t, { missingOptimizer: true });
  await build(f.options, f.dependencies);
  await build(f.options, f.dependencies);
  const call = f.calls.find((item) => item.tool === "perry" && item.args[0] === "compile");
  assert.equal(call.args[1], f.options.game);
  assert.equal(call.options.cwd, f.gameDir);
  assert.equal(call.options.shell, false);
  assert.equal(fs.readFileSync(path.join(f.output, "assets", "hello.txt"), "utf8"), "fixture asset");
  assert.equal(fs.readFileSync(path.join(f.output, "pkg", "bloom_web_bg.wasm"), "utf8"), "fixture wasm");
  assert.equal(fs.existsSync(path.join(f.output, "pkg", "pkg")), false);
  assert.match(fs.readFileSync(path.join(f.output, "index.html"), "utf8"), /__bloomReady.then/);
});

test("an installed optimizer failure stops assembly", async (t) => {
  const f = fixture(t, { failTool: "wasm-opt" });
  await assert.rejects(() => build(f.options, f.dependencies), (error) => error.exitCode === 23);
  assert.equal(fs.existsSync(f.output), false);
});

test("explicitly configured optimizer is required", async (t) => {
  const f = fixture(t, { missingOptimizer: true });
  f.dependencies.env.BLOOM_WASM_OPT = "wasm-opt";
  await assert.rejects(() => build(f.options, f.dependencies), /executable not found/);
  assert.equal(fs.existsSync(f.output), false);
});

test("engine-only builds do not require Perry", async (t) => {
  const f = fixture(t);
  await build({ output: f.output }, f.dependencies);
  assert.equal(f.calls.some((call) => call.tool === "perry"), false);
  assert.equal(fs.readFileSync(path.join(f.output, "index.html"), "utf8"), "fixture index.html");
});

test("unsupported Perry output aborts instead of publishing an ungated game", () => {
  assert.throws(() => splice("<html>different format</html>"), /perry-root/);
  assert.throws(() => splice('<div id="perry-root"></div>'), /boot block/);
  assert.throws(() => splice('<div id="perry-root"></div>window.__perryWasmB64'), /boot block shape/);
  const html = splice(perryHtml);
  assert.match(html, /window.__bloomReady.then\(\(\) => bootPerryWasm\("AA=="\)\).catch/);
  assert.match(html, /function bootPerryWasm\(wasmBase64\)/);
});
