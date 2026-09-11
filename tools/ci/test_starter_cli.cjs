"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { spawnSync } = require("node:child_process");
const { test } = require("node:test");
const { parseArgs, createProject, readProject, assetManifest } = require("../cli/bloom.cjs");
const { prepareToolchain, execute } = require("../cli/toolchain.cjs");
const { serve } = require("../cli/serve.cjs");

function temporary(t) {
  const parent = path.resolve(os.tmpdir());
  const root = fs.mkdtempSync(path.join(parent, "bloom-starter-test-"));
  t.after(() => {
    assert.equal(path.dirname(path.resolve(root)), parent);
    fs.rmSync(root, { recursive: true, force: true });
  });
  return root;
}

test("installed-style help needs no compiler and invalid commands fail", () => {
  const help = spawnSync(process.execPath, [path.resolve(__dirname, "../cli/bloom.cjs"), "--help"], { env: { ...process.env, PATH: "" }, encoding: "utf8", windowsHide: true });
  assert.equal(help.status, 0, help.stderr);
  assert.match(help.stdout, /bloom new/);
  for (const args of [["new"], ["run", "--typo"], ["new", "game", "--engine-package"], ["build", "--target", "android"], ["run", "--web", "--port", "0"]]) assert.throws(() => parseArgs(args));
});

test("scaffold keeps pinned manifests, separate lifecycle and assets; refuses existing projects", async (t) => {
  const root = temporary(t);
  const directory = path.join(root, "My game & Unicode-\u00e9");
  const archive = path.join(root, "fixture.tgz");
  fs.writeFileSync(archive, "owned package fixture");
  await createProject({ directory, engineArchive: archive, install: false });
  const config = readProject(directory);
  assert.equal(config.entry, fs.realpathSync(path.join(directory, "main.ts")));
  const source = fs.readFileSync(config.entry, "utf8");
  assert.match(source, /runGame\(\(dt\) => \{ update\(dt\); draw\(\); \}, cleanup\)/);
  assert.match(source, /readFile\("assets\/welcome.txt"\)/);
  assert.deepEqual(await assetManifest(path.join(directory, "assets")), ["assets/welcome.txt"]);
  await assert.rejects(() => createProject({ directory, install: false }), /already exists/);
  assert.equal(fs.readFileSync(config.entry, "utf8"), source);
  const manifestPath = path.join(directory, "package.json");
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  manifest.perry.allow.nativeLibrary = [];
  fs.writeFileSync(manifestPath, JSON.stringify(manifest));
  assert.throws(() => readProject(directory), /permissions/);
});

test("local package pin is copied into the project and missing archives fail before creation", async (t) => {
  const root = temporary(t);
  const missing = path.join(root, "missing-game");
  await assert.rejects(() => createProject({ directory: missing, engineArchive: path.join(root, "missing.tgz"), install: false }));
  assert.equal(fs.existsSync(missing), false);
  const archive = path.join(root, "test.tgz");
  fs.writeFileSync(archive, "owned package fixture");
  const directory = path.join(root, "local-game");
  await createProject({ directory, engineArchive: archive, install: false });
  assert.equal(fs.readFileSync(path.join(directory, ".bloom/engine.tgz"), "utf8"), "owned package fixture");
  const manifest = JSON.parse(fs.readFileSync(path.join(directory, "package.json")));
  assert.equal(manifest.dependencies["@bloomengine/engine"], "file:.bloom/engine.tgz");
  const configPath = path.join(directory, "bloom.json");
  const config = JSON.parse(fs.readFileSync(configPath));
  fs.writeFileSync(configPath, JSON.stringify({ ...config, perryVersion: "0.0.0" }));
  assert.throws(() => readProject(directory), /mismatch/);
  fs.writeFileSync(path.join(root, "outside.ts"), "// outside fixture");
  fs.writeFileSync(configPath, JSON.stringify({ ...config, entry: "../outside.ts" }));
  assert.throws(() => readProject(directory), /inside/);
});

test("tool failures retain the cause and compiler mismatches fail before Cargo", () => {
  assert.throws(() => execute(process.execPath, ["-e", "process.exit(23)"]), /exit 23/);
  assert.throws(() => prepareToolchain({ native: false, engineRoot: process.cwd(), env: { ...process.env, BLOOM_PERRY: process.execPath } }), /Compiler mismatch/);
});

test("development server presents generated files with WASM MIME and rejects missing routes", async (t) => {
  const root = temporary(t);
  fs.writeFileSync(path.join(root, "index.html"), "owned test page");
  fs.writeFileSync(path.join(root, "engine.wasm"), "fixture wasm");
  const server = await serve(root, 0);
  t.after(() => new Promise(resolve => server.close(resolve)));
  const base = `http://127.0.0.1:${server.address().port}`;
  assert.equal(await (await fetch(base)).text(), "owned test page");
  const wasm = await fetch(base + "/engine.wasm");
  assert.equal(wasm.headers.get("content-type"), "application/wasm");
  assert.equal((await fetch(base + "/missing.txt")).status, 404);
  assert.equal((await fetch(base, { method: "POST" })).status, 405);
});
