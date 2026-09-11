#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { PERRY_VERSION, execute, runNpm, installDependencies, prepareToolchain } = require("./toolchain.cjs");
const { serve } = require("./serve.cjs");
const engineRoot = path.resolve(__dirname, "../..");
const enginePackage = require(path.join(engineRoot, "package.json"));

async function assetManifest(directory, prefix = "assets") {
  const files = [];
  for (const entry of await fs.promises.readdir(directory, { withFileTypes: true })) {
    const name = `${prefix}/${entry.name}`;
    if (entry.isDirectory()) files.push(...await assetManifest(path.join(directory, entry.name), name));
    else if (entry.isFile()) files.push(name);
    else throw new Error(`Starter assets must be regular files or directories: ${name}`);
  }
  return files.sort();
}
const HELP = `Usage:
  bloom new <directory> [--no-install] [--engine-package <local.tgz>]
  bloom run [--web] [--port 8080]
  bloom build [--release] [--target native|web|windows|linux|macos]

Run build/run from the generated project directory. Native builds target the host.
Web run builds the same source and serves it at http://127.0.0.1:8080.
Requires Node.js 18+, Rust and Perry ${PERRY_VERSION}; web also needs wasm-pack.
Windows native builds prepare a matching compiler/runtime automatically unless
BLOOM_PERRY, PERRY_RUNTIME_DIR and PERRY_WORKSPACE_ROOT are explicitly configured.
Windows prerequisites: Python 3.12+, Git, Rust and Visual Studio C++ build tools.
Set BLOOM_PERRY, BLOOM_PYTHON or BLOOM_TOOLCHAIN_DIR to customize tool locations.
`;

function parseArgs(args) {
  if (!args.length || args.includes("--help") || args.includes("-h")) return { help: true };
  const [command, ...rest] = args;
  if (!["new", "run", "build"].includes(command)) throw new Error(`Unknown command: ${command}`);
  const options = { command, install: true, target: "native", port: 8080, release: false };
  for (let i = 0; i < rest.length; i += 1) {
    const arg = rest[i];
    if (arg === "--no-install" && command === "new") options.install = false;
    else if (arg === "--web" && command === "run") options.target = "web";
    else if (arg === "--release" && command === "build") options.release = true;
    else if ((arg === "--engine-package" && command === "new") || (arg === "--target" && command === "build") || (arg === "--port" && command === "run")) {
      if (!rest[i + 1] || rest[i + 1].startsWith("--")) throw new Error(`${arg} requires a value`);
      options[{ "--engine-package": "engineArchive", "--target": "target", "--port": "port" }[arg]] = rest[++i];
    } else if (command === "new" && !arg.startsWith("-") && !options.directory) options.directory = arg;
    else throw new Error(`Unexpected argument for ${command}: ${arg}`);
  }
  if (command === "new" && !options.directory) throw new Error("bloom new requires a directory");
  options.port = Number(options.port);
  if (!Number.isInteger(options.port) || options.port < 1 || options.port > 65535) throw new Error("--port must be an integer from 1 to 65535");
  if (!["native", "web", "windows", "linux", "macos"].includes(options.target)) throw new Error(`Unsupported starter target: ${options.target}. Use native or web; mobile packaging needs the target-specific SDK flow.`);
  return options;
}

async function createProject(options) {
  const directory = path.resolve(options.directory);
  if (fs.existsSync(directory)) throw new Error(`Directory already exists: ${directory}; choose a new project directory.`);
  const archive = options.engineArchive && path.resolve(options.engineArchive);
  if (archive && (!archive.endsWith(".tgz") || !fs.statSync(archive).isFile())) throw new Error("--engine-package must name an existing npm .tgz archive");
  await fs.promises.mkdir(path.join(directory, "assets"), { recursive: true });
  await fs.promises.mkdir(path.join(directory, ".bloom"));
  if (archive) {
    await fs.promises.copyFile(archive, path.join(directory, ".bloom/engine.tgz"));
  } else {
    // Pin the exact package running this command. A preview tarball can share
    // a version number with an older registry release that lacks these APIs.
    const packed = JSON.parse(runNpm(["pack", "--json", "--ignore-scripts", "--pack-destination", path.join(directory, ".bloom")], engineRoot,
      { encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] }));
    const filename = packed[0]?.filename;
    if (!filename || path.basename(filename) !== filename) throw new Error("npm pack did not report a package filename");
    await fs.promises.rename(path.join(directory, ".bloom", filename), path.join(directory, ".bloom/engine.tgz"));
  }
  const name = path.basename(directory).toLowerCase().replace(/[^a-z0-9-]/g, "-").replace(/^-+|-+$/g, "") || "bloom-game";
  const project = {
    name, version: "0.1.0", private: true, main: "main.ts",
    scripts: { start: "bloom run", web: "bloom run --web", build: "bloom build --release" },
    dependencies: { "@bloomengine/engine": "file:.bloom/engine.tgz" },
    perry: { allow: { nativeLibrary: ["@bloomengine/engine", "@bloomengine/engine/*"] } },
  };
  const config = { schema: 1, entry: "main.ts", bloomVersion: enginePackage.version, perryVersion: PERRY_VERSION };
  for (const [name, data] of [["package.json", project], ["bloom.json", config]]) {
    await fs.promises.writeFile(path.join(directory, name), JSON.stringify(data, null, 2) + "\n");
  }
  await fs.promises.copyFile(path.join(__dirname, "templates/main.ts"), path.join(directory, "main.ts"));
  await fs.promises.copyFile(path.join(__dirname, "templates/welcome.txt"), path.join(directory, "assets/welcome.txt"));
  await fs.promises.writeFile(path.join(directory, ".gitignore"), "node_modules/\ndist/\n");
  await fs.promises.copyFile(path.join(__dirname, "templates/README.md"), path.join(directory, "README.md"));
  if (options.install) installDependencies(directory);
  console.log(`Created ${directory}\nNext: cd into the directory${options.install ? "" : ", run npm install,"} and run npm start or npm run web.`);
  return directory;
}

function readProject(cwd) {
  let config;
  try { config = JSON.parse(fs.readFileSync(path.join(cwd, "bloom.json"), "utf8")); }
  catch (error) { throw new Error(`Cannot read bloom.json. Run from a generated project directory. ${error.message}`); }
  if (config.schema !== 1 || config.bloomVersion !== enginePackage.version || config.perryVersion !== PERRY_VERSION) throw new Error("Project/package mismatch: bloom.json must match the installed Bloom and compatible Perry versions. Reinstall the project's pinned dependency.");
  if (typeof config.entry !== "string" || !config.entry.endsWith(".ts")) throw new Error("bloom.json entry must be a TypeScript file");
  const entry = fs.realpathSync(path.resolve(cwd, config.entry));
  const relative = path.relative(fs.realpathSync(cwd), entry);
  if (relative === ".." || relative.startsWith(".." + path.sep) || path.isAbsolute(relative)) throw new Error("Project entry must stay inside the project directory");
  const manifest = JSON.parse(fs.readFileSync(path.join(cwd, "package.json"), "utf8"));
  const allow = manifest.perry?.allow?.nativeLibrary;
  if (!Array.isArray(allow) || !["@bloomengine/engine", "@bloomengine/engine/*"].every(value => allow.includes(value))) throw new Error("Project manifest is missing Bloom native-library permissions; restore the generated package.json perry.allow.nativeLibrary entries.");
  if (!fs.existsSync(path.join(cwd, "assets"))) throw new Error("Project asset directory is missing: assets");
  return { entry };
}

async function buildProject(options, cwd = process.cwd()) {
  const { entry } = readProject(cwd);
  const host = { win32: "windows", linux: "linux", darwin: "macos" }[process.platform];
  if (options.target !== "web" && options.target !== "native" && options.target !== host) throw new Error(`Target ${options.target} requires its native host; this host is ${host || process.platform}. Use --target web for a portable build.`);
  if (options.target !== "web" && !host) throw new Error(`Native starter builds are unsupported on ${process.platform}`);
  const toolchain = prepareToolchain({ native: options.target !== "web", engineRoot });
  if (options.target === "web") {
    const output = path.join(cwd, "dist/web");
    const assets = await assetManifest(path.join(cwd, "assets"));
    execute(process.execPath, [path.join(engineRoot, "native/web/build.cjs"), entry, "--output", output], { cwd, env: { ...toolchain.env, BLOOM_PERRY: toolchain.compiler } });
    // The glue prefetches this list before entering synchronous game code.
    await fs.promises.writeFile(path.join(output, "assets_manifest.json"), JSON.stringify({ files: assets }, null, 2) + "\n");
    if (options.command === "run") await serve(output, options.port);
    return output;
  }
  const output = path.join(cwd, "dist", host);
  await fs.promises.mkdir(output, { recursive: true });
  const temporary = await fs.promises.mkdtemp(path.join(output, ".build-"));
  const filename = host === "windows" ? "game.exe" : "game";
  try {
    const args = ["compile", entry, "-o", path.join(temporary, filename)];
    if (!options.release) args.push("--debug-symbols");
    execute(toolchain.compiler, args, { cwd, env: toolchain.env });
    const binary = path.join(temporary, filename);
    if (!fs.existsSync(binary) || fs.statSync(binary).size === 0) throw new Error("Compiler did not produce a fresh native executable");
    await fs.promises.cp(path.join(cwd, "assets"), path.join(temporary, "assets"), { recursive: true });
    await fs.promises.cp(temporary, output, { recursive: true });
  } finally {
    if (path.dirname(path.resolve(temporary)) !== path.resolve(output)) throw new Error("Build staging directory escaped the output directory");
    await fs.promises.rm(temporary, { recursive: true, force: true });
  }
  console.log(`Build complete: ${path.join(output, filename)}`);
  if (options.command === "run") execute(path.join(output, filename), [], { cwd: output, env: toolchain.env });
  return output;
}

async function main(args = process.argv.slice(2)) {
  try {
    const options = parseArgs(args);
    if (options.help) console.log(HELP);
    else if (options.command === "new") await createProject(options);
    else await buildProject(options);
    return 0;
  } catch (error) { console.error(`bloom: ${error.message}`); return 1; }
}

module.exports = { parseArgs, createProject, readProject, assetManifest, buildProject, main };
if (require.main === module) main().then(code => { process.exitCode = code; });
