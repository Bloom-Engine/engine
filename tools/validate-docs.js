#!/usr/bin/env node

// Fast documentation/package contract checks. Keep this dependency-free so it
// can run in the required `contracts` CI component.

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const read = (relative) => fs.readFileSync(path.join(root, relative), "utf8");
let failures = 0;

function fail(message) {
  console.error(`FAIL  ${message}`);
  failures += 1;
}

function walkMarkdown(relative = "") {
  const absolute = path.join(root, relative);
  const entries = fs.readdirSync(absolute, { withFileTypes: true });
  const result = [];
  for (const entry of entries) {
    const child = path.join(relative, entry.name);
    if (entry.isDirectory()) {
      if ([".git", "node_modules", "target"].includes(entry.name)) continue;
      if (child === "native/third_party" || child === "native/tvos/metal-patched") continue;
      result.push(...walkMarkdown(child));
    } else if (entry.name.endsWith(".md")) {
      result.push(child);
    }
  }
  return result;
}

const markdownFiles = walkMarkdown();
for (const relative of markdownFiles) {
  const source = read(relative);
  const links = source.matchAll(/!?\[[^\]]*\]\(([^)]+)\)/g);
  for (const match of links) {
    let target = match[1].trim().replace(/^<|>$/g, "");
    target = target.replace(/\s+["'][^"']*["']$/, "");
    if (!target || target.startsWith("#") || target.startsWith("/")) continue;
    if (/^(?:https?:|mailto:|tel:)/.test(target)) continue;
    target = target.split("#", 1)[0].split("?", 1)[0];
    try {
      target = decodeURIComponent(target);
    } catch {
      fail(`${relative}: malformed link target ${match[1]}`);
      continue;
    }
    const resolved = path.resolve(path.dirname(path.join(root, relative)), target);
    if (!resolved.startsWith(`${root}${path.sep}`) && resolved !== root) {
      fail(`${relative}: repository link escapes the checkout: ${match[1]}`);
    } else if (!fs.existsSync(resolved)) {
      fail(`${relative}: missing link target ${match[1]}`);
    }
  }
}

const currentDocs = markdownFiles.filter((relative) =>
  relative === "README.md" ||
  (relative.startsWith("docs/") &&
    !relative.startsWith("docs/evidence/") &&
    !relative.startsWith("docs/perf/") &&
    !relative.startsWith("docs/pt/") &&
    !relative.startsWith("docs/rfc/") &&
    relative !== "docs/tickets.md")
);
const currentText = currentDocs.map((relative) => `${relative}\n${read(relative)}`).join("\n");
for (const [label, pattern] of [
  ["removed Colors.RAYWHITE constant", /Colors\.RAYWHITE/],
  ["removed Platform.WATCH constant", /Platform\.WATCH\b/],
  ["numeric Camera3D projection", /projection\s*:\s*(?:number|[01](?:\.0)?\b)/],
  ["legacy @bloom package import", /from\s+['"]@bloom\//],
  ["undocumented local bloom import alias", /from\s+['"]bloom(?:\/|['"])/],
  ["retired wasm-pack installer URL", /rustwasm\.github\.io\/wasm-pack\/installer/],
]) {
  if (pattern.test(currentText)) fail(`current docs contain ${label}`);
}

const colorsSource = read("src/core/colors.ts");
const colorKeys = new Set(
  [...colorsSource.matchAll(/^\s{2}([A-Z][A-Z0-9_]+):\s+Color\./gm)].map((match) => match[1]),
);
for (const match of currentText.matchAll(/Colors\.([A-Z][A-Z0-9_]+)/g)) {
  if (!colorKeys.has(match[1])) fail(`current docs reference unknown Colors.${match[1]}`);
}

const qualityDocs = read("docs/quality-presets.md");
const qualitySource = read("native/shared/src/renderer/quality_preset.rs");
const ultraDocs = qualityDocs.match(/^\| Ultra \|[^\n]*\| ([0-9.]+) \| Full effect stack \|$/m)?.[1];
const ultraSource = qualitySource.match(/render_scale:\s*1\.0,[\s\S]*?composite_sharpen:\s*([0-9.]+)/)?.[1];
if (!ultraDocs || ultraDocs !== ultraSource) {
  fail(`Ultra sharpen docs (${ultraDocs || "missing"}) do not match source (${ultraSource || "missing"})`);
}

const packageJson = JSON.parse(read("package.json"));
if (packageJson.bin?.["bloom-web"] !== "native/web/build.sh") {
  fail("package.json does not expose the bloom-web command");
}

const pack = spawnSync("npm", ["pack", "--dry-run", "--json"], {
  cwd: root,
  encoding: "utf8",
});
if (pack.status !== 0) {
  fail(`npm pack --dry-run failed: ${pack.stderr.trim()}`);
} else {
  let packed = [];
  try {
    packed = JSON.parse(pack.stdout)[0].files.map((entry) => entry.path);
  } catch (error) {
    fail(`could not parse npm pack inventory: ${error.message}`);
  }
  for (const required of [
    "native/web/build.sh",
    "native/web/splice_game.py",
    "crates/bloom-geometry-format/Cargo.toml",
    "crates/bloom-scene-format/Cargo.toml",
    "native/third_party/bloom_jolt/CMakeLists.txt",
    "native/third_party/JoltPhysics/Build/CMakeLists.txt",
    "native/third_party/JoltPhysics/Jolt/Jolt.h",
  ]) {
    if (!packed.includes(required)) fail(`npm package omits ${required}`);
  }
}

const help = spawnSync("bash", ["native/web/build.sh", "--help"], {
  cwd: root,
  encoding: "utf8",
});
if (help.status !== 0 || !help.stdout.includes("--output")) {
  fail("bloom-web help/argument parsing is not usable");
}

console.log(`${markdownFiles.length} Markdown files checked; ${failures} failures`);
process.exit(failures === 0 ? 0 : 1);
