#!/usr/bin/env node
// Prevent local and PR command inventories from silently diverging.

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const read = (relative) =>
  fs.readFileSync(path.join(root, relative), "utf8").replace(/\r\n/g, "\n");

const expectedLanes = new Map([
  ["quick", ["contracts", "lint", "shared-tests", "wasm-check", "quality-contract", "example-inventory"]],
  ["full", ["contracts", "lint", "shared-tests", "wasm-check", "quality-contract", "example-inventory", "host-build", "wasm-build"]],
  ["web", ["wasm-check", "wasm-build", "browser-smoke"]],
  ["cross", ["target-check"]],
  ["hardware", ["example-compile", "quality-check", "quality-faults", "quality-run"]],
]);

const listing = spawnSync("bash", ["scripts/ci-check.sh", "--list"], {
  cwd: root,
  encoding: "utf8",
});
if (listing.status !== 0) {
  console.error(listing.stderr || "ci-check.sh --list failed");
  process.exit(1);
}

const actualLanes = new Map(
  listing.stdout.trim().split("\n").map((line) => {
    const [lane, components] = line.split("\t");
    return [lane, components.split(/\s+/)];
  }),
);

let failures = 0;
for (const [lane, expected] of expectedLanes) {
  const actual = actualLanes.get(lane);
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    console.error(`FAIL  ${lane} lane: expected ${expected.join(" ")}, got ${(actual || []).join(" ")}`);
    failures += 1;
  }
}

const testWorkflow = read(".github/workflows/test.yml");
const qualityWorkflow = read(".github/workflows/quality.yml");
const releaseWorkflow = read(".github/workflows/release.yml");
const failureAction = read(".github/actions/upload-ci-failure/action.yml");
const workflowCommands = [
  "./scripts/ci-check.sh --quick --component shared-tests",
  "./scripts/ci-check.sh --quick --component contracts",
  "./scripts/ci-check.sh --quick --component lint",
  "./scripts/ci-check.sh --full --component host-build",
  "./scripts/ci-check.sh --web --component wasm-check",
  "./scripts/ci-check.sh --web --component wasm-build",
  "./scripts/ci-check.sh --web --component browser-smoke",
  "./scripts/ci-check.sh --cross --component target-check",
  "./scripts/ci-check.sh --quick --component quality-contract",
  "./scripts/ci-check.sh --hardware --component example-compile",
  "./scripts/ci-check.sh --hardware --component quality-check",
  "./scripts/ci-check.sh --hardware --component quality-faults",
  "./scripts/ci-check.sh --hardware --component quality-run",
];

for (const command of workflowCommands) {
  const workflow = command.includes("quality-") || command.includes("example-compile")
    ? qualityWorkflow
    : testWorkflow;
  if (!workflow.includes(command)) {
    console.error(`FAIL  workflow does not delegate through: ${command}`);
    failures += 1;
  }
}

for (const [name, workflow] of [
  ["test.yml", testWorkflow],
  ["quality.yml", qualityWorkflow],
]) {
  if (/continue-on-error\s*:\s*true/.test(workflow)) {
    console.error(`FAIL  ${name} contains an advisory required step`);
    failures += 1;
  }
}

for (const evidencePath of [
  "target/ci",
  "native/shared/tests/golden/**/*.actual.png",
  "tools/quality/out",
]) {
  if (!failureAction.includes(evidencePath)) {
    console.error(`FAIL  failure evidence action omits: ${evidencePath}`);
    failures += 1;
  }
}
if ((testWorkflow.match(/uses: \.\/\.github\/actions\/upload-ci-failure/g) || []).length !== 9) {
  console.error("FAIL  every Tests job must upload failure evidence");
  failures += 1;
}
if (!qualityWorkflow.includes("uses: ./.github/actions/upload-ci-failure")) {
  console.error("FAIL  quality contract job must upload failure evidence");
  failures += 1;
}
for (const requiredReleaseText of [
  "resolve-release:",
  "SHA: ${{ needs.resolve-release.outputs.sha }}",
  "ref: ${{ needs.resolve-release.outputs.sha }}",
  'if [ "$type" != "commit" ]',
]) {
  if (!releaseWorkflow.includes(requiredReleaseText)) {
    console.error(`FAIL  release workflow omits exact-ref contract: ${requiredReleaseText}`);
    failures += 1;
  }
}
if (releaseWorkflow.includes("bypassing test gate")) {
  console.error("FAIL  manual releases may not bypass the exact-SHA test gate");
  failures += 1;
}

const duplicatedTestCommands = [
  "node tools/validate-ffi.js",
  "node tools/check-file-lines.js",
  "cargo fmt --check",
  "cargo clippy --release",
  "cargo test --release",
  "cargo check --target wasm32-unknown-unknown",
  "wasm-pack build --release --target web",
];
for (const command of duplicatedTestCommands) {
  if (testWorkflow.includes(command)) {
    console.error(`FAIL  test.yml duplicates ci-check.sh command: ${command}`);
    failures += 1;
  }
}

console.log(`${failures} failures`);
process.exit(failures === 0 ? 0 : 1);
