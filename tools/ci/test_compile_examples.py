"""The example gate must reject false compiler success and retain partial failures."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tools.ci import compile_examples


class NativeExampleGateTests(unittest.TestCase):
    def test_inventory_rejects_palette_names_that_link_but_fail_at_runtime(self):
        palette = (compile_examples.REPO_ROOT / "src/core/colors.ts").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="bloom-palette-gate-") as directory:
            root = Path(directory).resolve()
            (root / "src/core").mkdir(parents=True)
            (root / "src/core/colors.ts").write_text(palette, encoding="utf-8")
            example = root / "examples/pong"
            example.mkdir(parents=True)
            (example / "package.json").write_text("{}")
            manifest = root / "examples.json"
            manifest.write_text(json.dumps({"schema": "bloom-canonical-examples-v1", "examples": ["examples/pong"]}))
            with mock.patch.object(compile_examples, "REPO_ROOT", root), \
                 mock.patch.object(compile_examples, "MANIFEST_PATH", manifest):
                (example / "main.ts").write_text("clearBackground(Colors.Black); drawRect(Colors.White);")
                _, failures = compile_examples.load_inventory()
                self.assertEqual(len(failures), 1)
                self.assertIn("['Black', 'White']", failures[0])
                (example / "main.ts").write_text("clearBackground(Colors.BLACK); drawRect(Colors.WHITE);")
                self.assertEqual(compile_examples.load_inventory()[1], [])

    def exercise(self, outcomes):
        with tempfile.TemporaryDirectory(prefix="bloom-example-gate-") as directory:
            root = Path(directory).resolve()
            self.assertEqual(root.parent, Path(tempfile.gettempdir()).resolve())
            examples = [f"examples/case-{i}" for i in range(len(outcomes))]
            out = root / "out"
            (out / "bin").mkdir(parents=True)
            # Valid-looking executable left by a previous successful invocation.
            stale = out / "bin" / ("case-0.exe" if os.name == "nt" else "case-0")
            stale.write_bytes(b"MZ" + b"previous build" * 8)
            commands = []

            def compiler(command, **kwargs):
                if command[1:] == ["--version"]:
                    return subprocess.CompletedProcess(command, 0, stdout="perry test")
                commands.append(command)
                outcome = outcomes[len(commands) - 1]
                if outcome == "timeout":
                    raise subprocess.TimeoutExpired(command, 1)
                if outcome == "link-error":
                    kwargs["stderr"].write("undefined symbol: removed_palette\n")
                    return subprocess.CompletedProcess(command, 1)
                if outcome in ("duplicate-msvc", "duplicate-lld"):
                    prefix = "warning LNK4006:" if outcome == "duplicate-msvc" else "lld-link: warning: duplicate symbol:"
                    kwargs["stderr"].write(prefix + " perry_global_index_ts__0\n")
                    Path(command[command.index("-o") + 1]).write_bytes(b"MZ" + bytes(range(64)))
                if isinstance(outcome, bytes):
                    Path(command[command.index("-o") + 1]).write_bytes(outcome)
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(compile_examples, "REPO_ROOT", root), \
                 mock.patch.object(compile_examples, "load_inventory", return_value=(examples, [])), \
                 mock.patch.object(compile_examples, "ensure_engine_dependency"), \
                 mock.patch.object(compile_examples.shutil, "which", return_value="perry-test"), \
                 mock.patch.object(compile_examples.subprocess, "run", side_effect=compiler), \
                 mock.patch("sys.argv", ["compile_examples.py", "--out", str(out)]), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                result = compile_examples.main()
            report = json.loads((out / "result.json").read_text(encoding="utf-8"))
            return result, report, commands

    def test_zero_exit_without_new_executable_rejects_stale_success(self):
        result, report, commands = self.exercise([None, b"COFF object code" * 8])
        self.assertEqual(result, 1)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(len(report["failures"]), 2)
        self.assertTrue(all("sha256" not in row for row in report["examples"]))
        self.assertTrue(all("--no-link" not in command for command in commands))

    def test_link_failure_and_timeout_do_not_hide_later_results(self):
        executable = b"MZ" + bytes(range(64))
        result, report, _ = self.exercise(["link-error", "timeout", executable])
        self.assertEqual(result, 1)
        self.assertEqual([row["status"] for row in report["examples"]], ["fail", "fail", "pass"])
        self.assertEqual(report["examples"][2]["sha256"], hashlib.sha256(executable).hexdigest())

    def test_success_records_native_link_mode_and_artifact_hash(self):
        executable = b"\x7fELF" + bytes(range(64))
        result, report, _ = self.exercise([executable])
        self.assertEqual(result, 0)
        self.assertEqual(report["status"], "pass")
        record = report["examples"][0]
        self.assertEqual(record["mode"], "native-compile-link")
        self.assertEqual(record["bytes"], len(executable))
        self.assertEqual(record["sha256"], hashlib.sha256(executable).hexdigest())

    def test_zero_exit_with_merged_module_globals_is_not_success(self):
        result, report, _ = self.exercise(["duplicate-msvc", "duplicate-lld", b"MZ" + bytes(range(64))])
        self.assertEqual(result, 1)
        self.assertEqual([row["status"] for row in report["examples"]], ["fail", "fail", "pass"])
        self.assertTrue(all("duplicate Perry module globals" in row["error"] for row in report["examples"][:2]))
        compile_examples.reject_duplicate_module_globals("warning LNK4006: rustc_demangle already defined")


if __name__ == "__main__":
    unittest.main()
