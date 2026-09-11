import unittest

from tools.ci.compiled_web_smoke import validate_state
from tools.ci.compile_web_game import validate_compilation


class CompiledGameAcceptanceTests(unittest.TestCase):
    def test_unresolved_imports_and_empty_game_cannot_pass_compilation(self):
        imports = [{"module": "ffi", "name": name} for name in
                   ["bloom_init_window", "bloom_set_target_fps", "bloom_set_direct_2d_mode",
                    "bloom_run_game_with_cleanup", "bloom_write_file", "bloom_draw_rect"]]
        validate_compilation("Generating WebAssembly", imports)
        with self.assertRaisesRegex(RuntimeError, "could not resolve"):
            validate_compilation("Warning: Could not resolve import '@bloomengine/engine/core'", imports)
        with self.assertRaisesRegex(RuntimeError, "missing required"):
            validate_compilation("Generating WebAssembly", [])

    def test_success_requires_render_progress_and_exactly_one_cleanup(self):
        state = {"frames": "8", "cleanups": "1", "expectedFault": None, "errors": []}
        validate_state("game", state)
        for change in ({"frames": None}, {"cleanups": None}, {"cleanups": "2"}, {"errors": ["WASM Error: unreachable"]}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_state("game", {**state, **change})

    def test_failure_control_requires_entering_its_fault_and_an_actual_error(self):
        state = {"frames": None, "cleanups": None, "expectedFault": "BLOOM_EXPECTED_STARTUP_FAILURE", "errors": ["RuntimeError: BLOOM_EXPECTED_STARTUP_FAILURE"]}
        validate_state("trap-control", state)
        for change in ({"expectedFault": None}, {"errors": []}, {"errors": ["unrelated error"]}, {"frames": "8"}, {"cleanups": "1"}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_state("trap-control", {**state, **change})


if __name__ == "__main__":
    unittest.main()
