import unittest

from tools.ci.compiled_web_smoke import validate_state


class CompiledGameAcceptanceTests(unittest.TestCase):
    def test_success_requires_render_progress_and_exactly_one_cleanup(self):
        state = {"frames": "8", "cleanups": "1", "expectedFault": None, "errors": []}
        validate_state("game", state)
        for change in ({"frames": None}, {"cleanups": None}, {"cleanups": "2"}, {"errors": ["WASM Error: unreachable"]}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_state("game", {**state, **change})

    def test_failure_control_requires_entering_its_fault_and_an_actual_error(self):
        state = {"frames": None, "cleanups": None, "expectedFault": "BLOOM_EXPECTED_STARTUP_FAILURE", "errors": ["WASM Error: unreachable"]}
        validate_state("trap-control", state)
        for change in ({"expectedFault": None}, {"errors": []}, {"frames": "8"}, {"cleanups": "1"}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_state("trap-control", {**state, **change})


if __name__ == "__main__":
    unittest.main()
