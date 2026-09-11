import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.ci.fixed_step_smoke import EXPECTED, validate_observations, validate_game_lifecycle


class FixedStepEvidenceTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node is required for the WASM reporting harness')
    def test_wasm_harness_waits_for_boot_and_rejects_late_errors(self):
        # Harness controls only. Actual native/WASM compiler acceptance uses
        # the real Perry-generated runtime and observations in fixed_step_smoke.
        emit = 'console.log(' + json.dumps('BLOOM_FIXED_STEP_RESULT:' + json.dumps(EXPECTED)) + ');'
        with tempfile.TemporaryDirectory(prefix='bloom-clock-controls-') as directory:
            page = Path(directory) / 'control.html'
            helper = Path(__file__).with_name('perry_wasm_console.cjs')
            for body, success in ((emit, True), (emit + emit, False), ('', False),
                                  (emit + 'console.error("late failure");', False),
                                  (emit + 'throw new Error("late boot failure");', False)):
                page.write_text('<script>function __bitsToJsValue(bits) {return bits;} async function bootPerryWasm() {' + body + '}</script>'
                                '<script>window.__perryWasmB64 = "AGFzbQEAAAA=";'
                                'bootPerryWasm().catch(e => console.error(e));</script>', encoding='utf-8')
                result = subprocess.run(['node', str(helper), str(page)], capture_output=True, text=True, timeout=15)
                with self.subTest(body=body):
                    self.assertEqual(result.returncode == 0, success, result.stderr)

    def test_actual_game_requires_every_hook_and_valid_interpolation(self):
        validate_game_lifecycle('1,8,8,13,13,0.3', expected_frames=8)
        validate_game_lifecycle('1,9,9,15,15,0')
        for value in (None, '', '1,8,8,13,13', '0,8,8,13,13,0', '1,7,8,13,13,0',
                      '1,8,8,0,0,0', '1,8,8,13,12,0', '1,8,8,13,13,NaN', '1,8,8,13,13,1',
                      '1,9,9,15,15,0'):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                validate_game_lifecycle(value, expected_frames=8)

    def test_rejects_missing_duplicate_wrong_ticks_and_nonfinite_results(self):
        line = "BLOOM_FIXED_STEP_RESULT:" + json.dumps(EXPECTED)
        validate_observations(line)
        for output in ("", line + "\n" + line):
            with self.assertRaises(RuntimeError):
                validate_observations(output)
        for change in ({"variedTicks": 49}, {"cappedSteps": 12}, {"stopEvents": "FUDC"},
                       {"events": "IUDFFUDCC"}, {"longAlpha": float("nan")}, {"invalidConfig": 1}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_observations("BLOOM_FIXED_STEP_RESULT:" + json.dumps({**EXPECTED, **change}))


if __name__ == "__main__":
    unittest.main()
