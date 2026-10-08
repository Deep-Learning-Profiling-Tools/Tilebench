"""Tests for the MI300X paper-figure data package (artifacts/paper_figures/amd/MI300X/).

No GPU, profiler or benchmark run is needed. The package checks use only tracked files; the checks
that re-read the git-ignored profiling reports are skipped when /root/Tilebench/outputs is absent.
Runs under pytest or `python3 -m unittest tests.test_paper_figures_mi300x`.
"""
import gzip
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PF = REPO / "scripts" / "paper_figures"
sys.path.insert(0, str(PF))

from case_identity import canonical_params_json, case_id, normalize_dtype, parse_params_cell  # noqa: E402
from mi300x_common import OUT_DIR, write_csv  # noqa: E402

OUTPUTS = Path(os.environ.get("TILEBENCH_OUTPUTS_ROOT", "/root/Tilebench/outputs"))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_pf_{name}", PF / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CaseIdentityTest(unittest.TestCase):
    def test_simple_cell(self):
        self.assertEqual(parse_params_cell("cols=512, rows=512"), {"cols": 512, "rows": 512})

    def test_quoted_and_nested_commas(self):
        p = parse_params_cell("shape=(4, 8), name='a,b', d={'x': [1, 2]}, f=0.5")
        self.assertEqual(p, {"shape": [4, 8], "name": "a,b", "d": {"x": [1, 2]}, "f": 0.5})

    def test_key_order_does_not_change_identity(self):
        a = case_id("Vector-Add", "FP16", parse_params_cell("n=1, m=2"))
        b = case_id("vector_add", "fp16", parse_params_cell("m=2, n=1"))
        self.assertEqual(a, b)
        self.assertEqual(canonical_params_json({"n": 1, "m": 2}), '{"m":2,"n":1}')

    def test_fp8_variants_stay_distinct(self):
        self.assertNotEqual(normalize_dtype("fp8_e4m3fn"), normalize_dtype("fp8_e4m3fnuz"))
        self.assertNotEqual(case_id("matmul", "fp8_e4m3fn", {"K": 1}), case_id("matmul", "fp8_e4m3fnuz", {"K": 1}))

    def test_unbalanced_cell_rejected(self):
        with self.assertRaises(ValueError):
            parse_params_cell("shape=(4, 8")


class DeterministicWriteTest(unittest.TestCase):
    def test_gzip_csv_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.csv.gz", Path(d) / "b.csv.gz"
            rows = [{"x": "1", "y": "a,b"}, {"x": "0.1234", "y": ""}]
            write_csv(a, ["x", "y"], rows, gz=True)
            write_csv(b, ["x", "y"], rows, gz=True)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(gzip.decompress(a.read_bytes()).decode(), 'x,y\n1,"a,b"\n0.1234,\n')


@unittest.skipUnless((OUT_DIR / "benchmark_cases.csv").exists(), "MI300X package not generated")
class PackageValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v = _load("validate_mi300x")
        with tempfile.TemporaryDirectory() as d:
            # validate a copy so the committed qa_summary.json is not rewritten by the test
            import shutil
            out = Path(d) / "MI300X"
            shutil.copytree(OUT_DIR, out)
            with redirect_stdout(io.StringIO()):
                cls.summary = v.run(out, OUTPUTS if OUTPUTS.exists() else None)

    def test_all_checks_pass(self):
        failed = {k: c for k, c in self.summary["checks"].items() if not c["pass"]}
        self.assertEqual(failed, {})

    def test_expected_coverage(self):
        c = self.summary["counts"]
        self.assertEqual(c["operators"], 45)
        self.assertEqual(c["operator_dtype_pairs_formal"], 109)
        self.assertEqual(c["operator_dtype_pairs_profiled"], 109)

    def test_fp8_omission_is_documented(self):
        self.assertEqual(self.summary["missing_entries"]["known_exclusion"]["dtype"], "fp8_e4m3fn")


if __name__ == "__main__":
    unittest.main()
