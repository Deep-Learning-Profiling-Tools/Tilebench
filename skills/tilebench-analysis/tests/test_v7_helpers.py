"""Portable arithmetic and disassembly regressions; no GPU or NCU installation needed."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


def load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


csv_ratios = load("csv_ratios")
sass_listing = load("sass_listing")


class CsvTests(unittest.TestCase):
    def analyze(self, values, references):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurements.csv"
            path.write_text("shape,dtype,candidate_us,reference_us,library_us\nS,fp32," + values + "\n")
            return csv_ratios.compare_references(path, {"shape": "S", "dtype": "fp32"},
                                                  "candidate_us", references)

    def test_arbitrary_column_names_and_reference_scope(self):
        result = self.analyze("20,10,5", ["reference_us", "library_us"])
        ranking = result["reference_ranking"]
        self.assertEqual(ranking["fastest_reference_columns"], ["library_us"])
        self.assertEqual(ranking["target_over_fastest"], 4)
        self.assertEqual(result["comparisons"][0]["ratio"], 2)

    def test_ties_preserved(self):
        result = self.analyze("20,10,10", ["reference_us", "library_us"])
        self.assertEqual(result["reference_ranking"]["fastest_reference_columns"],
                         ["reference_us", "library_us"])

    def test_invalid_reference_not_dropped(self):
        result = self.analyze("20,10,0", ["reference_us", "library_us"])
        self.assertEqual(result["reference_ranking"]["status"], "unresolved")

    def test_target_cannot_be_reference(self):
        with self.assertRaises(ValueError):
            self.analyze("20,10,5", ["candidate_us"])

    def test_stored_ratio_does_not_pick_arbitrary_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratios.csv"
            path.write_text("shape,a,b,c,ratio\nS,2.00,1.00,1.00,2.00\n")
            result = csv_ratios.analyze_csv(path, {"shape": "S"}, [], ["ratio"], ["a", "b", "c"])
            self.assertEqual(result["stored_ratio_checks"][0]["status"], "ambiguous")


class SassTests(unittest.TestCase):
    def test_uniform_predication_and_nops(self):
        result = sass_listing.parse_listing("Kernel Name: sample\n"
                                            "0x0000 @!UP0 LDG.E.U16 R1, [R2];\n"
                                            "0x0010 @P2 IADD3 R3, R4, 1, RZ;\n"
                                            "0x0020 NOP;\n")
        kernel = result["kernels"][0]
        self.assertEqual(kernel["instruction_count"], 3)
        self.assertEqual(kernel["non_nop_instruction_count"], 2)
        self.assertEqual(kernel["predicates"], {"@!UP0": 1, "@P2": 1})

    def test_multiple_actions_keep_separate_address_spaces(self):
        result = sass_listing.parse_listing("Function: first\n/*0000*/ MOV R1, R2;\n"
                                            "Function: second\n/*0000*/ NOP;\n")
        self.assertEqual([item["instruction_count"] for item in result["kernels"]], [1, 1])
        self.assertFalse(any(item["code"] == "duplicate_address" for item in result["warnings"]))

    def test_duplicate_addresses_not_silently_accepted(self):
        result = sass_listing.parse_listing("Function: sample\n/*0000*/ MOV R1, R2;\n"
                                            "/*0000*/ MOV R2, R3;\n")
        self.assertTrue(any(item["code"] == "duplicate_address" for item in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
