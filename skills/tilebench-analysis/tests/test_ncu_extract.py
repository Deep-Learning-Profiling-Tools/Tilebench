import importlib.util
import json
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ncu_extract.py"
spec = importlib.util.spec_from_file_location("ncu_extract", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Metric:
    def __init__(self, value, unit="inst"):
        self.number, self.units = value, unit

    def value(self):
        if isinstance(self.number, Exception):
            raise self.number
        return self.number

    def unit(self):
        return self.units


class Action:
    def __init__(self, name, metrics):
        self.kernel, self.metrics = name, metrics

    def name(self):
        return self.kernel

    def metric_names(self):
        return list(self.metrics)

    def __getitem__(self, name):
        return self.metrics[name]


class Range:
    def __init__(self, actions):
        self.actions = actions

    def num_actions(self):
        return len(self.actions)

    def action_by_idx(self, index):
        return self.actions[index]


class Report:
    def __init__(self, ranges):
        self.ranges = ranges

    def num_ranges(self):
        return len(self.ranges)

    def range_by_idx(self, index):
        return self.ranges[index]


class ExtractionTests(unittest.TestCase):
    def test_exact_values_units_and_every_action(self):
        report = Report([Range([Action("partial", {"inst": Metric(2**60 + 1)}),
                                Action("reduce", {"time": Metric(0.125, "nsecond")})]),
                         Range([Action("other", {"device": Metric("B200", "")})])])
        result = module.extract(report, "evidence/run.ncu-rep")
        self.assertEqual(len(result["actions"]), 3)
        self.assertEqual(result["metrics"][0]["value"], 2**60 + 1)
        self.assertIsInstance(result["metrics"][0]["value"], int)
        self.assertEqual(result["metrics"][1]["unit"], "nsecond")
        self.assertEqual(result["metrics"][2]["range_index"], 1)
        self.assertEqual(result["metrics"][2]["action_index"], 0)
        self.assertEqual(json.loads(json.dumps(result))["metrics"][2]["value"], "B200")

    def test_selection_missing_is_not_zero_and_inventory_is_complete(self):
        report = Report([Range([Action("kernel", {"inst": Metric(0), "time": Metric(2)})])])
        result = module.extract(report, "run.ncu-rep", ["inst", "missing", "inst"])
        self.assertEqual(result["actions"][0]["metric_names"], ["inst", "time"])
        self.assertEqual(len(result["metrics"]), 1)
        self.assertEqual(result["metrics"][0]["value"], 0)
        self.assertEqual(result["errors"][0]["error"], "metric_not_collected")

    def test_extraction_failures_and_nonfinite_values_are_explicit(self):
        metrics = {"bad": Metric(RuntimeError("unsupported")),
                   "nan": Metric(float("nan")), "complex": Metric([1, 2])}
        result = module.extract(Report([Range([Action("kernel", metrics)])]), "run.ncu-rep")
        self.assertEqual(result["metrics"], [])
        self.assertEqual(len(result["errors"]), 3)
        json.dumps(result, allow_nan=False)

    def test_empty_report_is_not_an_invented_action(self):
        result = module.extract(Report([]), "empty.ncu-rep")
        self.assertEqual(result["actions"], [])
        self.assertEqual(result["metrics"], [])


if __name__ == "__main__":
    unittest.main()
