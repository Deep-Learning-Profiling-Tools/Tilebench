"""The committed Trn2 results, results/TRN2/csv/: PyTorch eager vs NKI on the same Trainium2
device (docs/developer_guide.md, Trn2 (NKI) results).

- 45 default CSVs on the case grid of the GPU namespaces.
- speedup_nki is nan exactly when a status is not ok, and every such row names why
  (na_scope, na_reason); a row with a speedup names nothing.
- The autotune file covers selected cases only and is never an <op>_autotune.csv.
- No GPU CSV carries an NKI column.
"""
import csv
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRN2 = REPO / "results" / "TRN2" / "csv"
COLUMNS = ["params", "dtype", "torch_ms", "nki_ms", "speedup_nki", "torch_status", "nki_status",
           "na_scope", "na_reason"]
NA_REASONS = {"pytorch_cpu_fallback", "pytorch_cpu_round_trip", "pytorch_correctness_failure",
              "pytorch_compiler_runtime_failure", "nki_unsupported", "nki_compile_failure"}
NA_OPERATORS = {"argmax", "2d_max_pooling", "radix_sort", "flash_attention", "block_sparse_attention"}
AUTOTUNE = TRN2 / "autotune_selected_cases" / "nki_autotune_largest_case_per_dtype.csv"


def read(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def default_csvs():
    return sorted(TRN2.glob("*_default.csv"))


def operator_of(path):
    return path.name[:-len("_default.csv")]


def test_one_default_csv_per_operator_and_no_full_autotune_claim():
    ops = {operator_of(p) for p in default_csvs()}
    assert len(ops) == 45
    assert ops == {p.name[:-len("_default.csv")] for p in (REPO / "results" / "B200" / "csv").glob("*_default.csv")}
    assert not list(TRN2.glob("*_autotune.csv"))          # the Trn2 autotune run is no full sweep
    assert sum(len(read(p)[1]) for p in default_csvs()) == 2200


@pytest.mark.parametrize("path", default_csvs(), ids=operator_of)
def test_default_csv(path):
    header, rows = read(path)
    assert header == COLUMNS
    gpu = read(REPO / "results" / "B200" / "csv" / path.name)[1]
    norm = lambda d: "fp32" if d == "float32" else d     # noqa: E731  (B200 kl_divergence label)
    assert [(r["params"], r["dtype"]) for r in rows] == [(r["params"], norm(r["dtype"])) for r in gpu]
    for r in rows:
        assert r["torch_status"] in ("ok", "failed", "unresolved"), r
        assert r["nki_status"] in ("ok", "failed", "unsupported"), r
        valid = (r["torch_status"], r["nki_status"]) == ("ok", "ok")
        assert valid == (r["speedup_nki"] != "nan") == (r["na_scope"] == "") == (r["na_reason"] == ""), r
        if valid:
            t, n, s = float(r["torch_ms"]), float(r["nki_ms"]), float(r["speedup_nki"])
            assert s == pytest.approx(t / n, rel=0.01, abs=0.006), r     # 4 / 2 decimals written
        else:
            assert r["na_scope"] in ("operator", "dtype", "input"), r
            assert set(r["na_reason"].split("+")) <= NA_REASONS, r
            if r["torch_status"] != "ok":
                assert r["torch_ms"] == "nan", r
            if r["nki_status"] == "unsupported":
                assert "nki_unsupported" in r["na_reason"], r
            if r["nki_status"] == "failed":
                assert "nki_compile_failure" in r["na_reason"], r
    if operator_of(path) in NA_OPERATORS:
        assert all(r["na_scope"] == "operator" for r in rows)
    else:
        assert any(r["na_scope"] == "" for r in rows)
        assert not any(r["na_scope"] == "operator" for r in rows)


def test_dtype_scope_covers_the_whole_dtype():
    for path in default_csvs():
        rows = read(path)[1]
        for dt in {r["dtype"] for r in rows if r["na_scope"] == "dtype"}:
            assert all(r["na_scope"] == "dtype" for r in rows if r["dtype"] == dt), (path.name, dt)


def test_autotune_selected_cases():
    header, rows = read(AUTOTUNE)
    assert header == ["operator", "params", "dtype", "selection", "torch_ms", "default_nki_ms",
                      "autotune_nki_ms", "autotune_gain", "speedup_default", "speedup_autotune",
                      "selected_config"]
    assert len(rows) == 72 and len({r["operator"] for r in rows}) == 29
    for r in rows:
        default = {(d["params"], d["dtype"]): d for d in read(TRN2 / f"{r['operator']}_default.csv")[1]}
        d = default[(r["params"], r["dtype"])]                            # a case of the default sweep
        assert d["nki_ms"] == r["default_nki_ms"] and d["speedup_nki"] == r["speedup_default"]
        assert r["selection"] in ("largest case of the dtype",
                                  "largest case of the dtype with a valid PyTorch baseline")
        a, b = float(r["default_nki_ms"]), float(r["autotune_nki_ms"])
        assert float(r["autotune_gain"]) == pytest.approx(a / b, rel=0.01, abs=0.002)
        assert float(r["speedup_autotune"]) == pytest.approx(float(r["torch_ms"]) / b, rel=0.01, abs=0.006)
    per = {}
    for r in rows:
        per.setdefault((r["operator"], r["dtype"]), []).append(r["selection"])
    assert all(v.count("largest case of the dtype") == 1 for v in per.values()), per


@pytest.mark.parametrize("gpu", ["B200", "GH200", "MI300X"])
def test_gpu_csvs_carry_no_nki_columns(gpu):
    paths = sorted((REPO / "results" / gpu / "csv").glob("**/*.csv"))
    assert paths
    for p in paths:
        assert not [c for c in read(p)[0] if "nki" in c], p.name
