"""A case whose input generation or torch reference rejects it with
ValueError is skipped like RuntimeError/TypeError already were; the other
cases of the operator still run and are returned. Any other exception still
stops the run. The operator, generator and timer are mocked, so the tests run
on any host."""
import sys
import types

import pytest
import yaml

from tilebench.core import engine

MSG = "Expected b.dtype() == at::kFloat8_e4m3fnuz, got: c10::Float8_e4m3fn"


@pytest.fixture
def fake_operator(tmp_path, monkeypatch):
    """Cases n=1 (A), n=2 (B), n=3 (C) of a torch-only operator. state.bad_ref
    / state.bad_input name the cases whose torch reference / input generator
    raises ValueError."""
    cfg = {"benchmark": {"warmup": 1, "repeat": 1, "use_cuda_graph": False, "flush_l2": False},
           "test_cases": [{"n": 1, "dtype": "fp32"}, {"n": 2, "dtype": "fp32"}, {"n": 3, "dtype": "fp32"}]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    state = types.SimpleNamespace(bad_ref=set(), bad_input=set(), timed=[])

    def generate(n, dtype):
        if n in state.bad_input:
            raise ValueError(f"cannot build inputs for n={n}")
        return (n,)

    def reference(n):
        if n in state.bad_ref:
            raise ValueError(MSG)
        return n

    monkeypatch.setitem(sys.modules, "tilebench.benchmarks.operators.fakeop.impl_torch",
                        types.SimpleNamespace(run=reference))
    monkeypatch.setattr(engine, "operator_config", lambda op: path)
    monkeypatch.setattr(engine, "get_generator", lambda op: generate)
    monkeypatch.setattr(engine, "infer_problem_size", lambda op, params: params["n"])
    monkeypatch.setattr(engine, "HAS_CUDA", True)
    monkeypatch.setattr(engine, "_sync", lambda: None)
    monkeypatch.setattr(engine, "report_benchmark",
                        lambda fn, args, **kw: state.timed.append(args) or {"mean": 1.0})
    state.impl_key = "tilebench.benchmarks.operators.fakeop.impl_torch"
    return state


def test_a_torch_reference_value_error_skips_only_that_case(fake_operator, capsys):
    fake_operator.bad_ref = {2}
    results = engine.run_benchmark_suite("fakeop", enabled_backends=set())
    assert [r["params"]["n"] for r in results] == [1, 3]                 # A and C kept
    assert fake_operator.timed == [(1,), (3,)]                           # B never timed
    out = capsys.readouterr().out
    assert f"Skipped: dtype=fp32 not supported by torch (ValueError: {MSG})" in out


def test_an_input_generation_value_error_skips_only_that_case(fake_operator, capsys):
    fake_operator.bad_input = {2}
    results = engine.run_benchmark_suite("fakeop", enabled_backends=set())
    assert [r["params"]["n"] for r in results] == [1, 3]
    assert "Skipped: dtype=fp32 not supported for input generation (ValueError: cannot build inputs for n=2)" \
        in capsys.readouterr().out


def test_every_case_passing_is_unchanged(fake_operator):
    results = engine.run_benchmark_suite("fakeop", enabled_backends=set())
    assert [r["params"]["n"] for r in results] == [1, 2, 3]
    assert all(r["torch_ms"] == 1.0 for r in results)


def test_other_exceptions_from_the_torch_reference_still_stop_the_run(fake_operator, monkeypatch):
    monkeypatch.setitem(sys.modules, fake_operator.impl_key, types.SimpleNamespace(run=lambda n: 1 / 0))
    with pytest.raises(ZeroDivisionError):
        engine.run_benchmark_suite("fakeop", enabled_backends=set())
