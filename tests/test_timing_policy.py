"""Timing policy: the config requests CUDA-graph replay, the runtime decides
whether the timed region really uses it (eager on ROCm), and a torch
reference that rejects a case with ValueError skips only that case.

Every device and profiler here is mocked, so the tests run on any host."""
import sys
import types

import pytest
import torch
import yaml

from tilebench import hardware
from tilebench.core import engine, timer

from test_hardware import B200_L2, MB, device  # noqa: F401  (fixture)


@pytest.fixture
def hip(monkeypatch):
    def set_hip(version):
        monkeypatch.setattr(torch.version, "hip", version)
    return set_hip


# --------------------------------------------------------------------------
# effective_use_cuda_graph / timing_mode
# --------------------------------------------------------------------------

def test_nvidia_keeps_the_requested_graph(hip):
    hip(None)
    assert timer.effective_use_cuda_graph(True) is True
    assert timer.effective_use_cuda_graph(False) is False
    assert timer.timing_mode(True) == {"requested_use_cuda_graph": True,
                                       "effective_use_cuda_graph": True,
                                       "timing_execution_mode": "graph", "timing_note": None}


def test_rocm_runs_a_requested_graph_eagerly(hip):
    hip("7.1.25424")
    assert timer.effective_use_cuda_graph(True) is False
    mode = timer.timing_mode(True)
    assert mode["requested_use_cuda_graph"] is True and mode["effective_use_cuda_graph"] is False
    assert mode["timing_execution_mode"] == "eager"
    assert "HIP Graph" in mode["timing_note"]


def test_rocm_without_a_requested_graph_is_eager_and_unremarked(hip):
    hip("7.1.25424")
    assert timer.effective_use_cuda_graph(False) is False
    assert timer.timing_mode(False) == {"requested_use_cuda_graph": False,
                                        "effective_use_cuda_graph": False,
                                        "timing_execution_mode": "eager", "timing_note": None}


def test_the_policy_does_not_read_the_result_label():
    import inspect
    assert list(inspect.signature(timer.effective_use_cuda_graph).parameters) == ["requested"]


# --------------------------------------------------------------------------
# report_benchmark: which runner the timed region gets
# --------------------------------------------------------------------------

class FakeProton:
    """Records nothing on a GPU; returns one 'launch' scope worth 1 us/call."""
    def __init__(self):
        self.repeat = 0

    def start(self, **kw):
        return 1

    def scope(self, name):
        import contextlib
        return contextlib.nullcontext()

    def finalize(self, session=None):
        pass


@pytest.fixture
def no_gpu_timer(monkeypatch):
    """report_benchmark without a GPU: fake Proton/hatchet, no-op sync."""
    monkeypatch.setattr(timer, "proton", FakeProton())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    hatchet = {"frame": {"name": "ROOT"}, "children": [
        {"frame": {"name": "launch"}, "metrics": {}, "children": [
            {"frame": {"name": "k"}, "metrics": {"device_type": "HIP", "time (ns)": 3000.0}}]}]}
    monkeypatch.setattr(timer, "_load_profile_data", lambda base: (hatchet, "/nonexistent"))


def _forbid_graph_capture(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("graph capture must not happen")
    monkeypatch.setattr(torch.cuda, "CUDAGraph", boom)
    monkeypatch.setattr(torch.cuda, "graph", boom)


def test_rocm_fallback_never_captures_a_graph(hip, no_gpu_timer, monkeypatch):
    hip("7.1.25424")
    _forbid_graph_capture(monkeypatch)
    calls = []
    res = timer.report_benchmark(lambda: calls.append(1), (), warmup=2, repeat=3,
                                 use_cuda_graph=True, flush_l2=False)
    assert len(calls) == 2 + 3          # warmup + every timed call ran the function itself
    assert res == {"mean": 0.001}       # 3000 ns / 3 repeats


def test_rocm_passes_eager_to_the_runner(hip, no_gpu_timer, monkeypatch):
    hip("7.1.25424")
    seen = []
    real = timer._prepare_runner
    monkeypatch.setattr(timer, "_prepare_runner",
                        lambda f, a, kw, use_cuda_graph: seen.append(use_cuda_graph) or real(f, a, kw, use_cuda_graph))
    timer.report_benchmark(lambda: None, (), warmup=0, repeat=1, use_cuda_graph=True)
    assert seen == [False]


def test_nvidia_still_replays_the_requested_graph(hip, no_gpu_timer, monkeypatch):
    hip(None)
    seen = []
    monkeypatch.setattr(timer, "_prepare_runner",
                        lambda f, a, kw, use_cuda_graph: seen.append(use_cuda_graph) or (lambda: None))
    timer.report_benchmark(lambda: None, (), warmup=0, repeat=1, use_cuda_graph=True)
    assert seen == [True]
    seen.clear()
    timer.report_benchmark(lambda: None, (), warmup=0, repeat=1, use_cuda_graph=False)
    assert seen == [False]


# --------------------------------------------------------------------------
# cache eviction is untouched by the timing policy
# --------------------------------------------------------------------------

def test_b200_eviction_is_unchanged(device):
    device(name="NVIDIA B200", capability=(10, 0), l2=B200_L2)
    assert hardware.last_level_cache_bytes() == B200_L2
    assert timer._flush_l2_buffer_mb() == 253


def test_cdna3_eviction_is_unchanged(device):
    device(name="AMD Instinct MI300X", hip="7.1.25424", gcn="gfx942:sramecc+:xnack-", l2=4 * MB)
    assert hardware.last_level_cache_bytes() == 256 * MB
    assert timer._flush_l2_buffer_mb() == 512


# --------------------------------------------------------------------------
# engine: one timing notice per process, mode attached to every case,
# a ValueError from the torch reference skips that case only
# --------------------------------------------------------------------------

@pytest.fixture
def fake_operator(tmp_path, monkeypatch):
    """Three cases n=1,2,3 of a torch-only operator; the torch reference of
    the case listed in fake_operator.bad raises ValueError."""
    cfg = {"benchmark": {"warmup": 1, "repeat": 1, "use_cuda_graph": True, "flush_l2": False},
           "test_cases": [{"n": 1, "dtype": "fp32"}, {"n": 2, "dtype": "fp32"}, {"n": 3, "dtype": "fp32"}]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    state = types.SimpleNamespace(bad={2}, timed=[])

    def ref(n):
        if n in state.bad:
            raise ValueError("Expected b.dtype() == at::kFloat8_e4m3fnuz, got: c10::Float8_e4m3fn")
        return n
    impl_torch = types.SimpleNamespace(run=ref)
    monkeypatch.setitem(sys.modules, "tilebench.benchmarks.operators.fakeop.impl_torch", impl_torch)
    monkeypatch.setattr(engine, "operator_config", lambda op: path)
    monkeypatch.setattr(engine, "get_generator", lambda op: (lambda n, dtype: (n,)))
    monkeypatch.setattr(engine, "infer_problem_size", lambda op, params: params["n"])
    monkeypatch.setattr(engine, "HAS_CUDA", True)
    monkeypatch.setattr(engine, "_sync", lambda: None)
    monkeypatch.setattr(engine, "report_benchmark",
                        lambda fn, args, **kw: state.timed.append((args, kw["use_cuda_graph"])) or {"mean": 1.0})
    monkeypatch.setattr(engine, "_timing_fallback_announced", False)
    return state


def test_a_torch_value_error_skips_only_that_case(fake_operator, hip, capsys):
    hip(None)
    results = engine.run_benchmark_suite("fakeop", enabled_backends=set())
    assert [r["params"]["n"] for r in results] == [1, 3]                 # A and C kept
    out = capsys.readouterr().out
    assert "Skipped: dtype=fp32 not supported by torch (ValueError: Expected b.dtype()" in out
    assert [args for args, _ in fake_operator.timed] == [(1,), (3,)]     # B never timed


def test_other_exceptions_from_the_torch_reference_still_propagate(fake_operator, hip, monkeypatch):
    hip(None)
    monkeypatch.setitem(sys.modules, "tilebench.benchmarks.operators.fakeop.impl_torch",
                        types.SimpleNamespace(run=lambda n: 1 / 0))
    with pytest.raises(ZeroDivisionError):
        engine.run_benchmark_suite("fakeop", enabled_backends=set())


def test_engine_records_and_announces_the_rocm_fallback_once(fake_operator, hip, capsys):
    hip("7.1.25424")
    fake_operator.bad = set()
    first = engine.run_benchmark_suite("fakeop", enabled_backends=set())
    second = engine.run_benchmark_suite("fakeop", enabled_backends=set())
    out = capsys.readouterr().out
    assert out.count("ROCm timing fallback") == 1                        # once per process
    for r in first + second:
        assert r["timing"]["requested_use_cuda_graph"] is True
        assert r["timing"]["effective_use_cuda_graph"] is False
        assert r["timing"]["timing_execution_mode"] == "eager"
    # the engine still passes the requested value; report_benchmark applies the policy
    assert {g for _, g in fake_operator.timed} == {True}


def test_engine_on_nvidia_reports_graph_and_prints_nothing(fake_operator, hip, capsys):
    hip(None)
    fake_operator.bad = set()
    results = engine.run_benchmark_suite("fakeop", enabled_backends=set())
    assert "ROCm timing fallback" not in capsys.readouterr().out
    assert all(r["timing"]["timing_execution_mode"] == "graph" for r in results)


# --------------------------------------------------------------------------
# where the requested/effective mode is recorded
# --------------------------------------------------------------------------

from test_results_layout import fake_results, load_script, results, run_bench  # noqa: E402,F401

ROCM_MODE = {"requested_use_cuda_graph": True, "effective_use_cuda_graph": False,
             "timing_execution_mode": "eager", "timing_note": timer.HIP_GRAPH_TIMING_NOTE}


def test_run_bench_sidecar_records_the_timing_mode_and_the_log_keeps_its_format(run_bench, results, monkeypatch):
    import json

    def engine_with_mode(operator, benchmark_overrides=None, enabled_backends=None, logs_dir=None):
        rows = fake_results(enabled_backends)
        for r in rows:
            r["timing"] = dict(ROCM_MODE)
        return rows
    monkeypatch.setattr(run_bench.module, "run_benchmark_suite", engine_with_mode)
    run_bench("--gpu", "SMOKE-MI300X", "--operator", "mul2", "--tile-language", "triton")
    name = "mul2_default_triton.json"
    record = json.loads((results / "SMOKE-MI300X/logs/provenance" / name).read_text())
    assert record["timing"] == ROCM_MODE
    assert set(record["run"]) == {"script", "operator", "mode", "backends", "overrides", "timing_log",
                                  "autotune_log", "summary_csv", "tilelang_autotuner_log",
                                  "tilelang_autotuner_log_note"}          # run block unchanged
    log = json.loads((results / "SMOKE-MI300X/logs/time_measurement_logs" / name).read_text())
    assert isinstance(log, list) and all("timing" not in case for case in log)


def test_run_bench_all_records_the_timing_mode_per_operator(results, tmp_path, monkeypatch):
    import json
    mod = load_script("run_bench_all")

    def engine_with_mode(op, enabled_backends=None, logs_dir=None):
        rows = fake_results({"triton"})
        for r in rows:
            r["timing"] = dict(ROCM_MODE)
        return rows
    monkeypatch.setattr(mod, "run_benchmark_suite", engine_with_mode)
    monkeypatch.setattr(sys, "argv", ["run_bench_all.py", "--gpu", "SMOKE-MI300X", "--results-root",
                                      str(tmp_path / "runs"), "--operators", "mul2", "relu",
                                      "--tile-language", "triton"])
    assert mod.main() == 0
    (summary,) = (tmp_path / "runs").glob("*/summary.json")
    manifest = json.loads(summary.read_text())
    assert manifest["provenance"]["timing"] == {"mul2": ROCM_MODE, "relu": ROCM_MODE}
    assert manifest["provenance"]["run"] == {"script": "scripts/run_bench_all.py",
                                             "tile_language": "triton", "backends": ["triton"]}
