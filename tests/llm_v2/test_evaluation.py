"""Timing protocol (mocked Proton/CUDA), anti-cache checks (CPU torch), launcher env scrubbing."""
import json
import sys
import types

import pytest
import torch

from tilebench.core import timer as core_timer
from tilebench.llm.v2.evaluation import anticache, timing
from tilebench.llm.v2.evaluation.adapters import AdapterNotReady, require_adapter
from tilebench.llm.v2.evaluation.launcher import scrubbed_env


class FakeProton:
    """Records scopes and hands back a hatchet tree with 7/8/9 us per scope."""
    def __init__(self):
        self.scopes = []
        self.started = 0

    def start(self, name, context, data, backend):
        self.started += 1
        self.base = name
        return 1

    class _Scope:
        def __init__(self, outer, name):
            self.outer, self.name = outer, name

        def __enter__(self):
            self.outer.scopes.append(self.name)

        def __exit__(self, *a):
            pass

    def scope(self, name):
        return self._Scope(self, name)

    def finalize(self, session):
        pass


@pytest.fixture
def fake_cuda(monkeypatch, tmp_path):
    fp = FakeProton()
    monkeypatch.setattr(core_timer, "proton", fp)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(core_timer, "_flush_l2_cache", lambda flush_mb=None: calls.append("flush"))
    monkeypatch.setattr(core_timer, "_flush_l2_buffer_mb", lambda: 253)
    calls = []

    def load(base):
        tree = {"frame": {"name": "root"}, "children": [
            {"frame": {"name": n}, "metrics": {}, "children": [{"frame": {"name": "k"}, "metrics": {"device_type": "CUDA", "time (ns)": (7 + i) * 1000}, "children": []}]}
            for i, n in enumerate(fp.scopes)]}
        return tree, str(tmp_path / "p.json")
    monkeypatch.setattr(core_timer, "_load_profile_data", load)
    monkeypatch.setattr(core_timer, "_build_profile_base", lambda kind, d, l: str(tmp_path / "base"))
    return fp, calls


def test_one_warmup_three_timed_with_flush_and_samples(fake_cuda, monkeypatch):
    fp, calls = fake_cuda
    monkeypatch.setattr(torch.version, "hip", "7.1")              # ROCm: eager, no capture attempted
    runs = []
    rec = timing.measure(lambda: runs.append(1), warmup=1, repeat=3, use_cuda_graph=True, flush=True)
    assert len(runs) == 4 and calls.count("flush") == 4           # 1 warmup + 3 timed, each flushed
    assert rec.samples_ms == [0.007, 0.008, 0.009] and abs(rec.mean_ms - 0.008) < 1e-12
    assert rec.scope_names == ["launch_1", "launch_2", "launch_3"] and fp.started == 1
    assert rec.timing_execution_mode == "eager" and rec.capture_succeeded is None and rec.graph_prep_runs == 0
    assert rec.requested_use_cuda_graph and not rec.effective_use_cuda_graph


def test_graph_capture_failure_is_recorded_not_hidden(fake_cuda, monkeypatch):
    fp, calls = fake_cuda
    monkeypatch.setattr(torch.version, "hip", None)

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("capture unsupported")
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Boom)
    runs = []
    rec = timing.measure(lambda: runs.append(1), warmup=1, repeat=3, use_cuda_graph=True, flush=False)
    assert rec.capture_succeeded is False and "capture unsupported" in rec.capture_error
    assert rec.timing_execution_mode == "eager" and rec.graph_prep_runs == timing.GRAPH_PREP_RUNS + 1
    assert len(runs) == 1 + timing.GRAPH_PREP_RUNS + 3


def test_graph_capture_success_replays_the_graph(fake_cuda, monkeypatch):
    fp, calls = fake_cuda
    monkeypatch.setattr(torch.version, "hip", None)
    replays = []

    class G:
        def replay(self):
            replays.append(1)

    class Ctx:
        def __init__(self, g):
            pass

        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(torch.cuda, "CUDAGraph", G)
    monkeypatch.setattr(torch.cuda, "graph", Ctx)
    runs = []
    rec = timing.measure(lambda: runs.append(1), warmup=1, repeat=3, use_cuda_graph=True, flush=False)
    assert rec.capture_succeeded and rec.timing_execution_mode == "graph" and len(replays) == 3
    assert len(runs) == 1 + timing.GRAPH_PREP_RUNS + 1          # warmup + prep + capture; timed runs are replays


def test_restore_hook_runs_outside_scopes(fake_cuda, monkeypatch):
    fp, calls = fake_cuda
    monkeypatch.setattr(torch.version, "hip", "7.1")
    order = []
    timing.measure(lambda: order.append("run"), warmup=1, repeat=3, use_cuda_graph=False, flush=False,
                   before_launch=lambda: order.append("restore"))
    assert order == ["restore", "run"] * 4


# ---- anti-cache on CPU --------------------------------------------------

def _gen():
    return (torch.randn(64), torch.randn(64))


def test_fresh_inputs_catch_output_cache():
    cache = {}

    def cached(a, b):
        key = a.data_ptr()
        if key not in cache:
            cache[key] = a + b
        return cache[key]
    res = anticache.run_numerical_checks(cached, lambda a, b: a + b, _gen, atol=1e-6, rtol=1e-6,
                                         restore_required=False, sync=lambda: None)
    names = [r.name for r in res]
    assert names[0] == "fresh_storage" and res[0].ok
    assert names[1] == "same_address_new_values" and not res[1].ok


def test_honest_implementation_passes_all_three():
    res = anticache.run_numerical_checks(lambda a, b: a + b, lambda a, b: a + b, _gen, atol=1e-6, rtol=1e-6,
                                         restore_required=False, sync=lambda: None)
    assert [r.ok for r in res] == [True, True, True]


def test_undeclared_input_mutation_fails_and_declared_is_restored():
    def mutating(a, b):
        a.add_(1)
        return a + b
    res = anticache.run_numerical_checks(mutating, lambda a, b: a + 1 + b, _gen, atol=1e-6, rtol=1e-6,
                                         restore_required=False, sync=lambda: None)
    assert not res[0].ok and res[0].inputs_mutated == [0]

    def sort_inplace(x):
        x.copy_(torch.sort(x)[0])
        return x.clone()
    res2 = anticache.run_numerical_checks(sort_inplace, lambda x: torch.sort(x)[0], lambda: (torch.randn(32),),
                                          atol=0, rtol=0, restore_required=True, sync=lambda: None)
    assert [r.ok for r in res2] == [True, True, True] and res2[2].inputs_mutated == [0]


def test_launcher_scrubs_secrets_and_redirects_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("HUGGING_FACE", "hf-x")
    monkeypatch.setenv("HOME", "/home/u")
    env = scrubbed_env(tmp_path)
    assert "OPENAI_API_KEY" not in env and "HUGGING_FACE" not in env and env["HOME"] == "/home/u"
    assert env["TILEBENCH_V2_SANDBOX"] == "1" and env["TRITON_CACHE_DIR"].startswith(str(tmp_path))


def test_worker_refuses_to_run_outside_sandbox(tmp_path, monkeypatch):
    from tilebench.llm.v2.evaluation import worker
    monkeypatch.delenv("TILEBENCH_V2_SANDBOX", raising=False)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({}))
    assert worker.main(["--job", str(job), "--out", str(tmp_path / "o.json")]) == 2


def test_nki_adapter_is_not_ready():
    assert require_adapter("proton_cuda_graph")["ready"]
    with pytest.raises(AdapterNotReady, match="NKI_HANDOFF"):
        require_adapter("neuron_runtime_trace")


def test_worker_classifies_without_gpu(tmp_path, monkeypatch):
    """The worker must not initialise a GPU on import and must report
    infrastructure_incomplete, not unsupported, when no device is visible."""
    from tilebench.llm.v2.evaluation import worker
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    res = worker.run_job({"operator": "vector_add", "dtype": "fp16", "params": {"n": 16}, "dsl": "triton",
                          "source_path": str(tmp_path / "x.py"), "atol": 1e-3, "rtol": 1e-3})
    assert res["status"] == "infrastructure_incomplete" and "no CUDA/HIP device" in res["diagnostic"]
