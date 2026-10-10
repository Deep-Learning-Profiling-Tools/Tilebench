"""Opt-in prepared timing (rope): run() is split into an untimed restore and a timed launch.

report_prepared_benchmark(launch, restore) runs restore() before every launch, ahead of the
cache flush and outside both the Proton scope and the captured graph; the engine uses it only
for impls that define prepare_timed_run(), every other impl is still timed by
timer.report_benchmark. The timer tests mock Proton, the flush and graph capture, so they run
on any host; the rope tests at the end need a CUDA device."""
import contextlib
import sys
import types

import pytest
import torch
import yaml

from tilebench.core import engine, prepared_timer, timer


class RecordingProton:
    def __init__(self, events):
        self.events = events

    def start(self, **kw):
        return 1

    @contextlib.contextmanager
    def scope(self, name):
        self.events.append("scope+")
        yield
        self.events.append("scope-")

    def finalize(self, session=None):
        pass


@pytest.fixture
def events(monkeypatch):
    ev = []
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(timer, "proton", RecordingProton(ev))
    monkeypatch.setattr(timer, "_flush_l2_cache", lambda *a, **k: ev.append("flush"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    hatchet = {"frame": {"name": "ROOT"}, "children": [
        {"frame": {"name": "launch"}, "metrics": {}, "children": [
            {"frame": {"name": "k"}, "metrics": {"device_type": "CUDA", "time (ns)": 3000.0}}]}]}
    monkeypatch.setattr(timer, "_load_profile_data", lambda base: (hatchet, "/nonexistent"))
    return ev


@pytest.fixture
def capture(monkeypatch, events):
    """torch.cuda.graph / CUDAGraph stand-ins: a replay is recorded as 'replay'; restoring
    while a graph is being captured is an error."""
    state = types.SimpleNamespace(capturing=False)

    class Graph:
        def replay(self):
            events.append("replay")

    @contextlib.contextmanager
    def graph(g):
        state.capturing = True
        events.append("capture+")
        yield
        events.append("capture-")
        state.capturing = False

    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", graph)
    return state


def test_eager_restore_precedes_the_flush_and_stays_outside_the_scope(events):
    prepared_timer.report_prepared_benchmark(
        lambda: events.append("launch"), lambda: events.append("restore"),
        warmup=2, repeat=3, use_cuda_graph=False, flush_l2=True)
    assert events == (["restore", "flush", "launch"] * 2
                      + ["restore", "flush", "scope+", "launch", "scope-"] * 3)


def test_graph_holds_only_the_launch_and_every_replay_is_restored_first(events, capture):
    def restore():
        assert not capture.capturing, "restore must not be captured into the timed graph"
        events.append("restore")

    res = prepared_timer.report_prepared_benchmark(
        lambda: events.append("launch"), restore, warmup=1, repeat=2, use_cuda_graph=True, flush_l2=True)
    assert events == (["restore", "flush", "launch"]                       # warmup
                      + ["restore", "launch"] * 3 + ["restore"]            # graph preparation
                      + ["capture+", "launch", "capture-"]                 # graph = launch only
                      + ["restore", "flush", "scope+", "replay", "scope-"] * 2)
    assert res == {"mean": 0.0015}                                         # 3000 ns / 2 repeats


def test_rocm_restores_before_every_eager_launch(events, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.1.25424")

    def no_graph(*a, **k):
        raise AssertionError("ROCm must not capture a graph")
    monkeypatch.setattr(torch.cuda, "CUDAGraph", no_graph)
    monkeypatch.setattr(torch.cuda, "graph", no_graph)
    prepared_timer.report_prepared_benchmark(
        lambda: events.append("launch"), lambda: events.append("restore"),
        warmup=1, repeat=2, use_cuda_graph=True, flush_l2=True)
    assert events == (["restore", "flush", "launch"]
                      + ["restore", "flush", "scope+", "launch", "scope-"] * 2)


# --------------------------------------------------------------------------
# engine: prepare_timed_run is opt-in; run() is still what gets verified
# --------------------------------------------------------------------------

@pytest.fixture
def fake_op(tmp_path, monkeypatch):
    cfg = {"benchmark": {"warmup": 1, "repeat": 1, "use_cuda_graph": True, "flush_l2": True},
           "test_cases": [{"n": 2, "dtype": "fp32"}]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    st = types.SimpleNamespace(timed=[], prepared=[], calls=[])
    restore = lambda: None                                 # noqa: E731
    launch = lambda: None                                  # noqa: E731

    def prepared(*inputs, autotune=False):
        st.calls.append((inputs, autotune))
        return launch, restore

    plain = types.SimpleNamespace(run=lambda n, autotune=False: n)
    split = types.SimpleNamespace(run=lambda n, autotune=False: n, prepare_timed_run=prepared)
    pkg = "tilebench.benchmarks.operators.fakeop"
    monkeypatch.setitem(sys.modules, f"{pkg}.impl_torch", types.SimpleNamespace(run=lambda n: n))
    monkeypatch.setitem(sys.modules, f"{pkg}.impl_triton", split)
    monkeypatch.setitem(sys.modules, f"{pkg}.impl_tilelang", plain)
    monkeypatch.setattr(engine, "operator_config", lambda op: path)
    monkeypatch.setattr(engine, "get_generator", lambda op: (lambda n, dtype: (n,)))
    monkeypatch.setattr(engine, "infer_problem_size", lambda op, params: params["n"])
    monkeypatch.setattr(engine, "HAS_CUDA", True)
    monkeypatch.setattr(engine, "_sync", lambda: None)
    monkeypatch.setattr(engine, "verify", lambda out, ref, **kw: (True, ""))
    monkeypatch.setattr(engine, "report_benchmark",
                        lambda fn, args, **kw: st.timed.append((fn, args, kw)) or {"mean": 1.0})
    monkeypatch.setattr(engine, "report_prepared_benchmark",
                        lambda la, re, **kw: st.prepared.append((la, re, kw)) or {"mean": 2.0})
    st.launch, st.restore, st.plain = launch, restore, plain
    return st


def test_engine_uses_prepared_timing_only_for_impls_that_define_it(fake_op):
    res = engine.run_benchmark_suite("fakeop", enabled_backends={"triton", "tilelang"},
                                     benchmark_overrides={"autotune": True})
    (la, re, kw), = fake_op.prepared                                         # triton: prepared
    assert (la, re) == (fake_op.launch, fake_op.restore)
    assert fake_op.calls == [((2,), True)]                                   # same inputs + autotune
    assert kw["flush_l2"] is True and kw["use_cuda_graph"] is True and kw["repeat"] == 1
    torch_t, tilelang_t = fake_op.timed                                      # torch, tilelang: run()
    assert torch_t[1] == (2,)
    assert tilelang_t[0] is fake_op.plain.run and tilelang_t[2]["kwargs"] == {"autotune": True}
    assert res[0]["triton_ms"] == 2.0 and res[0]["tilelang_ms"] == 1.0


# --------------------------------------------------------------------------
# rope on a CUDA device: the prepared launch is run()'s kernel, restore() prevents accumulation,
# and the timed scope holds the rotation kernel only
# --------------------------------------------------------------------------

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                                reason="needs a CUDA device")


def _rope(backend):
    impl = pytest.importorskip(f"tilebench.benchmarks.operators.rope.impl_{backend}")
    from tilebench.data.tensors import generate_rope_inputs
    torch.manual_seed(0)
    q, cos, sin = generate_rope_inputs(1, 1024, 32, 128, dtype=torch.float16, device="cuda")
    return impl, q, cos, sin


@needs_cuda
@pytest.mark.parametrize("backend", ["triton", "cutile", "tilelang"])
@pytest.mark.parametrize("autotune", [False, True])
def test_rope_prepared_launch_matches_run_and_restores(backend, autotune):
    impl, q, cos, sin = _rope(backend)
    q0 = q.clone()
    expected = impl.run(q, cos, sin, autotune=autotune)
    launch, restore = impl.prepare_timed_run(q, cos, sin, autotune=autotune)
    for _ in range(2):
        out = restore()                                    # Tensor.copy_ returns the output buffer
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        launch()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == before     # no clone/scratch inside the launch
        assert torch.equal(out, expected)
    launch()                                               # without restore() it rotates again
    torch.cuda.synchronize()
    assert not torch.equal(out, expected)
    assert torch.equal(q, q0)                              # the input itself is never modified


@needs_cuda
@pytest.mark.parametrize("backend", ["triton", "cutile", "tilelang"])
def test_rope_proton_scope_holds_only_the_rotation_kernel(backend, tmp_path):
    impl, q, cos, sin = _rope(backend)
    if timer.proton is None:
        pytest.skip("triton.profiler (proton) not available")
    launch, restore = impl.prepare_timed_run(q, cos, sin)
    prepared_timer.report_prepared_benchmark(
        launch, restore, warmup=1, repeat=3, use_cuda_graph=True, flush_l2=True,
        keep_proton_files=True, proton_output_dir=str(tmp_path), proton_file_label="rope")
    tree, _ = timer._load_profile_data(str(tmp_path / "tilebench_proton_tree_rope"))
    kernels = []

    def collect(node, inside):
        name = (node.get("frame") or {}).get("name", "")
        inside = inside or name == "launch"
        m = node.get("metrics") or {}
        if inside and str(m.get("device_type", "")).upper() == "CUDA":
            kernels.append((name, m.get("count")))
        for c in node.get("children", []):
            collect(c, inside)
    for root in (tree if isinstance(tree, list) else [tree]):
        collect(root, False)
    assert kernels and all(n.startswith("rope_embedding") for n, _ in kernels), kernels
    assert sum(c for _, c in kernels) == 3                 # one rotation kernel per timed replay
