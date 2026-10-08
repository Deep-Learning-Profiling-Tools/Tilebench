"""cuTile autotune: every ct.tune.exhaustive_search runs with the crash-isolation
guard, while the search space, its order, the launch callbacks and the per-shape
tuning cache stay exactly as the operator passed them.

exhaustive_search is replaced by a recorder, so these run on any host with
cuda-tile installed; the last test tunes a real kernel and needs a CUDA device.
"""
from types import SimpleNamespace

import pytest

ct = pytest.importorskip("cuda.tile")
torch = pytest.importorskip("torch")

from tilebench.core import cutile_autotune  # noqa: E402
from tilebench.core.cutile_autotune import CRASH_ISOLATION_TIMEOUT_SEC, CutileAutotuner  # noqa: E402


@pytest.fixture
def search(monkeypatch):
    """Record every exhaustive_search call; the best config is the last candidate."""
    calls = []

    def fake(search_space, stream, *args, **kwargs):
        calls.append(SimpleNamespace(search_space=search_space, stream=stream, args=args, kwargs=kwargs))
        best = search_space[-1]
        return SimpleNamespace(best=SimpleNamespace(config=best, mean_us=float(len(calls))),
                               successes=[], failures=[])
    monkeypatch.setattr(ct.tune, "exhaustive_search", fake)
    return calls


def _tune(tuner, key, space, **kw):
    callbacks = dict(grid_fn=lambda cfg: (1, 1, 1), args_fn=lambda cfg: (), hints_fn=lambda cfg: {})
    callbacks.update(kw)
    return tuner.tune_or_cached(shape_key=key, search_space=space, stream="stream", **callbacks), callbacks


def test_crash_isolation_guard_is_60_seconds():
    assert CRASH_ISOLATION_TIMEOUT_SEC == 60


def test_tune_or_cached_passes_the_guard_and_everything_else_unchanged(search):
    kernel = object()
    space = [SimpleNamespace(tile=t, occupancy=o) for t in (32, 64) for o in (4, 8)]
    snapshot = list(space)
    best, callbacks = _tune(CutileAutotuner(kernel), (1, 2, 3), space)

    (call,) = search
    assert call.kwargs["single_run_timeout_sec"] == CRASH_ISOLATION_TIMEOUT_SEC
    assert call.search_space is space and space == snapshot          # same list, same order
    assert call.stream == "stream" and call.kwargs["kernel"] is kernel
    for name, fn in callbacks.items():
        assert call.kwargs[name] is fn
    assert best is space[-1]                                         # result.best.config


def test_tuning_is_still_cached_per_shape(search):
    tuner = CutileAutotuner(object())
    space = [SimpleNamespace(occupancy=o) for o in (4, 8, 16)]
    first, _ = _tune(tuner, ("a",), space)
    again, _ = _tune(tuner, ("a",), space)
    other, _ = _tune(tuner, ("b",), space)
    assert first is again is other is space[-1]
    assert len(search) == 2                                          # ("a",) tuned once


def test_top_k_direct_search_passes_the_guard(search, monkeypatch):
    impl = pytest.importorskip("tilebench.benchmarks.operators.top_k_selection.impl_cutile")
    monkeypatch.setattr(impl, "_autotune_cache", {})
    x = torch.zeros(8192)
    cfg = impl._tune(x, k=16, K2=16, stream="stream")
    assert search and all(c.kwargs["single_run_timeout_sec"] == CRASH_ISOLATION_TIMEOUT_SEC for c in search)
    assert all(c.search_space is impl._OCC_SPACE for c in search)
    assert cfg.block in impl._BLOCKS and cfg.occupancy == impl._OCC_SPACE[-1].occupancy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_real_isolated_search_returns_a_verified_config():
    from tilebench.benchmarks.operators.batched_matmul import impl_cutile, impl_torch
    from tilebench.data.tensors import GENERATORS
    assert cutile_autotune.ct.tune.exhaustive_search is ct.tune.exhaustive_search
    inputs = GENERATORS["batched_matmul"](4, 64, dtype=torch.float16)
    out = impl_cutile.run(*inputs, autotune=True)
    torch.testing.assert_close(out.float(), impl_torch.run(*inputs).float(), atol=1.0, rtol=1e-2)
    cfg = impl_cutile.get_last_config()
    assert SimpleNamespace(**cfg) in impl_cutile._SEARCH_SPACE
