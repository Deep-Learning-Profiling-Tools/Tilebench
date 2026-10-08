"""destindex TileLang: the nope and rope copies launch one kernel twice and are
tuned separately, so each launch reads its own default config, which the
profiling replay fills with that launch's winner.
"""
import importlib

import pytest
import torch

from tilebench.profiling.replay import apply_winner

OP = "tilebench.benchmarks.operators.destindex"
# formal GH200 autotune winner of the fp32 sweep-max case: the two launches differ
FP32_WINNER = {"nope_BLOCK_SIZE": 512, "nope_threads": 64, "rope_BLOCK_SIZE": 1024, "rope_threads": 64}


@pytest.fixture
def impl(monkeypatch):
    mod = importlib.reload(importlib.import_module(f"{OP}.impl_tilelang"))
    yield mod
    importlib.reload(mod)                         # later tests see the pristine defaults


def launches(mod, monkeypatch):
    """Run the default path with the kernel replaced by a recorder (no GPU needed)."""
    calls = []
    monkeypatch.setattr(mod, "copy_by_dest_kernel", lambda *a, **kw: calls.append(
        (kw["head_dim"], kw["BLOCK_SIZE"], kw["threads"])))
    s = 8
    mod.run(torch.zeros(s, 12, 128), torch.zeros(s, 1, 64), torch.arange(s),
            torch.zeros(s, 12, 128), torch.zeros(s, 1, 64))
    return calls                                  # [(head_dim, BLOCK_SIZE, threads)] in launch order


def test_default_launches_keep_the_original_config(impl, monkeypatch):
    assert impl._DEFAULT_NOPE_CONFIG == impl._DEFAULT_ROPE_CONFIG == {"BLOCK_SIZE": 1024, "threads": 128}
    assert launches(impl, monkeypatch) == [(128, 1024, 128), (64, 1024, 128)]


def test_fp32_winner_replays_per_launch(impl, monkeypatch):
    apply_winner(impl, dict(FP32_WINNER), torch.float32, strict=True)
    assert launches(impl, monkeypatch) == [(128, 512, 64), (64, 1024, 64)]       # nope, then rope


def test_identical_winners_of_the_other_dtypes_replay(impl, monkeypatch):
    apply_winner(impl, {"nope_BLOCK_SIZE": 1024, "nope_threads": 64,
                        "rope_BLOCK_SIZE": 1024, "rope_threads": 64}, torch.float16, strict=True)
    assert launches(impl, monkeypatch) == [(128, 1024, 64), (64, 1024, 64)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int8])
@pytest.mark.parametrize("winner", [None, FP32_WINNER])
def test_destindex_stays_correct(impl, dtype, winner):
    from tilebench.data.tensors import GENERATORS
    ref_mod = importlib.import_module(f"{OP}.impl_torch")
    if winner:
        apply_winner(impl, dict(winner), dtype, strict=True)
    inputs = GENERATORS["destindex"](batch_size=1, seq_len=2048, kv_nope_head_num=12, kv_rope_head_num=1,
                                     kv_nope_head_dim=128, kv_rope_head_dim=64, dtype=dtype)
    out, ref = impl.run(*inputs), ref_mod.run(*inputs)
    for o, r in zip(out, ref):
        assert torch.equal(o, r)
