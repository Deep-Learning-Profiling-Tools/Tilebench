"""bitonic_sort: pad_kernel is not tuned and always launches with the fixed
_PAD_BLOCK, in the default path, the autotune path and when a profiler harness
replays a winner into _DEFAULT_CONFIG; only bitonic_step_kernel takes the
winner. The kernels are replaced by recorders, so the test runs on any host."""
import pytest
import torch

import tilebench.benchmarks.operators.bitonic_sort.impl_triton as bitonic
from tilebench.profiling.replay import apply_winner


class _Recorder:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.log.append((self.name, grid, kwargs.get("BLOCK")))
        return launch


@pytest.fixture
def launches(monkeypatch):
    log = []
    monkeypatch.setattr(bitonic, "pad_kernel", _Recorder(log, "pad"))
    monkeypatch.setattr(bitonic, "bitonic_step_kernel", _Recorder(log, "step"))
    monkeypatch.setattr(bitonic, "_bitonic_step_kernel_autotuned", _Recorder(log, "step_autotuned"))
    monkeypatch.setattr(bitonic, "_DEFAULT_CONFIG", bitonic._DEFAULT_CONFIG)  # restored after the test
    return log


def _run(autotune=False):
    data = torch.zeros(3000)
    bitonic.run(data, data.numel(), autotune=autotune)


def test_pad_block_is_the_builtin_default_block():
    assert bitonic._PAD_BLOCK == 1024 == bitonic._DEFAULT_CONFIG["BLOCK"]


@pytest.mark.parametrize("winner_block", [512, 1024, 2048, 4096])
def test_replayed_winner_changes_only_the_step_kernel(launches, winner_block):
    apply_winner(bitonic, {"BLOCK": winner_block, "num_warps": 8})
    _run()
    pads = [l for l in launches if l[0] == "pad"]
    steps = [l for l in launches if l[0] == "step"]
    assert pads == [("pad", (4,), 1024)]                 # M = 4096
    assert steps and all(l[2] == winner_block and l[1] == (4096 // winner_block,) for l in steps)


@pytest.mark.parametrize("autotune", [False, True])
def test_default_and_autotune_paths_pad_with_1024(launches, autotune):
    _run(autotune)
    assert [l for l in launches if l[0] == "pad"] == [("pad", (4,), 1024)]
