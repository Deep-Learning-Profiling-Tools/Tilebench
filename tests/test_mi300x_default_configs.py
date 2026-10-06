"""batched_matmul / streamk_matmul: the non-autotuned path uses a CDNA3-safe
fixed config only on CDNA3 (the original NVIDIA default needs more than
gfx942's 64 KiB of LDS for fp32). The NVIDIA defaults and the autotune search
spaces are unchanged, and the CDNA3 default is one of the existing candidates.
The architecture is mocked, so the tests run on any host."""
import pytest

import tilebench.benchmarks.operators.batched_matmul.impl_triton as bmm
import tilebench.benchmarks.operators.streamk_matmul.impl_triton as streamk

# The defaults as they were before the CDNA3 fallback (exp/mi300x 5e9a2c29).
BMM_DEFAULT = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
               "GROUPSIZE": 8, "num_warps": 4, "num_stages": 4}
STREAMK_DEFAULT = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8,
                   "num_warps": 8, "num_stages": 3}


@pytest.mark.parametrize("mod", [bmm, streamk])
@pytest.mark.parametrize("arch", ["blackwell", "hopper", None])
def test_non_cdna3_keeps_the_original_default(mod, arch, monkeypatch):
    monkeypatch.setattr(mod, "detect_arch", lambda: arch)
    assert mod._default_config() is mod._DEFAULT_CONFIG


@pytest.mark.parametrize("mod", [bmm, streamk])
def test_cdna3_uses_the_cdna3_default(mod, monkeypatch):
    monkeypatch.setattr(mod, "detect_arch", lambda: "cdna3")
    assert mod._default_config() is mod._CDNA3_DEFAULT_CONFIG


def test_the_nvidia_defaults_are_unchanged():
    assert bmm._DEFAULT_CONFIG == BMM_DEFAULT
    assert streamk._DEFAULT_CONFIG == STREAMK_DEFAULT


def test_bmm_cdna3_default_only_lowers_the_pipeline_depth_and_is_a_candidate():
    assert bmm._CDNA3_DEFAULT_CONFIG == {**BMM_DEFAULT, "num_stages": 3}
    cands = {(c.kwargs["BLOCK_SIZE_M"], c.kwargs["BLOCK_SIZE_N"], c.kwargs["BLOCK_SIZE_K"],
              c.kwargs["GROUPSIZE"], c.num_warps, c.num_stages) for c in bmm._bmm_kernel_autotuned.configs}
    d = bmm._CDNA3_DEFAULT_CONFIG
    assert (d["BLOCK_SIZE_M"], d["BLOCK_SIZE_N"], d["BLOCK_SIZE_K"], d["GROUPSIZE"],
            d["num_warps"], d["num_stages"]) in cands


def test_streamk_cdna3_default_only_lowers_the_k_tile_and_is_a_candidate():
    assert streamk._CDNA3_DEFAULT_CONFIG == {**STREAMK_DEFAULT, "BLOCK_K": 32}
    assert streamk._CDNA3_DEFAULT_CONFIG in streamk._SEARCH_SPACE


def test_the_autotune_search_spaces_are_unchanged():
    expected = [(bm, bn, bk, gs, nw, ns)
                for bm in [32, 64, 128] for bn in [32, 64, 128] for bk in [32, 64]
                for gs in [1, 8] for nw in [4, 8] for ns in [2, 3, 4]
                if (bm * bk + bn * bk) * 4 * ns + bm * bn * 4 <= 220_000]
    got = [(c.kwargs["BLOCK_SIZE_M"], c.kwargs["BLOCK_SIZE_N"], c.kwargs["BLOCK_SIZE_K"],
            c.kwargs["GROUPSIZE"], c.num_warps, c.num_stages) for c in bmm._bmm_kernel_autotuned.configs]
    assert got == expected
    assert streamk._SEARCH_SPACE == [
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8, "num_warps": nw, "num_stages": 3}
        for bm in (64, 128) for bn in (128, 256) for bk in (32, 64) for nw in (4, 8)]


# Precedence of the non-autotuned config: a replayed autotune winner (the
# profiler harness rebinds _DEFAULT_CONFIG through replay.apply_winner) wins
# over the CDNA3 fallback, which wins over the builtin default.
BMM_WINNER = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
              "GROUPSIZE": 1, "num_warps": 4, "num_stages": 2}
STREAMK_WINNER = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8,
                  "num_warps": 4, "num_stages": 3}


@pytest.mark.parametrize("mod,winner", [(bmm, BMM_WINNER), (streamk, STREAMK_WINNER)])
@pytest.mark.parametrize("arch", ["cdna3", "blackwell", "hopper", None])
def test_a_replayed_winner_is_used_as_given(mod, winner, arch, monkeypatch):
    from tilebench.profiling.replay import apply_winner
    monkeypatch.setattr(mod, "detect_arch", lambda: arch)
    monkeypatch.setattr(mod, "_DEFAULT_CONFIG", mod._DEFAULT_CONFIG)  # restored after the test
    apply_winner(mod, dict(winner))
    assert mod._default_config() == winner


@pytest.mark.parametrize("mod", [bmm, streamk])
def test_the_builtin_default_is_the_sentinel(mod):
    assert mod._BUILTIN_DEFAULT_CONFIG is mod._DEFAULT_CONFIG
