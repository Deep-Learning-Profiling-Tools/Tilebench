"""Top-k via hierarchical block-topk tournament reduction.

Algorithm (mirrors impl_cutile.py):
  1. Split the input into BLOCK_SIZE-wide chunks; each CTA computes its
     chunk's local top-k' (k' = next_pow2(k)) with `tl.topk` — an
     in-register bitonic top-k — and writes k' sorted-descending
     candidates to a (num_blocks, k') buffer.
  2. Repeat the same kernel on the flattened candidate buffer until one
     block remains; its first k values are the answer.

Levels shrink by a factor of BLOCK_SIZE / k' per round, so configs are
constrained to BLOCK_SIZE >= 2*k' (otherwise the candidate count would
not decrease; e.g. k=1024 with BLOCK_SIZE=1024 recurses forever).
Total data touched ~ N * (1 + k'/B + ...) ≈ 1.1-1.5 passes, and the
launch count drops from the old full bitonic sort's log²(2N)/2 = 210 to
2-6 — the same complexity class as torch.topk's RadixSelect.

The tuner is hand-rolled (streamk pattern): it times the WHOLE level
hierarchy per config on scratch buffers, because the level structure
(grid sizes, buffer shapes) depends on BLOCK_SIZE, which
triton.autotune cannot express (output allocation happens between
launches). do_bench warmup=1/rep=3 is the repo-wide autotune budget.
"""
import torch
import triton
import triton.language as tl
from triton.testing import do_bench


_DEFAULT_CONFIG = {
    "BLOCK_SIZE": 2048,
    "num_warps": 8,
}

# Tile dim is 1:1 with impl_cutile.py's search space; num_warps is
# Triton's scheduling knob like cuTile's occupancy. Configs with
# BLOCK_SIZE < 2*k' are filtered out at tune time.
_SEARCH_SPACE = [
    {"BLOCK_SIZE": bs, "num_warps": nw}
    for bs in (1024, 2048, 4096)
    for nw in (4, 8)
]

# (N, k) -> winning config dict, filled by _tune_pipeline.
_autotune_cache: dict = {}
_last_autotune_config: dict = {}


@triton.jit
def block_topk_kernel(
    input_ptr, out_ptr, n,
    K2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + offs, mask=offs < n, other=-float("inf"))
    if K2 == 1:
        tl.store(out_ptr + pid, tl.max(x))
    else:
        offs_out = pid * K2 + tl.arange(0, K2)
        tl.store(out_ptr + offs_out, tl.topk(x, K2))


def _next_pow2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _run_hierarchy(x: torch.Tensor, k: int, K2: int, cfg: dict) -> torch.Tensor:
    B, nw = cfg["BLOCK_SIZE"], cfg["num_warps"]
    cur, n = x, x.numel()
    while True:
        nb = triton.cdiv(n, B)
        out = torch.empty((nb, K2), device=x.device, dtype=x.dtype)
        block_topk_kernel[(nb,)](cur, out, n, K2=K2, BLOCK_SIZE=B,
                                 num_warps=nw)
        if nb == 1:
            return out[0, :k]
        cur, n = out.view(-1), nb * K2


def _tune_pipeline(x: torch.Tensor, k: int, K2: int) -> dict:
    key = (x.numel(), k)
    cached = _autotune_cache.get(key)
    if cached is not None:
        return cached
    best_cfg, best_ms = None, float("inf")
    for cfg in _SEARCH_SPACE:
        if cfg["BLOCK_SIZE"] < 2 * K2:
            continue   # level sizes would not shrink
        ms = do_bench(lambda: _run_hierarchy(x, k, K2, cfg), warmup=1, rep=3)
        if ms < best_ms:
            best_cfg, best_ms = cfg, ms
    _autotune_cache[key] = best_cfg
    return best_cfg


def run(input: torch.Tensor, N: int, k: int,
        block_size: int = None, autotune: bool = False, **kwargs):
    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.float32
    assert 1 <= k <= N

    input = input.contiguous()
    K2 = _next_pow2(k)

    if autotune:
        cfg = _tune_pipeline(input, k, K2)
        _last_autotune_config.clear()
        _last_autotune_config.update(cfg)
    else:
        cfg = dict(_DEFAULT_CONFIG)
        if block_size is not None:
            cfg["BLOCK_SIZE"] = int(block_size)
        # progress guarantee: candidates must shrink between levels
        cfg["BLOCK_SIZE"] = max(cfg["BLOCK_SIZE"], 2 * K2)

    return _run_hierarchy(input, k, K2, cfg)


def get_last_config() -> dict | None:
    # Hand-rolled tuner (no triton.autotune wrapper to read best_config
    # from) — mutable-dict pattern, same as the cuTile side.
    return dict(_last_autotune_config) if _last_autotune_config else None
