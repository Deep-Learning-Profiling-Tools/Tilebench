"""Top-k via hierarchical block-topk tournament reduction: each CTA writes
its chunk's sorted top-k' (k' = next_power_of_2(k)) candidates via tl.topk,
the kernel re-launches on the candidates until one block remains.
Configs keep BLOCK_SIZE >= 2*k' so every level shrinks."""
import torch
import triton
import triton.language as tl
from triton.testing import do_bench


_DEFAULT_CONFIG = {
    "BLOCK_SIZE": 2048,
    "num_warps": 8,
}

# Tile dim mirrors impl_cutile.py's block values; num_warps is Triton's
# scheduling knob. BLOCK_SIZE < 2*k' is skipped at tune time.
_SEARCH_SPACE = [
    {"BLOCK_SIZE": bs, "num_warps": nw}
    for bs in (1024, 2048, 4096)
    for nw in (4, 8)
]

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
    """Time the whole level hierarchy per config (the level structure —
    grid sizes and buffer shapes between launches — depends on BLOCK_SIZE,
    which triton.autotune cannot express). do_bench warmup=1/rep=3 is the
    repo-wide autotune budget."""
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
    K2 = triton.next_power_of_2(k)

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
