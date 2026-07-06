"""Top-k via hierarchical block-topk tournament reduction (cuTile).

Same two-phase structure as impl_triton.py: each CTA computes its
BLOCK-wide chunk's local top-k' (k' = next_pow2(k)) sorted descending,
writes k' candidates to a (num_blocks, k') buffer, and the same kernel
is re-launched on the flattened candidates until one block remains.
Configs keep BLOCK >= 2*k' so every level shrinks.

The block-local top-k' is an in-tile bitonic sorting network built from
ct.reshape / ct.extract / ct.minimum / ct.maximum / ct.cat (cuTile has
no sort/topk primitive — Triton uses tl.topk, which lowers to the same
kind of in-register network). The network's log²(B) compare-exchange
stages need tile SHAPES that change per stage, and cuTile 1.3 offers no
in-language way to unroll them statically: in-kernel `for range(...)`
and `while` both compile to runtime IR loops (loop variables cannot
feed constant reshape shapes), and iterating a Python tuple is rejected
("cannot create constant from value of type tuple"). The kernels are
therefore GENERATED as straight-line source per (B, k') and imported
from a temp module (the tracer needs inspect-able source, so exec()
alone is not enough). This is itself an expressibility data point:
static metaprogramming that Triton gets from tl.constexpr + built-in
primitives requires source generation in cuTile.

Compile cost is one-time per (B, k') variant (~2-7 s), paid inside the
engine's warmup, and cached for the process lifetime.
"""
import importlib.util
import tempfile
from types import SimpleNamespace

import cuda.tile as ct
import torch
# Timing-only import: keeps the tuning clock identical to impl_triton.py.
from triton.testing import do_bench

_DEFAULT_CONFIG = SimpleNamespace(block=2048, occupancy=8)

# Tile dim is 1:1 with impl_triton.py's search space; occupancy is
# cuTile's scheduling knob like Triton's num_warps. Configs with
# block < 2*k' are filtered out at tune time.
_SEARCH_SPACE = [
    SimpleNamespace(block=bs, occupancy=occ)
    for bs in (1024, 2048, 4096)
    for occ in (4, 8, 16)
]

_last_autotune_config: dict = {}
_autotune_cache: dict = {}

# Generated-module machinery: one straight-line kernel per (B, K2),
# written into a temp dir that lives for the process lifetime.
_gen_dir = tempfile.TemporaryDirectory(prefix="cutile_topk_gen_")
_kernel_cache: dict = {}
_hinted_cache: dict = {}


def _gen_source(B: int, K2: int) -> str:
    """Emit the fully unrolled descending bitonic sort + top-k' extract."""
    L = ["import cuda.tile as ct", "", "", "@ct.kernel",
         "def block_topk_kernel(inp, out2d):",
         "    bid = ct.bid(0)",
         f"    x = ct.load(inp, index=(bid,), shape=({B},), "
         "padding_mode=ct.PaddingMode.NEG_INF)"]
    for kb in range(1, B.bit_length()):
        ksz = 1 << kb
        for jj in range(kb):
            j = 1 << (kb - 1 - jj)
            G = B // (2 * j)
            L += [f"    x3 = ct.reshape(x, ({G}, 2, {j}))",
                  f"    a = ct.extract(x3, index=(0, 0, 0), shape=({G}, 1, {j}))",
                  f"    b = ct.extract(x3, index=(0, 1, 0), shape=({G}, 1, {j}))",
                  "    lo = ct.minimum(a, b)",
                  "    hi = ct.maximum(a, b)",
                  f"    m = ((ct.arange({G}, dtype=ct.int32) * {2 * j}) & {ksz}) == 0",
                  f"    m3 = ct.reshape(m, ({G}, 1, 1))",
                  "    first = ct.where(m3, hi, lo)",
                  "    second = ct.where(m3, lo, hi)",
                  f"    x = ct.reshape(ct.cat((first, second), axis=1), ({B},))"]
    L += [f"    top = ct.extract(x, index=(0,), shape=({K2},))",
          f"    ct.store(out2d, index=(bid, 0), tile=ct.reshape(top, (1, {K2})))",
          ""]
    return "\n".join(L)


def _make_kernel(B: int, K2: int):
    key = (B, K2)
    if key not in _kernel_cache:
        name = f"cutile_topk_gen_b{B}_k{K2}"
        path = f"{_gen_dir.name}/{name}.py"
        with open(path, "w") as f:
            f.write(_gen_source(B, K2))
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _kernel_cache[key] = mod.block_topk_kernel
    return _kernel_cache[key]


def _kernel_with_hints(B: int, K2: int, occupancy: int):
    """replace_hints result cached per (B, K2, occupancy) — required for
    CUDA-graph stability, same rationale as core.cutile_autotune."""
    key = (B, K2, occupancy)
    if key not in _hinted_cache:
        _hinted_cache[key] = _make_kernel(B, K2).replace_hints(occupancy=occupancy)
    return _hinted_cache[key]


def _next_pow2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _run_hierarchy(x: torch.Tensor, k: int, K2: int, cfg, stream) -> torch.Tensor:
    B = cfg.block
    kern = _kernel_with_hints(B, K2, cfg.occupancy)
    cur, n = x, x.numel()
    while True:
        nb = (n + B - 1) // B
        out = torch.empty((nb, K2), device=x.device, dtype=x.dtype)
        ct.launch(stream, (nb, 1, 1), kern, (cur, out))
        if nb == 1:
            return out[0, :k]
        cur, n = out.reshape(-1), nb * K2


def _tune_pipeline(x: torch.Tensor, k: int, K2: int, stream):
    key = (x.numel(), k)
    cached = _autotune_cache.get(key)
    if cached is not None:
        return cached
    best_cfg, best_ms = None, float("inf")
    for cfg in _SEARCH_SPACE:
        if cfg.block < 2 * K2:
            continue   # level sizes would not shrink
        ms = do_bench(lambda: _run_hierarchy(x, k, K2, cfg, stream),
                      warmup=1, rep=3)
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
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tune_pipeline(input, k, K2, stream)
        _last_autotune_config.clear()
        _last_autotune_config.update({"block": cfg.block,
                                      "occupancy": cfg.occupancy})
    else:
        blk = int(block_size) if block_size is not None else _DEFAULT_CONFIG.block
        # progress guarantee: candidates must shrink between levels
        cfg = SimpleNamespace(block=max(blk, 2 * K2),
                              occupancy=_DEFAULT_CONFIG.occupancy)

    return _run_hierarchy(input, k, K2, cfg, stream)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
