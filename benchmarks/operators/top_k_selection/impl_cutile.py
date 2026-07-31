from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(block=2048, occupancy=8)

_BLOCKS = (1024, 2048, 4096)
_OCC_SPACE = [SimpleNamespace(occupancy=occ) for occ in (4, 8, 16)]

_last_config: dict = {}
_autotune_cache: dict = {}


def _next_pow2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _make_bitonic_stages(B: int):
    stages = []
    for kb in range(1, B.bit_length()):
        ksz = 1 << kb
        for jj in range(kb):
            j = 1 << (kb - 1 - jj)
            G = B // (2 * j)
            stages.append((G, j, ksz))
    return tuple(stages)


_STAGES_1024 = _make_bitonic_stages(1024)
_STAGES_2048 = _make_bitonic_stages(2048)
_STAGES_4096 = _make_bitonic_stages(4096)


def _sort_desc_1024(x):
    for G, j, ksz in ct.static_iter(_STAGES_1024):
        x3 = ct.reshape(x, (G, 2, j))

        a = ct.extract(x3, index=(0, 0, 0), shape=(G, 1, j))
        b = ct.extract(x3, index=(0, 1, 0), shape=(G, 1, j))

        lo = ct.minimum(a, b)
        hi = ct.maximum(a, b)

        m = ((ct.arange(G, dtype=ct.int32) * (2 * j)) & ksz) == 0
        m3 = ct.reshape(m, (G, 1, 1))

        first = ct.where(m3, hi, lo)
        second = ct.where(m3, lo, hi)

        x = ct.reshape(ct.cat((first, second), axis=1), (1024,))

    return x


def _sort_desc_2048(x):
    for G, j, ksz in ct.static_iter(_STAGES_2048):
        x3 = ct.reshape(x, (G, 2, j))

        a = ct.extract(x3, index=(0, 0, 0), shape=(G, 1, j))
        b = ct.extract(x3, index=(0, 1, 0), shape=(G, 1, j))

        lo = ct.minimum(a, b)
        hi = ct.maximum(a, b)

        m = ((ct.arange(G, dtype=ct.int32) * (2 * j)) & ksz) == 0
        m3 = ct.reshape(m, (G, 1, 1))

        first = ct.where(m3, hi, lo)
        second = ct.where(m3, lo, hi)

        x = ct.reshape(ct.cat((first, second), axis=1), (2048,))

    return x


def _sort_desc_4096(x):
    for G, j, ksz in ct.static_iter(_STAGES_4096):
        x3 = ct.reshape(x, (G, 2, j))

        a = ct.extract(x3, index=(0, 0, 0), shape=(G, 1, j))
        b = ct.extract(x3, index=(0, 1, 0), shape=(G, 1, j))

        lo = ct.minimum(a, b)
        hi = ct.maximum(a, b)

        m = ((ct.arange(G, dtype=ct.int32) * (2 * j)) & ksz) == 0
        m3 = ct.reshape(m, (G, 1, 1))

        first = ct.where(m3, hi, lo)
        second = ct.where(m3, lo, hi)

        x = ct.reshape(ct.cat((first, second), axis=1), (4096,))

    return x


@ct.kernel
def block_topk_kernel_b1024(inp, out2d, K2: ConstInt):
    bid = ct.bid(0)

    x = ct.load(
        inp,
        index=(bid,),
        shape=(1024,),
        padding_mode=ct.PaddingMode.NEG_INF,
    )

    x = _sort_desc_1024(x)

    top = ct.extract(x, index=(0,), shape=(K2,))
    ct.store(out2d, index=(bid, 0), tile=ct.reshape(top, (1, K2)))


@ct.kernel
def block_topk_kernel_b2048(inp, out2d, K2: ConstInt):
    bid = ct.bid(0)

    x = ct.load(
        inp,
        index=(bid,),
        shape=(2048,),
        padding_mode=ct.PaddingMode.NEG_INF,
    )

    x = _sort_desc_2048(x)

    top = ct.extract(x, index=(0,), shape=(K2,))
    ct.store(out2d, index=(bid, 0), tile=ct.reshape(top, (1, K2)))


@ct.kernel
def block_topk_kernel_b4096(inp, out2d, K2: ConstInt):
    bid = ct.bid(0)

    x = ct.load(
        inp,
        index=(bid,),
        shape=(4096,),
        padding_mode=ct.PaddingMode.NEG_INF,
    )

    x = _sort_desc_4096(x)

    top = ct.extract(x, index=(0,), shape=(K2,))
    ct.store(out2d, index=(bid, 0), tile=ct.reshape(top, (1, K2)))


KERNELS = {
    1024: block_topk_kernel_b1024,
    2048: block_topk_kernel_b2048,
    4096: block_topk_kernel_b4096,
}

_tuners = {B: CutileAutotuner(KERNELS[B]) for B in _BLOCKS}


def _resolve_block(block_size: int | None, K2: int) -> int:
    requested = int(block_size) if block_size is not None else _DEFAULT_CONFIG.block
    min_block = max(requested, 2 * K2)

    for B in _BLOCKS:
        if B >= min_block:
            return B

    raise ValueError(
        f"Unsupported top-k size: k'={K2}. "
        f"Need block >= {2 * K2}, but supported blocks are {_BLOCKS}."
    )


def _run_hierarchy(x: torch.Tensor, k: int, K2: int, cfg, stream) -> torch.Tensor:
    B = int(cfg.block)
    kernel = _tuners[B].kernel_with_hints(occupancy=cfg.occupancy)

    cur = x
    n = x.numel()

    while True:
        nb = ct.cdiv(n, B)
        out = torch.empty((nb, K2), device=x.device, dtype=x.dtype)

        ct.launch(stream, (nb, 1, 1), kernel, (cur, out, K2))

        if nb == 1:
            return out[0, :k]

        cur = out.reshape(-1)
        n = nb * K2


def _tune(x: torch.Tensor, k: int, K2: int, stream) -> SimpleNamespace:
    key = (x.numel(), K2, str(x.dtype))
    cached = _autotune_cache.get(key)
    if cached is not None:
        return cached

    n = x.numel()
    best_us = None
    best_cfg = None

    for B in _BLOCKS:
        if B < 2 * K2:
            continue

        nb = ct.cdiv(n, B)
        scratch = torch.empty((nb, K2), device=x.device, dtype=x.dtype)

        result = ct.tune.exhaustive_search(
            _OCC_SPACE,
            stream,
            grid_fn=lambda cfg: (nb, 1, 1),
            kernel=_tuners[B].kernel,
            args_fn=lambda cfg: (x, scratch, K2),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )

        mean_us = result.best.mean_us
        if best_us is None or mean_us < best_us:
            best_us = mean_us
            best_cfg = SimpleNamespace(
                block=B,
                occupancy=result.best.config.occupancy,
            )

    if best_cfg is None:
        raise ValueError(
            f"No viable cuTile top-k config for k={k}, k'={K2}. "
            f"Need some block >= {2 * K2}, supported blocks are {_BLOCKS}."
        )

    _autotune_cache[key] = best_cfg
    return best_cfg


def run(
    input: torch.Tensor,
    N: int,
    k: int,
    block_size: int = None,
    autotune: bool = False,
    **kwargs,
):
    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.float32
    assert 1 <= k <= N

    x = input.contiguous()
    K2 = _next_pow2(k)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tune(x, k, K2, stream)
    else:
        cfg = SimpleNamespace(
            block=_resolve_block(block_size, K2),
            occupancy=_DEFAULT_CONFIG.occupancy,
        )

    _last_config.clear()
    _last_config.update(
        {
            "block": int(cfg.block),
            "occupancy": int(cfg.occupancy),
            "K2": int(K2),
        }
    )

    return _run_hierarchy(x, k, K2, cfg, stream)


def get_last_config() -> dict | None:
    return dict(_last_config) if _last_config else None
