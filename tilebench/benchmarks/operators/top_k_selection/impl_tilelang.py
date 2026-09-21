import torch
import tilelang
import tilelang.language as T
from tilelang.math import next_power_of_2
from triton.testing import do_bench


_DEFAULT_CONFIG = {"BLOCK_SIZE": 2048, "threads": 256}
_BLOCKS = (1024, 2048, 4096)
_THREADS = (128, 256)
_autotune_cache: dict = {}
_last_autotune_config: dict = {}


def topk_configs():
    return [
        dict(BLOCK_SIZE=bs, threads=nt)
        for bs in _BLOCKS
        for nt in _THREADS
    ]


@tilelang.jit
def block_topk_kernel(
    n,
    K2,
    dtype,
    BLOCK_SIZE: int = 2048,
    threads: int = 256,
):
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    @T.prim_func
    def main(
        input: T.Tensor((n,), dtype),
        output: T.Tensor((num_blocks, K2), dtype),
    ):
        workspace = T.alloc_shared((BLOCK_SIZE,), dtype)
        neg_inf = -T.infinity(dtype)
        with T.Kernel(num_blocks, threads=threads) as pid:
            T.annotate_safe_value({input: neg_inf})
            for local_idx in T.Parallel(BLOCK_SIZE):
                workspace[local_idx] = input[pid * BLOCK_SIZE + local_idx]

            T.sync_threads()

            for kb in T.unroll(1, BLOCK_SIZE.bit_length(), explicit=True):
                for jj in T.unroll(BLOCK_SIZE.bit_length() - 1, explicit=True):
                    if jj < kb:
                        stage = T.shift_left(1, kb)
                        safe_shift = T.max(kb - 1 - jj, 0)
                        stride = T.shift_left(1, safe_shift)
                        for pair_idx in T.Parallel(BLOCK_SIZE // 2):
                            group = pair_idx // stride
                            offset = pair_idx % stride
                            left = group * (2 * stride) + offset
                            right = left + stride

                            a = workspace[left]
                            b = workspace[right]
                            hi = T.Select(a > b, a, b)
                            lo = T.Select(a > b, b, a)
                            left_desc = (left & stage) == 0

                            workspace[left] = T.Select(left_desc, hi, lo)
                            workspace[right] = T.Select(left_desc, lo, hi)
                        T.sync_threads()

            for local_idx in T.Parallel(K2):
                output[pid, local_idx] = workspace[local_idx]

    return main


def _resolve_block(block_size: int | None, K2: int) -> int:
    requested = int(block_size) if block_size is not None else _DEFAULT_CONFIG["BLOCK_SIZE"]
    min_block = max(requested, 2 * K2)

    for block in _BLOCKS:
        if block >= min_block:
            return block

    raise ValueError(
        f"Unsupported top-k size: k'={K2}. "
        f"Need block >= {2 * K2}, but supported blocks are {_BLOCKS}."
    )


def _run_hierarchy(x: torch.Tensor, k: int, K2: int, cfg: dict) -> torch.Tensor:
    block_size = int(cfg["BLOCK_SIZE"])
    threads = int(cfg["threads"])
    dtype = str(x.dtype).removeprefix("torch.")

    cur = x
    n = x.numel()

    while True:
        nb = (n + block_size - 1) // block_size
        out = torch.empty((nb, K2), device=x.device, dtype=x.dtype)
        kernel = block_topk_kernel(
            n,
            K2,
            dtype,
            BLOCK_SIZE=block_size,
            threads=threads,
        )
        kernel(cur, out)

        if nb == 1:
            return out[0, :k]

        cur = out.reshape(-1)
        n = nb * K2


def _tune_pipeline(x: torch.Tensor, k: int, K2: int) -> dict:
    key = (x.numel(), k, str(x.dtype))
    cached = _autotune_cache.get(key)
    if cached is not None:
        return cached

    best_cfg = None
    best_ms = float("inf")
    for cfg in topk_configs():
        if cfg["BLOCK_SIZE"] < 2 * K2:
            continue
        ms = do_bench(lambda: _run_hierarchy(x, k, K2, cfg), warmup=1, rep=3)
        if ms < best_ms:
            best_cfg = cfg
            best_ms = ms

    if best_cfg is None:
        raise ValueError(
            f"No viable TileLang top-k config for k={k}, k'={K2}. "
            f"Need block >= {2 * K2}, supported blocks are {_BLOCKS}."
        )

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
    K2 = next_power_of_2(k)

    if autotune:
        cfg = _tune_pipeline(input, k, K2)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(cfg))
    else:
        _last_autotune_config.clear()
        cfg = dict(_DEFAULT_CONFIG)
        cfg["BLOCK_SIZE"] = _resolve_block(block_size, K2)

    return _run_hierarchy(input, k, K2, cfg)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
