import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs


_DEFAULT_CONFIG = {"BLOCK": 1024, "threads": 128}
_last_autotune_config: dict = {}


def bitonic_step_configs():
    return [
        dict(BLOCK=bs, threads=nt)
        for bs in [512, 1024, 2048, 4096]
        for nt in [64, 128, 256]
    ]


def _next_pow2(n: int) -> int:
    return 1 << ((n - 1).bit_length()) if n > 1 else 1


@tilelang.jit
def pad_kernel(data, work, dtype, BLOCK: int = 1024, threads: int = 128):
    N, M = T.const("N, M")
    data: T.Tensor((N,), dtype)
    work: T.Tensor((M,), dtype)

    inf = T.infinity(dtype)
    with T.Kernel(T.ceildiv(M, BLOCK), threads=threads) as pid:
        for local_idx in T.Parallel(BLOCK):
            offs = pid * BLOCK + local_idx
            if offs < M:
                work[offs] = T.if_then_else(offs < N, data[offs], inf)


@tilelang.autotune(configs=bitonic_step_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def bitonic_step_kernel(M, dtype, BLOCK: int = 1024, threads: int = 128):
    @T.prim_func
    def main(
        work: T.Tensor((M,), dtype),
        k: T.int32,
        j: T.int32,
    ):
        zero = T.cast(0, dtype)
        with T.Kernel(T.ceildiv(M, BLOCK), threads=threads) as pid:
            for local_idx in T.Parallel(BLOCK):
                offs = pid * BLOCK + local_idx
                ixj = T.bitwise_xor(offs, j)
                active = (ixj > offs) and (ixj < M) and (offs < M)

                a = T.if_then_else(active, work[offs], zero)
                b = T.if_then_else(active, work[ixj], zero)
                ascending = T.bitwise_and(offs, k) == 0
                swap = T.if_then_else(ascending, a > b, a < b)
                new_a = T.if_then_else(swap, b, a)
                new_b = T.if_then_else(swap, a, b)

                if active:
                    work[offs] = new_a
                    work[ixj] = new_b

    return main


def run(data: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    if N <= 1:
        return data.clone()

    dtype = str(data.dtype).removeprefix("torch.")
    M = _next_pow2(N)
    work = torch.empty((M,), device=data.device, dtype=data.dtype)

    cfg = dict(_DEFAULT_CONFIG)
    pad_kernel(
        data, work, dtype,
        BLOCK=cfg["BLOCK"],
        threads=cfg["threads"],
    )

    if autotune:
        with set_autotune_inputs(work, 2, 1):
            kernel = bitonic_step_kernel(M, dtype)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
    else:
        _last_autotune_config.clear()
        kernel = bitonic_step_kernel(
            M, dtype,
            BLOCK=cfg["BLOCK"],
            threads=cfg["threads"],
        )

    k = 2
    while k <= M:
        j = k // 2
        while j > 0:
            kernel(work, k, j)
            j //= 2
        k *= 2

    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
