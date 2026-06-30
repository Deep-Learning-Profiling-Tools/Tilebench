import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs


_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "threads": 128,
    "num_stages": 2,
}
_last_autotune_config: dict = {}


def bmm_configs():
    block_m = [32, 64, 128]
    block_n = [32, 64, 128]
    block_k = [32, 64]
    group_size_m = [1, 8]
    threads = [128]
    num_stages = [2]
    return [
        dict(
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
            GROUP_SIZE_M=gs,
            threads=nt,
            num_stages=ns,
        )
        for bm in block_m
        for bn in block_n
        for bk in block_k
        for gs in group_size_m
        for nt in threads
        for ns in num_stages
    ]


@tilelang.autotune(configs=bmm_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def bmm_kernel(
    A,
    B,
    C,
    dtype,
    BLOCK_SIZE_M: int = 64,
    BLOCK_SIZE_N: int = 64,
    BLOCK_SIZE_K: int = 32,
    GROUP_SIZE_M: int = 8,
    threads: int = 128,
    num_stages: int = 2,
):
    BATCH, M, N, K = T.const("BATCH, M, N, K")

    A: T.Tensor((BATCH, M, K), dtype)
    B: T.Tensor((BATCH, K, N), dtype)
    C: T.Tensor((BATCH, M, N), dtype)

    with T.Kernel(
        T.ceildiv(M, BLOCK_SIZE_M),
        T.ceildiv(N, BLOCK_SIZE_N),
        BATCH,
        threads=threads,
    ) as (pid_m, pid_n, pid_b):
        T.use_swizzle(panel_size=GROUP_SIZE_M, enable=True)

        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        A_tile = T.alloc_shared((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype)
        B_tile = T.alloc_shared((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype)
        acc = T.alloc_fragment((BLOCK_SIZE_M, BLOCK_SIZE_N), "float32")

        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K, BLOCK_SIZE_K), num_stages=num_stages):
            T.copy(A[pid_b, start_m, k * BLOCK_SIZE_K], A_tile)
            T.copy(B[pid_b, k * BLOCK_SIZE_K, start_n], B_tile)
            T.gemm(A_tile, B_tile, acc)

        T.copy(acc, C[pid_b, start_m, start_n])


def run(
    A: torch.Tensor,
    B: torch.Tensor,
    BATCH: int,
    M: int,
    N: int,
    K: int,
    block_size: int = 1024,
    autotune: bool = False,
    **kwargs,
):
    A_3d = A.contiguous().view(BATCH, M, K)
    B_3d = B.contiguous().view(BATCH, K, N)
    C_3d = torch.empty((BATCH, M, N), dtype=A.dtype, device=A.device)
    dtype = str(A.dtype).removeprefix("torch.")

    if autotune:
        with set_autotune_inputs(A_3d, B_3d, C_3d):
            tuned_kernel = bmm_kernel.compile(A_3d, B_3d, C_3d, dtype=dtype)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(tuned_kernel.config or {}))
        tuned_kernel(A_3d, B_3d, C_3d)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        bmm_kernel(
            A_3d,
            B_3d,
            C_3d,
            dtype=dtype,
            BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
            GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )

    return C_3d.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
