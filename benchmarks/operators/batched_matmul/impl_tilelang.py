import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs


_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUPSIZE": 8,
    "threads": 128,
    "num_stages": 2,
}
_last_autotune_config: dict = {}


def bmm_configs():
    block_m = [32, 64, 128]
    block_n = [32, 64, 128]
    block_k = [32, 64]
    group_size_m = [1, 8]
    threads = [128, 256]
    num_stages = [2, 3, 4]

    def fits_triton_smem_budget(bm, bn, bk, ns):
        return (bm * bk + bn * bk) * 4 * ns + bm * bn * 4 <= 220_000

    def runtime_timeout_prone(bm, bn, bk, gs, nt, ns):
        # Leave compile-time failures to TileLang's autotuner. Only prune
        # shapes observed to benchmark-timeout / poison the CUDA context at
        # BATCH=32, M=N=K=352.
        if bm == 32 and bk == 32:
            return True
        if bm == 32 and bk == 64 and bn in (64, 128):
            return True
        if bm == 128 and bn in (32, 64):
            return True
        if bm == 64 and bn == 64:
            return True
        return False

    return [
        dict(
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
            GROUPSIZE=gs,
            threads=nt,
            num_stages=ns,
        )
        for bm in block_m
        for bn in block_n
        for bk in block_k
        for gs in group_size_m
        for nt in threads
        for ns in num_stages
        if fits_triton_smem_budget(bm, bn, bk, ns)
        if not runtime_timeout_prone(bm, bn, bk, gs, nt, ns)
    ]


@tilelang.autotune(configs=bmm_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False},
)
def bmm_kernel(
    A,
    B,
    C,
    dtype,
    BLOCK_SIZE_M: int = 64,
    BLOCK_SIZE_N: int = 64,
    BLOCK_SIZE_K: int = 32,
    GROUPSIZE: int = 8,
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
        T.use_swizzle(panel_size=GROUPSIZE, order="row", enable=GROUPSIZE > 1)

        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        A_tile = T.alloc_shared((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype)
        B_tile = T.alloc_shared((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype)
        acc = T.alloc_fragment((BLOCK_SIZE_M, BLOCK_SIZE_N), "float32")
        use_tmem = dtype != "float32"
        if use_tmem:
            acc_tmem = T.alloc_tmem((BLOCK_SIZE_M, BLOCK_SIZE_N), "float32")
            mbar = T.alloc_barrier(1)
        else:
            T.clear(acc)

        for k in T.Pipelined(T.ceildiv(K, BLOCK_SIZE_K), num_stages=num_stages):
            T.copy(A[pid_b, start_m, k * BLOCK_SIZE_K], A_tile)
            T.copy(B[pid_b, k * BLOCK_SIZE_K, start_n], B_tile)
            if use_tmem:
                T.gemm(A_tile, B_tile, acc_tmem, mbar=mbar, clear_accum=k == 0)
                T.sync_threads()
            else:
                T.gemm(A_tile, B_tile, acc)

        if use_tmem:
            T.sync_threads()
            T.copy(acc_tmem, acc)
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
            GROUPSIZE=cfg["GROUPSIZE"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )

    return C_3d.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
