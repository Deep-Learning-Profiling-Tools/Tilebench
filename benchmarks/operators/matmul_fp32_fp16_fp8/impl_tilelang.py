import torch 
import tilelang 
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIGS = {
    torch.float32: {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8, "threads": 256, "num_stages": 3,
    },
    torch.float16: {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8, "threads": 256, "num_stages": 3,
    },
    torch.float8_e4m3fn: {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8, "threads": 256, "num_stages": 3,
    },
    torch.float8_e5m2: {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8, "threads": 256, "num_stages": 3,
    },
}

_last_autotune_config: dict = {}

def matmul_configs():
    BLOCK_SIZE_M = [128, 256]
    BLOCK_SIZE_N = [128, 256]
    BLOCK_SIZE_K = [128, 256]
    GROUP_SIZE_M = [64, 128]
    threads = [128, 256]
    stages = [3]
    return [
        dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, 
             GROUP_SIZE_M=gs, threads=nt, num_stages=ns)
        for bm in BLOCK_SIZE_M
        for bn in BLOCK_SIZE_N
        for bk in BLOCK_SIZE_K
        for gs in GROUP_SIZE_M
        for nt in threads
        for ns in stages
    ]

@tilelang.autotune(configs=matmul_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit

def matmul_kernel(
    a, b, c, dtype,
    BLOCK_SIZE_M: int = 128,
    BLOCK_SIZE_N: int = 128,
    BLOCK_SIZE_K: int = 32, 
    GROUP_SIZE_M: int = 8,
    threads: int = 256,
    num_stages: int = 3,
):

    M, K, N = T.const("M, K, N")

    a: T.Tensor((M, K), dtype)
    b: T.Tensor((K, N), dtype)
    c: T.Tensor((M, N), dtype)

    with T.Kernel(T.ceildiv(M, BLOCK_SIZE_M), 
                  T.ceildiv(N, BLOCK_SIZE_N), threads=threads) as (pid_m, pid_n):
        T.use_swizzle(panel_size=GROUP_SIZE_M, enable = True)
        # C tile is shape [BLOCK_SIZE_M, BLOCK_SIZE_N]
        # each index [i, j] in C tile needs whole row / col in A and B
        # BLOCK_SIZE_K is used to compute elements in BLOCK along A / B
        # what do we want in SMEM? each thread computes part of the tile
        # each thread needs access to what similar elements?
        # Most likely the A and B tile that we are @ then adding to acc
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        a_tile = T.alloc_shared((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype)
        b_tile = T.alloc_shared((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype)
        acc = T.alloc_fragment((BLOCK_SIZE_M, BLOCK_SIZE_N), "float32")
        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(K, BLOCK_SIZE_K), num_stages=num_stages):
            T.copy(a[start_m, k * BLOCK_SIZE_K], a_tile)
            T.copy(b[k * BLOCK_SIZE_K, start_n], b_tile)
            T.gemm(a_tile, b_tile, acc)

        T.copy(acc, c[start_m, start_n])


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    a = a.contiguous()
    b = b.contiguous()
    M, _ = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    dtype = str(a.dtype).removeprefix("torch.")

    if autotune:
        with set_autotune_inputs(a, b, c):
            tuned_kernel = matmul_kernel.compile(a, b, c, dtype=dtype)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(tuned_kernel.config or {}))
        tuned_kernel(a, b, c)
    else:
        _last_autotune_config.clear()
        if a.dtype not in _DEFAULT_CONFIGS:
            raise ValueError(f"No default config for dtype {a.dtype}")
        cfg = _DEFAULT_CONFIGS[a.dtype]
        matmul_kernel(
            a, b, c,
            dtype=dtype,
            BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
            GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )

    return c


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
