import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs


_DEFAULT_KV_CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_K": 32,
    "threads": 128,
    "num_stages": 3,
}


_DEFAULT_OUT_CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_K": 32,
    "threads": 128,
    "num_stages": 3,
}


_TILE_SHAPES = [
    (bm, bn, bk)
    for bm in [32, 64, 128]
    for bn in [32, 64, 128]
    for bk in [32, 64]
]


_last_autotune_config: dict = {}


def kv_kernel_configs():
    return [
        dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, threads=nw * 32, num_stages=ns)
        for bm, bn, bk in _TILE_SHAPES
        for nw in [4, 8]
        for ns in [2, 3]
    ]


def out_kernel_configs():
    return [
        dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, threads=nw * 32, num_stages=ns)
        for bm, bn, bk in _TILE_SHAPES
        for nw in [4, 8]
        for ns in [2, 3]
    ]


def _phi(x):
    return T.Select(x > 0.0, x + 1.0, T.exp(x))


@tilelang.jit
def phi_kernel(Y, X, dtype, BLOCK_M: int = 32, BLOCK_D: int = 32, threads: int = 128):
    M, D = T.const("M, D")
    Y: T.Tensor((M, D), dtype)
    X: T.Tensor((M, D), dtype)

    with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(D, BLOCK_D), threads=threads) as (pid_m, pid_d):
        m_start = pid_m * BLOCK_M
        d_start = pid_d * BLOCK_D

        x_frag = T.alloc_fragment((BLOCK_M, BLOCK_D), "float32")
        y_frag = T.alloc_fragment((BLOCK_M, BLOCK_D), dtype)

        T.annotate_safe_value({X: 0.0})
        T.copy(X[m_start, d_start], x_frag)

        for i, j in T.Parallel(BLOCK_M, BLOCK_D):
            y_frag[i, j] = T.cast(_phi(x_frag[i, j]), dtype)

        T.copy(y_frag, Y[m_start, d_start])


@tilelang.autotune(configs=kv_kernel_configs(), rep=100, warmup=20, timeout=60)
@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def kv_kernel(
    S,
    PhiK,
    V,
    mma_dtype,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
    threads: int = 128,
    num_stages: int = 3,
):
    M, D = T.const("M, D")
    S: T.Tensor((D, D), "float32")
    PhiK: T.Tensor((M, D), "float32")
    V: T.Tensor((M, D), "float32")

    with T.Kernel(T.ceildiv(D, BLOCK_M), T.ceildiv(D, BLOCK_N), threads=threads) as (pid_m, pid_n):
        d_m = pid_m * BLOCK_M
        d_n = pid_n * BLOCK_N

        k_shared = T.alloc_shared((BLOCK_M, BLOCK_K), mma_dtype)
        v_shared = T.alloc_shared((BLOCK_N, BLOCK_K), mma_dtype)
        acc = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")

        T.annotate_safe_value({PhiK: 0.0, V: 0.0})
        T.clear(acc)

        for k_tile in T.Pipelined(T.ceildiv(M, BLOCK_K), num_stages=num_stages):
            m_start = k_tile * BLOCK_K

            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                k_shared[i, j] = T.cast(PhiK[m_start + j, d_m + i], mma_dtype)

            for i, j in T.Parallel(BLOCK_N, BLOCK_K):
                v_shared[i, j] = T.cast(V[m_start + j, d_n + i], mma_dtype)

            T.gemm(k_shared, v_shared, acc, transpose_B=True)

        T.copy(acc, S[d_m, d_n])


@tilelang.jit
def z_kernel(Z, PhiK, dtype, BLOCK_M: int = 32, BLOCK_D: int = 32, threads: int = 128):
    M, D = T.const("M, D")
    Z: T.Tensor((D,), "float32")
    PhiK: T.Tensor((M, D), dtype)

    with T.Kernel(T.ceildiv(D, BLOCK_D), threads=threads) as pid_d:
        d_start = pid_d * BLOCK_D

        k_frag = T.alloc_fragment((BLOCK_M, BLOCK_D), "float32")
        tile_sum = T.alloc_fragment((BLOCK_D,), "float32")
        acc = T.alloc_fragment((BLOCK_D,), "float32")

        T.annotate_safe_value({PhiK: 0.0})
        T.clear(acc)

        for m_start in T.serial(0, M, BLOCK_M):
            T.copy(PhiK[m_start, d_start], k_frag)
            T.reduce_sum(k_frag, tile_sum, dim=0, clear=True)

            for j in T.Parallel(BLOCK_D):
                acc[j] = acc[j] + tile_sum[j]

        T.copy(acc, Z[d_start])


@tilelang.autotune(configs=out_kernel_configs(), rep=100, warmup=20, timeout=60)
@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def out_kernel(
    O,
    PhiQ,
    S,
    Z,
    mma_dtype,
    eps,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
    threads: int = 128,
    num_stages: int = 3,
):
    M, D = T.const("M, D")
    O: T.Tensor((M, D), "float32")
    PhiQ: T.Tensor((M, D), "float32")
    S: T.Tensor((D, D), "float32")
    Z: T.Tensor((D,), "float32")

    with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(D, BLOCK_N), threads=threads) as (pid_m, pid_n):
        m_start = pid_m * BLOCK_M
        d_start = pid_n * BLOCK_N

        q_frag = T.alloc_fragment((BLOCK_M, BLOCK_K), "float32")
        z_frag = T.alloc_fragment((BLOCK_K,), "float32")
        q_shared = T.alloc_shared((BLOCK_M, BLOCK_K), mma_dtype)
        s_shared = T.alloc_shared((BLOCK_N, BLOCK_K), mma_dtype)
        numer = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
        denom = T.alloc_fragment((BLOCK_M,), "float32")
        denom_prod = T.alloc_fragment((BLOCK_M, BLOCK_K), "float32")
        denom_delta = T.alloc_fragment((BLOCK_M,), "float32")
        out = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")

        T.annotate_safe_value({PhiQ: 0.0, S: 0.0, Z: 0.0})
        T.clear(numer)
        T.clear(denom)

        for k_tile in T.Pipelined(T.ceildiv(D, BLOCK_K), num_stages=num_stages):
            k_start = k_tile * BLOCK_K

            T.copy(PhiQ[m_start, k_start], q_frag)

            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                q_shared[i, j] = T.cast(q_frag[i, j], mma_dtype)

            for i, j in T.Parallel(BLOCK_N, BLOCK_K):
                s_shared[i, j] = T.cast(S[k_start + j, d_start + i], mma_dtype)

            T.gemm(q_shared, s_shared, numer, transpose_B=True)

            T.copy(Z[k_start], z_frag)
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                denom_prod[i, j] = q_frag[i, j] * z_frag[j]

            T.reduce_sum(denom_prod, denom_delta, dim=1, clear=True)
            for i in T.Parallel(BLOCK_M):
                denom[i] = denom[i] + denom_delta[i]

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            out[i, j] = numer[i, j] / (denom[i] + T.cast(eps, "float32"))

        T.copy(out, O[m_start, d_start])


def _launch_phi(Q: torch.Tensor, K: torch.Tensor):
    PhiQ = torch.empty_like(Q)
    PhiK = torch.empty_like(K)

    phi_kernel(PhiQ, Q, "float32", BLOCK_M=32, BLOCK_D=32, threads=128)
    phi_kernel(PhiK, K, "float32", BLOCK_M=32, BLOCK_D=32, threads=128)
    return PhiQ, PhiK


def _launch_z(PhiK: torch.Tensor):
    D = PhiK.shape[1]
    Z = torch.empty((D,), device=PhiK.device, dtype=torch.float32)

    z_kernel(Z, PhiK, "float32", BLOCK_M=32, BLOCK_D=32, threads=128)
    return Z


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float = 1e-6,
        block_size: int = None, autotune: bool = False, **kwargs):
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.shape == K.shape == V.shape
    assert Q.dtype == K.dtype == V.dtype == torch.float32

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    M, D = Q.shape
    S = torch.empty((D, D), device=Q.device, dtype=torch.float32)
    O = torch.empty((M, D), device=Q.device, dtype=torch.float32)
    mma_dtype = T.tfloat32

    PhiQ, PhiK = _launch_phi(Q, K)

    if autotune:
        with set_autotune_inputs(S, PhiK, V):
            compiled_kv = kv_kernel.compile(
                S, PhiK, V,
                mma_dtype=mma_dtype,
            )
        kv_config = dict(compiled_kv.config or {})
        compiled_kv(S, PhiK, V)
    else:
        kv_config = {}
        kv_cfg = dict(_DEFAULT_KV_CONFIG)
        kv_kernel(
            S, PhiK, V, mma_dtype,
            BLOCK_M=kv_cfg["BLOCK_M"],
            BLOCK_N=kv_cfg["BLOCK_N"],
            BLOCK_K=kv_cfg["BLOCK_K"],
            threads=kv_cfg["threads"],
            num_stages=kv_cfg["num_stages"],
        )

    Z = _launch_z(PhiK)

    if autotune:
        with set_autotune_inputs(O, PhiQ, S, Z):
            compiled_out = out_kernel.compile(
                O, PhiQ, S, Z,
                mma_dtype=mma_dtype,
                eps=float(eps),
            )
        out_config = dict(compiled_out.config or {})
        compiled_out(O, PhiQ, S, Z)
        _last_autotune_config.clear()
        _last_autotune_config.update({
            **{f"kv_{key}": value for key, value in kv_config.items()},
            **{f"out_{key}": value for key, value in out_config.items()},
        })
    else:
        _last_autotune_config.clear()
        out_cfg = dict(_DEFAULT_OUT_CONFIG)
        out_kernel(
            O, PhiQ, S, Z, mma_dtype, float(eps),
            BLOCK_M=out_cfg["BLOCK_M"],
            BLOCK_N=out_cfg["BLOCK_N"],
            BLOCK_K=out_cfg["BLOCK_K"],
            threads=out_cfg["threads"],
            num_stages=out_cfg["num_stages"],
        )

    return O


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
