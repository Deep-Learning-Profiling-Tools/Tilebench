import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs


_DEFAULT_CONFIG = {
    "KV_BLOCK_M": 32,
    "BLOCK_M": 32,
    "BLOCK_D": 16,
    "threads": 128,
}
_last_autotune_config: dict = {}


def out_kernel_configs():
    return [
        dict(BLOCK_M=bm, BLOCK_D=bd, threads=nt)
        for bm in [16, 32, 64]
        for bd in [8, 16, 32]
        for nt in [32, 64, 128]
    ]


def _phi(x):
    return T.if_then_else(x > 0.0, x + 1.0, T.exp(x))


@tilelang.jit
def kv_kernel(S, K, V, dtype, BLOCK_M: int = 32, threads: int = 32):
    M, D = T.const("M, D")
    S: T.Tensor((D, D), "float32")
    K: T.Tensor((M, D), dtype)
    V: T.Tensor((M, D), dtype)

    with T.Kernel(D, D, threads=threads) as (pid_d0, pid_d1):
        k_reg = T.alloc_fragment((BLOCK_M, 1), "float32")
        v_reg = T.alloc_fragment((BLOCK_M, 1), "float32")
        prod = T.alloc_fragment((BLOCK_M,), "float32")
        tile_sum = T.alloc_fragment((1,), "float32")
        acc = T.alloc_var("float32", init=0.0)

        for m_start in T.serial(0, M, BLOCK_M):
            T.copy(K[m_start : m_start + BLOCK_M, pid_d0 : pid_d0 + 1], k_reg)
            T.copy(V[m_start : m_start + BLOCK_M, pid_d1 : pid_d1 + 1], v_reg)

            for i in T.Parallel(BLOCK_M):
                prod[i] = _phi(k_reg[i, 0]) * v_reg[i, 0]

            T.reduce_sum(prod, tile_sum, dim=0, clear=True)
            acc = acc + tile_sum[0]

        S[pid_d0, pid_d1] = acc


@tilelang.jit
def z_kernel(Z, K, dtype, BLOCK_M: int = 32, threads: int = 32):
    M, D = T.const("M, D")
    Z: T.Tensor((D,), "float32")
    K: T.Tensor((M, D), dtype)

    with T.Kernel(D, threads=threads) as pid_d:
        k_reg = T.alloc_fragment((BLOCK_M,), "float32")
        phi_reg = T.alloc_fragment((BLOCK_M,), "float32")
        tile_sum = T.alloc_fragment((1,), "float32")
        acc = T.alloc_var("float32", init=0.0)

        T.annotate_safe_value({K: -T.infinity("float32")})

        for m_start in T.serial(0, M, BLOCK_M):
            for i in T.Parallel(BLOCK_M):
                k_reg[i] = T.cast(K[m_start + i, pid_d], "float32")

            for i in T.Parallel(BLOCK_M):
                phi_reg[i] = _phi(k_reg[i])

            T.reduce_sum(phi_reg, tile_sum, dim=0, clear=True)
            acc = acc + tile_sum[0]

        Z[pid_d] = acc


@tilelang.autotune(configs=out_kernel_configs(), rep=100, warmup=20, timeout=60)
@tilelang.jit
def out_kernel(
    O,
    Q,
    S,
    Z,
    dtype,
    eps,
    BLOCK_M: int = 32,
    BLOCK_D: int = 16,
    threads: int = 128,
):
    M, D = T.const("M, D")
    O: T.Tensor((M, D), dtype)
    Q: T.Tensor((M, D), dtype)
    S: T.Tensor((D, D), "float32")
    Z: T.Tensor((D,), "float32")

    with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(D, BLOCK_D), threads=threads) as (pid_m, pid_do):
        m_start = pid_m * BLOCK_M
        d_start = pid_do * BLOCK_D

        q_reg = T.alloc_fragment((BLOCK_M, 1), "float32")
        phi_q = T.alloc_fragment((BLOCK_M,), "float32")
        s_reg = T.alloc_fragment((1, BLOCK_D), "float32")
        numer = T.alloc_fragment((BLOCK_M, BLOCK_D), "float32")
        denom = T.alloc_fragment((BLOCK_M,), "float32")
        out = T.alloc_fragment((BLOCK_M, BLOCK_D), dtype)

        T.fill(numer, 0.0)
        T.fill(denom, 0.0)

        for d_idx in T.serial(D):
            z_val = Z[d_idx]
            T.copy(Q[m_start : m_start + BLOCK_M, d_idx : d_idx + 1], q_reg)
            T.copy(S[d_idx : d_idx + 1, d_start : d_start + BLOCK_D], s_reg)

            for i in T.Parallel(BLOCK_M):
                phi_q[i] = _phi(q_reg[i, 0])
                denom[i] = denom[i] + phi_q[i] * z_val

            for i, j in T.Parallel(BLOCK_M, BLOCK_D):
                numer[i, j] = numer[i, j] + phi_q[i] * s_reg[0, j]

        for i, j in T.Parallel(BLOCK_M, BLOCK_D):
            out[i, j] = T.Cast(dtype, numer[i, j] / (denom[i] + T.cast(eps, "float32")))

        T.copy(out, O[m_start : m_start + BLOCK_M, d_start : d_start + BLOCK_D])


def _launch_kv_z(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, M: int, D: int, KV_BLOCK_M: int):
    S = torch.empty((D, D), device=Q.device, dtype=torch.float32)
    Z = torch.empty((D,), device=Q.device, dtype=torch.float32)

    kv_kernel(
        S, K, V, "float32",
        BLOCK_M=KV_BLOCK_M,
        threads=32,
    )
    z_kernel(
        Z, K, "float32",
        BLOCK_M=KV_BLOCK_M,
        threads=32,
    )
    return S, Z


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float = 1e-6,
        block_size: int = None, autotune: bool = False, **kwargs):
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.shape == K.shape == V.shape
    assert Q.dtype == K.dtype == V.dtype == torch.float32

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    M, D = Q.shape
    O = torch.empty((M, D), device=Q.device, dtype=torch.float32)

    cfg = dict(_DEFAULT_CONFIG)
    S, Z = _launch_kv_z(Q, K, V, M, D, cfg["KV_BLOCK_M"])

    if autotune:
        with set_autotune_inputs(O, Q, S, Z):
            kernel = out_kernel.compile(
                O, Q, S, Z,
                dtype="float32",
                eps=float(eps),
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(O, Q, S, Z)
    else:
        _last_autotune_config.clear()
        BLOCK_M = int(block_size) if block_size is not None else cfg["BLOCK_M"]
        out_kernel(
            O, Q, S, Z, "float32", float(eps),
            BLOCK_M=BLOCK_M,
            BLOCK_D=cfg["BLOCK_D"],
            threads=cfg["threads"],
        )

    return O


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
