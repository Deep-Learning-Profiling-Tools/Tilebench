import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs


_DEFAULT_CONFIG = {"BLOCK_M": 128, "BLOCK_N": 64, "threads": 256, "num_stages": 3}
_last_autotune_config: dict = {}


def flash_attention_configs():
    return [
        dict(BLOCK_M=bm, BLOCK_N=bn, threads=nt, num_stages=ns)
        for bm in [64, 128]
        for bn in [32, 64, 128]
        for nt in [64, 128, 256]
        for ns in [2, 3, 4]
    ]


@tilelang.autotune(configs=flash_attention_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def flash_attention_kernel(
    batch,
    heads,
    seq_len,
    dim,
    dtype,
    is_causal,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    threads: int = 256,
    num_stages: int = 4,
):
    qk_scale = (1.0 / dim) ** 0.5 * 1.44269504
    accum_dtype = "float32"

    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, seq_len, dim), dtype),
        K: T.Tensor((batch, heads, seq_len, dim), dtype),
        V: T.Tensor((batch, heads, seq_len, dim), dtype),
        O: T.Tensor((batch, heads, seq_len, dim), dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, BLOCK_M), heads, batch, threads=threads) as (pid_m, pid_h, pid_b):
            q_shared = T.alloc_shared((BLOCK_M, dim), dtype)
            k_shared = T.alloc_shared((BLOCK_N, dim), dtype)
            v_shared = T.alloc_shared((BLOCK_N, dim), dtype)

            qk = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            qk_tmem = T.alloc_tmem((BLOCK_M, BLOCK_N), accum_dtype)
            p_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
            acc = T.alloc_fragment((BLOCK_M, dim), accum_dtype)
            pv = T.alloc_fragment((BLOCK_M, dim), accum_dtype)
            pv_tmem = T.alloc_tmem((BLOCK_M, dim), accum_dtype)
            qk_mbar = T.alloc_barrier(1)
            pv_mbar = T.alloc_barrier(1)
            m_i = T.alloc_fragment((BLOCK_M,), accum_dtype)
            m_ij = T.alloc_fragment((BLOCK_M,), accum_dtype)
            alpha = T.alloc_fragment((BLOCK_M,), accum_dtype)
            l_ij = T.alloc_fragment((BLOCK_M,), accum_dtype)
            l_i = T.alloc_fragment((BLOCK_M,), accum_dtype)

            T.copy(Q[pid_b, pid_h, pid_m * BLOCK_M : (pid_m + 1) * BLOCK_M, :], q_shared)
            for i, j in T.Parallel(BLOCK_M, dim):
                q_shared[i, j] = T.cast(T.cast(q_shared[i, j], accum_dtype) * qk_scale, dtype)
            T.fill(acc, 0)
            T.fill(l_i, 0)
            T.fill(m_i, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_len, BLOCK_N), T.ceildiv((pid_m + 1) * BLOCK_M, BLOCK_N))
                if is_causal
                else T.ceildiv(seq_len, BLOCK_N)
            )

            for k_tile in T.serial(loop_range):
                T.copy(K[pid_b, pid_h, k_tile * BLOCK_N : (k_tile + 1) * BLOCK_N, :], k_shared)
                T.gemm(
                    q_shared,
                    k_shared,
                    qk_tmem,
                    transpose_B=True,
                    mbar=qk_mbar,
                    clear_accum=True,
                )
                T.copy(qk_tmem, qk)
                if is_causal:
                    for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                        qk[i, j] = T.Select(
                            pid_m * BLOCK_M + i >= k_tile * BLOCK_N + j,
                            qk[i, j],
                            -T.infinity(accum_dtype),
                        )
                else:
                    for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                        qk[i, j] = T.Select(
                            k_tile * BLOCK_N + j >= seq_len,
                            -T.infinity(accum_dtype),
                            qk[i, j],
                        )

                T.fill(m_ij, -T.infinity(accum_dtype))
                T.reduce_max(qk, m_ij, dim=1, clear=False)
                for i in T.Parallel(BLOCK_M):
                    m_ij[i] = T.max(m_i[i], m_ij[i])
                    alpha[i] = T.exp2(m_i[i] - m_ij[i])

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    qk[i, j] = T.exp2(qk[i, j] - m_ij[i])

                T.reduce_sum(qk, l_ij, dim=1)
                for i in T.Parallel(BLOCK_M):
                    l_i[i] = l_i[i] * alpha[i] + l_ij[i]

                T.copy(qk, p_shared)
                for i, j in T.Parallel(BLOCK_M, dim):
                    acc[i, j] *= alpha[i]

                T.copy(V[pid_b, pid_h, k_tile * BLOCK_N : (k_tile + 1) * BLOCK_N, :], v_shared)
                T.gemm(p_shared, v_shared, pv_tmem, mbar=pv_mbar, clear_accum=True)
                T.copy(pv_tmem, pv)
                for i, j in T.Parallel(BLOCK_M, dim):
                    acc[i, j] += pv[i, j]
                T.copy(m_ij, m_i)

            for i, j in T.Parallel(BLOCK_M, dim):
                acc[i, j] /= l_i[i]

            T.copy(acc, O[pid_b, pid_h, pid_m * BLOCK_M : (pid_m + 1) * BLOCK_M, :])

    return main


def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
        autotune: bool = False, **kwargs):
    assert q.shape == k.shape == v.shape
    assert q.ndim == 4

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch, heads, seq_len, dim = q.shape
    o = torch.empty_like(q)
    dtype = str(q.dtype).removeprefix("torch.")

    if autotune:
        with set_autotune_inputs(q, k, v, o):
            kernel = flash_attention_kernel(batch, heads, seq_len, dim, dtype, bool(causal))
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(q, k, v, o)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        kernel = flash_attention_kernel(
            batch,
            heads,
            seq_len,
            dim,
            dtype,
            bool(causal),
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )
        kernel(q, k, v, o)

    return o


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
