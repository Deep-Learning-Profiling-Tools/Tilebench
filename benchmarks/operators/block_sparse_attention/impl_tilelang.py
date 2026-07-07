import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs


_DEFAULT_CONFIG = {"threads": 128, "num_stages": 2}
_last_autotune_config: dict = {}


def block_sparse_attention_configs():
    return [
        dict(threads=nt, num_stages=ns)
        for nt in [128, 256]
        for ns in [1, 2]
    ]


@tilelang.autotune(configs=block_sparse_attention_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def block_sparse_attention_kernel(
    batch,
    num_heads,
    num_kv_heads,
    total_seq_len,
    head_dim,
    csr_row_len,
    csr_col_len,
    num_layout,
    softmax_scale,
    dtype,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_D: int = 128,
    NUM_D_BLOCKS: int = 1,
    threads: int = 128,
    num_stages: int = 2,
):
    accum_dtype = "float32"
    total_d = BLOCK_D * NUM_D_BLOCKS
    csr_row_storage = num_layout * csr_row_len
    csr_col_storage = num_layout * csr_col_len

    @T.prim_func
    def main(
        Q: T.Tensor((batch, num_heads, total_seq_len, head_dim), dtype),
        K: T.Tensor((batch, num_kv_heads, total_seq_len, head_dim), dtype),
        V: T.Tensor((batch, num_kv_heads, total_seq_len, head_dim), dtype),
        row_indices: T.Tensor((csr_row_storage,), "int32"),
        col_indices: T.Tensor((csr_col_storage,), "int32"),
        O: T.Tensor((batch, num_heads, total_seq_len, head_dim), dtype),
    ):
        with T.Kernel(T.ceildiv(total_seq_len, BLOCK_M), batch * num_heads, threads=threads) as (
            start_m,
            off_bh,
        ):
            off_h = off_bh % num_heads
            off_b = off_bh // num_heads
            head_groups = num_heads // num_kv_heads
            off_h_kv = off_h // head_groups
            layout_h = off_h % num_layout

            q_shared = T.alloc_shared((BLOCK_M, total_d), dtype)
            k_shared = T.alloc_shared((BLOCK_N, total_d), dtype)
            v_shared = T.alloc_shared((BLOCK_N, total_d), dtype)
            o_shared = T.alloc_shared((BLOCK_M, total_d), dtype)

            qk = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            p_cast = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
            acc = T.alloc_fragment((BLOCK_M, total_d), accum_dtype)
            scores_max = T.alloc_fragment((BLOCK_M,), accum_dtype)
            scores_max_prev = T.alloc_fragment((BLOCK_M,), accum_dtype)
            scores_scale = T.alloc_fragment((BLOCK_M,), accum_dtype)
            scores_sum = T.alloc_fragment((BLOCK_M,), accum_dtype)
            logsum = T.alloc_fragment((BLOCK_M,), accum_dtype)

            T.copy(
                Q[
                    off_b,
                    off_h,
                    start_m * BLOCK_M : (start_m + 1) * BLOCK_M,
                    0:total_d,
                ],
                q_shared,
            )
            T.fill(acc, 0.0)
            T.fill(logsum, 0.0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            start_l = T.alloc_var("int32")
            end_l = T.alloc_var("int32")
            l = T.alloc_var("int32")
            col_idx = T.alloc_var("int32")
            start_n = T.alloc_var("int32")

            start_l = row_indices[layout_h * csr_row_len + start_m]
            end_l = row_indices[layout_h * csr_row_len + start_m + 1]
            l = start_l

            while l < end_l:
                col_idx = col_indices[layout_h * csr_col_len + l]
                start_n = col_idx * BLOCK_N

                T.copy(K[off_b, off_h_kv, start_n : start_n + BLOCK_N, 0:total_d], k_shared)

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    qk[i, j] = T.if_then_else(
                        start_m * BLOCK_M + i >= start_n + j,
                        0.0,
                        -T.infinity(accum_dtype),
                    )

                T.gemm(q_shared, k_shared, qk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    qk[i, j] *= T.cast(softmax_scale, accum_dtype)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(qk, scores_max, dim=1, clear=False)

                for i in T.Parallel(BLOCK_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    qk[i, j] = T.if_then_else(
                        start_m * BLOCK_M + i >= start_n + j,
                        T.exp(qk[i, j] - scores_max[i]),
                        0.0,
                    )

                T.reduce_sum(qk, scores_sum, dim=1)
                for i in T.Parallel(BLOCK_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    scores_scale[i] = T.if_then_else(logsum[i] > 0.0, 1.0 / logsum[i], 0.0)

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    qk[i, j] *= scores_scale[i]

                T.copy(qk, p_cast)

                for i, j in T.Parallel(BLOCK_M, total_d):
                    acc[i, j] *= T.if_then_else(
                        logsum[i] > 0.0,
                        (logsum[i] - scores_sum[i]) * scores_scale[i],
                        1.0,
                    )

                T.copy(V[off_b, off_h_kv, start_n : start_n + BLOCK_N, 0:total_d], v_shared)
                T.gemm(p_cast, v_shared, acc, policy=T.GemmWarpPolicy.FullRow)

                l += 1

            T.copy(acc, o_shared)
            T.copy(
                o_shared,
                O[
                    off_b,
                    off_h,
                    start_m * BLOCK_M : (start_m + 1) * BLOCK_M,
                    0:total_d,
                ],
            )

    return main


def run(
    Q,
    K,
    V,
    layout_csr_row_indices,
    layout_csr_col_indices,
    layout_csr_row_stride_h,
    layout_csr_col_stride_h,
    num_layout,
    softmax_scale,
    num_heads,
    num_kv_heads,
    total_seq_len,
    BLOCK_M,
    EVEN_M,
    BLOCK_N,
    EVEN_N,
    BLOCK_D,
    NUM_D_BLOCKS,
    block_size: int = None,
    autotune: bool = False,
    **kwargs,
):
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.ndim == K.ndim == V.ndim == 4
    assert total_seq_len == Q.shape[2]
    assert Q.shape[0] == K.shape[0] == V.shape[0]
    assert Q.shape[1] == num_heads
    assert K.shape[1] == V.shape[1] == num_kv_heads
    assert Q.shape[-1] == K.shape[-1] == V.shape[-1]
    assert Q.shape[-1] == BLOCK_D * NUM_D_BLOCKS
    assert num_heads % num_kv_heads == 0

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    layout_csr_row_indices = layout_csr_row_indices.contiguous()
    layout_csr_col_indices = layout_csr_col_indices.contiguous()

    batch = Q.shape[0]
    head_dim = Q.shape[-1]
    dtype = str(Q.dtype).removeprefix("torch.")
    out = torch.empty((batch, num_heads, total_seq_len, head_dim), device=Q.device, dtype=Q.dtype)

    cfg = dict(_DEFAULT_CONFIG)
    if block_size is not None:
        cfg["threads"] = int(block_size)

    if autotune:
        with set_autotune_inputs(Q, K, V, layout_csr_row_indices, layout_csr_col_indices, out):
            kernel = block_sparse_attention_kernel(
                batch,
                num_heads,
                num_kv_heads,
                total_seq_len,
                head_dim,
                int(layout_csr_row_stride_h),
                int(layout_csr_col_stride_h),
                int(num_layout),
                float(softmax_scale),
                dtype,
                BLOCK_M=int(BLOCK_M),
                BLOCK_N=int(BLOCK_N),
                BLOCK_D=int(BLOCK_D),
                NUM_D_BLOCKS=int(NUM_D_BLOCKS),
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(Q, K, V, layout_csr_row_indices, layout_csr_col_indices, out)
    else:
        _last_autotune_config.clear()
        kernel = block_sparse_attention_kernel(
            batch,
            num_heads,
            num_kv_heads,
            total_seq_len,
            head_dim,
            int(layout_csr_row_stride_h),
            int(layout_csr_col_stride_h),
            int(num_layout),
            float(softmax_scale),
            dtype,
            BLOCK_M=int(BLOCK_M),
            BLOCK_N=int(BLOCK_N),
            BLOCK_D=int(BLOCK_D),
            NUM_D_BLOCKS=int(NUM_D_BLOCKS),
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )
        kernel(Q, K, V, layout_csr_row_indices, layout_csr_col_indices, out)

    return out


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
