import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"ROPE_GROUP_SIZE": 16, "threads": 64}
_last_autotune_config: dict = {}


def rope_config():
    return [
        dict(ROPE_GROUP_SIZE=gs, threads=nt)
        for gs in [1, 2, 4, 8, 16]
        for nt in [64, 128, 256]
    ]


@tilelang.autotune(configs=rope_config(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def rope_embedding(Q, cos, sin, dtype, seq_len,
                   ROPE_GROUP_SIZE: int = 16,
                   threads: int = 64):
    M, n_heads, half_dim = T.const("M, n_heads, half_dim")

    Q: T.Tensor((M, n_heads, 2, half_dim), dtype)
    cos: T.Tensor((seq_len, half_dim), dtype)
    sin: T.Tensor((seq_len, half_dim), dtype)

    with T.Kernel(M, T.ceildiv(n_heads, ROPE_GROUP_SIZE), threads=threads) as (row_id, group_id):
        q1_tile = T.alloc_fragment((ROPE_GROUP_SIZE, half_dim), dtype)
        q2_tile = T.alloc_fragment((ROPE_GROUP_SIZE, half_dim), dtype)
        cos_tile = T.alloc_fragment((half_dim,), dtype)
        sin_tile = T.alloc_fragment((half_dim,), dtype)

        seq_idx = row_id % seq_len
        head_start = group_id * ROPE_GROUP_SIZE
        T.annotate_safe_value({Q: 0.0})
        T.copy(cos[seq_idx, 0:half_dim], cos_tile)
        T.copy(sin[seq_idx, 0:half_dim], sin_tile)
        T.copy(Q[row_id, head_start:head_start + ROPE_GROUP_SIZE, 0, 0:half_dim], q1_tile)
        T.copy(Q[row_id, head_start:head_start + ROPE_GROUP_SIZE, 1, 0:half_dim], q2_tile)

        for h, d in T.Parallel(ROPE_GROUP_SIZE, half_dim):
            q1 = q1_tile[h, d]
            q2 = q2_tile[h, d]
            c = cos_tile[d]
            s = sin_tile[d]
            q1_tile[h, d] = q1 * c - q2 * s
            q2_tile[h, d] = q2 * c + q1 * s

        T.copy(q1_tile, Q[row_id, head_start:head_start + ROPE_GROUP_SIZE, 0, 0:half_dim])
        T.copy(q2_tile, Q[row_id, head_start:head_start + ROPE_GROUP_SIZE, 1, 0:half_dim])


def run(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        block_size: int = None, autotune: bool = False):
    output = q.clone().contiguous()
    batch, seq_len, n_heads, head_dim = output.shape
    half_dim = head_dim // 2
    dtype = str(output.dtype).removeprefix("torch.")

    output_view = output.view(batch * seq_len, n_heads, 2, half_dim)
    cos_view = cos.view(seq_len, half_dim)
    sin_view = sin.view(seq_len, half_dim)

    if autotune:
        tmp = output.clone()
        tmp_view = tmp.view(batch * seq_len, n_heads, 2, half_dim)
        with set_autotune_inputs(tmp_view, cos_view, sin_view):
            kernel = rope_embedding.compile(
                tmp_view, cos_view, sin_view,
                dtype=dtype,
                seq_len=seq_len,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(output_view, cos_view, sin_view)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        rope_embedding(
            output_view, cos_view, sin_view,
            dtype=dtype,
            seq_len=seq_len,
            ROPE_GROUP_SIZE=cfg["ROPE_GROUP_SIZE"],
            threads=cfg["threads"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
