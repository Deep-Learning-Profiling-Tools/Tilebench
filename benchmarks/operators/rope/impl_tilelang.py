import torch  
import tilelang  
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"ROPE_GROUP_SIZE": 4, "threads": 128}
_last_autotune_config: dict = {}


def rope_config():
    ROPE_GROUP_SIZE = [1, 2, 4, 8, 16]
    threads = [64, 128, 256]
    return [
        dict(ROPE_GROUP_SIZE=gs, threads=nt)
        for gs in ROPE_GROUP_SIZE
        for nt in threads
    ]
@tilelang.autotune(configs=rope_config(), warmup = 20, rep = 100, timeout = 60)
@tilelang.jit
def rope_embedding(Q, cos, sin, dtype, seq_len, 
                   ROPE_GROUP_SIZE: int = 4, 
                   threads: int = 128):
    M, n_heads, head_dim = T.const("M, n_heads, head_dim")

    Q: T.Tensor((M, n_heads, head_dim), dtype)
    cos: T.Tensor((seq_len, head_dim // 2), dtype)
    sin: T.Tensor((seq_len, head_dim // 2), dtype)
    with T.Kernel(M, T.ceildiv(n_heads, ROPE_GROUP_SIZE), threads=threads) as (row_id, group_id):
        #one CTA basicaly does rotation for part of a token
        #ROPE_GROUP_SIZE * head_dim elemetns per CTA -- need to split this among registers
        q1_tile = T.alloc_fragment((head_dim // 2,), dtype)
        q2_tile = T.alloc_fragment((head_dim // 2,), dtype)
        cos_tile = T.alloc_fragment((head_dim // 2,), dtype)
        sin_tile = T.alloc_fragment((head_dim // 2,), dtype)
        seq_idx = row_id % seq_len
        T.copy(cos[seq_idx, 0: head_dim // 2], cos_tile)
        T.copy(sin[seq_idx, 0: head_dim // 2], sin_tile)
        head_start = ROPE_GROUP_SIZE * group_id
        for k in T.Serial(0, ROPE_GROUP_SIZE):
            head_id = head_start + k
            if head_id < n_heads:
                T.copy(Q[row_id, head_id, 0: head_dim // 2], q1_tile)
                T.copy(Q[row_id, head_id, head_dim // 2: head_dim], q2_tile)
                for i in T.Parallel(head_dim // 2):
                    q1 = q1_tile[i]
                    q2 = q2_tile[i]
                    c = cos_tile[i]
                    s = sin_tile[i]

                    q1_tile[i] = q1 * c - q2 * s
                    q2_tile[i] = q2 * c + q1 * s
                
                T.copy(q1_tile, Q[row_id, head_id, 0: head_dim // 2])
                T.copy(q2_tile, Q[row_id, head_id, head_dim // 2: head_dim])


def run(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        block_size: int = None, autotune: bool = False):
    output = q.clone().contiguous()
    batch, seq_len, n_heads, head_dim = output.shape
    half_dim = head_dim // 2
    dtype = str(output.dtype).removeprefix("torch.")

    output_view = output.view(batch * seq_len, n_heads, head_dim)
    cos_view = cos.view(seq_len, half_dim)
    sin_view = sin.view(seq_len, half_dim)

    if autotune:
        tmp = output.clone()
        tmp_view = tmp.view(batch * seq_len, n_heads, head_dim)
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
