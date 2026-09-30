import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
from tilelang.math import next_power_of_2


_DEFAULT_CONFIG = {"threads": 128, "num_stages": 2}
_last_autotune_config: dict = {}


def flash_decode_stage2_configs():
    return [
        dict(threads=nt, num_stages=ns)
        for nt in [32, 64, 128]
        for ns in [2, 3, 4]
    ]


@tilelang.autotune(configs=flash_decode_stage2_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flash_decode_stage2_kernel(
    mid_o,
    mid_o_lse,
    b_seqlen,
    output,
    dtype,
    BLOCK_SEQ: int,
    BLOCK_DMODEL: int,
    threads: int = 128,
    num_stages: int = 2,
):
    batch, head_num, num_blocks, head_dim = T.const("batch, head_num, num_blocks, head_dim")
    mid_o: T.Tensor((batch, head_num, num_blocks, head_dim), dtype)
    mid_o_lse: T.Tensor((batch, head_num, num_blocks), dtype)
    b_seqlen: T.Tensor((batch,), "int32")
    output: T.Tensor((batch, head_num, head_dim), dtype)

    with T.Kernel(batch, head_num, threads=threads) as (cur_batch, cur_head):
        acc = T.alloc_fragment((BLOCK_DMODEL,), "float32")
        tv = T.alloc_fragment((BLOCK_DMODEL,), "float32")
        out = T.alloc_fragment((BLOCK_DMODEL,), dtype)
        sum_exp = T.alloc_var("float32", init=0.0)
        max_logic = T.alloc_var("float32", init=-T.infinity("float32"))
        exp_logic = T.alloc_var("float32")

        T.fill(acc, 0.0)
        T.fill(out, T.Cast(dtype, 0.0))

        cur_batch_seq_len = b_seqlen[cur_batch]
        block_n_size = T.Select(
            cur_batch_seq_len <= 0,
            0,
            (cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ,
        )

        for block_seq_n in T.Pipelined(block_n_size, num_stages=num_stages):
            T.copy(mid_o[cur_batch, cur_head, block_seq_n, 0:BLOCK_DMODEL], tv)
            tlogic = T.cast(mid_o_lse[cur_batch, cur_head, block_seq_n], "float32")
            new_max_logic = T.max(tlogic, max_logic)
            old_scale = T.exp(max_logic - new_max_logic)
            exp_logic = T.exp(tlogic - new_max_logic)

            for d in T.Parallel(BLOCK_DMODEL):
                acc[d] = acc[d] * old_scale + exp_logic * tv[d]

            sum_exp = sum_exp * old_scale + exp_logic
            max_logic = new_max_logic

        if cur_batch_seq_len > 0:
            for d in T.Parallel(BLOCK_DMODEL):
                out[d] = T.cast(acc[d] / sum_exp, dtype)

        T.copy(out, output[cur_batch, cur_head, 0:BLOCK_DMODEL])


def run(mid_o, mid_o_lse, b_seqlen, block_seq_tensor,
        block_size: int = None, autotune: bool = False, **kwargs):
    if isinstance(block_seq_tensor, torch.Tensor):
        block_seq = int(block_seq_tensor.item())
    else:
        block_seq = int(block_seq_tensor)

    mid_o = mid_o.contiguous()
    mid_o_lse = mid_o_lse.contiguous()
    b_seqlen = b_seqlen.contiguous()

    batch, head_num, num_blocks, head_dim = mid_o.shape
    output = torch.empty((batch, head_num, head_dim), dtype=mid_o.dtype, device=mid_o.device)

    dtype = str(mid_o.dtype).removeprefix("torch.")
    block_dmodel = next_power_of_2(head_dim)

    if autotune:
        with set_autotune_inputs(mid_o, mid_o_lse, b_seqlen, output):
            kernel = flash_decode_stage2_kernel.compile(
                mid_o, mid_o_lse, b_seqlen, output,
                dtype=dtype,
                BLOCK_SEQ=block_seq,
                BLOCK_DMODEL=block_dmodel,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(mid_o, mid_o_lse, b_seqlen, output)
    else:
        _last_autotune_config.clear()
        cfg = dict(_DEFAULT_CONFIG)
        flash_decode_stage2_kernel(
            mid_o, mid_o_lse, b_seqlen, output, dtype,
            BLOCK_SEQ=block_seq,
            BLOCK_DMODEL=block_dmodel,
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
