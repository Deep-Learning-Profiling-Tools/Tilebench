import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"num_warps": 4, "num_stages": 2}


@triton.jit
def fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,
    Mid_O_LogExpSum,
    Out,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    stride_obs,
    stride_oh,
    stride_od,
    head_dim,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os, mask=offs_d < head_dim, other=0.0)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)

        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    tl.store(Out + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp, mask=offs_d < head_dim)
    return


_fwd_kernel_flash_decode_stage2_autotuned = triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [1, 2, 4]
        for ns in [2, 3, 4]
    ],
    key=["head_dim"],
)(fwd_kernel_flash_decode_stage2)


def run(mid_o, mid_o_lse, b_seqlen, block_seq_tensor, block_size: int = None, autotune: bool = False):

    if isinstance(block_seq_tensor, torch.Tensor):
        block_seq = block_seq_tensor.item()
    else:
        block_seq = block_seq_tensor


    batch, head_num, num_blocks, head_dim = mid_o.shape


    output = torch.empty((batch, head_num, head_dim), dtype=mid_o.dtype, device=mid_o.device)


    BLOCK_DMODEL = triton.next_power_of_2(head_dim)


    grid = (batch, head_num)

    if autotune:
        _fwd_kernel_flash_decode_stage2_autotuned[grid](
            B_Seqlen=b_seqlen,
            Mid_O=mid_o,
            Mid_O_LogExpSum=mid_o_lse,
            Out=output,
            stride_mid_ob=mid_o.stride(0),
            stride_mid_oh=mid_o.stride(1),
            stride_mid_os=mid_o.stride(2),
            stride_mid_od=mid_o.stride(3),
            stride_mid_o_eb=mid_o_lse.stride(0),
            stride_mid_o_eh=mid_o_lse.stride(1),
            stride_mid_o_es=mid_o_lse.stride(2),
            stride_obs=output.stride(0),
            stride_oh=output.stride(1),
            stride_od=output.stride(2),
            head_dim=head_dim,
            BLOCK_SEQ=block_seq,
            BLOCK_DMODEL=BLOCK_DMODEL,
        )
    else:
        cfg = _DEFAULT_CONFIG
        fwd_kernel_flash_decode_stage2[grid](
            B_Seqlen=b_seqlen,
            Mid_O=mid_o,
            Mid_O_LogExpSum=mid_o_lse,
            Out=output,

            stride_mid_ob=mid_o.stride(0),
            stride_mid_oh=mid_o.stride(1),
            stride_mid_os=mid_o.stride(2),
            stride_mid_od=mid_o.stride(3),

            stride_mid_o_eb=mid_o_lse.stride(0),
            stride_mid_o_eh=mid_o_lse.stride(1),
            stride_mid_o_es=mid_o_lse.stride(2),

            stride_obs=output.stride(0),
            stride_oh=output.stride(1),
            stride_od=output.stride(2),

            head_dim=head_dim,
            BLOCK_SEQ=block_seq,
            BLOCK_DMODEL=BLOCK_DMODEL,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return output


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Profile Flash Decode Stage 2 cuTile Kernel")
    parser.add_argument("-b", "--batch", type=int, default=1, help="Batch size")
    parser.add_argument("-h_q", "--heads", type=int, default=32, help="Number of Query Heads")
    parser.add_argument("-s", "--seq-len", type=int, default=4096, help="Sequence Length (Context Length)")
    parser.add_argument("-d", "--head-dim", type=int, default=128, help="Head Dimension")
    parser.add_argument("--block-seq", type=int, default=128, help="Stage 1 Block Size (Split size)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "bfloat16"], help="Data type")

    args = parser.parse_args()


    device = "cuda"
    dtype = getattr(torch, args.dtype)

    batch = args.batch
    heads = args.heads
    seq_len = args.seq_len
    head_dim = args.head_dim
    block_seq = args.block_seq

    print(f"Profiling Config: Batch={batch}, Heads={heads}, SeqLen={seq_len}, Dim={head_dim}, BlockSeq={block_seq}, Dtype={dtype}")


    num_blocks = (seq_len + block_seq - 1) // block_seq


    b_seqlen = torch.full((batch,), seq_len, dtype=torch.int32, device=device)


    mid_o = torch.randn((batch, heads, num_blocks, head_dim), dtype=dtype, device=device)


    mid_o_lse = torch.randn((batch, heads, num_blocks), dtype=dtype, device=device)


    block_seq_tensor = torch.tensor(block_seq, dtype=torch.int32, device='cpu')


    print("Warming up...")
    for _ in range(10):
        run(mid_o, mid_o_lse, b_seqlen, block_seq_tensor)
    torch.cuda.synchronize()


    print("Starting Profile Run...")


    torch.cuda.nvtx.range_push("FlashDecodeStage2_Triton")
    run(mid_o, mid_o_lse, b_seqlen, block_seq_tensor)
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    print("Done.")

def get_last_config() -> dict | None:
    cfg = getattr(_fwd_kernel_flash_decode_stage2_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"num_warps": cfg.num_warps, "num_stages": cfg.num_stages}
