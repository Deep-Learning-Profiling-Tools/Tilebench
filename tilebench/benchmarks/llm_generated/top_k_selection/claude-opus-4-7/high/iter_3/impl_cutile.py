import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

_CHUNK = 4096
_LOG_CHUNK = 12


@ct.kernel(occupancy=2)
def _topk_stage_kernel(in_arr, out_arr, K: ConstInt):
    pid = ct.bid(0)
    x = ct.load(in_arr, index=(pid,), shape=(_CHUNK,),
                padding_mode=ct.PaddingMode.NEG_INF)

    # Bitonic sort, descending overall. 2D-reshape compare-swap:
    # View x as (nrows, block_len=2*sub_j); left half vs right half.
    # Direction (descending vs ascending) is determined by which outer
    # stage_k-group each row belongs to (group parity).
    for stage_log in range(1, _LOG_CHUNK + 1):
        for sub_log in range(stage_log - 1, -1, -1):
            sub_j = 1 << sub_log
            block_len = 2 * sub_j
            nrows = _CHUNK // block_len

            a = ct.reshape(x, (nrows, block_len))
            left = ct.extract(a, (0, 0), shape=(nrows, sub_j))
            right = ct.extract(a, (0, sub_j), shape=(nrows, sub_j))

            mn = ct.minimum(left, right)
            mx = ct.maximum(left, right)

            # log2(npairs) = stage_log - sub_log - 1
            shift = stage_log - sub_log - 1
            row_idx = ct.arange(nrows, dtype=np.int32)
            if shift == 0:
                group_idx = row_idx
            else:
                group_idx = ct.bitwise_rshift(row_idx, shift)
            is_desc = (group_idx & 1) == 0
            is_desc_2d = ct.reshape(is_desc, (nrows, 1))

            new_left = ct.where(is_desc_2d, mx, mn)
            new_right = ct.where(is_desc_2d, mn, mx)

            new_a = ct.cat((new_left, new_right), axis=1)
            x = ct.reshape(new_a, (_CHUNK,))

    top_k = ct.extract(x, (0,), shape=(K,))
    ct.store(out_arr, index=(pid,), tile=top_k)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    assert input.is_cuda
    assert input.dtype == torch.float32
    output = torch.empty(k, dtype=torch.float32, device=input.device)

    CHUNK = _CHUNK

    stages = []
    cur_size = N
    while True:
        num_blocks = (cur_size + CHUNK - 1) // CHUNK
        if num_blocks <= 1:
            stages.append((cur_size, k, 1))
            break
        next_size = num_blocks * k
        stages.append((cur_size, next_size, num_blocks))
        cur_size = next_size

    max_inter = k
    for _, out_size, num_blocks in stages:
        if num_blocks > 1:
            max_inter = max(max_inter, out_size)

    if len(stages) > 1:
        buf_a = torch.empty(max_inter, dtype=torch.float32, device=input.device)
        buf_b = torch.empty(max_inter, dtype=torch.float32, device=input.device)
    else:
        buf_a = buf_b = None

    stream = torch.cuda.current_stream()

    src = input
    use_a = True
    for in_size, out_size, num_blocks in stages:
        if num_blocks == 1:
            dst = output
        else:
            base = buf_a if use_a else buf_b
            dst = base[:out_size]

        if src.numel() != in_size:
            src_view = src[:in_size]
        else:
            src_view = src

        ct.launch(stream, (num_blocks, 1, 1), _topk_stage_kernel, (src_view, dst, k))
        src = dst
        use_a = not use_a

    _LAST_CFG.clear()
    _LAST_CFG.update({"CHUNK": CHUNK, "occupancy": 2})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
