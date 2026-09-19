import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

_CHUNK = 4096
_LOG_CHUNK = 12  # log2(4096)


@ct.kernel
def _topk_stage_kernel(in_arr, out_arr, K: ConstInt):
    pid = ct.bid(0)
    x = ct.load(in_arr, index=(pid,), shape=(_CHUNK,),
                padding_mode=ct.PaddingMode.NEG_INF)

    # Iterative bitonic sort -> descending output.
    # Outer loop: sequence size doubles (k_log = 1 .. LOG_CHUNK).
    # Inner loop: comparison distance halves (j_log = k_log-1 .. 0).
    for k_log in range(1, _LOG_CHUNK + 1):
        for j_log in range(k_log - 1, -1, -1):
            j = 1 << j_log
            block_len = j << 1
            nrows = _CHUNK // block_len

            a = ct.reshape(x, (nrows, block_len))
            left = ct.extract(a, (0, 0), shape=(nrows, j))
            right = ct.extract(a, (0, j), shape=(nrows, j))

            mn = ct.minimum(left, right)
            mx = ct.maximum(left, right)

            row_idx = ct.arange(nrows, dtype=ct.int32)
            shift_amt = k_log - j_log - 1
            seq_idx = row_idx >> shift_amt  # shift by 0 is identity
            descending = (seq_idx & 1) == 0  # bool, shape (nrows,)
            descending = ct.reshape(descending, (nrows, 1))
            descending = ct.broadcast_to(descending, (nrows, j))

            new_left = ct.where(descending, mx, mn)
            new_right = ct.where(descending, mn, mx)

            combined = ct.cat((new_left, new_right), axis=1)
            x = ct.reshape(combined, (_CHUNK,))

    top_k = ct.extract(x, (0,), shape=(K,))
    ct.store(out_arr, index=(pid,), tile=top_k)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    assert input.is_cuda
    assert input.dtype == torch.float32
    output = torch.empty(k, dtype=torch.float32, device=input.device)
    CHUNK = _CHUNK
    stream = torch.cuda.current_stream()

    stages = []
    cur_size = N
    while True:
        num_blocks = (cur_size + CHUNK - 1) // CHUNK
        if num_blocks <= 1:
            stages.append((cur_size, k, 1, True))
            break
        next_size = num_blocks * k
        stages.append((cur_size, next_size, num_blocks, False))
        cur_size = next_size

    if len(stages) == 1:
        ct.launch(stream, (1, 1, 1), _topk_stage_kernel, (input, output, k))
    else:
        max_inter = max(s[1] for s in stages if not s[3])
        buf_a = torch.empty(max_inter, dtype=torch.float32, device=input.device)
        buf_b = torch.empty(max_inter, dtype=torch.float32, device=input.device)

        src = input
        use_a_dst = True
        for in_size, out_size, num_blocks, is_last in stages:
            if is_last:
                dst = output
            else:
                dst = buf_a if use_a_dst else buf_b
                use_a_dst = not use_a_dst
            ct.launch(stream, (num_blocks, 1, 1), _topk_stage_kernel, (src, dst, k))
            src = dst

    _LAST_CFG.clear()
    _LAST_CFG.update({"CHUNK": CHUNK})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
