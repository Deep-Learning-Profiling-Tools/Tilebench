import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

_CHUNK = 2048
_LOG_CHUNK = 11  # log2(2048)


@ct.kernel
def _topk_stage_kernel(in_arr, out_arr, K: ConstInt):
    pid = ct.bid(0)
    x = ct.load(in_arr, index=(pid,), shape=(_CHUNK,),
                padding_mode=ct.PaddingMode.NEG_INF)

    # Bitonic sort, descending.
    for stage_log in range(1, _LOG_CHUNK + 1):
        for sub_log in range(stage_log - 1, -1, -1):
            sub_j = 1 << sub_log
            block_len = 2 * sub_j
            nrows = _CHUNK // block_len
            shift = stage_log - sub_log - 1

            a = ct.reshape(x, (nrows, block_len))
            left = ct.extract(a, (0, 0), shape=(nrows, sub_j))
            right = ct.extract(a, (0, sub_j), shape=(nrows, sub_j))

            mn = ct.minimum(left, right)
            mx = ct.maximum(left, right)

            row_idx = ct.arange(nrows, dtype=np.int32)
            if shift > 0:
                shifted = row_idx >> shift
            else:
                shifted = row_idx
            parity = shifted & 1
            is_desc_1d = parity == 0
            is_desc_2d = ct.reshape(is_desc_1d, (nrows, 1))
            is_desc_b = ct.broadcast_to(is_desc_2d, (nrows, sub_j))

            new_left = ct.where(is_desc_b, mx, mn)
            new_right = ct.where(is_desc_b, mn, mx)

            new_a = ct.cat((new_left, new_right), axis=1)
            x = ct.reshape(new_a, (_CHUNK,))

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
