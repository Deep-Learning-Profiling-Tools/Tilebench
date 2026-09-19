import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _bitonic_pass_kernel(src, dst, size, stride, TILE: ConstInt):
    bid = ct.bid(0)
    offs = ct.arange(TILE, dtype=np.int32)
    idx = bid * TILE + offs

    val = ct.load(src, index=(bid,), shape=(TILE,),
                  padding_mode=ct.PaddingMode.NEG_INF)
    partner = idx ^ stride
    other = ct.gather(src, partner, padding_value=-np.inf, check_bounds=True)

    v_max = ct.maximum(val, other)
    v_min = ct.minimum(val, other)

    descending_segment = (idx & size) == 0
    lower_partner = (idx & stride) == 0
    take_max = descending_segment == lower_partner

    out = ct.where(take_max, v_max, v_min)
    ct.store(dst, index=(bid,), tile=out)


@ct.kernel
def _copy_topk_kernel(src, output, TILE: ConstInt):
    bid = ct.bid(0)
    vals = ct.load(src, index=(bid,), shape=(TILE,),
                   padding_mode=ct.PaddingMode.ZERO)
    ct.store(output, index=(bid,), tile=vals)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()
    output = torch.empty((k,), device=x.device, dtype=torch.float32)

    TILE = 1024
    occupancy = 8

    sort_len = 1 << ((N - 1).bit_length())
    temp_a = torch.empty((sort_len,), device=x.device, dtype=torch.float32)
    temp_b = torch.empty((sort_len,), device=x.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    pass_kernel = _bitonic_pass_kernel.with_hints(occupancy=occupancy)
    copy_kernel = _copy_topk_kernel.with_hints(occupancy=occupancy)

    src = x
    dst = temp_a
    first_pass = True
    pass_count = 0

    log_n = sort_len.bit_length() - 1
    grid_sort = (ct.cdiv(sort_len, TILE), 1, 1)

    for p in range(1, log_n + 1):
        size = 1 << p
        for q in range(p, 0, -1):
            stride = 1 << (q - 1)
            ct.launch(stream, grid_sort, pass_kernel, (src, dst, size, stride, TILE))
            pass_count += 1

            if first_pass:
                src = temp_a
                dst = temp_b
                first_pass = False
            else:
                src, dst = dst, src

    grid_copy = (ct.cdiv(k, TILE), 1, 1)
    ct.launch(stream, grid_copy, copy_kernel, (src, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
        "sort_len": sort_len,
        "passes": pass_count,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
