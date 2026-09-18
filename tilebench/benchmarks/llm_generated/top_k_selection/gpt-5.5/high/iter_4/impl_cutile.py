import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _bitonic_local_pass_kernel(src, dst, size, STRIDE: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)

    vals = ct.load(
        src,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.NEG_INF,
        allow_tma=False,
        latency=1,
    )

    GROUPS = TILE // (2 * STRIDE)

    x = ct.reshape(vals, (GROUPS, 2, STRIDE))
    lo = ct.extract(x, (0, 0, 0), shape=(GROUPS, 1, STRIDE))
    hi = ct.extract(x, (0, 1, 0), shape=(GROUPS, 1, STRIDE))

    v_max = ct.maximum(lo, hi)
    v_min = ct.minimum(lo, hi)

    gids = ct.arange(GROUPS, dtype=np.int32)[:, None, None]
    group_base = bid * TILE + gids * (2 * STRIDE)
    descending_segment = (group_base & size) == 0

    new_lo = ct.where(descending_segment, v_max, v_min)
    new_hi = ct.where(descending_segment, v_min, v_max)

    y = ct.cat((new_lo, new_hi), 1)
    out = ct.reshape(y, (TILE,))

    ct.store(dst, index=(bid,), tile=out, allow_tma=False)


@ct.kernel(occupancy=8)
def _bitonic_global_pass_kernel(src, dst, size, stride, TILE: ConstInt):
    bid = ct.bid(0)
    offs = ct.arange(TILE, dtype=np.int32)
    idx = bid * TILE + offs

    val = ct.load(
        src,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.NEG_INF,
        allow_tma=False,
        latency=1,
    )

    partner_bid = bid ^ (stride // TILE)
    other = ct.load(
        src,
        index=(partner_bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.NEG_INF,
        allow_tma=False,
        latency=1,
    )

    v_max = ct.maximum(val, other)
    v_min = ct.minimum(val, other)

    descending_segment = (idx & size) == 0
    lower_partner = (idx & stride) == 0
    take_max = descending_segment == lower_partner

    out = ct.where(take_max, v_max, v_min)
    ct.store(dst, index=(bid,), tile=out, allow_tma=False)


@ct.kernel(occupancy=8)
def _copy_topk_kernel(src, output, TILE: ConstInt):
    bid = ct.bid(0)
    vals = ct.load(
        src,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
        latency=1,
    )
    ct.store(output, index=(bid,), tile=vals, allow_tma=False)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()
    output = torch.empty((k,), device=x.device, dtype=torch.float32)

    TILE = 2048
    occupancy = 8

    sort_len = 1 << ((N - 1).bit_length())
    temp_a = torch.empty((sort_len,), device=x.device, dtype=torch.float32)
    temp_b = torch.empty((sort_len,), device=x.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    src = x
    dst = temp_a
    first_pass = True
    pass_count = 0
    local_passes = 0
    global_passes = 0

    log_n = sort_len.bit_length() - 1
    grid_sort = (ct.cdiv(sort_len, TILE), 1, 1)

    for p in range(1, log_n + 1):
        size = 1 << p
        for q in range(p, 0, -1):
            stride = 1 << (q - 1)

            if stride < TILE:
                ct.launch(
                    stream,
                    grid_sort,
                    _bitonic_local_pass_kernel,
                    (src, dst, size, stride, TILE),
                )
                local_passes += 1
            else:
                ct.launch(
                    stream,
                    grid_sort,
                    _bitonic_global_pass_kernel,
                    (src, dst, size, stride, TILE),
                )
                global_passes += 1

            pass_count += 1

            if first_pass:
                src = temp_a
                dst = temp_b
                first_pass = False
            else:
                src, dst = dst, src

    grid_copy = (ct.cdiv(k, TILE), 1, 1)
    ct.launch(stream, grid_copy, _copy_topk_kernel, (src, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
        "sort_len": sort_len,
        "passes": pass_count,
        "local_passes": local_passes,
        "global_passes": global_passes,
        "coalesced_global": 1,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
