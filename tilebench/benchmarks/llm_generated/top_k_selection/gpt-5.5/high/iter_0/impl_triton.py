import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _bitonic_sort_desc(vals, BLOCK_SIZE: tl.constexpr, LOG_BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK_SIZE)

    for p in tl.static_range(1, LOG_BLOCK + 1):
        size = 1 << p
        for q in tl.static_range(p, 0, -1):
            stride = 1 << (q - 1)
            other = tl.gather(vals, idx ^ stride, axis=0)

            v_min = tl.minimum(vals, other)
            v_max = tl.maximum(vals, other)

            descending_segment = (idx & size) == 0
            lower_partner = (idx & stride) == 0
            take_max = descending_segment == lower_partner

            vals = tl.where(take_max, v_max, v_min)

    return vals


@triton.jit
def _topk_stage_kernel(in_ptr, out_ptr, n_elements, K,
                       BLOCK_SIZE: tl.constexpr, LOG_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    in_offs = pid * BLOCK_SIZE + offs
    mask = in_offs < n_elements

    vals = tl.load(in_ptr + in_offs, mask=mask, other=-float("inf"))
    vals = _bitonic_sort_desc(vals, BLOCK_SIZE, LOG_BLOCK)

    store_mask = offs < K
    tl.store(out_ptr + pid * K + offs, vals, mask=store_mask)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    x = input.contiguous()

    BLOCK_SIZE = 2048
    LOG_BLOCK = 11
    num_warps = 8
    num_stages = 4

    cur = x
    n_cur = N
    stage_launches = 0

    while True:
        num_blocks = triton.cdiv(n_cur, BLOCK_SIZE)

        if num_blocks == 1:
            output = torch.empty((k,), device=x.device, dtype=torch.float32)
            _topk_stage_kernel[(1,)](
                cur, output, n_cur, k,
                BLOCK_SIZE=BLOCK_SIZE,
                LOG_BLOCK=LOG_BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            stage_launches += 1
            break

        tmp_len = num_blocks * k
        tmp = torch.empty((tmp_len,), device=x.device, dtype=torch.float32)
        _topk_stage_kernel[(num_blocks,)](
            cur, tmp, n_cur, k,
            BLOCK_SIZE=BLOCK_SIZE,
            LOG_BLOCK=LOG_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        stage_launches += 1
        cur = tmp
        n_cur = tmp_len

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "LOG_BLOCK": LOG_BLOCK,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "k": k,
        "stage_launches": stage_launches,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
