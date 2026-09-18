import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}

_TILE = 2048
_LOAD = 1024


@triton.jit
def _topk_persistent_kernel(in_ptr, out_ptr, N,
                            K: tl.constexpr,
                            TILE: tl.constexpr,
                            LOAD: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    NEG_INF: tl.constexpr = float('-inf')

    running = tl.full((TILE,), NEG_INF, dtype=tl.float32)
    idx = tl.arange(0, TILE)

    elems_per_block = tl.cdiv(N, num_progs)
    start = pid * elems_per_block
    end = tl.minimum(start + elems_per_block, N)

    # Process LOAD elements per iteration; the working tile has size TILE = K + LOAD' >= 2K.
    # Layout: positions [0,LOAD) hold previous-iter top-K (rest -inf), positions [LOAD,TILE)
    # receive newly-loaded elements. After `tl.maximum`, the tile equals concat(running_first_K, new_data).
    for offset in range(start, end, LOAD):
        new_offs = offset + (idx - LOAD)
        valid = (idx >= LOAD) & (new_offs < end)
        new_tile = tl.load(in_ptr + new_offs, mask=valid, other=NEG_INF)
        combined = tl.maximum(running, new_tile)
        sorted_c = tl.sort(combined, descending=True)
        running = tl.where(idx < K, sorted_c, NEG_INF)

    # Store first K of running to out_ptr[pid*K : pid*K + K]. Mask suppresses the extra TILE-K positions.
    out_offs = pid * K + idx
    out_mask = idx < K
    tl.store(out_ptr + out_offs, running, mask=out_mask)


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    assert input.is_cuda
    assert input.dtype == torch.float32
    assert 1 <= k <= N
    output = torch.empty(k, dtype=torch.float32, device=input.device)

    TILE = _TILE
    LOAD = _LOAD
    num_warps = 8
    num_stages = 2
    target_per_block = 8192  # tune: elements per block per stage

    cur = input
    cur_size = N

    intermediates = []
    while True:
        num_persist = max(1, (cur_size + target_per_block - 1) // target_per_block)
        num_persist = min(num_persist, 256)

        if num_persist == 1:
            _topk_persistent_kernel[(1,)](
                cur, output, cur_size,
                K=k, TILE=TILE, LOAD=LOAD,
                num_warps=num_warps, num_stages=num_stages,
            )
            break

        intermediate = torch.empty(num_persist * k, dtype=torch.float32, device=input.device)
        intermediates.append(intermediate)
        _topk_persistent_kernel[(num_persist,)](
            cur, intermediate, cur_size,
            K=k, TILE=TILE, LOAD=LOAD,
            num_warps=num_warps, num_stages=num_stages,
        )
        cur = intermediate
        cur_size = num_persist * k

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE, "LOAD": LOAD,
        "target_per_block": target_per_block,
        "num_warps": num_warps, "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
