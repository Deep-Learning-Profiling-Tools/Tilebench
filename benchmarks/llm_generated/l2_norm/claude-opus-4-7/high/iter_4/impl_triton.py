import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}

_NUM_SM = None
def _num_sm():
    global _NUM_SM
    if _NUM_SM is None:
        _NUM_SM = torch.cuda.get_device_properties(0).multi_processor_count
    return _NUM_SM


@triton.jit
def _l2_norm_kernel(x_ptr, out_ptr, M, K, eps, stride_row,
                    BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    for row in tl.range(pid, M, num_progs):
        x_row = x_ptr + row * stride_row
        o_row = out_ptr + row * stride_row

        # Pass 1: sum of squares (fp32). evict_last keeps data in L2 for pass 2.
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            x = tl.load(x_row + offs, mask=mask, other=0.0,
                        eviction_policy="evict_last").to(tl.float32)
            acc += x * x
        total = tl.sum(acc, axis=0)
        inv = 1.0 / tl.maximum(tl.sqrt(total), eps)

        # Pass 2: normalize. evict_first since we're done with this row.
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            x = tl.load(x_row + offs, mask=mask, other=0.0,
                        eviction_policy="evict_first").to(tl.float32)
            y = x * inv
            tl.store(o_row + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1]).contiguous()
    M, K = x_flat.shape
    out_flat = torch.empty_like(x_flat)

    BLOCK_K = 2048
    num_warps = 8
    num_stages = 2

    # Persistent grid: NUM_SM * occupancy programs, each handling multiple rows.
    # This eliminates wave imbalance (2048 rows on 148 SMs would have a half-empty
    # final wave).
    grid_size = min(M, _num_sm() * 4)
    grid = (grid_size,)

    _l2_norm_kernel[grid](
        x_flat, out_flat, M, K, float(eps), x_flat.stride(0),
        BLOCK_K=BLOCK_K, num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_K": BLOCK_K,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "grid_size": grid_size,
        "persistent": True,
    })
    return out_flat.reshape(shape)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
