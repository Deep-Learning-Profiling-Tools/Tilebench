import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _l2_norm_kernel(x_ptr, out_ptr, K, eps, stride_row,
                    BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_row
    o_row = out_ptr + row * stride_row

    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    # Single-pass: load row once, keep in registers, compute norm, write.
    x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
    sq = x * x
    total = tl.sum(sq, axis=0)
    inv = 1.0 / tl.maximum(tl.sqrt(total), eps)
    y = x * inv
    tl.store(o_row + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1]).contiguous()
    M, K = x_flat.shape
    out_flat = torch.empty_like(x_flat)

    # K is at most 10240 in this benchmark; 16384 covers the whole row in one tile.
    BLOCK_K = triton.next_power_of_2(K)
    if BLOCK_K < 256:
        BLOCK_K = 256
    num_warps = 8
    num_stages = 1

    grid = (M,)
    _l2_norm_kernel[grid](
        x_flat, out_flat, K, float(eps), x_flat.stride(0),
        BLOCK_K=BLOCK_K, num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_K": BLOCK_K,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "single_pass": True,
    })
    return out_flat.reshape(shape)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
