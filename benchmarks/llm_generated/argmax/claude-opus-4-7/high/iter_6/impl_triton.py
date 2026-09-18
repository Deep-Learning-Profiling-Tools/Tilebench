import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


def _alloc(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc)


@triton.jit
def _argmax_kernel(x_ptr, out_ptr, M, N, stride_m,
                   BLOCK_N: tl.constexpr):
    # TMA descriptor for the full 2D matrix; one row per CTA.
    desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[M, N],
        strides=[stride_m, 1],
        block_shape=[1, BLOCK_N],
    )
    pid = tl.program_id(0)

    NEG_INF: tl.constexpr = float('-inf')
    best_val = tl.full([1], NEG_INF, dtype=tl.float32)
    best_idx = tl.full([1], 0, dtype=tl.int64)

    for n_start in tl.range(0, N, BLOCK_N, num_stages=3):
        x2d = desc.load([pid, n_start])               # [1, BLOCK_N]
        x = tl.reshape(x2d, [BLOCK_N]).to(tl.float32)

        # Mask OOB elements to -inf (TMA OOB padding may be 0 by default).
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.where(mask, x, NEG_INF)

        local_max = tl.max(x, axis=0, keep_dims=True)
        local_arg = tl.argmax(x, axis=0, keep_dims=True)
        local_global = local_arg.to(tl.int64) + n_start

        # Strict '>' ensures earliest occurrence wins on ties across tiles.
        new_better = local_max > best_val
        best_val = tl.where(new_better, local_max, best_val)
        best_idx = tl.where(new_better, local_global, best_idx)

    tl.store(out_ptr + pid + tl.arange(0, 1), best_idx)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim in (1, -1)
    assert x.ndim == 2
    x = x.contiguous()
    M, N = x.shape
    output = torch.empty(M, dtype=torch.int64, device=x.device)

    BLOCK_N = 4096
    num_warps = 8
    num_stages = 3

    grid = (M,)
    _argmax_kernel[grid](
        x, output, M, N, x.stride(0),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_N": BLOCK_N,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "tma": True,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
