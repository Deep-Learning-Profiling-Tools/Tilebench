import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _dequantize_rowwise_kernel(
    x_ptr,
    state_ptr,
    out_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = offs_m[:, None] * cols + offs_n[None, :]

    if USE_MASK:
        mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)
        x_i8 = tl.load(
            x_ptr + offsets,
            mask=mask,
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        scale = tl.load(
            state_ptr + offs_m,
            mask=offs_m < rows,
            other=0.0,
        ).to(tl.float32)
        scale = scale * 0.007874015748031496
        y = x_i8 * scale[:, None]
        tl.store(
            out_ptr + offsets,
            y,
            mask=mask,
            cache_modifier=".cs",
            eviction_policy="evict_first",
        )
    else:
        x_i8 = tl.load(
            x_ptr + offsets,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        scale = tl.load(state_ptr + offs_m).to(tl.float32)
        scale = scale * 0.007874015748031496
        y = x_i8 * scale[:, None]
        tl.store(
            out_ptr + offsets,
            y,
            cache_modifier=".cs",
            eviction_policy="evict_first",
        )


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    rows = x.shape[0]
    cols = x.shape[1]
    output = torch.empty((rows, cols), device=x.device, dtype=torch.float16)

    BLOCK_M = 4
    BLOCK_N = 1024
    num_warps = 8
    num_stages = 2
    USE_MASK = (rows % BLOCK_M) != 0 or (cols % BLOCK_N) != 0

    grid = (triton.cdiv(cols, BLOCK_N), triton.cdiv(rows, BLOCK_M))
    _dequantize_rowwise_kernel[grid](
        x,
        state_x,
        output,
        rows=rows,
        cols=cols,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        USE_MASK=USE_MASK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "USE_MASK": USE_MASK,
            "GRID_N_FAST": True,
            "CACHE_STREAMING": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
