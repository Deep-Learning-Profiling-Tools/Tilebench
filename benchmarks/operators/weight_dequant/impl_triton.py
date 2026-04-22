import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4}


@triton.jit
def dequant_kernel(X, S, Y, M: tl.constexpr, N: tl.constexpr,
                   S_COLS: tl.constexpr, TILE_SIZE: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N

    row = offsets // N
    col = offsets % N

    s_row = row // TILE_SIZE
    s_col = col // TILE_SIZE

    x = tl.load(X + offsets, mask=mask).to(tl.float32)
    scale = tl.load(S + s_row * S_COLS + s_col, mask=mask).to(tl.float32)

    y = x * scale

    tl.store(Y + offsets, y, mask=mask)


_dequant_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [256, 512, 1024, 2048, 4096]
        for nw in [4, 8]
    ],
    key=["M", "N"],
)(dequant_kernel)


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output = torch.empty(M, N, dtype=X.dtype, device=X.device)
    S_COLS = triton.cdiv(N, TILE_SIZE)
    total = M * N

    if autotune:
        grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
        _dequant_kernel_autotuned[grid](X, S, output, M, N, S_COLS, TILE_SIZE)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(total, cfg["BLOCK_SIZE"]),)
        dequant_kernel[grid](
            X, S, output, M, N,
            S_COLS, TILE_SIZE, cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_dequant_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
