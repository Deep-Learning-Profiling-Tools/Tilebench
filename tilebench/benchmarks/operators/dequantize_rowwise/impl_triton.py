import torch
import triton
import triton.language as tl


_DEFAULT_CONFIG = {"CHUNK": 512, "num_warps": 4}

_INV_127 = tl.constexpr(1.0 / 127.0)


@triton.jit
def dequantize_rowwise_kernel(
    x_ptr, state_ptr, out_ptr,
    ROWS, COLS,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)

    cols = chunk * CHUNK + tl.arange(0, CHUNK)
    mask = cols < COLS
    offsets = row * COLS + cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    scale = tl.load(state_ptr + row).to(tl.float32)

    out = x * scale * _INV_127
    tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)


_dequantize_rowwise_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"CHUNK": ch}, num_warps=nw)
        for ch in [256, 512, 1024]
        for nw in [2, 4, 8]
    ],
    key=["ROWS", "COLS"],
)(dequantize_rowwise_kernel)


def run(x: torch.Tensor, state_x: torch.Tensor,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    rows, cols = x.shape
    output = torch.empty(rows, cols, device=x.device, dtype=torch.float16)

    if autotune:
        grid = lambda meta: (rows, triton.cdiv(cols, meta["CHUNK"]))
        _dequantize_rowwise_kernel_autotuned[grid](
            x, state_x, output, rows, cols,
        )
    else:
        cfg = _DEFAULT_CONFIG
        grid = (rows, triton.cdiv(cols, cfg["CHUNK"]))
        dequantize_rowwise_kernel[grid](
            x, state_x, output, rows, cols,
            CHUNK=cfg["CHUNK"],
            num_warps=cfg["num_warps"],
        )
    return output


def get_last_config() -> dict | None:
    cfg = getattr(_dequantize_rowwise_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"CHUNK": cfg.kwargs["CHUNK"], "num_warps": cfg.num_warps}
