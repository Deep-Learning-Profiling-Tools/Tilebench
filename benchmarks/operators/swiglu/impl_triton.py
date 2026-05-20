import torch
import triton
import triton.language as tl

from core.triton_tma import ensure_tma_available

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4}


@triton.jit
def _swiglu_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE])
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE])

    x = x_desc.load([block_start])
    y = y_desc.load([block_start])
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    out = x_f32 * tl.sigmoid(x_f32) * y_f32
    out_desc.store([block_start], out.to(x.dtype))


_swiglu_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [512, 1024, 2048]
        for nw in [2, 4, 8]
    ],
    key=["n_elements"],
)(_swiglu_kernel)


def run(x: torch.Tensor, y: torch.Tensor,
        block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    ensure_tma_available()
    assert x.shape == y.shape
    x_flat = x.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)
    output = torch.empty_like(x_flat)
    n_elements = x_flat.numel()

    if autotune:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _swiglu_kernel_autotuned[grid](x_flat, y_flat, output, n_elements)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(n_elements, cfg["BLOCK_SIZE"]),)
        _swiglu_kernel[grid](
            x_flat, y_flat, output, n_elements,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
        )
    return output.view(x.shape)


def get_last_config() -> dict | None:
    cfg = getattr(_swiglu_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
