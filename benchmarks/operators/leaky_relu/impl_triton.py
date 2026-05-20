import torch
import triton
import triton.language as tl

from core.triton_tma import ensure_tma_available

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 2}


@triton.jit
def _leaky_relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[n_elements, 1], strides=[1, 1], block_shape=[BLOCK_SIZE, 1])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[n_elements, 1], strides=[1, 1], block_shape=[BLOCK_SIZE, 1])
    x = x_desc.load([block_start, 0])
    y = tl.where(x > 0, x, 0.01 * x)
    y_desc.store([block_start, 0], y)


_leaky_relu_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in [1024, 2048, 4096, 8192]
        for nw in [4, 8]
        for ns in [1, 2]
    ],
    key=["n_elements"],
)(_leaky_relu_kernel)


def run(input: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    ensure_tma_available()
    input_flat = input.contiguous().view(-1)
    output = torch.empty_like(input_flat)

    if autotune:
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        _leaky_relu_kernel_autotuned[grid](input_flat, output, N)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(N, cfg["BLOCK_SIZE"]),)
        _leaky_relu_kernel[grid](
            input_flat, output, N,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return output.view(input.shape)


def get_last_config() -> dict | None:
    cfg = getattr(_leaky_relu_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
