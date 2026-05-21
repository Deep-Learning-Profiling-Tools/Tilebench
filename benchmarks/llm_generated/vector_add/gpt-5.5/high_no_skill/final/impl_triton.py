import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0)
    out_vals = x_vals + y_vals

    tl.store(out_ptr + offsets, out_vals, mask=mask)


_vector_add_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in [256, 512, 1024, 2048]
        for nw in [2, 4, 8]
        for ns in [1, 2]
    ],
    key=["n_elements"],
)(_vector_add_kernel)


def run(x, y, autotune: bool = False):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _vector_add_kernel_autotuned[grid](x, y, output, n_elements)
    return output


def get_last_config() -> dict | None:
    cfg = getattr(_vector_add_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
