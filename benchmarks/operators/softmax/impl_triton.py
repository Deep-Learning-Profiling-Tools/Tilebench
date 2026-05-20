import torch
import triton
import triton.language as tl

from core.triton_tma import ensure_tma_available

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4}


@triton.jit
def softmax_online_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    input_desc = tl.make_tensor_descriptor(
        input_ptr + row_idx * input_row_stride,
        shape=[n_cols],
        strides=[1],
        block_shape=[BLOCK_SIZE],
    )
    output_desc = tl.make_tensor_descriptor(
        output_ptr + row_idx * output_row_stride,
        shape=[n_cols],
        strides=[1],
        block_shape=[BLOCK_SIZE],
    )

    # Pass 1: online max + sum
    m = -float('inf')
    l = 0.0
    for col_start in range(0, n_cols, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = input_desc.load([col_start]).to(tl.float32)
        x = tl.where(mask, x, -float('inf'))
        block_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, block_max)
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    # Pass 2: normalize and write back
    for col_start in range(0, n_cols, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = input_desc.load([col_start]).to(tl.float32)
        x = tl.where(mask, x, 0.0)
        y = tl.exp(x - m) / l
        output_desc.store([col_start], y.to(output_ptr.dtype.element_ty))


_softmax_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [512, 1024, 2048]
        for nw in [4, 8]
    ],
    warmup=5,
    rep=20,
    key=["n_cols"],
)(softmax_online_kernel)


def run(x: torch.Tensor, block_size: int = None, autotune: bool = False):
    ensure_tma_available()
    x = x.contiguous()
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    grid = (n_rows,)

    if autotune:
        _softmax_kernel_autotuned[grid](
            output, x,
            x.stride(0), output.stride(0),
            n_cols,
        )
    else:
        cfg = _DEFAULT_CONFIG
        softmax_online_kernel[grid](
            output, x,
            x.stride(0), output.stride(0),
            n_cols,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_softmax_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
        "num_warps": cfg.num_warps,
    }
