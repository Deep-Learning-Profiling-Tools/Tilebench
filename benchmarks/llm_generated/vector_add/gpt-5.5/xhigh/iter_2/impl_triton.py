import torch
import triton
import triton.language as tl


_MAX_BLOCK_SIZE = 4096


@triton.jit
def _vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_units,
    DTYPE_CODE: tl.constexpr,
    MODE: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if MODE == 1:
        # int8 fast path: operate on four packed int8 lanes per int32 word.
        if MASKED:
            mask = offsets < n_units
            x_vals = tl.load(
                x_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint32)
            y_vals = tl.load(
                y_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint32)
        else:
            x_vals = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.uint32)
            y_vals = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.uint32)

        lo_mask = tl.full((BLOCK_SIZE,), 0x00FF00FF, tl.uint32)
        hi_mask = tl.full((BLOCK_SIZE,), 0xFF00FF00, tl.uint32)

        lo = (x_vals & lo_mask) + (y_vals & lo_mask)
        hi = (x_vals & hi_mask) + (y_vals & hi_mask)
        out_vals = (lo & lo_mask) | (hi & hi_mask)

        if MASKED:
            tl.store(out_ptr + offsets, out_vals, mask=mask, eviction_policy="evict_first")
        else:
            tl.store(out_ptr + offsets, out_vals, eviction_policy="evict_first")
    else:
        if MASKED:
            mask = offsets < n_units
            x_vals = tl.load(
                x_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            )
            y_vals = tl.load(
                y_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            )
            tl.store(out_ptr + offsets, x_vals + y_vals, mask=mask, eviction_policy="evict_first")
        else:
            x_vals = tl.load(x_ptr + offsets, eviction_policy="evict_first")
            y_vals = tl.load(y_ptr + offsets, eviction_policy="evict_first")
            tl.store(out_ptr + offsets, x_vals + y_vals, eviction_policy="evict_first")


_TRITON_CONFIGS = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
    for bs, warps in [
        (64, [2]),
        (128, [2, 4]),
        (256, [2, 4]),
        (512, [2, 4]),
        (1024, [4, 8]),
        (2048, [4, 8]),
        (4096, [4, 8]),
    ]
    for nw in warps
    for ns in [1, 3]
]

_vector_add_kernel_autotuned = triton.autotune(
    configs=_TRITON_CONFIGS,
    key=["n_units", "DTYPE_CODE", "MODE", "MASKED"],
)(_vector_add_kernel)


def _dtype_code(dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    if dtype == torch.int8:
        return 3
    return 4


def run(x, y, autotune: bool = False):
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    output = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return output

    dtype_code = _dtype_code(x.dtype)
    mode = 0
    n_units = n_elements
    x_arg = x
    y_arg = y
    out_arg = output

    if (
        x.dtype == torch.int8
        and n_elements % 4 == 0
        and x.is_contiguous()
        and y.is_contiguous()
        and output.is_contiguous()
    ):
        mode = 1
        n_units = n_elements // 4
        x_arg = x.view(torch.int32)
        y_arg = y.view(torch.int32)
        out_arg = output.view(torch.int32)

    masked = (n_units % _MAX_BLOCK_SIZE) != 0
    grid = lambda meta: (triton.cdiv(n_units, meta["BLOCK_SIZE"]),)

    _vector_add_kernel_autotuned[grid](
        x_arg,
        y_arg,
        out_arg,
        n_units,
        dtype_code,
        mode,
        masked,
    )
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
