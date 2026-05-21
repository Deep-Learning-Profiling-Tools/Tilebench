import torch
import triton
import triton.language as tl


_MAX_BLOCK_SIZE = 256


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
        # int8 packed as int64: 8 independent wrapping byte additions/word.
        if MASKED:
            mask = offsets < n_units
            x_vals = tl.load(
                x_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
            y_vals = tl.load(
                y_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
        else:
            x_vals = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)
            y_vals = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)

        lo_mask = tl.full((BLOCK_SIZE,), 0x00FF00FF00FF00FF, tl.uint64)
        hi_mask = lo_mask << 8

        lo = (x_vals & lo_mask) + (y_vals & lo_mask)
        hi = (x_vals & hi_mask) + (y_vals & hi_mask)
        out_vals = (lo & lo_mask) | (hi & hi_mask)

        if MASKED:
            tl.store(out_ptr + offsets, out_vals, mask=mask, eviction_policy="evict_first")
        else:
            tl.store(out_ptr + offsets, out_vals, eviction_policy="evict_first")

    elif MODE == 2:
        # fp32 packed as int64: 2 float32 additions/word with one 64-bit load/store.
        if MASKED:
            mask = offsets < n_units
            x_vals = tl.load(
                x_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
            y_vals = tl.load(
                y_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
        else:
            x_vals = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)
            y_vals = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)

        mask32 = tl.full((BLOCK_SIZE,), 0xFFFFFFFF, tl.uint64)

        x_lo_bits = (x_vals & mask32).to(tl.uint32)
        y_lo_bits = (y_vals & mask32).to(tl.uint32)
        x_hi_bits = (x_vals >> 32).to(tl.uint32)
        y_hi_bits = (y_vals >> 32).to(tl.uint32)

        x_lo = x_lo_bits.to(tl.float32, bitcast=True)
        y_lo = y_lo_bits.to(tl.float32, bitcast=True)
        x_hi = x_hi_bits.to(tl.float32, bitcast=True)
        y_hi = y_hi_bits.to(tl.float32, bitcast=True)

        out_lo_bits = (x_lo + y_lo).to(tl.uint32, bitcast=True)
        out_hi_bits = (x_hi + y_hi).to(tl.uint32, bitcast=True)
        out_vals = out_lo_bits.to(tl.uint64) | (out_hi_bits.to(tl.uint64) << 32)

        if MASKED:
            tl.store(out_ptr + offsets, out_vals, mask=mask, eviction_policy="evict_first")
        else:
            tl.store(out_ptr + offsets, out_vals, eviction_policy="evict_first")

    elif MODE == 3:
        # fp16 packed as int64: 4 fp16 additions/word via two f16x2 instructions.
        if MASKED:
            mask = offsets < n_units
            x_vals = tl.load(
                x_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
            y_vals = tl.load(
                y_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
        else:
            x_vals = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)
            y_vals = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)

        mask32 = tl.full((BLOCK_SIZE,), 0xFFFFFFFF, tl.uint64)
        x_lo = (x_vals & mask32).to(tl.uint32)
        y_lo = (y_vals & mask32).to(tl.uint32)
        x_hi = (x_vals >> 32).to(tl.uint32)
        y_hi = (y_vals >> 32).to(tl.uint32)

        out_lo = tl.inline_asm_elementwise(
            "add.rn.f16x2 $0, $1, $2;",
            constraints="=r,r,r",
            args=[x_lo, y_lo],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        out_hi = tl.inline_asm_elementwise(
            "add.rn.f16x2 $0, $1, $2;",
            constraints="=r,r,r",
            args=[x_hi, y_hi],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        out_vals = out_lo.to(tl.uint64) | (out_hi.to(tl.uint64) << 32)

        if MASKED:
            tl.store(out_ptr + offsets, out_vals, mask=mask, eviction_policy="evict_first")
        else:
            tl.store(out_ptr + offsets, out_vals, eviction_policy="evict_first")

    elif MODE == 4:
        # bf16 packed as int64: 4 bf16 additions/word via two bf16x2 instructions.
        if MASKED:
            mask = offsets < n_units
            x_vals = tl.load(
                x_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
            y_vals = tl.load(
                y_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
        else:
            x_vals = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)
            y_vals = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)

        mask32 = tl.full((BLOCK_SIZE,), 0xFFFFFFFF, tl.uint64)
        x_lo = (x_vals & mask32).to(tl.uint32)
        y_lo = (y_vals & mask32).to(tl.uint32)
        x_hi = (x_vals >> 32).to(tl.uint32)
        y_hi = (y_vals >> 32).to(tl.uint32)

        out_lo = tl.inline_asm_elementwise(
            "add.rn.bf16x2 $0, $1, $2;",
            constraints="=r,r,r",
            args=[x_lo, y_lo],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        out_hi = tl.inline_asm_elementwise(
            "add.rn.bf16x2 $0, $1, $2;",
            constraints="=r,r,r",
            args=[x_hi, y_hi],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        out_vals = out_lo.to(tl.uint64) | (out_hi.to(tl.uint64) << 32)

        if MASKED:
            tl.store(out_ptr + offsets, out_vals, mask=mask, eviction_policy="evict_first")
        else:
            tl.store(out_ptr + offsets, out_vals, eviction_policy="evict_first")

    else:
        # Generic scalar fallback.
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
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
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

    if x.dtype == torch.int8 and n_elements % 8 == 0:
        mode = 1
        n_units = n_elements // 8
        x_arg = x.view(torch.int64)
        y_arg = y.view(torch.int64)
        out_arg = output.view(torch.int64)
    elif x.dtype == torch.float32 and n_elements % 2 == 0:
        mode = 2
        n_units = n_elements // 2
        x_arg = x.view(torch.int64)
        y_arg = y.view(torch.int64)
        out_arg = output.view(torch.int64)
    elif x.dtype == torch.float16 and n_elements % 4 == 0:
        mode = 3
        n_units = n_elements // 4
        x_arg = x.view(torch.int64)
        y_arg = y.view(torch.int64)
        out_arg = output.view(torch.int64)
    elif x.dtype == torch.bfloat16 and n_elements % 4 == 0:
        mode = 4
        n_units = n_elements // 4
        x_arg = x.view(torch.int64)
        y_arg = y.view(torch.int64)
        out_arg = output.view(torch.int64)

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
