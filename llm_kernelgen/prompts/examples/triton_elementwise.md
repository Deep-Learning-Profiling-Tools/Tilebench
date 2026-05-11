# Train Example: Triton Elementwise (vector_add)

**Operator:** `vector_add`
**Backend:** Triton
**Pattern:** Vectorised elementwise with masking

This example shows how to implement a simple elementwise Triton kernel
with correct masking for arbitrary (non-power-of-two) lengths.

```python
import torch
import triton
import triton.language as tl

_last_config: dict | None = None


@triton.jit
def _vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n: int,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}),
        triton.Config({"BLOCK": 512}),
        triton.Config({"BLOCK": 1024}),
    ],
    key=["n"],
)
@triton.jit
def _vector_add_kernel_at(
    x_ptr, y_ptr, out_ptr, n: int, BLOCK: tl.constexpr
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


def run(x: torch.Tensor, y: torch.Tensor,
        block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    global _last_config
    n   = x.numel()
    out = torch.empty_like(x)

    if autotune:
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        _vector_add_kernel_at[grid](x, y, out, n)
        _last_config = {k: v for k, v in
                        _vector_add_kernel_at.best_config.kwargs.items()}
    else:
        grid = (triton.cdiv(n, block_size),)
        _vector_add_kernel[grid](x, y, out, n, BLOCK=block_size)
        _last_config = None

    return out


def get_last_config() -> dict | None:
    return _last_config
```
