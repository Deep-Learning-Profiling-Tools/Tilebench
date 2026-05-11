# Train Example: Triton Fused Ops (relu)

**Operator:** `relu`
**Backend:** Triton
**Pattern:** Simple elementwise with dtype-aware casting

This example shows dtype handling and a fused activation pattern.

```python
import torch
import triton
import triton.language as tl

_last_config: dict | None = None


@triton.jit
def _relu_kernel(
    x_ptr,
    out_ptr,
    n: int,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    x   = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tl.where(x > 0, x, tl.zeros_like(x))
    tl.store(out_ptr + offs, out, mask=mask)


def run(x: torch.Tensor,
        block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    global _last_config
    n   = x.numel()
    out = torch.empty_like(x)

    grid = (triton.cdiv(n, block_size),)
    _relu_kernel[grid](x, out, n, BLOCK=block_size)
    _last_config = None
    return out


def get_last_config() -> dict | None:
    return _last_config
```
