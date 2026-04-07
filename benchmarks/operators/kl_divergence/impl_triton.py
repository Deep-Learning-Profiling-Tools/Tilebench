import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 2}


@triton.jit
def _kl_divergence_kernel(p_ptr, q_ptr, out_ptr, n_elements, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(q_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    p = tl.maximum(p, eps)
    q = tl.maximum(q, eps)
    out = p * (tl.log(p) - tl.log(q))
    tl.store(out_ptr + offsets, out, mask=mask)


_kl_divergence_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [256, 512, 1024, 2048, 4096, 8192]
        for nw in [4, 8, 16]
    ],
    key=["n_elements"],
)(_kl_divergence_kernel)


def run(p: torch.Tensor, q: torch.Tensor, eps: float,
        block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    if p.shape != q.shape:
        raise ValueError("Input tensors must have the same shape.")
    p = p.contiguous()
    q = q.contiguous()
    out = torch.empty(p.shape, device=p.device, dtype=torch.float32)
    n_elements = p.numel()
    if autotune:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _kl_divergence_kernel_autotuned[grid](p, q, out, n_elements, eps)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(n_elements, cfg["BLOCK_SIZE"]),)
        _kl_divergence_kernel[grid](
            p, q, out, n_elements, eps,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return out


def get_last_config() -> dict | None:
    cfg = getattr(_kl_divergence_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
