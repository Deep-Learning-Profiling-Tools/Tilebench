import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _argmax_combine(v1, i1, v2, i2):
    better = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    return tl.where(better, v1, v2), tl.where(better, i1, i2)


@triton.jit
def _argmax_p1(x_ptr, pv_ptr, pi_ptr, N, CHUNK, stride_m,
               BLOCK_N: tl.constexpr, NSPLIT: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)

    start = pid_s * CHUNK
    end = tl.minimum(start + CHUNK, N)

    row_ptr = x_ptr + pid_m * stride_m
    NEG_INF: tl.constexpr = float('-inf')

    best_val = tl.full([1], NEG_INF, dtype=tl.float32)
    best_idx = tl.full([1], 0, dtype=tl.int32)

    for n_start in tl.range(start, end, BLOCK_N, num_stages=3):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < end
        x = tl.load(row_ptr + offs, mask=mask, other=NEG_INF).to(tl.float32)
        idxs = offs.to(tl.int32)
        lm, la = tl.reduce((x, idxs), 0, _argmax_combine, keep_dims=True)
        better = (lm > best_val) | ((lm == best_val) & (la < best_idx))
        best_val = tl.where(better, lm, best_val)
        best_idx = tl.where(better, la, best_idx)

    out_off = pid_m * NSPLIT + pid_s
    tl.store(pv_ptr + out_off + tl.arange(0, 1), best_val)
    tl.store(pi_ptr + out_off + tl.arange(0, 1), best_idx)


@triton.jit
def _argmax_p2(pv_ptr, pi_ptr, out_ptr, NSPLIT: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, NSPLIT)
    vals = tl.load(pv_ptr + pid * NSPLIT + offs)
    idxs = tl.load(pi_ptr + pid * NSPLIT + offs)
    _, best = tl.reduce((vals, idxs), 0, _argmax_combine)
    tl.store(out_ptr + pid + tl.arange(0, 1), best.to(tl.int64))


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim in (1, -1)
    assert x.ndim == 2
    x = x.contiguous()
    M, N = x.shape
    output = torch.empty(M, dtype=torch.int64, device=x.device)

    BLOCK_N = 2048
    NSPLIT = 4
    num_warps = 4
    num_stages = 3
    CHUNK = triton.cdiv(N, NSPLIT)

    pv = torch.empty(M * NSPLIT, dtype=torch.float32, device=x.device)
    pi = torch.empty(M * NSPLIT, dtype=torch.int32, device=x.device)

    _argmax_p1[(M, NSPLIT)](
        x, pv, pi, N, CHUNK, x.stride(0),
        BLOCK_N=BLOCK_N, NSPLIT=NSPLIT,
        num_warps=num_warps, num_stages=num_stages,
    )

    _argmax_p2[(M,)](pv, pi, output, NSPLIT=NSPLIT, num_warps=1)

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_N": BLOCK_N,
        "NSPLIT": NSPLIT,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
