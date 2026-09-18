import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _process_one_row(
    x_ptr,
    out_ptr,
    eps,
    row,
    K: tl.constexpr,
    BLOCK_MAIN: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    base = row * K
    offs0 = tl.arange(0, BLOCK_MAIN)

    if K == BLOCK_MAIN + BLOCK_TAIL:
        x0 = tl.load(x_ptr + base + offs0, cache_modifier=".cg")
        x0_f = x0.to(tl.float32)
        sum_sq = tl.sum(x0_f * x0_f, axis=0)

        offs1 = tl.arange(0, BLOCK_TAIL)
        k1 = BLOCK_MAIN + offs1
        x1 = tl.load(x_ptr + base + k1, cache_modifier=".cg")
        x1_f = x1.to(tl.float32)
        sum_sq += tl.sum(x1_f * x1_f, axis=0)

        eps_f = sum_sq * 0.0 + eps
        rstd = tl.rsqrt(tl.maximum(sum_sq, eps_f * eps_f))

        tl.store(out_ptr + base + offs0, x0 * rstd, cache_modifier=".cs")
        tl.store(out_ptr + base + k1, x1 * rstd, cache_modifier=".cs")

    elif K > BLOCK_MAIN:
        x0 = tl.load(x_ptr + base + offs0, cache_modifier=".cg")
        x0_f = x0.to(tl.float32)
        sum_sq = tl.sum(x0_f * x0_f, axis=0)

        offs1 = tl.arange(0, BLOCK_TAIL)
        k1 = BLOCK_MAIN + offs1
        mask1 = k1 < K
        x1 = tl.load(
            x_ptr + base + k1,
            mask=mask1,
            other=0.0,
            cache_modifier=".cg",
        )
        x1_f = x1.to(tl.float32)
        sum_sq += tl.sum(x1_f * x1_f, axis=0)

        eps_f = sum_sq * 0.0 + eps
        rstd = tl.rsqrt(tl.maximum(sum_sq, eps_f * eps_f))

        tl.store(out_ptr + base + offs0, x0 * rstd, cache_modifier=".cs")
        tl.store(
            out_ptr + base + k1,
            x1 * rstd,
            mask=mask1,
            cache_modifier=".cs",
        )

    elif K == BLOCK_MAIN:
        x0 = tl.load(x_ptr + base + offs0, cache_modifier=".cg")
        x0_f = x0.to(tl.float32)
        sum_sq = tl.sum(x0_f * x0_f, axis=0)

        eps_f = sum_sq * 0.0 + eps
        rstd = tl.rsqrt(tl.maximum(sum_sq, eps_f * eps_f))

        tl.store(out_ptr + base + offs0, x0 * rstd, cache_modifier=".cs")

    else:
        mask0 = offs0 < K
        x0 = tl.load(
            x_ptr + base + offs0,
            mask=mask0,
            other=0.0,
            cache_modifier=".cg",
        )
        x0_f = x0.to(tl.float32)
        sum_sq = tl.sum(x0_f * x0_f, axis=0)

        eps_f = sum_sq * 0.0 + eps
        rstd = tl.rsqrt(tl.maximum(sum_sq, eps_f * eps_f))

        tl.store(
            out_ptr + base + offs0,
            x0 * rstd,
            mask=mask0,
            cache_modifier=".cs",
        )


@triton.jit
def _l2_norm_kernel(
    x_ptr,
    out_ptr,
    eps,
    K: tl.constexpr,
    BLOCK_MAIN: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    GRID_ROWS: tl.constexpr,
    DO_SECOND_ROW: tl.constexpr,
):
    pid = tl.program_id(0)

    _process_one_row(
        x_ptr,
        out_ptr,
        eps,
        pid,
        K,
        BLOCK_MAIN,
        BLOCK_TAIL,
    )

    if DO_SECOND_ROW:
        _process_one_row(
            x_ptr,
            out_ptr,
            eps,
            pid + GRID_ROWS,
            K,
            BLOCK_MAIN,
            BLOCK_TAIL,
        )


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]
    K = x.shape[2]
    rows = batch * M

    BLOCK_MAIN = 8192
    BLOCK_TAIL = 2048
    num_warps = 16
    num_stages = 3

    if rows % 2 == 0:
        ROWS_PER_CTA = 2
        grid_rows = rows // 2
        DO_SECOND_ROW = True
    else:
        ROWS_PER_CTA = 1
        grid_rows = rows
        DO_SECOND_ROW = False

    grid = (grid_rows,)
    _l2_norm_kernel[grid](
        x,
        output,
        eps,
        K=K,
        BLOCK_MAIN=BLOCK_MAIN,
        BLOCK_TAIL=BLOCK_TAIL,
        GRID_ROWS=grid_rows,
        DO_SECOND_ROW=DO_SECOND_ROW,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_MAIN": BLOCK_MAIN,
            "BLOCK_TAIL": BLOCK_TAIL,
            "ROWS_PER_CTA": ROWS_PER_CTA,
            "GRID_ROWS": grid_rows,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "LOAD_CACHE_CG": 1,
            "STORE_CACHE_CS": 1,
            "KEEP_INPUT_DTYPE": 1,
            "MATH_RSQRT": 1,
            "FULL_K_FASTPATH": 1,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
