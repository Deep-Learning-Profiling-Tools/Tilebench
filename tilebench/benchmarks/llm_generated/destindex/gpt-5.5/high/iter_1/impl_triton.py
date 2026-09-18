import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


@triton.jit
def _destindex_kernel(
    kv_nope,
    kv_rope,
    dest_loc,
    out_nope,
    out_rope,
    T,
    ROW_NOPE: tl.constexpr,
    ROW_ROPE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_COL: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    EVEN_T: tl.constexpr,
    EVEN_NOPE: tl.constexpr,
    EVEN_ROPE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)

    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = pid_c * BLOCK_COL + tl.arange(0, BLOCK_COL)

    if EVEN_T:
        dests = tl.load(dest_loc + rows)
    else:
        row_mask = rows < T
        dests = tl.load(dest_loc + rows, mask=row_mask, other=0)

    if EVEN_NOPE:
        vals_nope = tl.load(
            kv_nope + rows[:, None] * ROW_NOPE + cols[None, :],
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        tl.store(
            out_nope + dests[:, None] * ROW_NOPE + cols[None, :],
            vals_nope,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
    else:
        row_mask = rows < T
        mask_nope = row_mask[:, None] & (cols[None, :] < ROW_NOPE)
        vals_nope = tl.load(
            kv_nope + rows[:, None] * ROW_NOPE + cols[None, :],
            mask=mask_nope,
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        tl.store(
            out_nope + dests[:, None] * ROW_NOPE + cols[None, :],
            vals_nope,
            mask=mask_nope,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )

    if pid_c == 0:
        rcols = tl.arange(0, BLOCK_ROPE)
        if EVEN_ROPE:
            vals_rope = tl.load(
                kv_rope + rows[:, None] * ROW_ROPE + rcols[None, :],
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )
            tl.store(
                out_rope + dests[:, None] * ROW_ROPE + rcols[None, :],
                vals_rope,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )
        else:
            row_mask = rows < T
            mask_rope = row_mask[:, None] & (rcols[None, :] < ROW_ROPE)
            vals_rope = tl.load(
                kv_rope + rows[:, None] * ROW_ROPE + rcols[None, :],
                mask=mask_rope,
                other=0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )
            tl.store(
                out_rope + dests[:, None] * ROW_ROPE + rcols[None, :],
                vals_rope,
                mask=mask_rope,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )


def run(
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    dest_loc: torch.Tensor,
    o_nope: torch.Tensor,
    o_rope: torch.Tensor,
):
    out_nope = torch.empty_like(o_nope)
    out_rope = torch.empty_like(o_rope)

    T = kv_nope.shape[0]
    row_nope = kv_nope.numel() // T
    row_rope = kv_rope.numel() // T

    BLOCK_T = 8
    BLOCK_COL = 512
    BLOCK_ROPE = _next_power_of_2(row_rope)

    EVEN_T = (T % BLOCK_T) == 0
    EVEN_NOPE = EVEN_T and ((row_nope % BLOCK_COL) == 0)
    EVEN_ROPE = EVEN_T and (row_rope == BLOCK_ROPE)

    num_warps = 8
    num_stages = 2

    grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(row_nope, BLOCK_COL))
    _destindex_kernel[grid](
        kv_nope,
        kv_rope,
        dest_loc,
        out_nope,
        out_rope,
        T,
        ROW_NOPE=row_nope,
        ROW_ROPE=row_rope,
        BLOCK_T=BLOCK_T,
        BLOCK_COL=BLOCK_COL,
        BLOCK_ROPE=BLOCK_ROPE,
        EVEN_T=EVEN_T,
        EVEN_NOPE=EVEN_NOPE,
        EVEN_ROPE=EVEN_ROPE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_T": BLOCK_T,
            "BLOCK_COL": BLOCK_COL,
            "BLOCK_ROPE": BLOCK_ROPE,
            "EVEN_T": EVEN_T,
            "EVEN_NOPE": EVEN_NOPE,
            "EVEN_ROPE": EVEN_ROPE,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    )
    return out_nope, out_rope


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
