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
    BLOCK_NOPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)

    offs_nope = tl.arange(0, BLOCK_NOPE)
    mask_nope = offs_nope < ROW_NOPE

    offs_rope = tl.arange(0, BLOCK_ROPE)
    mask_rope = offs_rope < ROW_ROPE

    for t in tl.range(pid, T, nprog, num_stages=1):
        dest = tl.load(
            dest_loc + t,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )

        vals_nope = tl.load(
            kv_nope + t * ROW_NOPE + offs_nope,
            mask=mask_nope,
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        tl.store(
            out_nope + dest * ROW_NOPE + offs_nope,
            vals_nope,
            mask=mask_nope,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )

        vals_rope = tl.load(
            kv_rope + t * ROW_ROPE + offs_rope,
            mask=mask_rope,
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        tl.store(
            out_rope + dest * ROW_ROPE + offs_rope,
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

    BLOCK_NOPE = max(512, _next_power_of_2(row_nope))
    BLOCK_ROPE = max(512, _next_power_of_2(row_rope))
    MAX_CTAS = 16384
    grid_ctas = min(T, MAX_CTAS)

    num_warps = 8
    num_stages = 2

    _destindex_kernel[(grid_ctas,)](
        kv_nope,
        kv_rope,
        dest_loc,
        out_nope,
        out_rope,
        T,
        ROW_NOPE=row_nope,
        ROW_ROPE=row_rope,
        BLOCK_NOPE=BLOCK_NOPE,
        BLOCK_ROPE=BLOCK_ROPE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_NOPE": BLOCK_NOPE,
            "BLOCK_ROPE": BLOCK_ROPE,
            "MAX_CTAS": MAX_CTAS,
            "GRID_CTAS": grid_ctas,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    )
    return out_nope, out_rope


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
