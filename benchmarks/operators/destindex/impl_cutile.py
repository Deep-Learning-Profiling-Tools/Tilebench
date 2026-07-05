"""cuTile implementation of destindex using ct.gather + ct.scatter.

Operation: out[dest_loc[token_id], head_id, :] = kv[token_id, head_id, :]

ct.store() only accepts static indices, so the dynamic scatter index
dest_loc[token_id] cannot be used with it. ct.scatter() supports integer
tile or scalar indices (including runtime-loaded values), which is exactly
what we need here.

Algorithm (one CTA per (token, head) pair) — mirrors Triton's d-axis loop:
  1. dest_index = ct.gather(dest_loc, token_id)                ← runtime scalar
  2. for d_start in range(0, HEAD_DIM, BLOCK_D):
       offsets = d_start + ct.arange(BLOCK_D)
       kv_vals = ct.gather(kv,  (token_id,   head_id, offsets), padding_value=0.0)
       ct.scatter(out, (dest_index, head_id, offsets), kv_vals)
"""
from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

# Field names match get_last_config()'s FLAT keys exactly: the NCU harness
# replays the autotune winner by merging that dict into _DEFAULT_CONFIG, so
# nested/mismatched names would silently profile the default config instead.
_DEFAULT_CONFIG = SimpleNamespace(
    nope_block_d=64, nope_occupancy=8,
    rope_block_d=64, rope_occupancy=8,
)

_SEARCH_SPACE = [
    SimpleNamespace(block_d=bd, occupancy=occ)
    for bd in [32, 64, 128]
    for occ in [2, 4, 8, 16]
]


@ct.kernel
def copy_by_dest_kernel(kv, dest_loc, out, HEAD_DIM: ConstInt, BLOCK_D: ConstInt):
    """One CTA copies one (token, head) slice, tiling the head dim in BLOCK_D chunks."""
    token_id = ct.bid(0)
    head_id  = ct.bid(1)

    # Load the destination row index from dest_loc at runtime.
    dest_index = ct.gather(dest_loc, token_id)

    for d_start in range(0, HEAD_DIM, BLOCK_D):
        offsets = d_start + ct.arange(BLOCK_D, dtype=ct.int32)
        # padding_value=0 (int literal) is dtype-agnostic — cuTile auto-casts it
        # to kv.dtype, working for fp16/bf16/fp32 and int8 alike. Handles BLOCK_D
        # > HEAD_DIM; OOB scatter writes are silently dropped by cuTile, so
        # out-of-range lanes produce no side effect.
        kv_vals = ct.gather(kv, (token_id, head_id, offsets), padding_value=0)
        ct.scatter(out, (dest_index, head_id, offsets), kv_vals)


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
# Same kernel is launched twice (nope + rope) with different shapes, so the
# two shape_keys share this single tuner's caches.
_tuner = CutileAutotuner(copy_by_dest_kernel)


def run(
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    dest_loc: torch.Tensor,
    o_nope: torch.Tensor,
    o_rope: torch.Tensor,
    autotune: bool = False,
):

    out_nope = o_nope.clone()
    out_rope = o_rope.clone()

    seq_len, nope_head_num, nope_head_dim = kv_nope.shape
    _, rope_head_num, rope_head_dim = kv_rope.shape

    stream = torch.cuda.current_stream()

    if autotune:
        nope_cfg = _tuner.tune_or_cached(
            shape_key=("nope", seq_len, nope_head_num, nope_head_dim),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (seq_len, nope_head_num, 1),
            args_fn=lambda cfg: (kv_nope, dest_loc, out_nope, nope_head_dim, cfg.block_d),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        rope_cfg = _tuner.tune_or_cached(
            shape_key=("rope", seq_len, rope_head_num, rope_head_dim),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (seq_len, rope_head_num, 1),
            args_fn=lambda cfg: (kv_rope, dest_loc, out_rope, rope_head_dim, cfg.block_d),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "nope_block_d": nope_cfg.block_d, "nope_occupancy": nope_cfg.occupancy,
            "rope_block_d": rope_cfg.block_d, "rope_occupancy": rope_cfg.occupancy,
        })
    else:
        nope_cfg = SimpleNamespace(block_d=_DEFAULT_CONFIG.nope_block_d,
                                   occupancy=_DEFAULT_CONFIG.nope_occupancy)
        rope_cfg = SimpleNamespace(block_d=_DEFAULT_CONFIG.rope_block_d,
                                   occupancy=_DEFAULT_CONFIG.rope_occupancy)

    nope_kernel = _tuner.kernel_with_hints(occupancy=nope_cfg.occupancy)
    rope_kernel = _tuner.kernel_with_hints(occupancy=rope_cfg.occupancy)
    ct.launch(stream, (seq_len, nope_head_num, 1), nope_kernel,
              (kv_nope, dest_loc, out_nope, nope_head_dim, nope_cfg.block_d))
    ct.launch(stream, (seq_len, rope_head_num, 1), rope_kernel,
              (kv_rope, dest_loc, out_rope, rope_head_dim, rope_cfg.block_d))

    return out_nope, out_rope


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
