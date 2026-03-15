"""cuTile implementation of destindex using ct.gather + ct.scatter.

Operation: out[dest_loc[token_id], head_id, :] = kv[token_id, head_id, :]

ct.store() only accepts static indices, so the dynamic scatter index
dest_loc[token_id] cannot be used with it. ct.scatter() supports integer
tile or scalar indices (including runtime-loaded values), which is exactly
what we need here.

Algorithm (one CTA per (token, head) pair):
  1. dest_index = ct.gather(dest_loc, token_id)       ← runtime scalar
  2. kv_vals    = ct.gather(kv, (token_id, head_id, offsets))
  3. ct.scatter(out, (dest_index, head_id, offsets), kv_vals)
"""
from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(occupancy=2)

_SEARCH_SPACE = [SimpleNamespace(occupancy=occ) for occ in [1, 2, 4, 8]]


@ct.kernel
def _copy_by_dest_kernel(kv, dest_loc, out, HEAD_DIM: ConstInt):
    """One CTA copies one (token, head) slice to a dynamically-indexed destination."""
    token_id = ct.bid(0)
    head_id  = ct.bid(1)

    # Load the destination row index from dest_loc at runtime.
    dest_index = ct.gather(dest_loc, token_id)

    # Column offsets for the head dimension.
    offsets = ct.arange(HEAD_DIM, dtype=np.int32)

    # Gather KV values for this (token, head) slice.
    kv_vals = ct.gather(kv, (token_id, head_id, offsets))

    # Scatter to the output at the dynamic destination index.
    ct.scatter(out, (dest_index, head_id, offsets), kv_vals)


def run(
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    dest_loc: torch.Tensor,
    o_nope: torch.Tensor,
    o_rope: torch.Tensor,
    autotune: bool = False,
):
    global _last_autotune_config

    out_nope = o_nope.clone()
    out_rope = o_rope.clone()

    seq_len, nope_head_num, nope_head_dim = kv_nope.shape
    _, rope_head_num, rope_head_dim = kv_rope.shape

    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        nope_result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (seq_len, nope_head_num, 1),
            kernel=_copy_by_dest_kernel,
            args_fn=lambda cfg: (kv_nope, dest_loc, out_nope, nope_head_dim),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        rope_result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (seq_len, rope_head_num, 1),
            kernel=_copy_by_dest_kernel,
            args_fn=lambda cfg: (kv_rope, dest_loc, out_rope, rope_head_dim),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {
            "nope": {"occupancy": nope_result.tuned_config.occupancy},
            "rope": {"occupancy": rope_result.tuned_config.occupancy},
        }
    else:
        ct.launch(stream, (seq_len, nope_head_num, 1), _copy_by_dest_kernel,
                  (kv_nope, dest_loc, out_nope, nope_head_dim))
        ct.launch(stream, (seq_len, rope_head_num, 1), _copy_by_dest_kernel,
                  (kv_rope, dest_loc, out_rope, rope_head_dim))

    return out_nope, out_rope


def get_last_config() -> dict | None:
    return _last_autotune_config
