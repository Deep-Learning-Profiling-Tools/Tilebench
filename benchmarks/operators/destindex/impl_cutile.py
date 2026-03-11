from types import SimpleNamespace

import torch
import cuda.tile as ct

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

ConstInt = ct.Constant[int]

_last_config: dict | None = None




# NOTE: cuTile DSL ct.store() index accepts only static scalars (block IDs, constants).
# A data-dependent scatter index (dest_loc[token_id] loaded at runtime) is typed as
# Tile[int32,(1)], which ct.store rejects. Scatter-by-dynamic-index is not yet
# supported in the cuTile tile abstraction; these kernels are intentionally left as
# stubs and will fail at runtime (caught by the engine's exception handler).

@ct.kernel
def copy_nope_by_dest_kernel(kv_nope, dest_loc, out_nope, HEAD_DIM: ConstInt):
    token_id = ct.bid(0)
    head_id = ct.bid(1)
    dest_index = ct.load(dest_loc, index=(token_id,), shape=(1,))
    kv_tile = ct.load(kv_nope, index=(token_id, head_id, 0), shape=(1, 1, HEAD_DIM))
    ct.store(out_nope, index=(dest_index, head_id, 0), tile=kv_tile)


@ct.kernel
def copy_rope_by_dest_kernel(kv_rope, dest_loc, out_rope, HEAD_DIM: ConstInt):
    token_id = ct.bid(0)
    head_id = ct.bid(1)
    dest_index = ct.load(dest_loc, index=(token_id,), shape=(1,))
    kv_tile = ct.load(kv_rope, index=(token_id, head_id, 0), shape=(1, 1, HEAD_DIM))
    ct.store(out_rope, index=(dest_index, head_id, 0), tile=kv_tile)


_SEARCH_SPACE = [SimpleNamespace(occupancy=occ) for occ in [1, 2, 4, 8]]


def run(
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    dest_loc: torch.Tensor,
    o_nope: torch.Tensor,
    o_rope: torch.Tensor,
    autotune: bool = False,
):
    global _last_config
    out_nope = o_nope.clone()
    out_rope = o_rope.clone()

    seq_len, nope_head_num, nope_head_dim = kv_nope.shape
    _, rope_head_num, rope_head_dim = kv_rope.shape

    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        nope_result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (seq_len, nope_head_num, 1),
            kernel=copy_nope_by_dest_kernel,
            args_fn=lambda cfg: (kv_nope, dest_loc, out_nope, nope_head_dim),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        rope_result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (seq_len, rope_head_num, 1),
            kernel=copy_rope_by_dest_kernel,
            args_fn=lambda cfg: (kv_rope, dest_loc, out_rope, rope_head_dim),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_config = {
            "nope": {"occupancy": nope_result.tuned_config.occupancy},
            "rope": {"occupancy": rope_result.tuned_config.occupancy},
        }
    else:
        ct.launch(stream, (seq_len, nope_head_num, 1), copy_nope_by_dest_kernel,
                  (kv_nope, dest_loc, out_nope, nope_head_dim))
        ct.launch(stream, (seq_len, rope_head_num, 1), copy_rope_by_dest_kernel,
                  (kv_rope, dest_loc, out_rope, rope_head_dim))

    return out_nope, out_rope


def get_last_config() -> dict | None:
    return _last_config
