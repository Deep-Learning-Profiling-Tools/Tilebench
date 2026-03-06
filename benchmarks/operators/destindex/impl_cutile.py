import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]


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


def run(
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    dest_loc: torch.Tensor,
    o_nope: torch.Tensor,
    o_rope: torch.Tensor,
    block_size: int = None,
):
    del block_size
    out_nope = o_nope.clone()
    out_rope = o_rope.clone()

    seq_len, nope_head_num, nope_head_dim = kv_nope.shape
    _, rope_head_num, rope_head_dim = kv_rope.shape

    grid_nope = (seq_len, nope_head_num, 1)
    grid_rope = (seq_len, rope_head_num, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid_nope,
        copy_nope_by_dest_kernel,
        (kv_nope, dest_loc, out_nope, nope_head_dim),
    )
    ct.launch(
        torch.cuda.current_stream(),
        grid_rope,
        copy_rope_by_dest_kernel,
        (kv_rope, dest_loc, out_rope, rope_head_dim),
    )
    return out_nope, out_rope
