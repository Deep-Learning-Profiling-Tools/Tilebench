import torch


def run(
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    dest_loc: torch.Tensor,
    o_nope: torch.Tensor,
    o_rope: torch.Tensor,
):
    out_nope = o_nope.clone()
    out_rope = o_rope.clone()
    dest = dest_loc.to(torch.long)
    out_nope.index_copy_(0, dest, kv_nope)
    out_rope.index_copy_(0, dest, kv_rope)
    return out_nope, out_rope
