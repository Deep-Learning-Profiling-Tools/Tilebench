from types import SimpleNamespace

import cuda.tile as ct
import torch

from tilebench.core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}


_DEFAULT_CONFIG = SimpleNamespace(
    nope_block_size=1024, nope_occupancy=4,
    rope_block_size=512, rope_occupancy=4,
)

_SEARCH_SPACE = [
    SimpleNamespace(block_size=bs, occupancy=occ)
    for bs in [256, 512, 1024]
    for occ in [4, 8, 16]
]

_out_cache = torch.utils.weak.WeakTensorKeyDictionary()


def _cached_out(o: torch.Tensor) -> torch.Tensor:
    out = _out_cache.get(o)
    if out is None:
        out = o.clone()
        _out_cache[o] = out
    return out


@ct.kernel
def copy_by_dest_kernel(
    kv_flat,
    dest_loc,
    out_flat,
    head_num: ConstInt,
    head_dim: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    pid = ct.bid(0)
    offs = pid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    total = kv_flat.shape[0]
    mask = offs < total

    d = offs % head_dim
    tmp = offs // head_dim
    head = tmp % head_num
    token = tmp // head_num


    dest = ct.astype(ct.gather(dest_loc, token, padding_value=0), ct.int32)
    dst_off = (dest * head_num + head) * head_dim + d
    dst_off = ct.where(mask, dst_off, -1)

    v = ct.load(kv_flat, index=(pid,), shape=(BLOCK_SIZE,),
                padding_mode=ct.PaddingMode.ZERO)
    ct.scatter(out_flat, dst_off, v)


_tuner = CutileAutotuner(copy_by_dest_kernel)


def run(
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    dest_loc: torch.Tensor,
    o_nope: torch.Tensor,
    o_rope: torch.Tensor,
    autotune: bool = False,
):
    out_nope = _cached_out(o_nope)
    out_rope = _cached_out(o_rope)

    _, nope_head_num, nope_head_dim = kv_nope.shape
    _, rope_head_num, rope_head_dim = kv_rope.shape
    kv_nope_flat = kv_nope.contiguous().view(-1)
    kv_rope_flat = kv_rope.contiguous().view(-1)
    out_nope_flat = out_nope.view(-1)
    out_rope_flat = out_rope.view(-1)
    nope_total = kv_nope_flat.numel()
    rope_total = kv_rope_flat.numel()

    stream = torch.cuda.current_stream()

    if autotune:
        nope_cfg = _tuner.tune_or_cached(
            shape_key=("nope", nope_total, nope_head_num, nope_head_dim,
                       str(kv_nope.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (ct.cdiv(nope_total, cfg.block_size), 1, 1),
            args_fn=lambda cfg: (kv_nope_flat, dest_loc, out_nope_flat,
                                 nope_head_num, nope_head_dim, cfg.block_size),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        rope_cfg = _tuner.tune_or_cached(
            shape_key=("rope", rope_total, rope_head_num, rope_head_dim,
                       str(kv_rope.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (ct.cdiv(rope_total, cfg.block_size), 1, 1),
            args_fn=lambda cfg: (kv_rope_flat, dest_loc, out_rope_flat,
                                 rope_head_num, rope_head_dim, cfg.block_size),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "nope_block_size": nope_cfg.block_size,
            "nope_occupancy":  nope_cfg.occupancy,
            "rope_block_size": rope_cfg.block_size,
            "rope_occupancy":  rope_cfg.occupancy,
        })
    else:
        nope_cfg = SimpleNamespace(block_size=_DEFAULT_CONFIG.nope_block_size,
                                   occupancy=_DEFAULT_CONFIG.nope_occupancy)
        rope_cfg = SimpleNamespace(block_size=_DEFAULT_CONFIG.rope_block_size,
                                   occupancy=_DEFAULT_CONFIG.rope_occupancy)

    kernel_nope = _tuner.kernel_with_hints(occupancy=nope_cfg.occupancy)
    ct.launch(stream, (ct.cdiv(nope_total, nope_cfg.block_size), 1, 1),
              kernel_nope,
              (kv_nope_flat, dest_loc, out_nope_flat,
               nope_head_num, nope_head_dim, nope_cfg.block_size))

    kernel_rope = _tuner.kernel_with_hints(occupancy=rope_cfg.occupancy)
    ct.launch(stream, (ct.cdiv(rope_total, rope_cfg.block_size), 1, 1),
              kernel_rope,
              (kv_rope_flat, dest_loc, out_rope_flat,
               rope_head_num, rope_head_dim, rope_cfg.block_size))

    return out_nope, out_rope


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
