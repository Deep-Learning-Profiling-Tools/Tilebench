import os

import torch 
import tilelang 
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
_DEFAULT_CONFIG = {"BLOCK_DMODEL": 64, "threads": 32, "num_stages": 2}
_last_autotune_config: dict = {}
_kernel_cache: dict = {}

def destindex_config():
    BLOCK_DMODEL = [32, 64, 128]
    threads = [32, 64, 128]
    #num_stages = [1, 2]
    return [
        dict(BLOCK_DMODEL=bd, threads=nt)
        for bd in BLOCK_DMODEL
        for nt in threads
        #for ns in num_stages
    ]
@tilelang.autotune(configs=destindex_config(), warmup = 20, rep = 100, timeout = 60)
@tilelang.jit
def copy_by_dest_kernel(
        kv,
        dest,
        out,
        dtype,
        BLOCK_DMODEL: int = 64,
        threads: int = 128
):
    seq_len, head_num, head_dim = T.const("seq_len, head_num, head_dim")

    kv: T.Tensor((seq_len, head_num, head_dim), dtype)
    dest: T.Tensor((seq_len,), "int32")
    out: T.Tensor((seq_len, head_num, head_dim), dtype)
    with T.Kernel(seq_len, head_num, threads=threads) as (token_id, head_id):
        local_out = T.alloc_fragment((BLOCK_DMODEL,), dtype)
        #.copy(dest[seq_len, head_num], local_dest)
        dest_index = dest[token_id]
        for start in T.serial(0, head_dim, BLOCK_DMODEL):
            T.copy(kv[token_id, head_id, start], local_out)
            T.copy(local_out, out[dest_index, head_id, start])


def _kernel_key(kv: torch.Tensor, dest_loc: torch.Tensor, out: torch.Tensor, dtype: str):
    return (
        dtype,
        tuple(kv.shape),
        tuple(kv.stride()),
        tuple(dest_loc.shape),
        tuple(dest_loc.stride()),
        tuple(out.shape),
        tuple(out.stride()),
    )


def _kernel_cache_disabled() -> bool:
    return os.environ.get("TILEBENCH_DESTINDEX_DISABLE_KERNEL_CACHE") == "1"
    
def _launch_copy(
    kv: torch.Tensor,
    dest_loc: torch.Tensor,
    out: torch.Tensor,
    autotune: bool,
    label: str,
):
    #seq_len, head_num, head_dim = kv.shape
    dtype = str(kv.dtype).removeprefix("torch.")
    if autotune:
        key = _kernel_key(kv, dest_loc, out, dtype)
        kernel = None if _kernel_cache_disabled() else _kernel_cache.get(key)
        if kernel is None:
            with set_autotune_inputs(kv, dest_loc, out):
                kernel = copy_by_dest_kernel.compile(
                    kv, dest_loc, out, dtype=dtype
                )
            if not _kernel_cache_disabled():
                _kernel_cache[key] = kernel
        _last_autotune_config[label] = {
            "shape": tuple(kv.shape),
            **dict(kernel.config or {}),
        }
        kernel(kv, dest_loc, out)

    else:
        cfg = _DEFAULT_CONFIG
        copy_by_dest_kernel(
            kv, dest_loc, out, dtype=dtype,
            BLOCK_DMODEL=cfg["BLOCK_DMODEL"],
            threads=cfg["threads"]
        )


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
    _last_autotune_config.clear()
    _launch_copy(kv_nope, dest_loc, out_nope, autotune, "nope")
    _launch_copy(kv_rope, dest_loc, out_rope, autotune, "rope")
    return out_nope, out_rope

def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
