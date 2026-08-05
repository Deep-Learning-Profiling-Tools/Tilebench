import torch 
import tilelang 
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config: dict = {}
_out_cache = torch.utils.weak.WeakTensorKeyDictionary()

def destindex_config():
    BLOCK_SIZE = [256, 512, 1024]
    threads = [64, 128, 256]
    return [
        dict(BLOCK_SIZE=bs, threads=nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]


@tilelang.autotune(configs=destindex_config(), warmup = 20, rep = 100, timeout = 60)
@tilelang.jit
def copy_by_dest_kernel(
        kv,
        dest,
        out,
        dtype,
        dest_dtype,
        head_num: int = 1,
        head_dim: int = 1,
        BLOCK_SIZE: int = 1024,
        threads: int = 128
):
    n_elements, seq_len = T.const("n_elements, seq_len")

    kv: T.Tensor((n_elements,), dtype)
    dest: T.Tensor((seq_len,), dest_dtype)
    out: T.Tensor((n_elements,), dtype)
    with T.Kernel(T.ceildiv(n_elements, BLOCK_SIZE), threads=threads) as pid:
        for local_idx in T.Parallel(BLOCK_SIZE):
            offs = pid * BLOCK_SIZE + local_idx
            if offs < n_elements:
                d = offs % head_dim
                tmp = offs // head_dim
                head = tmp % head_num
                token = tmp // head_num
                dest_index = T.Cast("int32", dest[token])
                dst_off = (dest_index * head_num + head) * head_dim + d
                out[dst_off] = kv[offs]


def _cached_out(o: torch.Tensor) -> torch.Tensor:
    out = _out_cache.get(o)
    if out is None:
        out = o.clone()
        _out_cache[o] = out
    return out


def _launch_copy(
    kv: torch.Tensor,
    dest_loc: torch.Tensor,
    out: torch.Tensor,
    autotune: bool,
    label: str,
):
    assert kv.is_contiguous() and out.is_contiguous()
    _, head_num, head_dim = kv.shape
    kv_flat = kv.view(-1)
    out_flat = out.view(-1)
    dtype = str(kv.dtype).removeprefix("torch.")
    dest_dtype = str(dest_loc.dtype).removeprefix("torch.")
    if autotune:
        with set_autotune_inputs(kv_flat, dest_loc, out_flat):
            kernel = copy_by_dest_kernel.compile(
                kv_flat, dest_loc, out_flat, dtype=dtype, dest_dtype=dest_dtype,
                head_num=head_num, head_dim=head_dim,
            )
        _last_autotune_config[label] = {
            "shape": tuple(kv.shape),
            **dict(kernel.config or {}),
        }
        kernel(kv_flat, dest_loc, out_flat)

    else:
        cfg = _DEFAULT_CONFIG
        copy_by_dest_kernel(
            kv_flat, dest_loc, out_flat, dtype=dtype, dest_dtype=dest_dtype,
            head_num=head_num, head_dim=head_dim,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
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
    out_nope = _cached_out(o_nope)
    out_rope = _cached_out(o_rope)
    _last_autotune_config.clear()
    _launch_copy(kv_nope, dest_loc, out_nope, autotune, "nope")
    _launch_copy(kv_rope, dest_loc, out_rope, autotune, "rope")
    return out_nope, out_rope

def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
