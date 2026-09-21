import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
from tilelang.math import next_power_of_2

_DEFAULT_CONFIG = {"threads": 128}
_last_autotune_config: dict = {}


def moe_topk_gating_configs():
    threads=[32, 64, 128]
    return [
        dict(threads=nt)
        for nt in threads
    ]

@tilelang.autotune(configs=moe_topk_gating_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def moe_topk_gating_kernel(logits, topk_w, topk_idx, dtype, BLOCK_SIZE_E: int, BLOCK_SIZE_K: int, threads: int = 128):
    
    M, E, K = T.const("M, E, K")
    logits: T.Tensor((M, E), dtype)
    topk_w: T.Tensor((M, K), dtype)
    topk_idx: T.Tensor((M, K), "int32")
    with T.Kernel(M, threads=threads) as pid:
        logits_reg = T.alloc_fragment((BLOCK_SIZE_E,), "float32")
        topk_vals = T.alloc_fragment((BLOCK_SIZE_K,), "float32")
        topk_idxs = T.alloc_fragment((BLOCK_SIZE_K,), "int32")
        src_idx = T.alloc_fragment((BLOCK_SIZE_E,), "int32")
        curr_max_val = T.alloc_fragment((1,), "float32")
        curr_max_idx = T.alloc_fragment((1,), "int32")
        mx = T.alloc_fragment((1,), "float32")
        rs = T.alloc_fragment((1,), "float32")
        T.fill(logits_reg, -T.infinity("float32"))
        T.fill(topk_vals, -T.infinity("float32"))
        T.fill(topk_idxs, 0)
        T.copy(logits[pid : pid + 1, 0:E], logits_reg)

        for i in T.serial(K):
            T.reduce_max(logits_reg, curr_max_val, dim=0, clear=True)
            T.fill(src_idx, E)
            for j in T.Parallel(BLOCK_SIZE_E):
                src_idx[j] = T.Select(logits_reg[j] == curr_max_val[0], j, E)
            T.reduce_min(src_idx, curr_max_idx, dim=0, clear=True)

            for j in T.Parallel(BLOCK_SIZE_K):
                topk_vals[j] = T.Select(j == i, curr_max_val[0], topk_vals[j])
                topk_idxs[j] = T.Select(j == i, curr_max_idx[0], topk_idxs[j])
            for j in T.Parallel(BLOCK_SIZE_E):
                logits_reg[j] = T.Select(j == curr_max_idx[0], -T.infinity("float32"), logits_reg[j])

        T.reduce_max(topk_vals, mx, dim=0, clear=True)
        for i in T.Parallel(BLOCK_SIZE_K):
            topk_vals[i] = T.exp(topk_vals[i] - mx[0])
        T.reduce_sum(topk_vals, rs, dim=0, clear=True)
        for i in T.Parallel(BLOCK_SIZE_K):
            topk_vals[i] = topk_vals[i] / rs[0]
            topk_w[pid, i] = T.cast(topk_vals[i], dtype)

        T.copy(topk_idxs, topk_idx[pid: pid + 1, 0:K])


def run(logits: torch.Tensor, M: int, E: int, k: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    dtype = str(logits.dtype).removeprefix("torch.")
    topk_weights = torch.empty(M, k, dtype=logits.dtype, device=logits.device)
    topk_indices = torch.empty(M, k, dtype=torch.int32, device=logits.device)

    block_size_e = next_power_of_2(E)
    block_size_k = next_power_of_2(k)

    if autotune:
        with set_autotune_inputs(logits, topk_weights, topk_indices):
            kernel = moe_topk_gating_kernel.compile(
                logits, topk_weights, topk_indices,
                dtype=dtype,
                BLOCK_SIZE_E=block_size_e,
                BLOCK_SIZE_K=block_size_k,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(logits, topk_weights, topk_indices)
    else:
        _last_autotune_config.clear()
        cfg = dict(_DEFAULT_CONFIG)
        moe_topk_gating_kernel(
            logits, topk_weights, topk_indices, dtype,
            BLOCK_SIZE_E=block_size_e,
            BLOCK_SIZE_K=block_size_k,
            threads=cfg["threads"],
        )

    return (topk_weights, topk_indices)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
