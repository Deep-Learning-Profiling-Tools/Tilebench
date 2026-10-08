from types import SimpleNamespace

import cuda.tile as ct
import torch

from tilebench.core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}


_DEFAULT_CONFIGS = {
    torch.float32:        SimpleNamespace(tm=128,  tn=128, tk=32,  group_size_m=8, occupancy=16),
    torch.float16:        SimpleNamespace(tm=128, tn=128, tk=64,  group_size_m=8, occupancy=16),
    torch.float8_e4m3fn:  SimpleNamespace(tm=128, tn=256, tk=64, group_size_m=8, occupancy=16),
}


_SEARCH_SPACE = [
    SimpleNamespace(tm=tm, tn=tn, tk=tk, group_size_m=gs, occupancy=occ)
    for tm in [128, 256]
    for tn in [128, 256]
    for tk in [64, 128]
    for gs in [8]
    for occ in [4, 8]
] + [


    SimpleNamespace(tm=128, tn=128, tk=32, group_size_m=8, occupancy=occ)
    for occ in [4, 8]
]


@ct.kernel
def matmul_kernel(
    A, B, C,
    M, N,
    K: ConstInt,
    TM: ConstInt, TN: ConstInt, TK: ConstInt,
    GROUP_SIZE_M: ConstInt,
):
    pid = ct.bid(0)
    num_pid_m = ct.cdiv(M, TM)
    num_pid_n = ct.cdiv(N, TN)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    acc = ct.zeros((TM, TN), dtype=ct.float32)
    num_tiles_k = ct.cdiv(K, TK)


    mma_dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k in range(num_tiles_k):
        a_tile = ct.load(A, index=(pid_m, k), shape=(TM, TK),
                         padding_mode=ct.PaddingMode.ZERO).astype(mma_dtype)
        b_tile = ct.load(B, index=(k, pid_n), shape=(TK, TN),
                         padding_mode=ct.PaddingMode.ZERO).astype(mma_dtype)
        acc = ct.mma(a_tile, b_tile, acc)

    ct.store(C, index=(pid_m, pid_n), tile=ct.astype(acc, C.dtype))


_tuner = CutileAutotuner(matmul_kernel)


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False) -> torch.Tensor:

    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(M, N, K, str(a.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (
                ((M + cfg.tm - 1) // cfg.tm) * ((N + cfg.tn - 1) // cfg.tn),
                1, 1,
            ),
            args_fn=lambda cfg: (
                a, b, c, M, N, K, cfg.tm, cfg.tn, cfg.tk, cfg.group_size_m,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tm":           cfg.tm,
            "tn":           cfg.tn,
            "tk":           cfg.tk,
            "group_size_m": cfg.group_size_m,
            "occupancy":    cfg.occupancy,
        })
    else:
        if a.dtype not in _DEFAULT_CONFIGS:
            raise ValueError(f"No default config for dtype {a.dtype}")
        cfg = _DEFAULT_CONFIGS[a.dtype]

    grid = (
        ((M + cfg.tm - 1) // cfg.tm) * ((N + cfg.tn - 1) // cfg.tn),
        1, 1,
    )
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream, grid, kernel,
        (a, b, c, M, N, K, cfg.tm, cfg.tn, cfg.tk, cfg.group_size_m),
    )
    return c


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
