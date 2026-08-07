from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(
    tm=256, tn=64, tk=32, group_size_m=8, occupancy=16,
)

_SEARCH_SPACE = [
    SimpleNamespace(tm=tm, tn=tn, tk=tk, group_size_m=8, occupancy=occ)
    for tm in [64, 128, 256]
    for tn in [64, 128, 256]
    for tk in [32, 64]
    for occ in [4, 8, 16]
]


@ct.kernel
def matmul_kernel(
    A, B, C,
    M, N,
    K_b: ConstInt,
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

    acc = ct.zeros((TM, TN), dtype=ct.int32)
    num_tiles_kb = ct.cdiv(K_b, TK)


    for j in range(num_tiles_kb):
        B_tile = ct.load(
            B, index=(j, pid_n), shape=(TK, TN),
            padding_mode=ct.PaddingMode.ZERO,
        )
        B_int32 = ct.astype(B_tile, ct.int32)
        for i in range(4):
            k_a = i * num_tiles_kb + j
            A_tile = ct.load(
                A, index=(pid_m, k_a), shape=(TM, TK),
                padding_mode=ct.PaddingMode.ZERO,
            )
            mask_i = 3 << (2 * i)


            b_unpacked = ct.astype(((B_int32 & mask_i) >> (2 * i)) - 1, ct.int8)
            acc = ct.mma(A_tile, b_unpacked, acc)

    ct.store(C, index=(pid_m, pid_n), tile=acc)


_tuner = CutileAutotuner(matmul_kernel)


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False) -> torch.Tensor:

    assert a.shape[1] == b.shape[0] * 4, (
        "Incompatible dims: A's K must equal 4 * B's K_b (B is packed 4-per-byte)"
    )
    M, K = a.shape
    K_b, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)

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
                a, b, c, M, N, K_b, cfg.tm, cfg.tn, cfg.tk, cfg.group_size_m,
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
        cfg = _DEFAULT_CONFIG

    grid = (
        ((M + cfg.tm - 1) // cfg.tm) * ((N + cfg.tn - 1) // cfg.tn),
        1, 1,
    )
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream, grid, kernel,
        (a, b, c, M, N, K_b, cfg.tm, cfg.tn, cfg.tk, cfg.group_size_m),
    )
    return c


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
