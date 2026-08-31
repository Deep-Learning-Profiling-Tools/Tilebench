import functools
import os
import re
import subprocess

from types import SimpleNamespace

import torch

from core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

TILE_M = 128
TILE_K = 128
TILE_N = 512


@functools.lru_cache(maxsize=1)
def _lnc_degree() -> int:
    explicit = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "")
    if explicit.strip().isdigit():
        return int(explicit.strip())
    match = re.search(r"--lnc[=\s]+(\d+)", os.environ.get("NEURON_CC_FLAGS", ""))
    if match:
        return int(match.group(1))
    target = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "").strip().lower()
    if target in ("trn2", "gen3", "trn3", "gen4"):
        return 2
    try:
        out = subprocess.run(["neuron-ls"], capture_output=True, text=True, timeout=10).stdout
        lnc = re.search(r"logical-neuroncore-config:\s*(\d+)", out)
        if lnc:
            return int(lnc.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    return 1


if nki is not None:
    @nki.jit
    def matmul_kernel(lhs, rhs, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K,
                      NUM_CORES=1):
        M, K = lhs.shape
        K_rhs, N = rhs.shape

        BLOCK_M = TILE_M * TILES_IN_BLOCK_M
        BLOCK_N = TILE_N * TILES_IN_BLOCK_N
        BLOCK_K = TILE_K * TILES_IN_BLOCK_K

        assert K == K_rhs, "lhs and rhs must share the contraction dimension"
        assert M % BLOCK_M == 0, "M must be a multiple of BLOCK_M"
        assert N % BLOCK_N == 0, "N must be a multiple of BLOCK_N"
        assert K % BLOCK_K == 0, "K must be a multiple of BLOCK_K"

        NUM_BLOCK_M = M // BLOCK_M
        NUM_BLOCK_N = N // BLOCK_N
        NUM_BLOCK_K = K // BLOCK_K

        assert NUM_BLOCK_M % NUM_CORES == 0, "M blocks must divide across cores"
        BLOCKS_PER_CORE = NUM_BLOCK_M // NUM_CORES

        result = nl.ndarray((M, N), dtype=lhs.dtype, buffer=nl.shared_hbm)

        cast_on_store = lhs.dtype != nl.float32

        core = nl.program_id(0)
        for mi in range(BLOCKS_PER_CORE):
            m_row0 = (core * BLOCKS_PER_CORE + mi) * BLOCK_M

            result_tiles = nl.ndarray((TILE_M, TILES_IN_BLOCK_M, N),
                                      dtype=nl.float32, buffer=nl.sbuf)

            for k in range(NUM_BLOCK_K):
                lhsT_tiles = nl.ndarray((TILE_M, TILES_IN_BLOCK_M, BLOCK_K),
                                        dtype=lhs.dtype, buffer=nl.sbuf)

                for bm in range(TILES_IN_BLOCK_M):
                    nisa.dma_copy(
                        dst=lhsT_tiles[0:TILE_M, bm, 0:BLOCK_K],
                        src=lhs[nl.ds(m_row0 + bm * TILE_M, TILE_M),
                                k * BLOCK_K:(k + 1) * BLOCK_K],
                    )
                    for bk in range(TILES_IN_BLOCK_K):
                        col0 = bk * TILE_M
                        t_psum = nl.ndarray((TILE_K, TILE_M), dtype=lhs.dtype,
                                            buffer=nl.psum)
                        nisa.nc_transpose(
                            dst=t_psum,
                            data=lhsT_tiles[0:TILE_M, bm, col0:col0 + TILE_M],
                        )
                        nisa.tensor_copy(
                            dst=lhsT_tiles[0:TILE_M, bm, col0:col0 + TILE_M],
                            src=t_psum,
                        )

                for n in range(NUM_BLOCK_N):
                    rhs_tiles = nl.ndarray((TILE_K, TILES_IN_BLOCK_K, BLOCK_N),
                                           dtype=rhs.dtype, buffer=nl.sbuf)
                    for bk in range(TILES_IN_BLOCK_K):
                        row0 = (k * TILES_IN_BLOCK_K + bk) * TILE_K
                        nisa.dma_copy(
                            dst=rhs_tiles[0:TILE_K, bk, 0:BLOCK_N],
                            src=rhs[row0:row0 + TILE_K,
                                    n * BLOCK_N:(n + 1) * BLOCK_N],
                        )

                    for bm in range(TILES_IN_BLOCK_M):
                        for bn in range(TILES_IN_BLOCK_N):
                            res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32,
                                                  buffer=nl.psum)
                            for bk in range(TILES_IN_BLOCK_K):
                                nisa.nc_matmul(
                                    dst=res_psum,
                                    stationary=lhsT_tiles[0:TILE_K, bm,
                                                          bk * TILE_M:(bk + 1) * TILE_M],
                                    moving=rhs_tiles[0:TILE_K, bk,
                                                     bn * TILE_N:(bn + 1) * TILE_N],
                                    accumulate=(bk > 0),
                                )

                            col0 = n * BLOCK_N + bn * TILE_N
                            acc = result_tiles[0:TILE_M, bm, col0:col0 + TILE_N]
                            if k == 0:
                                nisa.tensor_copy(dst=acc, src=res_psum)
                            else:
                                nisa.tensor_tensor(dst=acc, data1=acc, data2=res_psum,
                                                   op=nl.add)

            for bm in range(TILES_IN_BLOCK_M):
                row0 = m_row0 + bm * TILE_M
                if cast_on_store:
                    out_tile = nl.ndarray((TILE_M, BLOCK_N), dtype=lhs.dtype,
                                          buffer=nl.sbuf)
                    for n in range(NUM_BLOCK_N):
                        col0 = n * BLOCK_N
                        nisa.tensor_copy(
                            dst=out_tile,
                            src=result_tiles[0:TILE_M, bm, col0:col0 + BLOCK_N],
                        )
                        nisa.dma_copy(dst=result[nl.ds(row0, TILE_M),
                                                 col0:col0 + BLOCK_N],
                                      src=out_tile)
                else:
                    nisa.dma_copy(dst=result[nl.ds(row0, TILE_M), 0:N],
                                  src=result_tiles[0:TILE_M, bm, 0:N])

        return result


def _pick_tiles_in_block(dim: int, tile: int, preferred: int) -> int:
    for t in range(preferred, 0, -1):
        if dim % (tile * t) == 0:
            return t
    return 1


_tuner = NkiAutotuner(matmul_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible dimensions")
    if a.dtype != b.dtype:
        raise ValueError("Incompatible dtypes")
    if a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise NotImplementedError(
            "matmul NKI: fp8 is not supported on trn2 (neuronxcc rejects "
            "F8E4M3FN before TRN3; no other fp8 dtype is in the sweep, so no "
            "fp8 path is kept)"
        )

    M, K = a.shape
    _, N = b.shape

    if M % TILE_M or K % TILE_K or N % TILE_N:
        raise NotImplementedError(
            f"matmul NKI: shape ({M}, {N}, {K}) must be a multiple of "
            f"({TILE_M}, {TILE_N}, {TILE_K})"
        )

    tib_m = _pick_tiles_in_block(M, TILE_M, 4)
    tib_n = _pick_tiles_in_block(N, TILE_N, 2)
    tib_k = _pick_tiles_in_block(K, TILE_K, 8)
    _default = SimpleNamespace(block_size_m=TILE_M * tib_m, block_size_n=TILE_N * tib_n,
                               block_size_k=TILE_K * tib_k)
    if autotune:
        _space = [SimpleNamespace(block_size_m=bm, block_size_n=bn, block_size_k=bk)
                  for bm in (256, 512) for bn in (512, 1024) for bk in (512, 1024)
                  if M % bm == 0 and N % bn == 0 and K % bk == 0]
        if not any(vars(c) == vars(_default) for c in _space):
            _space.append(_default)
        cfg = _tuner.tune_or_cached(
            shape_key=((M, N, K), str(a.dtype)),
            search_space=_space,
            args_fn=lambda cfg: (a, b, cfg.block_size_m // TILE_M, cfg.block_size_n // TILE_N,
                                cfg.block_size_k // TILE_K, 1),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _default
    tib_m, tib_n, tib_k = (cfg.block_size_m // TILE_M, cfg.block_size_n // TILE_N,
                           cfg.block_size_k // TILE_K)

    num_block_m = M // (TILE_M * tib_m)
    lnc = _lnc_degree()
    num_cores = lnc if num_block_m % lnc == 0 else 1

    return matmul_kernel[num_cores](a, b, tib_m, tib_n, tib_k, num_cores)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
