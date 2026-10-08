import functools
import os
import re
import subprocess
from types import SimpleNamespace

import torch

from tilebench.core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


TILE_M = 128
TILE_K = 128
TILE_N = 512
GROUP_M = 8
K_CHUNK = 8
MAX_A_PANEL_BYTES = 64 * 1024


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


def _cdiv(x, y):
    return (x + y - 1) // y


def _swizzle_tile(tile_id, M, N, TM, TN, GROUP):
    grid_m = _cdiv(M, TM)
    grid_n = _cdiv(N, TN)
    width = GROUP * grid_n
    group_id = tile_id // width
    group_size = min(grid_m - group_id * GROUP, GROUP)
    pid_m = group_id * GROUP + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n


def _streamk_partition(M, N, TM, TN, num_workers):
    total_tiles = _cdiv(M, TM) * _cdiv(N, TN)
    streamk_tiles = total_tiles % num_workers
    if total_tiles - streamk_tiles > num_workers:
        streamk_tiles += num_workers
    return total_tiles, streamk_tiles


def _worker_segments(streamk_tiles, iters_per_tile, num_workers, pid):
    total_iters_streamk = streamk_tiles * iters_per_tile
    total_full_iters = total_iters_streamk // num_workers
    total_partial_iters = total_iters_streamk % num_workers
    start_iter = pid * total_full_iters + min(pid, total_partial_iters)
    last_iter = (pid + 1) * total_full_iters + min(pid + 1, total_partial_iters)
    segments = []
    while start_iter < last_iter:
        tile_id = start_iter // iters_per_tile
        rem = iters_per_tile - (start_iter % iters_per_tile)
        end_iter = min(start_iter + rem, last_iter)
        k_begin = start_iter % iters_per_tile
        segments.append((tile_id, k_begin, k_begin + end_iter - start_iter))
        start_iter = end_iter
    return segments


def streamk_schedule(M, N, K, tn, group_m, num_workers):
    total_tiles, streamk_tiles = _streamk_partition(M, N, TILE_M, tn, num_workers)
    iters_per_tile = _cdiv(K, TILE_K)
    total_iters_streamk = streamk_tiles * iters_per_tile
    segments = [_worker_segments(streamk_tiles, iters_per_tile, num_workers, pid)
                for pid in range(num_workers)]
    return SimpleNamespace(
        total_tiles=total_tiles, streamk_tiles=streamk_tiles, iters_per_tile=iters_per_tile,
        iter_split=(total_iters_streamk // num_workers, total_iters_streamk % num_workers),
        segments=segments,
        partial_tiles=sorted({t for seg in segments for t, kb, ke in seg
                              if kb > 0 or ke < iters_per_tile}),
        full_tiles=[list(range(streamk_tiles + pid, total_tiles, num_workers))
                    for pid in range(num_workers)],
        coords=[_swizzle_tile(t, M, N, TILE_M, tn, group_m) for t in range(total_tiles)],
    )


if nki is not None:

    def _load_a_transposed(a, m0, ms, k_begin, k_end):
        M, K = a.shape
        n_iters = k_end - k_begin
        a_t = nl.ndarray((TILE_K, n_iters, TILE_M), dtype=a.dtype, buffer=nl.sbuf)
        for c0 in range(0, n_iters, K_CHUNK):
            nc = min(K_CHUNK, n_iters - c0)
            k_lo = (k_begin + c0) * TILE_K
            k_hi = min((k_begin + c0 + nc) * TILE_K, K)
            a_rows = nl.ndarray((TILE_M, K_CHUNK * TILE_K), dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=a_rows[0:ms, 0:k_hi - k_lo], src=a[m0:m0 + ms, k_lo:k_hi])
            for j in range(nc):
                ks = min(TILE_K, K - (k_lo + j * TILE_K))
                t_psum = nl.ndarray((TILE_K, TILE_M), dtype=a.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=t_psum[0:ks, 0:ms],
                                  data=a_rows[0:ms, j * TILE_K:j * TILE_K + ks])
                nisa.tensor_copy(dst=a_t[0:ks, c0 + j, 0:ms], src=t_psum[0:ks, 0:ms])
        return a_t

    def _mac_tile(a_t, a_t_k0, b, ms, n0, ns, k_begin, k_end, tn):
        K, N = b.shape
        acc = nl.ndarray((TILE_M, tn), dtype=nl.float32, buffer=nl.psum)
        for c0 in range(k_begin, k_end, K_CHUNK):
            nc = min(K_CHUNK, k_end - c0)
            n_full = min(nc, (K - c0 * TILE_K) // TILE_K)
            b_sb = nl.ndarray((TILE_K, K_CHUNK, tn), dtype=b.dtype, buffer=nl.sbuf)
            if n_full > 0:
                nisa.dma_copy(
                    dst=b_sb[0:TILE_K, 0:n_full, 0:ns],
                    src=b.ap(pattern=[[N, TILE_K], [TILE_K * N, n_full], [1, ns]],
                             offset=c0 * TILE_K * N + n0),
                )
            if n_full < nc:
                k_tail = (c0 + n_full) * TILE_K
                nisa.dma_copy(dst=b_sb[0:K - k_tail, n_full, 0:ns],
                              src=b[k_tail:K, n0:n0 + ns])
            for j in range(nc):
                kk = c0 + j
                ks = min(TILE_K, K - kk * TILE_K)
                nisa.nc_matmul(
                    dst=acc[0:ms, 0:ns],
                    stationary=a_t[0:ks, kk - a_t_k0, 0:ms],
                    moving=b_sb[0:ks, j, 0:ns],
                    accumulate=(kk > k_begin),
                )
        return acc

    def _store_tile(c, src, m0, ms, n0, ns, tn):
        out_sb = nl.ndarray((TILE_M, tn), dtype=c.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_sb[0:ms, 0:ns], src=src[0:ms, 0:ns])
        nisa.dma_copy(dst=c[m0:m0 + ms, n0:n0 + ns], src=out_sb[0:ms, 0:ns])

    @nki.jit
    def streamk_matmul_kernel(a, b, tn, group_m):
        M, K = a.shape
        K_b, N = b.shape
        assert K == K_b, "a and b must share the contraction dimension"
        num_workers = nl.num_programs()
        assert num_workers <= 2, "the fix-up below pairs at most two workers"
        pid = nl.program_id(0)

        c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
        total_tiles, streamk_tiles = _streamk_partition(M, N, TILE_M, tn, num_workers)
        iters_per_tile = _cdiv(K, TILE_K)

        segments = _worker_segments(streamk_tiles, iters_per_tile, num_workers, pid)
        part = nl.ndarray((TILE_M, tn), dtype=nl.float32, buffer=nl.sbuf)
        n_split = 0
        split_k_begin = 0
        split_m0 = 0
        split_ms = 0
        split_n0 = 0
        split_ns = 0
        for s in range(len(segments)):
            tile_id, k_begin, k_end = segments[s]
            pid_m, pid_n = _swizzle_tile(tile_id, M, N, TILE_M, tn, group_m)
            m0 = pid_m * TILE_M
            ms = min(TILE_M, M - m0)
            n0 = pid_n * tn
            ns = min(tn, N - n0)
            a_t = _load_a_transposed(a, m0, ms, k_begin, k_end)
            acc = _mac_tile(a_t, k_begin, b, ms, n0, ns, k_begin, k_end, tn)
            if k_begin == 0 and k_end == iters_per_tile:
                _store_tile(c, acc, m0, ms, n0, ns, tn)
            else:
                nisa.tensor_copy(dst=part[0:ms, 0:ns], src=acc[0:ms, 0:ns])
                n_split = n_split + 1
                split_k_begin = k_begin
                split_m0 = m0
                split_ms = ms
                split_n0 = n0
                split_ns = ns
        assert n_split <= 1, "with two workers at most one tile is split"

        row_cols = []
        for i in range(_cdiv(M, TILE_M)):
            row_cols.append([])
        row_order = []
        for tile_id in range(streamk_tiles + pid, total_tiles, num_workers):
            pid_m, pid_n = _swizzle_tile(tile_id, M, N, TILE_M, tn, group_m)
            if len(row_cols[pid_m]) == 0:
                row_order.append(pid_m)
            row_cols[pid_m].append(pid_n)
        for r in range(len(row_order)):
            pid_m = row_order[r]
            m0 = pid_m * TILE_M
            ms = min(TILE_M, M - m0)
            a_t = _load_a_transposed(a, m0, ms, 0, iters_per_tile)
            pid_ns = row_cols[pid_m]
            for j in range(len(pid_ns)):
                n0 = pid_ns[j] * tn
                ns = min(tn, N - n0)
                acc = _mac_tile(a_t, 0, b, ms, n0, ns, 0, iters_per_tile, tn)
                _store_tile(c, acc, m0, ms, n0, ns, tn)

        if n_split == 1:
            other = nl.ndarray((TILE_M, tn), dtype=nl.float32, buffer=nl.sbuf)
            nisa.sendrecv(src=part[0:split_ms, 0:split_ns], dst=other[0:split_ms, 0:split_ns],
                          send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)
            if split_k_begin == 0:
                total = nl.ndarray((TILE_M, tn), dtype=c.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=total[0:split_ms, 0:split_ns],
                                   data1=part[0:split_ms, 0:split_ns],
                                   data2=other[0:split_ms, 0:split_ns], op=nl.add)
                nisa.dma_copy(dst=c[split_m0:split_m0 + split_ms, split_n0:split_n0 + split_ns],
                              src=total[0:split_ms, 0:split_ns])
        return c


_DEFAULT_CONFIG = SimpleNamespace(tn=TILE_N, group_m=GROUP_M)
_SEARCH_SPACE = [SimpleNamespace(tn=tn, group_m=GROUP_M) for tn in (256, 512)]
_kernel = streamk_matmul_kernel[_lnc_degree()] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible dimensions")
    if a.dtype != b.dtype:
        raise ValueError("Incompatible dtypes")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(f"streamk_matmul NKI: unsupported dtype {a.dtype}")
    M, K = a.shape
    _, N = b.shape
    if _cdiv(K, TILE_K) * TILE_M * a.element_size() > MAX_A_PANEL_BYTES:
        raise NotImplementedError(
            f"streamk_matmul NKI: K={K} too large for the per-row A^T panel kept in SBUF")
    a = a.contiguous()
    b = b.contiguous()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=((M, N, K), str(a.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (a, b, cfg.tn, cfg.group_m),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(a, b, cfg.tn, cfg.group_m)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
