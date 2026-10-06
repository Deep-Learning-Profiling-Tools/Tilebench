import functools
import os
import re
import subprocess

import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from nki.isa.constants import oob_mode
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None
    PMAX = 128

COLS = 16
SKIP_INDEX = 1 << 30


@functools.lru_cache(maxsize=1)
def _lnc_degree() -> int:
    explicit = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "")
    if explicit.strip().isdigit():
        return int(explicit.strip())
    match = re.search(r"--lnc[=\s]+(\d+)", os.environ.get("NEURON_CC_FLAGS", ""))
    if match:
        return int(match.group(1))
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
    def histogram_kernel(values, num_bins):
        N = values.shape[0]
        num_programs = nl.num_programs()
        pid = nl.program_id(0)
        partial = nl.ndarray((num_programs * PMAX * num_bins, 1), dtype=nl.int32, buffer=nl.shared_hbm)

        zeros = nl.ndarray((PMAX, num_bins), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=zeros, value=0)
        nisa.dma_copy(dst=partial.ap(pattern=[[num_bins, PMAX], [1, num_bins]], offset=pid * PMAX * num_bins),
                      src=zeros)
        ones = nl.ndarray((PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1)
        row_base = nl.ndarray((PMAX, COLS), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=row_base, pattern=[[0, COLS]], offset=pid * PMAX * num_bins, channel_multiplier=num_bins)

        per_core = (N + num_programs - 1) // num_programs
        lo = min(N, pid * per_core)
        hi = min(N, lo + per_core)
        block = PMAX * COLS
        n_blocks = (hi - lo) // block

        v = nl.ndarray((PMAX, COLS), dtype=nl.int32, buffer=nl.sbuf)
        idx = nl.ndarray((PMAX, COLS), dtype=nl.int32, buffer=nl.sbuf)
        off = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=off, value=0)

        def _block(it):
            nisa.dma_copy(dst=v, src=values.ap(pattern=[[COLS, PMAX], [1, COLS]], offset=lo,
                                               scalar_offset=off, indirect_dim=0))
            nisa.tensor_tensor(dst=idx, data1=v, data2=row_base, op=nl.add)
            for j in range(COLS):
                dst = partial.ap(pattern=[[1, PMAX], [1, 1]], offset=0,
                                 vector_offset=idx[0:PMAX, j:j + 1], indirect_dim=0)
                nisa.dma_compute(dst=dst, srcs=[dst, ones], reduce_op=nl.add)
            nisa.tensor_scalar(dst=off, data=off, op0=nl.add, operand0=block)

        if n_blocks > 0:
            nl.fori_loop(0, n_blocks, _block)

        start = lo + n_blocks * block
        while start < hi:
            r = min(PMAX, hi - start)
            t = nl.ndarray((PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(dst=t, value=SKIP_INDEX)
            nisa.dma_copy(dst=t[0:r, 0:1], src=values.ap(pattern=[[1, r], [1, 1]], offset=start))
            nisa.tensor_tensor(dst=t, data1=t, data2=row_base[0:PMAX, 0:1], op=nl.add)
            dst = partial.ap(pattern=[[1, PMAX], [1, 1]], offset=0, vector_offset=t, indirect_dim=0)
            nisa.dma_compute(dst=dst, srcs=[dst, ones], reduce_op=nl.add, oob_mode=oob_mode.skip)
            start += r
        return partial


_kernel = histogram_kernel[_lnc_degree()] if nki is not None else None


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs) -> torch.Tensor:
    assert input.ndim == 1 and input.shape[0] == N and input.dtype == torch.int32
    partials = _kernel(input.view(N, 1), num_bins)
    return partials.view(-1, num_bins).sum(dim=0, dtype=torch.int32)


def get_last_config() -> dict | None:
    return None
