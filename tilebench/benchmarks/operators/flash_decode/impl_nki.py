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

P_MAX = 128
NEG_BIG = 1.0e30


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
    def flash_decode_kernel(mid_o, mid_o_lse, valid_blocks, block_chunk):
        B, H, NB, D = mid_o.shape
        assert D <= 512 and block_chunk <= P_MAX
        out = nl.ndarray((B, H, D), dtype=mid_o.dtype, buffer=nl.shared_hbm)

        pairs = B * H
        n_prog = nl.num_programs()
        pid = nl.program_id(0)
        per_core = (pairs + n_prog - 1) // n_prog
        p_lo = min(pairs, pid * per_core)
        p_hi = min(pairs, p_lo + per_core)

        block_iota = nl.ndarray((1, NB), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=block_iota, pattern=[[1, NB]], offset=0, channel_multiplier=0)
        block_iota_f = nl.ndarray((1, NB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=block_iota_f, src=block_iota)
        one = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=one, value=1.0)
        n_chunks = (NB + block_chunk - 1) // block_chunk

        for p in range(p_lo, p_hi):
            b = p // H

            valid_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=valid_f, src=valid_blocks.ap(pattern=[[1, 1], [1, 1]], offset=b))
            mask = nl.ndarray((1, NB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mask, data=block_iota_f, op0=nl.less, operand0=valid_f)

            lse_raw = nl.ndarray((1, NB), dtype=mid_o_lse.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=lse_raw, src=mid_o_lse.ap(pattern=[[NB, 1], [1, NB]], offset=p * NB))
            lse = nl.ndarray((1, NB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=lse, src=lse_raw)

            penalty = nl.ndarray((1, NB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=penalty, data=mask, op0=nl.subtract, operand0=1.0,
                               op1=nl.multiply, operand1=NEG_BIG)
            nisa.tensor_tensor(dst=lse, data1=lse, data2=mask, op=nl.multiply)
            nisa.tensor_tensor(dst=lse, data1=lse, data2=penalty, op=nl.add)

            gmax = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=gmax, op=nl.maximum, data=lse, axis=(1,))
            w = nl.ndarray((1, NB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=w, data=lse, op0=nl.subtract, operand0=gmax)
            nisa.activation(dst=w, op=nl.exp, data=w)
            nisa.tensor_tensor(dst=w, data1=w, data2=mask, op=nl.multiply)
            denom = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=denom, op=nl.add, data=w, axis=(1,))
            nisa.tensor_scalar(dst=denom, data=denom, op0=nl.add, operand0=1.0e-10)
            inv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv, data=denom)
            nisa.tensor_scalar(dst=w, data=w, op0=nl.multiply, operand0=inv)

            acc = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.psum)
            for c in range(n_chunks):
                c0 = c * block_chunk
                cs = min(block_chunk, NB - c0)
                wcol_ps = nl.ndarray((cs, 1), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=wcol_ps, stationary=w[0:1, c0:c0 + cs], moving=one)
                wcol = nl.ndarray((cs, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=wcol, src=wcol_ps)
                mo = nl.ndarray((cs, D), dtype=mid_o.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=mo, src=mid_o.ap(pattern=[[D, cs], [1, D]], offset=(p * NB + c0) * D))
                nisa.nc_matmul(dst=acc, stationary=wcol, moving=mo, accumulate=(c > 0))

            res = nl.ndarray((1, D), dtype=mid_o.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res, src=acc)
            nisa.dma_copy(dst=out.ap(pattern=[[D, 1], [1, D]], offset=p * D), src=res)

        return out


_DEFAULT_CONFIG = SimpleNamespace(block_chunk=P_MAX)
_SEARCH_SPACE = [SimpleNamespace(block_chunk=b) for b in (32, 64, 128)]
_LNC = _lnc_degree()
_kernel = flash_decode_kernel[_LNC] if nki is not None else None
_tuner = NkiAutotuner(_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(mid_o: torch.Tensor, mid_o_lse: torch.Tensor, b_seqlen: torch.Tensor,
        block_seq, block_size: int = None, autotune: bool = False, **kwargs) -> torch.Tensor:
    batch = mid_o.shape[0]
    valid = ((b_seqlen + block_seq - 1) // block_seq).to(torch.float32).reshape(batch, 1)
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(mid_o.shape), str(mid_o.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (mid_o, mid_o_lse, valid, cfg.block_chunk),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return _kernel(mid_o, mid_o_lse, valid, cfg.block_chunk)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
