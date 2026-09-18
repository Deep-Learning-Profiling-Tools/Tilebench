import functools
import os
import re
import subprocess

import torch

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None
    PMAX = 128


@functools.lru_cache(maxsize=1)
def _lnc_degree() -> int:
    """Logical-NeuronCore degree the kernel must be launched with.

    A kernel containing on-device control flow (``nl.dynamic_range``) only
    lowers correctly when the NKI launch degree matches the LNC the XLA
    module is compiled for -- launching an LNC=1 kernel into an LNC=2 module
    fails with ``[NCC_IXGM002] ... core 1 has 1 basic blocks`` (see
    block_sparse_attention/impl_nki.py, where this was first worked out).
    trn2/trn3 default to LNC=2 unless the compiler/runtime env says otherwise.
    """
    explicit = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "")
    if explicit.strip().isdigit():
        return int(explicit.strip())
    match = re.search(r"--lnc[=\s]+(\d+)", os.environ.get("NEURON_CC_FLAGS", ""))
    if match:
        return int(match.group(1))
    target = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "").strip().lower()
    if target in ("trn2", "gen3", "trn3", "gen4"):
        return 2
    # NEURON_PLATFORM_TARGET_OVERRIDE is rarely set in practice, so the branch
    # above rarely fires -- ask the instance directly rather than guessing LNC=1.
    try:
        out = subprocess.run(["neuron-ls"], capture_output=True, text=True, timeout=10).stdout
        lnc = re.search(r"logical-neuroncore-config:\s*(\d+)", out)
        if lnc:
            return int(lnc.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    return 1


def div_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


if nki is not None:

    @nki.jit
    def gaussian_blur_kernel(padded_input, kernel_bcast, rows, cols,
                             kernel_rows, kernel_cols):
        """2D convolution of a zero-padded image with a (kernel_rows, kernel_cols) tap set.

        Args:
            padded_input: (rows + kernel_rows - 1, cols + kernel_cols - 1) image,
                          already zero-padded by the caller.
            kernel_bcast: (kernel_rows * 128, kernel_cols) weight table; row
                          ``kh * 128 + p`` holds ``w[kh, :]`` for every p, so a
                          128-partition tile of tap row ``kh`` is one DMA.
            rows, cols:   output extent.

        Returns:
            (rows, cols) convolution result.
        """
        padded_cols = padded_input.shape[1]
        row_span = cols + kernel_cols - 1
        num_blocks = div_ceil(rows, PMAX)

        out = nl.ndarray((rows, cols), dtype=padded_input.dtype, buffer=nl.shared_hbm)

        # One set of buffers reused by every row block: allocating inside the
        # loop would multiply both SBUF pressure and compile time.
        acc = nl.ndarray((PMAX, cols), dtype=nl.float32, buffer=nl.sbuf)
        prod = nl.ndarray((PMAX, cols), dtype=nl.float32, buffer=nl.sbuf)
        row_tile = nl.ndarray((PMAX, row_span), dtype=nl.float32, buffer=nl.sbuf)
        w_tile = nl.ndarray((PMAX, kernel_cols), dtype=nl.float32, buffer=nl.sbuf)
        res = nl.ndarray((PMAX, cols), dtype=padded_input.dtype, buffer=nl.sbuf)
        src_off = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        tap_off = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)

        for i in range(num_blocks):
            base = i * PMAX
            p_size = min(PMAX, rows - base)

            nisa.memset(dst=acc[0:p_size, :], value=0.0)
            # Row of padded_input feeding tap row kh=0 of this block, and the
            # matching row of the broadcast weight table.
            nisa.memset(dst=src_off, value=base)
            nisa.memset(dst=tap_off, value=0)

            # kh is the reduction dimension: the body only accumulates into
            # SBUF, so nothing inside the dynamic region touches `out`.
            for _ in nl.dynamic_range(kernel_rows):
                nisa.dma_copy(
                    dst=row_tile[0:p_size, 0:row_span],
                    src=padded_input.ap(pattern=[[padded_cols, p_size], [1, row_span]],
                                        scalar_offset=src_off, indirect_dim=0),
                )
                nisa.dma_copy(
                    dst=w_tile[0:p_size, 0:kernel_cols],
                    src=kernel_bcast.ap(pattern=[[kernel_cols, p_size], [1, kernel_cols]],
                                        scalar_offset=tap_off, indirect_dim=0),
                )

                for kw in range(kernel_cols):
                    nisa.tensor_scalar(dst=prod[0:p_size, :],
                                       data=row_tile[0:p_size, kw:kw + cols],
                                       op0=nl.multiply, operand0=w_tile[0:p_size, kw:kw + 1])
                    nisa.tensor_tensor(dst=acc[0:p_size, :], data1=acc[0:p_size, :],
                                       data2=prod[0:p_size, :], op=nl.add)

                nisa.tensor_scalar(dst=src_off, data=src_off, op0=nl.add, operand0=1)
                nisa.tensor_scalar(dst=tap_off, data=tap_off, op0=nl.add, operand0=PMAX)

            nisa.tensor_copy(dst=res[0:p_size, :], src=acc[0:p_size, :])
            nisa.dma_copy(dst=out[base:base + p_size, 0:cols], src=res[0:p_size, :])

        return out


def run(input: torch.Tensor, kernel: torch.Tensor, input_rows: int, input_cols: int,
        kernel_rows: int, kernel_cols: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    x = input.reshape(input_rows, input_cols)
    w = kernel.reshape(kernel_rows, kernel_cols)
    pr, pc = kernel_rows // 2, kernel_cols // 2
    padded = torch.nn.functional.pad(x, (pc, pc, pr, pr))
    # Weight table replicated across the 128 partitions -- see module docstring.
    w_bcast = w.reshape(kernel_rows, 1, kernel_cols) \
               .expand(kernel_rows, PMAX, kernel_cols) \
               .reshape(kernel_rows * PMAX, kernel_cols)
    result = gaussian_blur_kernel[_lnc_degree()](
        padded, w_bcast, input_rows, input_cols, kernel_rows, kernel_cols)
    return result.reshape(-1)


def get_last_config() -> dict | None:
    return None
