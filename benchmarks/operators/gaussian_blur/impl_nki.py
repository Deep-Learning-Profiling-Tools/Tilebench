"""NKI 2D convolution ("gaussian_blur" -- the benchmark's kernel weights are an
arbitrary normalized random matrix, not a true separable Gaussian, so this
must implement full 2D conv semantics; a horizontal+vertical separable
rewrite would be mathematically wrong for the actual test data).

Compile time is the reason this file is shaped the way it is. A plain Python
``for i in range(num_blocks): for kh: for kw:`` triple-nested loop is fully
unrolled at trace time, so the emitted instruction count is
``O(num_blocks * kernel_rows * kernel_cols)``. Only ``num_blocks``
(= ceil(input_rows / 128)) grows with the benchmark's problem size, up to 80
at the largest configured ``input_rows``; ``kernel_rows * kernel_cols`` is
fixed (7*7 = 49 in this operator's config).

The fix is to put the ``kh`` (kernel-row) loop on ``nl.dynamic_range`` -- an
on-device loop whose body is emitted once -- so each row-block emits one tap
row instead of ``kernel_rows`` of them, i.e. ``O(num_blocks * kernel_cols)``
overall. Measured end-to-end compile at the largest configured case
(input_rows = 10240): ~65 s fully unrolled vs ~10 s here.

Which loop is made dynamic matters, and not only for the instruction count:

* ``nl.dynamic_range`` cannot be nested, so exactly one of the three loops can
  be dynamic.
* Making the *row-block* loop dynamic is what one would try first (it is the
  only loop whose trip count grows), but it does not compile.  A row block
  produces a disjoint slice of the output, so a dynamic row-block loop has to
  store to the kernel's ``nl.shared_hbm`` output tensor from *inside* the
  dynamic region, via ``.ap(scalar_offset=...)``.  neuronx-cc 2.26 rejects
  that: as soon as two such kernel invocations land in one XLA module -- which
  the benchmark harness does, it traces the correctness call and the timing
  call into a single HLO graph -- the backend fails with

      [INTERNAL_ERROR] [NCC_IESL012] Did not expect any shared internal
      tensors to be live on input to function sg0000 in subgraph 0, but
      found custom_call.1 live.

  (A single invocation per module compiles fine, which is why this only shows
  up under the benchmark harness and not in a one-shot script.)  Staging the
  dynamic-region stores through an ``nl.private_hbm`` buffer and copying it to
  the output afterwards does compile, but the extra full-tensor HBM round trip
  costs ~27% of run time on this memory-bound kernel.
* Making the *kh* loop dynamic sidesteps the restriction entirely: kh is a
  reduction dimension, so the dynamic region only accumulates into SBUF and
  reads HBM indirectly, and the one store per row block happens in the
  enclosing static Python loop -- the same shape as the working
  ``nl.dynamic_range`` kernels in block_sparse_attention/flash_attention.

Two details inherited from block_sparse_attention's rewrite:

* A ``nl.dynamic_range`` index is a hardware register with no arithmetic and
  cannot be passed to ``nl.ds()``. The two offsets the loop body needs (the
  source row in ``padded_input`` and the tap row in the weight table) are kept
  in 1x1 int32 SBUF scalars, advanced with ``nisa.tensor_scalar``, and applied
  via ``.ap(scalar_offset=..., indirect_dim=0)``.
* ``nisa.tensor_scalar``'s scalar operand must be a per-partition column, so
  the weights are handed to the kernel pre-broadcast as
  ``(kernel_rows * 128, kernel_cols)`` (row ``kh * 128 + p`` = ``w[kh, :]``).
  That also makes the dynamic tap-row lookup use the exact same
  ``.ap(scalar_offset=...)`` idiom as the input load. Building it costs one
  ``expand`` on a 7x7 tensor.
"""

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
