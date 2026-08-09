"""NKI 1D convolution (im2col + matmul) matching ``torch.nn.functional.conv1d``.

Reference semantics (see ``impl_torch.py``)::

    input : (batch, in_channels, in_L)
    weight: (out_channels, in_channels // groups, kL)
    out   : (batch, out_channels, out_L)
    out_L = (in_L + 2 * padding - kL) // stride + 1

Algorithm
---------
Same im2col-style formulation the GPU backends use, but the "gather" is
expressed as a *static affine* access pattern instead of an index tensor:

* The contraction axis (in_channels_per_group x kL) is decomposed exactly as in
  ``impl_triton.py`` -- ``ic_local``/``kl_idx`` -- but instead of materialising
  ``il_idx = ol_idx * stride + kl_idx - pad`` as an index tensor, the whole
  input window for a block of output positions is loaded once into SBUF with a
  single contiguous DMA, and each ``kl`` slice is then a *shifted view* of that
  window (offset ``kl``, element stride ``stride``).  NKI gathers are expensive
  and blow up compile time, affine access patterns are free.
* Each ``nisa.nc_matmul`` contracts over 128 input channels with the output
  positions as the moving free axis, accumulating in fp32 PSUM over the
  ``(ic_tile, kl)`` pairs -- i.e. ``acc += gathered_input @ gathered_weight``.
* Zero padding is handled by *peeling*: only the first/last L-blocks can touch
  out-of-range input, so those are emitted statically with a zeroed window and a
  clipped DMA.  Every interior block is fully in range and runs inside a single
  ``nl.dynamic_range`` on-device loop, which keeps the instruction count (and
  therefore compile time) constant as ``L`` sweeps from 128K to 2.6M -- a fully
  unrolled version would emit ~15k matmuls at the largest configured length.
* ``blocks_in_flight`` L-blocks share one loop iteration, each with its own
  window/output buffer, and all their loads are issued before the first matmul
  so the DMA queues overlap the Tensor Engine instead of alternating with it.

Two NKI details this kernel is built around (both cost a rewrite to discover):

* A ``nl.dynamic_range`` index is a hardware register that supports no
  arithmetic and cannot be passed to ``nl.ds()``. Runtime offsets are therefore
  kept in 1x1 int32 SBUF scalars advanced with ``nisa.tensor_scalar`` and
  applied via ``.ap(scalar_offset=..., indirect_dim=<L axis>)``.
* On-device control flow only lowers correctly when the kernel's launch degree
  matches the LNC the XLA module is compiled for, hence ``kernel[_lnc_degree()]``
  -- otherwise the backend fails with ``[NCC_IXGM002] ... 1 basic blocks``.

Supported: any stride, padding, kernel length, in/out channel counts and
fp32/fp16/bf16. ``batch > MAX_STATIC_BATCH`` or ``groups > MAX_STATIC_GROUPS``
raise ``NotImplementedError``: each batch item / group is a separately traced
body, so a large count would explode the instruction count (the benchmarked
config is batch=1, groups=1).
"""

import torch

from core.nki_timer import lnc_degree as _lnc_degree

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax          # 128 partitions
except ImportError:
    nki = None
    PMAX = 128

# PSUM bank free size in fp32 elements on trn2 (NeuronCore-v3): the moving
# operand of nc_matmul -- i.e. the number of output positions per matmul.
PSUM_FMAX = 512

# Output positions per L-block. Shrunk if the input window + output block would
# not fit the per-partition SBUF budget.
MAX_L_BLOCK = 8192
# Per-partition SBUF spent on the input windows + output blocks (trn2 has 192KB).
SBUF_BUDGET_BYTES = 144 * 1024
# Number of L-blocks in flight per dynamic-loop iteration, each with its own
# window/output buffer. All the loads of a group are issued before any of its
# matmuls, so DMA overlaps compute instead of strictly alternating with it.
MAX_BLOCKS_IN_FLIGHT = 4
# Smallest L-block worth using: 2048 columns keeps every window DMA well above
# the 2KB-contiguous-per-partition threshold where DMA efficiency drops off.
MIN_L_BLOCK = 2048

# Below this many interior block groups the on-device loop is not worth its
# overhead, so the blocks are unrolled at compile time instead.
MIN_DYNAMIC_GROUPS = 4

MAX_STATIC_BATCH = 8
MAX_STATIC_GROUPS = 4


def div_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def kernel_assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"[1d_conv NKI] {message}")


def _block_bytes(itemsize: int, n_ic_tiles: int, stride: int, kL: int, l_block: int) -> int:
    """Per-partition SBUF bytes for one in-flight block (window + output)."""
    window_cols = (l_block - 1) * stride + kL
    return itemsize * (n_ic_tiles * window_cols + l_block)


def _choose_tiling(itemsize: int, n_ic_tiles: int, stride: int, kL: int,
                   out_L: int) -> tuple[int, int]:
    """Pick (l_block, blocks_in_flight) fitting the per-partition SBUF budget.

    Overlap is worth more than block size here (the kernel is DMA/PE bound), so
    the largest number of in-flight blocks wins; the block length is then shrunk
    until the group fits SBUF, down to MIN_L_BLOCK to keep DMAs large.
    """
    needed = max(PSUM_FMAX, div_ceil(out_L, PSUM_FMAX) * PSUM_FMAX)
    l_cap = min(MAX_L_BLOCK, needed)
    l_floor = min(MIN_L_BLOCK, l_cap)

    for blocks_in_flight in (MAX_BLOCKS_IN_FLIGHT, 2, 1):
        l_block = l_cap
        while (l_block > l_floor
               and blocks_in_flight
               * _block_bytes(itemsize, n_ic_tiles, stride, kL, l_block)
               > SBUF_BUDGET_BYTES):
            # Halve, but keep l_block a whole number of PSUM tiles.
            l_block = max(l_floor, (l_block // 2) // PSUM_FMAX * PSUM_FMAX)
        if (blocks_in_flight
                * _block_bytes(itemsize, n_ic_tiles, stride, kL, l_block)
                <= SBUF_BUDGET_BYTES):
            return l_block, blocks_in_flight

    # Single block that still does not fit the budget: shrink to the PSUM tile.
    return PSUM_FMAX, 1


if nki is not None:

    def _load_window(input_hbm, windows, buf, cfg, b, g, l_start, blk,
                     in_off_sb=None, sub=0):
        """Load input[b, group-g channels, window] into SBUF, zero-padded.

        On the dynamic path (``in_off_sb`` given) the window start lives in an
        SBUF scalar that the on-device loop advances; ``sub`` is this block's
        index inside the in-flight group and rides along as a static offset.
        Such a block is guaranteed (by the caller's peeling) to lie fully inside
        the input. On the static path ``l_start`` is a plain Python int and the
        DMA is clipped to the valid input range, the rest of the window zeroed.
        """
        stride, pad, kL = cfg["stride"], cfg["pad"], cfg["kL"]
        in_ch_g, n_ic_tiles = cfg["in_ch_g"], cfg["n_ic_tiles"]
        in_channels, in_L = cfg["in_channels"], cfg["in_L"]
        w_len = (blk - 1) * stride + kL

        if in_off_sb != None:
            # A dynamic_range register supports no arithmetic and cannot be fed
            # to nl.ds(), so the window offset is carried in an SBUF scalar and
            # applied with .ap(scalar_offset=..., indirect_dim=<L axis>).
            sub_offset = sub * blk * stride
            for ic_tile in range(n_ic_tiles):
                ic_size = min(PMAX, in_ch_g - ic_tile * PMAX)
                ic_offset = g * in_ch_g + ic_tile * PMAX
                nisa.dma_copy(
                    dst=windows[0:ic_size, buf, ic_tile, 0:w_len],
                    src=input_hbm.ap(
                        pattern=[[in_L, ic_size], [1, w_len]],
                        offset=(b * in_channels + ic_offset) * in_L + sub_offset,
                        scalar_offset=in_off_sb,
                        indirect_dim=2,
                    ),
                )
            return

        il_start = l_start * stride - pad

        # Static (boundary) block: clip the DMA to the valid input range and
        # zero the columns that fall outside it -- that is the conv's zero pad.
        valid_lo = max(il_start, 0)
        valid_hi = min(il_start + w_len, in_L)
        n_valid = valid_hi - valid_lo
        dst_offset = valid_lo - il_start

        for ic_tile in range(n_ic_tiles):
            ic_size = min(PMAX, in_ch_g - ic_tile * PMAX)
            ic_offset = g * in_ch_g + ic_tile * PMAX
            if dst_offset > 0:
                nisa.memset(dst=windows[0:ic_size, buf, ic_tile, 0:dst_offset],
                            value=0.0)
            if dst_offset + n_valid < w_len:
                nisa.memset(
                    dst=windows[0:ic_size, buf, ic_tile, dst_offset + n_valid:w_len],
                    value=0.0,
                )
            if n_valid > 0:
                nisa.dma_copy(
                    dst=windows[0:ic_size, buf, ic_tile,
                                dst_offset:dst_offset + n_valid],
                    src=input_hbm[b, ic_offset:ic_offset + ic_size, valid_lo:valid_hi],
                )

    def _compute_block(output_hbm, weight_t, windows, out_sbs, buf, cfg, b, g,
                       l_start, blk, out_off_sb=None, sub=0):
        """Matmul an already-loaded window and store ``blk`` output positions."""
        stride, kL = cfg["stride"], cfg["kL"]
        in_ch_g, out_ch_g = cfg["in_ch_g"], cfg["out_ch_g"]
        n_ic_tiles, n_oc_tiles = cfg["n_ic_tiles"], cfg["n_oc_tiles"]
        window_cols, out_L = cfg["window_cols"], cfg["out_L"]
        win_pp = cfg["win_pp"]

        n_col_tiles = div_ceil(blk, PSUM_FMAX)
        for oc_tile in range(n_oc_tiles):
            oc_size = min(PMAX, out_ch_g - oc_tile * PMAX)

            for col_tile in range(n_col_tiles):
                col_offset = col_tile * PSUM_FMAX
                n_col = min(PSUM_FMAX, blk - col_offset)

                acc_psum = nl.ndarray((oc_size, n_col), dtype=nl.float32, buffer=nl.psum)
                for ic_tile in range(n_ic_tiles):
                    ic_size = min(PMAX, in_ch_g - ic_tile * PMAX)
                    for kl in range(kL):
                        w_idx = (oc_tile * n_ic_tiles + ic_tile) * kL + kl
                        nisa.nc_matmul(
                            dst=acc_psum,
                            stationary=weight_t[0:ic_size, w_idx, 0:oc_size],
                            # Shifted, strided view of the already-loaded window:
                            # column c holds input[.., c * stride + kl].
                            moving=windows.ap(
                                pattern=[[win_pp, ic_size], [stride, n_col]],
                                offset=(buf * n_ic_tiles + ic_tile) * window_cols
                                       + col_offset * stride + kl,
                            ),
                        )
                nisa.tensor_copy(
                    dst=out_sbs[0:oc_size, buf, col_offset:col_offset + n_col],
                    src=acc_psum,
                )

            oc_offset = g * out_ch_g + oc_tile * PMAX
            if out_off_sb != None:
                nisa.dma_copy(
                    dst=output_hbm.ap(
                        pattern=[[out_L, oc_size], [1, blk]],
                        offset=(b * cfg["out_channels"] + oc_offset) * out_L + sub * blk,
                        scalar_offset=out_off_sb,
                        indirect_dim=2,
                    ),
                    src=out_sbs[0:oc_size, buf, 0:blk],
                )
            else:
                nisa.dma_copy(
                    dst=output_hbm[b, oc_offset:oc_offset + oc_size,
                                   l_start:l_start + blk],
                    src=out_sbs[0:oc_size, buf, 0:blk],
                )

    def _process_l_block(input_hbm, output_hbm, weight_t, windows, out_sbs,
                         cfg, b, g, l_start, blk):
        """Static (compile-time offset) block: load then compute, buffer 0."""
        _load_window(input_hbm, windows, 0, cfg, b, g, l_start, blk)
        _compute_block(output_hbm, weight_t, windows, out_sbs, 0, cfg, b, g,
                       l_start, blk)

    def _prepare_weight(weight_hbm, cfg, g):
        """Transpose weight into per-(oc_tile, ic_tile, kl) stationary tiles.

        ``nc_matmul`` contracts along the partition axis, so the stationary
        operand must be laid out as ``[ic, oc]`` while HBM holds ``[oc, ic, kL]``.
        The weight is tiny, so it is loaded contiguously (fast DMA) and
        transposed on-chip once per kernel invocation rather than gathered.
        """
        kL, in_ch_g, out_ch_g = cfg["kL"], cfg["in_ch_g"], cfg["out_ch_g"]
        n_ic_tiles, n_oc_tiles = cfg["n_ic_tiles"], cfg["n_oc_tiles"]

        weight_t = nl.ndarray((PMAX, n_oc_tiles * n_ic_tiles * kL, PMAX),
                              dtype=cfg["dtype"], buffer=nl.sbuf)

        for oc_tile in range(n_oc_tiles):
            oc_size = min(PMAX, out_ch_g - oc_tile * PMAX)
            oc_offset = g * out_ch_g + oc_tile * PMAX

            w_raw = nl.ndarray((oc_size, in_ch_g, kL), dtype=cfg["dtype"], buffer=nl.sbuf)
            nisa.dma_copy(
                dst=w_raw,
                src=weight_hbm[oc_offset:oc_offset + oc_size, 0:in_ch_g, 0:kL],
            )
            w_f32 = nl.ndarray((oc_size, in_ch_g * kL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=w_f32, src=w_raw.ap(
                pattern=[[in_ch_g * kL, oc_size], [1, in_ch_g * kL]]))

            for ic_tile in range(n_ic_tiles):
                ic_size = min(PMAX, in_ch_g - ic_tile * PMAX)
                for kl in range(kL):
                    t_psum = nl.ndarray((ic_size, oc_size), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(
                        dst=t_psum,
                        data=w_f32.ap(
                            pattern=[[in_ch_g * kL, oc_size], [kL, ic_size]],
                            offset=ic_tile * PMAX * kL + kl,
                        ),
                    )
                    w_idx = (oc_tile * n_ic_tiles + ic_tile) * kL + kl
                    nisa.tensor_copy(
                        dst=weight_t[0:ic_size, w_idx, 0:oc_size],
                        src=t_psum,
                    )
        return weight_t

    @nki.jit
    def conv1d_kernel(input_hbm, weight_hbm, stride, padding, groups, l_block,
                      blocks_in_flight):
        """Batched, multi-channel, strided, zero-padded 1D convolution.

        Args:
            input_hbm:  (batch, in_channels, in_L)
            weight_hbm: (out_channels, in_channels // groups, kL)
            stride, padding, groups: conv parameters (compile-time constants)
            l_block: output positions per L-block (multiple of 512)
            blocks_in_flight: L-blocks per dynamic-loop iteration (own buffers)

        Returns:
            (batch, out_channels, out_L) with out_L = (in_L + 2*padding - kL)//stride + 1
        """
        batch, in_channels, in_L = input_hbm.shape
        out_channels, in_ch_g, kL = weight_hbm.shape
        out_L = (in_L + 2 * padding - kL) // stride + 1
        out_ch_g = out_channels // groups

        # NKI kernels cannot `raise`; the host-side run() performs the friendly
        # validation, these are the in-kernel invariants.
        assert in_ch_g * groups == in_channels
        assert out_channels % groups == 0
        assert l_block % PSUM_FMAX == 0

        cfg = {
            "stride": stride, "pad": padding, "kL": kL,
            "in_L": in_L, "out_L": out_L,
            "in_channels": in_channels, "out_channels": out_channels,
            "in_ch_g": in_ch_g, "out_ch_g": out_ch_g,
            "n_ic_tiles": div_ceil(in_ch_g, PMAX),
            "n_oc_tiles": div_ceil(out_ch_g, PMAX),
            "window_cols": (l_block - 1) * stride + kL,
            "win_pp": blocks_in_flight * div_ceil(in_ch_g, PMAX)
                      * ((l_block - 1) * stride + kL),
            "dtype": input_hbm.dtype,
        }

        output_hbm = nl.ndarray((batch, out_channels, out_L),
                                dtype=input_hbm.dtype, buffer=nl.shared_hbm)

        n_blocks = div_ceil(out_L, l_block)
        n_full_blocks = out_L // l_block

        # Blocks whose input window is entirely in range need no zero padding
        # and can therefore share one uniform on-device loop body.
        first_safe = div_ceil(padding, l_block * stride)
        last_safe = (in_L + padding - cfg["window_cols"]) // (l_block * stride)
        last_safe = min(last_safe, n_full_blocks - 1)
        if last_safe < first_safe:
            first_safe, last_safe = 0, -1

        n_safe = last_safe - first_safe + 1
        n_groups = n_safe // blocks_in_flight
        use_dynamic = n_groups >= MIN_DYNAMIC_GROUPS
        n_dynamic = n_groups * blocks_in_flight if use_dynamic else 0

        for g in range(groups):
            weight_t = _prepare_weight(weight_hbm, cfg, g)
            windows = nl.ndarray(
                (PMAX, blocks_in_flight, cfg["n_ic_tiles"], cfg["window_cols"]),
                dtype=input_hbm.dtype, buffer=nl.sbuf)
            out_sbs = nl.ndarray((PMAX, blocks_in_flight, l_block),
                                 dtype=input_hbm.dtype, buffer=nl.sbuf)

            for b in range(batch):
                # Head block(s): the only ones that can read left of the input.
                for i in range(0, first_safe):
                    _process_l_block(input_hbm, output_hbm, weight_t, windows,
                                     out_sbs, cfg, b, g, i * l_block,
                                     min(l_block, out_L - i * l_block))

                if use_dynamic:
                    # Interior blocks: one uniform body executed on-device, so
                    # the instruction count stays constant as in_L grows. All
                    # the loads of a group are issued before its first matmul so
                    # the DMA queues run ahead of the Tensor Engine.
                    in_off_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                    out_off_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.memset(dst=in_off_sb,
                                value=first_safe * l_block * stride - padding)
                    nisa.memset(dst=out_off_sb, value=first_safe * l_block)

                    for _ in nl.dynamic_range(n_groups):
                        for k in range(blocks_in_flight):
                            _load_window(input_hbm, windows, k, cfg, b, g, 0, l_block,
                                         in_off_sb=in_off_sb, sub=k)
                        for k in range(blocks_in_flight):
                            _compute_block(output_hbm, weight_t, windows, out_sbs, k,
                                           cfg, b, g, 0, l_block,
                                           out_off_sb=out_off_sb, sub=k)
                        nisa.tensor_scalar(dst=in_off_sb, data=in_off_sb, op0=nl.add,
                                           operand0=blocks_in_flight * l_block * stride)
                        nisa.tensor_scalar(dst=out_off_sb, data=out_off_sb, op0=nl.add,
                                           operand0=blocks_in_flight * l_block)

                # Interior blocks the on-device loop did not cover.
                for i in range(first_safe + n_dynamic, last_safe + 1):
                    _process_l_block(input_hbm, output_hbm, weight_t, windows,
                                     out_sbs, cfg, b, g, i * l_block, l_block)

                # Tail block(s): can read past the end of the input.
                for i in range(last_safe + 1, n_blocks):
                    _process_l_block(input_hbm, output_hbm, weight_t, windows,
                                     out_sbs, cfg, b, g, i * l_block,
                                     min(l_block, out_L - i * l_block))

        return output_hbm


def run(input: torch.Tensor, weight: torch.Tensor,
        stride: int = 1, padding: int = 1, groups: int = 1,
        block_size: int = 1024, autotune: bool = False,
        lnc_degree: int | None = None, **kwargs) -> torch.Tensor:
    kernel_assert(input.dim() == 3, "input must be (batch, in_channels, in_L)")
    kernel_assert(weight.dim() == 3, "weight must be (out_channels, in_channels/groups, kL)")

    batch, in_channels, in_L = input.shape
    out_channels, in_ch_g, kL = weight.shape

    kernel_assert(in_ch_g * groups == in_channels,
                  "in_channels must equal groups * weight.shape[1]")
    kernel_assert(out_channels % groups == 0, "out_channels must be divisible by groups")
    kernel_assert(in_L + 2 * padding >= kL, "input (with padding) shorter than the kernel")

    if groups > MAX_STATIC_GROUPS:
        raise NotImplementedError(
            f"1d_conv NKI kernel supports groups <= {MAX_STATIC_GROUPS} "
            f"(each group is traced separately); got groups={groups}"
        )
    if batch > MAX_STATIC_BATCH:
        raise NotImplementedError(
            f"1d_conv NKI kernel supports batch <= {MAX_STATIC_BATCH} "
            f"(each batch item is traced separately); got batch={batch}"
        )

    out_L = (in_L + 2 * padding - kL) // stride + 1
    n_ic_tiles = div_ceil(in_ch_g, PMAX)
    l_block, blocks_in_flight = _choose_tiling(
        input.element_size(), n_ic_tiles, stride, kL, out_L)

    degree = lnc_degree if lnc_degree is not None else _lnc_degree()
    return conv1d_kernel[degree](input, weight, stride, padding, groups,
                                 l_block, blocks_in_flight)


def get_last_config() -> dict | None:
    return None
