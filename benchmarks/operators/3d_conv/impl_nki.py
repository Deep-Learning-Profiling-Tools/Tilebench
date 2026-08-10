import os
import re

import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None
    PMAX = 128

# PSUM bank free size in fp32 elements on trn2 (NeuronCore-v3): also the maximum
# free size of an nc_matmul moving operand.
PSUM_FMAX = 512

# Output rows (oh) computed per dynamic-loop iteration.  Larger amortises the
# window DMA over more matmuls (the window overlap costs (R + kH - 1)/R rows of
# redundant traffic) but multiplies the emitted instruction count, which is
# already paid once per `od`.
MAX_OH_BLOCK = 8
# Per-partition SBUF spent on the input window + output rows (trn2 has 192KB).
SBUF_BUDGET_BYTES = 120 * 1024

# Fewer iterations than this and an on-device loop is not worth its overhead,
# so the blocks are unrolled at compile time instead.
MIN_DYNAMIC_ITERS = 2

MAX_STATIC_BATCH = 8
MAX_STATIC_GROUPS = 4


def _lnc_degree() -> int:
    """Logical-NeuronCore degree the kernel must be launched with.

    The kernel contains on-device control flow (``nl.dynamic_range``), which the
    backend only lowers correctly when the NKI launch degree matches the LNC the
    XLA module is compiled for -- launching an LNC=1 kernel into an LNC=2 module
    fails with ``[NCC_IXGM002] ... core 1 has 1 basic blocks``.  trn2/trn3
    default to LNC=2 unless the compiler/runtime env says otherwise.
    """
    explicit = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "")
    if explicit.strip().isdigit():
        return int(explicit.strip())
    match = re.search(r"--lnc[=\s]+(\d+)", os.environ.get("NEURON_CC_FLAGS", ""))
    if match:
        return int(match.group(1))
    target = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "").strip().lower()
    return 2 if target in ("trn2", "gen3", "trn3", "gen4") else 1


def div_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def kernel_assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"[3d_conv NKI] {message}")


def _sbuf_bytes(itemsize: int, n_ic_tiles: int, kD: int, kH: int, stride: int,
                w_buf: int, out_W: int, oh_block: int) -> int:
    """Per-partition SBUF bytes for the input window + output staging buffer."""
    nh_buf = (oh_block - 1) * stride + kH
    return itemsize * (n_ic_tiles * kD * nh_buf * w_buf + oh_block * out_W)


def _choose_oh_block(itemsize: int, n_ic_tiles: int, kD: int, kH: int,
                     stride: int, w_buf: int, out_W: int, out_H: int) -> int:
    """Largest ``oh_block <= MAX_OH_BLOCK`` fitting the per-partition budget."""
    oh_block = min(MAX_OH_BLOCK, max(1, out_H))
    while oh_block > 1 and _sbuf_bytes(itemsize, n_ic_tiles, kD, kH, stride,
                                       w_buf, out_W, oh_block) > SBUF_BUDGET_BYTES:
        oh_block //= 2
    return oh_block


if nki is not None:

    def _prepare_weight(weight_hbm, cfg, g):
        """Transpose weight into per-(oc_tile, ic_tile, tap) stationary tiles.

        ``nc_matmul`` contracts along the partition axis, so the stationary
        operand must be laid out as ``[ic, oc]`` while HBM holds
        ``[oc, ic, kD * kH * kW]``.  The weight is tiny, so it is loaded
        contiguously (fast DMA) and transposed on-chip once per invocation
        rather than gathered.
        """
        n_taps, in_ch_g, out_ch_g = cfg["n_taps"], cfg["in_ch_g"], cfg["out_ch_g"]
        n_ic_tiles, n_oc_tiles = cfg["n_ic_tiles"], cfg["n_oc_tiles"]

        weight_t = nl.ndarray((PMAX, n_oc_tiles * n_ic_tiles * n_taps, PMAX),
                              dtype=cfg["dtype"], buffer=nl.sbuf)

        for oc_tile in range(n_oc_tiles):
            oc_size = min(PMAX, out_ch_g - oc_tile * PMAX)
            oc_offset = g * out_ch_g + oc_tile * PMAX

            w_raw = nl.ndarray((oc_size, in_ch_g, n_taps), dtype=cfg["dtype"],
                               buffer=nl.sbuf)
            nisa.dma_copy(
                dst=w_raw,
                src=weight_hbm[oc_offset:oc_offset + oc_size, 0:in_ch_g, 0:n_taps],
            )
            w_f32 = nl.ndarray((oc_size, in_ch_g * n_taps), dtype=nl.float32,
                               buffer=nl.sbuf)
            nisa.tensor_copy(dst=w_f32, src=w_raw.ap(
                pattern=[[in_ch_g * n_taps, oc_size], [1, in_ch_g * n_taps]]))

            for ic_tile in range(n_ic_tiles):
                ic_size = min(PMAX, in_ch_g - ic_tile * PMAX)
                for tap in range(n_taps):
                    t_psum = nl.ndarray((ic_size, oc_size), dtype=nl.float32,
                                        buffer=nl.psum)
                    nisa.nc_transpose(
                        dst=t_psum,
                        data=w_f32.ap(
                            pattern=[[in_ch_g * n_taps, oc_size], [n_taps, ic_size]],
                            offset=ic_tile * PMAX * n_taps + tap,
                        ),
                    )
                    w_idx = (oc_tile * n_ic_tiles + ic_tile) * n_taps + tap
                    nisa.tensor_copy(dst=weight_t[0:ic_size, w_idx, 0:oc_size],
                                     src=t_psum)
        return weight_t

    def _load_window(input_hbm, win, cfg, b, g, od, oh0, blk, h_off_sb=None):
        """Fill ``win`` with the input sub-volume for output block (od, oh0..).

        ``win`` is ``[ic, ic_tile, kd_slot, h, w]``; slot ``s`` holds input depth
        plane ``id = od * stride + s - pad`` and window row ``j`` holds input row
        ``ih = oh0 * stride + j - pad``, window column ``j`` holds ``iw = j - pad``.

        The whole buffer was zeroed at allocation and every DMA writes only the
        W-interior, so the W zero-padding is permanent.  Depth planes out of
        range are memset (``od`` is a compile-time constant).  On the dynamic
        path (``h_off_sb`` given) the caller guarantees every window row is in
        range, and the row offset lives in an SBUF scalar the on-device loop
        advances -- a ``dynamic_range`` register supports no arithmetic, so it
        is applied via ``.ap(scalar_offset=..., indirect_dim=...)``.
        """
        stride, pad, kD, kH = cfg["stride"], cfg["pad"], cfg["kD"], cfg["kH"]
        in_D, in_H, in_W = cfg["in_D"], cfg["in_H"], cfg["in_W"]
        in_HW, in_DHW = in_H * in_W, in_D * in_H * in_W
        in_channels, in_ch_g = cfg["in_channels"], cfg["in_ch_g"]
        n_ic_tiles = cfg["n_ic_tiles"]
        w_buf, n_valid_w = cfg["w_buf"], cfg["n_valid_w"]
        nh = (blk - 1) * stride + kH

        for s in range(kD):
            id_idx = od * stride + s - pad
            for ic_tile in range(n_ic_tiles):
                ic_size = min(PMAX, in_ch_g - ic_tile * PMAX)
                ic_offset = g * in_ch_g + ic_tile * PMAX
                base = (b * in_channels + ic_offset) * in_DHW + id_idx * in_HW

                if id_idx < 0 or id_idx >= in_D:
                    # Depth padding: this whole plane is zero.
                    nisa.memset(dst=win[0:ic_size, ic_tile, s, 0:nh, 0:w_buf],
                                value=0.0)
                    continue

                if h_off_sb != None:
                    nisa.dma_copy(
                        dst=win[0:ic_size, ic_tile, s, 0:nh, pad:pad + n_valid_w],
                        src=input_hbm.ap(
                            pattern=[[in_DHW, ic_size], [in_W, nh], [1, n_valid_w]],
                            offset=base,
                            scalar_offset=h_off_sb,
                            indirect_dim=2,
                        ),
                    )
                    continue

                # Static (H-boundary) block: clip the DMA to the valid input
                # rows and zero the rows that fall outside -- the H zero pad.
                ih0 = oh0 * stride - pad
                lo = max(ih0, 0)
                hi = min(ih0 + nh, in_H)
                dst_j = lo - ih0
                n_rows = hi - lo
                if dst_j > 0:
                    nisa.memset(dst=win[0:ic_size, ic_tile, s, 0:dst_j, 0:w_buf],
                                value=0.0)
                if dst_j + n_rows < nh:
                    nisa.memset(
                        dst=win[0:ic_size, ic_tile, s, dst_j + n_rows:nh, 0:w_buf],
                        value=0.0)
                if n_rows > 0:
                    nisa.dma_copy(
                        dst=win[0:ic_size, ic_tile, s, dst_j:dst_j + n_rows,
                                pad:pad + n_valid_w],
                        src=input_hbm.ap(
                            pattern=[[in_DHW, ic_size], [in_W, n_rows],
                                     [1, n_valid_w]],
                            offset=base + lo * in_W,
                        ),
                    )

    def _compute_block(output_hbm, weight_t, win, out_sb, cfg, b, g, od, oh0, blk,
                       out_off_sb=None):
        """Matmul an already-loaded window into ``blk`` output rows and store."""
        stride = cfg["stride"]
        kD, kH, kW, n_taps = cfg["kD"], cfg["kH"], cfg["kW"], cfg["n_taps"]
        in_ch_g, out_ch_g = cfg["in_ch_g"], cfg["out_ch_g"]
        n_ic_tiles, n_oc_tiles = cfg["n_ic_tiles"], cfg["n_oc_tiles"]
        out_W, out_HW, out_DHW = cfg["out_W"], cfg["out_HW"], cfg["out_DHW"]
        out_channels = cfg["out_channels"]
        w_buf, nh_buf, win_pp = cfg["w_buf"], cfg["nh_buf"], cfg["win_pp"]
        n_w_tiles = div_ceil(out_W, PSUM_FMAX)

        for oc_tile in range(n_oc_tiles):
            oc_size = min(PMAX, out_ch_g - oc_tile * PMAX)

            for r in range(blk):
                for w_tile in range(n_w_tiles):
                    w_off = w_tile * PSUM_FMAX
                    n_col = min(PSUM_FMAX, out_W - w_off)

                    acc_psum = nl.ndarray((oc_size, n_col), dtype=nl.float32,
                                          buffer=nl.psum)
                    for ic_tile in range(n_ic_tiles):
                        ic_size = min(PMAX, in_ch_g - ic_tile * PMAX)
                        for kd in range(kD):
                            for kh in range(kH):
                                for kw in range(kW):
                                    tap = (kd * kH + kh) * kW + kw
                                    w_idx = ((oc_tile * n_ic_tiles + ic_tile)
                                             * n_taps + tap)
                                    # Shifted, strided view of the loaded window:
                                    # column c holds input[.., od*s+kd-p,
                                    # (oh0+r)*s+kh-p, (w_off+c)*s+kw-p].
                                    free_off = ((((ic_tile * kD + kd) * nh_buf
                                                  + r * stride + kh) * w_buf)
                                                + kw + w_off * stride)
                                    nisa.nc_matmul(
                                        dst=acc_psum,
                                        stationary=weight_t[0:ic_size, w_idx,
                                                            0:oc_size],
                                        moving=win.ap(
                                            pattern=[[win_pp, ic_size],
                                                     [stride, n_col]],
                                            offset=free_off,
                                        ),
                                    )
                    nisa.tensor_copy(dst=out_sb[0:oc_size, r, w_off:w_off + n_col],
                                     src=acc_psum)

            oc_offset = g * out_ch_g + oc_tile * PMAX
            base = (b * out_channels + oc_offset) * out_DHW + od * out_HW
            if out_off_sb != None:
                nisa.dma_copy(
                    dst=output_hbm.ap(
                        pattern=[[out_DHW, oc_size], [1, blk * out_W]],
                        offset=base,
                        scalar_offset=out_off_sb,
                        indirect_dim=2,
                    ),
                    src=out_sb[0:oc_size, 0:blk, 0:out_W],
                )
            else:
                nisa.dma_copy(
                    dst=output_hbm.ap(
                        pattern=[[out_DHW, oc_size], [1, blk * out_W]],
                        offset=base + oh0 * out_W,
                    ),
                    src=out_sb[0:oc_size, 0:blk, 0:out_W],
                )

    def _static_row_block(input_hbm, output_hbm, weight_t, win, out_sb, cfg,
                          b, g, od, oh0, blk):
        _load_window(input_hbm, win, cfg, b, g, od, oh0, blk)
        _compute_block(output_hbm, weight_t, win, out_sb, cfg, b, g, od, oh0, blk)

    def _dynamic_row_loop(input_hbm, output_hbm, weight_t, win, out_sb, cfg,
                          b, g, od, oh_start, n_iter, blk, h_off_sb, out_off_sb):
        """``n_iter`` uniform on-device iterations of ``blk`` output rows each."""
        stride, pad, in_W = cfg["stride"], cfg["pad"], cfg["in_W"]
        out_W = cfg["out_W"]
        nisa.memset(dst=h_off_sb, value=(oh_start * stride - pad) * in_W)
        nisa.memset(dst=out_off_sb, value=oh_start * out_W)
        for _ in nl.dynamic_range(n_iter):
            _load_window(input_hbm, win, cfg, b, g, od, 0, blk, h_off_sb=h_off_sb)
            _compute_block(output_hbm, weight_t, win, out_sb, cfg, b, g, od, 0,
                           blk, out_off_sb=out_off_sb)
            nisa.tensor_scalar(dst=h_off_sb, data=h_off_sb, op0=nl.add,
                               operand0=blk * stride * in_W)
            nisa.tensor_scalar(dst=out_off_sb, data=out_off_sb, op0=nl.add,
                               operand0=blk * out_W)

    @nki.jit
    def conv3d_kernel(input_hbm, weight_hbm, in_D, in_H, in_W, kD, kH, kW,
                      stride, padding, groups, oh_block):
        """Batched, multi-channel, grouped, strided, zero-padded 3D convolution.

        Args:
            input_hbm:  (batch, in_channels, in_D * in_H * in_W)
            weight_hbm: (out_channels, in_channels // groups, kD * kH * kW)
            in_D, in_H, in_W: input spatial extents (compile-time constants)
            kD, kH, kW: kernel extents (compile-time constants)
            stride, padding, groups: conv parameters (compile-time constants)
            oh_block: output rows computed per dynamic-loop iteration

        Returns:
            (batch, out_channels, out_D * out_H * out_W)
        """
        batch, in_channels, _ = input_hbm.shape
        out_channels, in_ch_g, n_taps = weight_hbm.shape

        out_D = (in_D + 2 * padding - kD) // stride + 1
        out_H = (in_H + 2 * padding - kH) // stride + 1
        out_W = (in_W + 2 * padding - kW) // stride + 1
        out_ch_g = out_channels // groups

        # NKI kernels cannot `raise`; run() does the friendly validation, these
        # are the in-kernel invariants.
        assert in_ch_g * groups == in_channels
        assert out_channels % groups == 0
        assert n_taps == kD * kH * kW

        n_ic_tiles = div_ceil(in_ch_g, PMAX)
        n_oc_tiles = div_ceil(out_ch_g, PMAX)

        # W window: column j holds input column j - padding, so a whole output
        # row is the affine view [stride, out_W] at offset kw.
        w_buf = (out_W - 1) * stride + kW
        n_valid_w = max(0, min(w_buf - padding, in_W))
        nh_buf = (oh_block - 1) * stride + kH

        cfg = {
            "stride": stride, "pad": padding,
            "kD": kD, "kH": kH, "kW": kW, "n_taps": n_taps,
            "in_D": in_D, "in_H": in_H, "in_W": in_W,
            "out_W": out_W, "out_HW": out_H * out_W,
            "out_DHW": out_D * out_H * out_W,
            "in_channels": in_channels, "out_channels": out_channels,
            "in_ch_g": in_ch_g, "out_ch_g": out_ch_g,
            "n_ic_tiles": n_ic_tiles, "n_oc_tiles": n_oc_tiles,
            "w_buf": w_buf, "n_valid_w": n_valid_w, "nh_buf": nh_buf,
            "win_pp": n_ic_tiles * kD * nh_buf * w_buf,
            "dtype": input_hbm.dtype,
        }

        output_hbm = nl.ndarray((batch, out_channels, out_D * out_H * out_W),
                                dtype=input_hbm.dtype, buffer=nl.shared_hbm)

        # H rows whose window lies entirely inside the input need no padding and
        # can therefore share one uniform on-device loop body; the rest are
        # peeled into static blocks of a single row.
        oh_lo = min(div_ceil(padding, stride), out_H)
        oh_hi = min((in_H + padding - kH) // stride, out_H - 1)
        if oh_hi < oh_lo:
            safe_start, n_safe = 0, 0
        else:
            safe_start, n_safe = oh_lo, oh_hi - oh_lo + 1
        head_rows = range(0, safe_start) if n_safe else range(0, out_H)
        tail_rows = range(safe_start + n_safe, out_H) if n_safe else range(0, 0)

        n_full = n_safe // oh_block
        rem_start = safe_start + n_full * oh_block
        n_rem = n_safe - n_full * oh_block

        for g in range(groups):
            weight_t = _prepare_weight(weight_hbm, cfg, g)
            win = nl.ndarray((PMAX, n_ic_tiles, kD, nh_buf, w_buf),
                             dtype=input_hbm.dtype, buffer=nl.sbuf)
            out_sb = nl.ndarray((PMAX, oh_block, out_W),
                                dtype=input_hbm.dtype, buffer=nl.sbuf)
            # Separate scalar pairs for the main / remainder loops so the two
            # on-device loops of one `od` never alias the same SBUF scalar.
            h_off_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            out_off_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            h_off_rem_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            out_off_rem_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)

            # Zero once: the W border columns are never written again, which is
            # exactly the convolution's W zero padding.
            nisa.memset(dst=win[0:PMAX, 0:n_ic_tiles, 0:kD, 0:nh_buf, 0:w_buf],
                        value=0.0)

            for b in range(batch):
                for od in range(out_D):
                    for oh in head_rows:
                        _static_row_block(input_hbm, output_hbm, weight_t, win,
                                          out_sb, cfg, b, g, od, oh, 1)

                    if n_full >= MIN_DYNAMIC_ITERS:
                        _dynamic_row_loop(input_hbm, output_hbm, weight_t, win,
                                          out_sb, cfg, b, g, od, safe_start,
                                          n_full, oh_block, h_off_sb, out_off_sb)
                    else:
                        for i in range(n_full):
                            _static_row_block(input_hbm, output_hbm, weight_t,
                                              win, out_sb, cfg, b, g, od,
                                              safe_start + i * oh_block, oh_block)

                    if n_rem >= MIN_DYNAMIC_ITERS:
                        _dynamic_row_loop(input_hbm, output_hbm, weight_t, win,
                                          out_sb, cfg, b, g, od, rem_start,
                                          n_rem, 1, h_off_rem_sb, out_off_rem_sb)
                    else:
                        for i in range(n_rem):
                            _static_row_block(input_hbm, output_hbm, weight_t,
                                              win, out_sb, cfg, b, g, od,
                                              rem_start + i, 1)

                    for oh in tail_rows:
                        _static_row_block(input_hbm, output_hbm, weight_t, win,
                                          out_sb, cfg, b, g, od, oh, 1)

        return output_hbm


def run(input: torch.Tensor, weight: torch.Tensor,
        stride: int = 1, padding: int = 1, groups: int = 1,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    kernel_assert(input.dim() == 5,
                  "input must be (batch, in_channels, in_D, in_H, in_W)")
    kernel_assert(weight.dim() == 5,
                  "weight must be (out_channels, in_channels/groups, kD, kH, kW)")

    batch, in_channels, in_D, in_H, in_W = input.shape
    out_channels, in_ch_g, kD, kH, kW = weight.shape

    kernel_assert(in_ch_g * groups == in_channels,
                  "in_channels must equal groups * weight.shape[1]")
    kernel_assert(out_channels % groups == 0,
                  "out_channels must be divisible by groups")
    kernel_assert(in_D + 2 * padding >= kD and in_H + 2 * padding >= kH
                  and in_W + 2 * padding >= kW,
                  "input (with padding) smaller than the kernel")

    if groups > MAX_STATIC_GROUPS:
        raise NotImplementedError(
            f"3d_conv NKI kernel supports groups <= {MAX_STATIC_GROUPS} "
            f"(each group is traced separately); got groups={groups}"
        )
    if batch > MAX_STATIC_BATCH:
        raise NotImplementedError(
            f"3d_conv NKI kernel supports batch <= {MAX_STATIC_BATCH} "
            f"(each batch item is traced separately); got batch={batch}"
        )

    out_D = (in_D + 2 * padding - kD) // stride + 1
    out_H = (in_H + 2 * padding - kH) // stride + 1
    out_W = (in_W + 2 * padding - kW) // stride + 1

    n_ic_tiles = div_ceil(in_ch_g, PMAX)
    w_buf = (out_W - 1) * stride + kW
    oh_block = _choose_oh_block(input.element_size(), n_ic_tiles, kD, kH, stride,
                                w_buf, out_W, out_H)

    x = input.reshape(batch, in_channels, in_D * in_H * in_W)
    w = weight.reshape(out_channels, in_ch_g, kD * kH * kW)

    result = conv3d_kernel[_lnc_degree()](x, w, in_D, in_H, in_W, kD, kH, kW,
                                          stride, padding, groups, oh_block)
    return result.reshape(batch, out_channels, out_D, out_H, out_W)


def get_last_config() -> dict | None:
    return None
