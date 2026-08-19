import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax          # 128 partitions
except ImportError:
    nki = None
    PMAX = 128

# Per-partition SBUF spent on the input window + the output block (trn2 has 192KB).
SBUF_BUDGET_BYTES = 100 * 1024

# Output rows computed per block.  Larger amortises the window DMA over more taps
# (the block overlap costs (kernel_size - stride) redundant rows per block) but
# multiplies the SBUF footprint; capped so the emitted instruction count stays
# small even for the largest sweep point.
MAX_OH_BLOCK = 32


def div_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def kernel_assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"[2d_max_pooling NKI] {message}")


def _sbuf_bytes(itemsize: int, kernel_size: int, stride: int, w_buf: int,
                out_W: int, oh_block: int) -> int:
    """Per-partition SBUF bytes for the input window + the output block."""
    nh_buf = (oh_block - 1) * stride + kernel_size
    return itemsize * (nh_buf * w_buf + oh_block * out_W)


def _choose_oh_block(itemsize: int, kernel_size: int, stride: int, w_buf: int,
                     out_W: int, out_H: int) -> int:
    """Largest ``oh_block <= MAX_OH_BLOCK`` fitting the per-partition budget."""
    oh_block = min(MAX_OH_BLOCK, max(1, out_H))
    while oh_block > 1 and _sbuf_bytes(itemsize, kernel_size, stride, w_buf,
                                       out_W, oh_block) > SBUF_BUDGET_BYTES:
        oh_block //= 2
    return oh_block


if nki is not None:

    @nki.jit
    def max_pool2d_kernel(input_hbm, in_H, in_W, kernel_size, stride, padding,
                          oh_block):
        """2D max pooling over the trailing two axes of a flattened plane tensor.

        Args:
            input_hbm: (planes, in_H * in_W) -- ``planes`` is ``N * C``
            in_H, in_W: input spatial extents (compile-time constants)
            kernel_size, stride, padding: pooling parameters (compile-time constants)
            oh_block: output rows computed per block

        Returns:
            (planes, out_H * out_W)
        """
        planes, in_HW = input_hbm.shape

        out_H = (in_H + 2 * padding - kernel_size) // stride + 1
        out_W = (in_W + 2 * padding - kernel_size) // stride + 1
        out_HW = out_H * out_W

        # NKI kernels cannot `raise`; run() does the friendly validation, these
        # are the in-kernel invariants.
        assert in_HW == in_H * in_W
        assert out_H >= 1 and out_W >= 1

        # W window: column j holds input column j - padding, so a whole output row
        # of tap kw is the affine view [stride, out_W] at offset kw.
        w_buf = (out_W - 1) * stride + kernel_size
        n_valid_w = max(0, min(w_buf - padding, in_W))
        nh_buf = (oh_block - 1) * stride + kernel_size
        win_pp = nh_buf * w_buf

        n_plane_tiles = div_ceil(planes, PMAX)
        n_row_blocks = div_ceil(out_H, oh_block)

        output_hbm = nl.ndarray((planes, out_HW), dtype=input_hbm.dtype,
                                buffer=nl.shared_hbm)

        win = nl.ndarray((PMAX, nh_buf, w_buf), dtype=input_hbm.dtype,
                         buffer=nl.sbuf)
        out_sb = nl.ndarray((PMAX, oh_block, out_W), dtype=input_hbm.dtype,
                            buffer=nl.sbuf)

        # Zero-pad analogue for max pooling: the border columns are never written
        # again, so they stay -inf and can never win a max.
        nisa.memset(dst=win[0:PMAX, 0:nh_buf, 0:w_buf], value=float("-inf"))

        for p_tile in range(n_plane_tiles):
            p_start = p_tile * PMAX
            p_sz = min(PMAX, planes - p_start)

            for blk_idx in range(n_row_blocks):
                oh0 = blk_idx * oh_block
                blk = min(oh_block, out_H - oh0)
                nh = (blk - 1) * stride + kernel_size

                # --- load the input window rows, clipped to [0, in_H) ---------
                ih0 = oh0 * stride - padding
                lo = max(ih0, 0)
                hi = min(ih0 + nh, in_H)
                dst_j = lo - ih0
                n_rows = hi - lo
                if dst_j > 0:
                    nisa.memset(dst=win[0:p_sz, 0:dst_j, 0:w_buf],
                                value=float("-inf"))
                if dst_j + n_rows < nh:
                    nisa.memset(dst=win[0:p_sz, dst_j + n_rows:nh, 0:w_buf],
                                value=float("-inf"))
                if n_rows > 0:
                    nisa.dma_copy(
                        dst=win[0:p_sz, dst_j:dst_j + n_rows,
                                padding:padding + n_valid_w],
                        src=input_hbm.ap(
                            pattern=[[in_HW, p_sz], [in_W, n_rows],
                                     [1, n_valid_w]],
                            offset=p_start * in_HW + lo * in_W,
                        ),
                    )

                # --- reduce the kernel_size^2 taps with max -------------------
                acc = out_sb[0:p_sz, 0:blk, 0:out_W]
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        # Shifted, strided view of the loaded window: element
                        # (r, c) is input[.., (oh0+r)*stride+kh-pad,
                        # c*stride+kw-pad].
                        tap = win.ap(
                            pattern=[[win_pp, p_sz], [stride * w_buf, blk],
                                     [stride, out_W]],
                            offset=kh * w_buf + kw,
                        )
                        if kh == 0 and kw == 0:
                            nisa.tensor_copy(dst=acc, src=tap)
                        else:
                            nisa.tensor_tensor(dst=acc, data1=acc, data2=tap,
                                               op=nl.maximum)

                nisa.dma_copy(
                    dst=output_hbm.ap(
                        pattern=[[out_HW, p_sz], [1, blk * out_W]],
                        offset=p_start * out_HW + oh0 * out_W,
                    ),
                    src=acc,
                )

        return output_hbm


def run(input: torch.Tensor, N: int, C: int, H: int, W: int,
        kernel_size: int, stride: int, padding: int,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    kernel_assert(stride >= 1 and kernel_size >= 1 and padding >= 0,
                  "kernel_size/stride must be >= 1 and padding >= 0")
    kernel_assert(2 * padding <= kernel_size,
                  "padding must not exceed kernel_size / 2")
    kernel_assert(H + 2 * padding >= kernel_size and W + 2 * padding >= kernel_size,
                  "input (with padding) smaller than the pooling window")

    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1

    w_buf = (out_W - 1) * stride + kernel_size
    oh_block = _choose_oh_block(input.element_size(), kernel_size, stride, w_buf,
                                out_W, out_H)

    x = input.view(N * C, H * W)
    result = max_pool2d_kernel(x, H, W, kernel_size, stride, padding, oh_block)
    return result.reshape(-1)


def get_last_config() -> dict | None:
    return None
