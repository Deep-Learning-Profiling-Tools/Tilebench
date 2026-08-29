from types import SimpleNamespace

import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax          # 128 partitions
except ImportError:
    nki = None
    PMAX = 128

if nki is not None:
    from core.nki_autotune import NkiAutotuner

    @nki.jit
    def max_pool2d_kernel(input_hbm, in_H, in_W, kernel_size, stride, padding,
                          itemsize, oh_block):
        """2D max pooling over the trailing two axes of a flattened plane tensor.

        Args:
            input_hbm: (planes, in_H * in_W) -- ``planes`` is ``N * C``
            in_H, in_W: input spatial extents (compile-time constants)
            kernel_size, stride, padding: pooling parameters (compile-time constants)
            itemsize: bytes per element, for sizing the SBUF window buffer
            oh_block: output rows per block; 0 picks the largest block that fits
                the per-partition SBUF budget, a nonzero value (autotune
                candidate) is used as-is once validated against that budget

        Returns:
            (planes, out_H * out_W)
        """
        planes, in_HW = input_hbm.shape

        out_H = (in_H + 2 * padding - kernel_size) // stride + 1
        out_W = (in_W + 2 * padding - kernel_size) // stride + 1
        out_HW = out_H * out_W

        assert in_HW == in_H * in_W
        assert out_H >= 1 and out_W >= 1

        # W window: column j holds input column j - padding, so a whole output row
        # of tap kw is the affine view [stride, out_W] at offset kw.
        w_buf = (out_W - 1) * stride + kernel_size
        n_valid_w = max(0, min(w_buf - padding, in_W))

        # oh_block == 0: largest block (output rows per block) whose window +
        # output buffers fit the per-partition SBUF budget. sbuf_fmax_bytes
        # only resolves inside an active trace, which is here -- not in
        # run(). A nonzero oh_block (autotune candidate) is validated against
        # the same budget below instead of trusted blindly, so a candidate
        # that doesn't fit raises here and NkiAutotuner just skips it.
        if oh_block == 0:
            oh_block = max(1, out_H)
            while oh_block > 1:
                nh_buf = (oh_block - 1) * stride + kernel_size
                sbuf_bytes = itemsize * (nh_buf * w_buf + oh_block * out_W)
                if sbuf_bytes <= nl.tile_size.sbuf_fmax_bytes:
                    break
                oh_block //= 2

        nh_buf = (oh_block - 1) * stride + kernel_size
        sbuf_bytes = itemsize * (nh_buf * w_buf + oh_block * out_W)
        assert sbuf_bytes <= nl.tile_size.sbuf_fmax_bytes
        win_pp = nh_buf * w_buf

        n_plane_tiles = (planes + PMAX - 1) // PMAX
        n_row_blocks = (out_H + oh_block - 1) // oh_block

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


_tuner = NkiAutotuner(max_pool2d_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(input: torch.Tensor, N: int, C: int, H: int, W: int,
        kernel_size: int, stride: int, padding: int,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    x = input.view(N * C, H * W)
    itemsize = input.element_size()

    if autotune:
        out_H = (H + 2 * padding - kernel_size) // stride + 1
        search_space = [SimpleNamespace(oh_block=o)
                        for o in (8, 16, 32, 64) if o <= out_H]
        search_space.append(SimpleNamespace(oh_block=0))  # SBUF-fit fallback
        cfg = _tuner.tune_or_cached(
            shape_key=((N, C, H, W), kernel_size, stride, padding, str(input.dtype)),
            search_space=search_space,
            args_fn=lambda cfg: (x, H, W, kernel_size, stride, padding,
                                 itemsize, cfg.oh_block),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
        oh_block = cfg.oh_block
    else:
        oh_block = 0

    result = max_pool2d_kernel(x, H, W, kernel_size, stride, padding,
                               itemsize, oh_block)
    return result.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
