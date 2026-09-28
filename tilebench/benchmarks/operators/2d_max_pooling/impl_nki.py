from types import SimpleNamespace

import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None
    PMAX = 128

_DEFAULT_RC = (2, 128)
_SEARCH_RC = [(1, 128), (1, 256), (1, 512), (2, 128), (2, 256), (4, 128), (4, 256),
              (8, 64), (16, 512), (32, 512)]

if nki is not None:
    from tilebench.core.nki_autotune import NkiAutotuner

    @nki.jit
    def max_pool2d_kernel(input_hbm, in_H, in_W, kernel_size, stride, padding,
                          itemsize, oh_block, ow_block):
        planes, in_HW = input_hbm.shape

        out_H = (in_H + 2 * padding - kernel_size) // stride + 1
        out_W = (in_W + 2 * padding - kernel_size) // stride + 1
        out_HW = out_H * out_W

        assert in_HW == in_H * in_W
        assert out_H >= 1 and out_W >= 1

        oh_block = min(oh_block, out_H)
        ow_block = min(ow_block, out_W)

        nh_buf = (oh_block - 1) * stride + kernel_size
        w_buf = (ow_block - 1) * stride + kernel_size
        sbuf_bytes = itemsize * (nh_buf * w_buf + oh_block * ow_block)
        assert sbuf_bytes <= nl.tile_size.sbuf_fmax_bytes
        win_pp = nh_buf * w_buf

        n_plane_tiles = (planes + PMAX - 1) // PMAX
        n_row_blocks = (out_H + oh_block - 1) // oh_block
        n_col_blocks = (out_W + ow_block - 1) // ow_block

        output_hbm = nl.ndarray((planes, out_HW), dtype=input_hbm.dtype,
                                buffer=nl.shared_hbm)

        win = nl.ndarray((PMAX, nh_buf, w_buf), dtype=input_hbm.dtype, buffer=nl.sbuf)
        out_sb = nl.ndarray((PMAX, oh_block, ow_block), dtype=input_hbm.dtype,
                            buffer=nl.sbuf)

        for p_tile in range(n_plane_tiles):
            p_start = p_tile * PMAX
            p_sz = min(PMAX, planes - p_start)

            for blk_idx in range(n_row_blocks):
                oh0 = blk_idx * oh_block
                blk = min(oh_block, out_H - oh0)
                nh = (blk - 1) * stride + kernel_size
                ih0 = oh0 * stride - padding
                row_lo = max(ih0, 0)
                row_hi = min(ih0 + nh, in_H)
                row_dst = row_lo - ih0
                n_rows = row_hi - row_lo

                for cblk_idx in range(n_col_blocks):
                    ow0 = cblk_idx * ow_block
                    cw = min(ow_block, out_W - ow0)
                    nw = (cw - 1) * stride + kernel_size
                    iw0 = ow0 * stride - padding
                    col_lo = max(iw0, 0)
                    col_hi = min(iw0 + nw, in_W)
                    col_dst = col_lo - iw0
                    n_cols = col_hi - col_lo

                    nisa.memset(dst=win[0:p_sz, 0:nh, 0:nw], value=float("-inf"))
                    if n_rows > 0 and n_cols > 0:
                        nisa.dma_copy(
                            dst=win[0:p_sz, row_dst:row_dst + n_rows,
                                    col_dst:col_dst + n_cols],
                            src=input_hbm.ap(
                                pattern=[[in_HW, p_sz], [in_W, n_rows], [1, n_cols]],
                                offset=p_start * in_HW + row_lo * in_W + col_lo,
                            ),
                        )

                    acc = out_sb[0:p_sz, 0:blk, 0:cw]
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            tap = win.ap(
                                pattern=[[win_pp, p_sz], [stride * w_buf, blk],
                                         [stride, cw]],
                                offset=kh * w_buf + kw,
                            )
                            if kh == 0 and kw == 0:
                                nisa.tensor_copy(dst=acc, src=tap)
                            else:
                                nisa.tensor_tensor(dst=acc, data1=acc, data2=tap,
                                                   op=nl.maximum)

                    nisa.dma_copy(
                        dst=output_hbm.ap(
                            pattern=[[out_HW, p_sz], [out_W, blk], [1, cw]],
                            offset=p_start * out_HW + oh0 * out_W + ow0,
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
        cfg = _tuner.tune_or_cached(
            shape_key=((N, C, H, W), kernel_size, stride, padding, str(input.dtype)),
            search_space=[SimpleNamespace(oh_block=r, ow_block=c) for r, c in _SEARCH_RC],
            args_fn=lambda cfg: (x, H, W, kernel_size, stride, padding,
                                 itemsize, cfg.oh_block, cfg.ow_block),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
        oh_block, ow_block = cfg.oh_block, cfg.ow_block
    else:
        oh_block, ow_block = _DEFAULT_RC

    result = max_pool2d_kernel(x, H, W, kernel_size, stride, padding,
                               itemsize, oh_block, ow_block)
    return result.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
