"""NKI Conv2d (groups=1) via implicit GEMM on the Tensor Engine.

case_defaults fixes in_channels == out_channels == 128 == PMAX, so each of
the kernel_size^2 taps is exactly one nc_matmul: stationary = weight[:,:,kh,kw]
(in_channels x out_channels, K=in_channels on partitions), moving = the
(kh,kw)-shifted input row (in_channels x out_W). Accumulating over the 9
taps into one PSUM tile per output row mirrors the K-block accumulation in
matmul_fp32_fp16_fp8. kernel_size is fixed at 3 by config -- the kernel
takes the 9 per-tap weight slices as separate arguments (sliced on the host,
where kh/kw are plain Python ints) rather than indexing a 4D weight tensor
inside the kernel.
"""
import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def conv2d_kernel(input_flat, w0, w1, w2, w3, w4, w5, w6, w7, w8,
                       Hp, in_channels, out_channels, out_H, out_W, stride):
        # input_flat: (in_channels * Hp, Wp) -- channel and (padded) row
        # dims flattened so a per-tap row lookup is a single affine index:
        # channel c, row r -> flat row c * Hp + r.
        w_taps = (w0, w1, w2, w3, w4, w5, w6, w7, w8)
        ic_idx = nl.arange(in_channels)[:, None]
        oc_idx_stat = nl.arange(in_channels)[:, None]
        oc_idx_stat2 = nl.arange(out_channels)[None, :]
        oc_idx_out = nl.arange(out_channels)[:, None]

        hbm_result = nl.ndarray((out_channels, out_H * out_W), dtype=input_flat.dtype, buffer=nl.hbm)

        for oh in nl.affine_range(out_H):
            psum_row = nl.zeros((out_channels, out_W), dtype=nl.float32, buffer=nl.psum)

            # Manually unrolled 3x3 tap loop: a plain Python `for kh/kw in
            # range(3):` loop reusing one `moving`/`row_sel` variable across
            # iterations was found (empirically, via temp/probe_2dconv_debug*)
            # to silently drop all but one tap's contribution once both the
            # row (kh) AND column (kw) index vary across >=2 values each --
            # a scheduling bug in how the compiler tracks per-iteration SBUF
            # tile lifetimes for a reused buffer inside nested plain loops.
            # Fully unrolled, uniquely-named tiles sidestep it (verified
            # correct in isolation).
            row0 = ic_idx * Hp + (oh * stride + 0)
            row1 = ic_idx * Hp + (oh * stride + 1)
            row2 = ic_idx * Hp + (oh * stride + 2)
            col0 = nl.arange(out_W)[None, :] * stride + 0
            col1 = nl.arange(out_W)[None, :] * stride + 1
            col2 = nl.arange(out_W)[None, :] * stride + 2

            m00 = nl.ndarray((in_channels, out_W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=m00, src=input_flat[row0, col0])
            m01 = nl.ndarray((in_channels, out_W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=m01, src=input_flat[row0, col1])
            m02 = nl.ndarray((in_channels, out_W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=m02, src=input_flat[row0, col2])
            m10 = nl.ndarray((in_channels, out_W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=m10, src=input_flat[row1, col0])
            m11 = nl.ndarray((in_channels, out_W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=m11, src=input_flat[row1, col1])
            m12 = nl.ndarray((in_channels, out_W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=m12, src=input_flat[row1, col2])
            m20 = nl.ndarray((in_channels, out_W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=m20, src=input_flat[row2, col0])
            m21 = nl.ndarray((in_channels, out_W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=m21, src=input_flat[row2, col1])
            m22 = nl.ndarray((in_channels, out_W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=m22, src=input_flat[row2, col2])

            s00 = nl.load(w_taps[0][oc_idx_stat, oc_idx_stat2], dtype=nl.float32)
            s01 = nl.load(w_taps[1][oc_idx_stat, oc_idx_stat2], dtype=nl.float32)
            s02 = nl.load(w_taps[2][oc_idx_stat, oc_idx_stat2], dtype=nl.float32)
            s10 = nl.load(w_taps[3][oc_idx_stat, oc_idx_stat2], dtype=nl.float32)
            s11 = nl.load(w_taps[4][oc_idx_stat, oc_idx_stat2], dtype=nl.float32)
            s12 = nl.load(w_taps[5][oc_idx_stat, oc_idx_stat2], dtype=nl.float32)
            s20 = nl.load(w_taps[6][oc_idx_stat, oc_idx_stat2], dtype=nl.float32)
            s21 = nl.load(w_taps[7][oc_idx_stat, oc_idx_stat2], dtype=nl.float32)
            s22 = nl.load(w_taps[8][oc_idx_stat, oc_idx_stat2], dtype=nl.float32)

            psum_row += nisa.nc_matmul(s00, m00)
            psum_row += nisa.nc_matmul(s01, m01)
            psum_row += nisa.nc_matmul(s02, m02)
            psum_row += nisa.nc_matmul(s10, m10)
            psum_row += nisa.nc_matmul(s11, m11)
            psum_row += nisa.nc_matmul(s12, m12)
            psum_row += nisa.nc_matmul(s20, m20)
            psum_row += nisa.nc_matmul(s21, m21)
            psum_row += nisa.nc_matmul(s22, m22)

            result_tile = nl.copy(psum_row, dtype=input_flat.dtype)
            nl.store(hbm_result[oc_idx_out, oh * out_W + nl.arange(out_W)[None, :]], value=result_tile)

        return hbm_result


def run(input: torch.Tensor, weight: torch.Tensor, stride: int = 1, padding: int = 1,
        groups: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if groups != 1:
        raise NotImplementedError("2d_conv NKI: only groups=1 is supported")
    batch, in_channels, H, W = input.shape
    out_channels, in_channels_g, kernel_size, kernel_size_w = weight.shape
    if batch != 1:
        raise NotImplementedError("2d_conv NKI: only batch=1 is supported")
    if kernel_size != 3 or kernel_size_w != 3:
        raise NotImplementedError("2d_conv NKI: only kernel_size=3 is supported")
    if in_channels != PMAX or out_channels != PMAX:
        raise NotImplementedError(f"2d_conv NKI: in/out_channels must equal {PMAX}")

    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1

    x = input.reshape(in_channels, H, W)
    padded = torch.nn.functional.pad(x, (padding, padding, padding, padding))
    Hp = H + 2 * padding
    Wp = W + 2 * padding
    input_flat = padded.reshape(in_channels * Hp, Wp)

    # (in_channels, out_channels) per tap, K=in_channels on partitions.
    weight_permuted = weight.permute(1, 0, 2, 3).contiguous()
    w_taps = [weight_permuted[:, :, kh, kw].contiguous()
              for kh in range(3) for kw in range(3)]

    result = conv2d_kernel(input_flat, *w_taps, Hp, in_channels, out_channels,
                            out_H, out_W, stride)
    return result.reshape(out_channels, out_H, out_W).reshape(-1)


def get_last_config() -> dict | None:
    return None
