import torch
import cuda.tile as ct
import numpy as np

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _conv2d_kernel(
    input_flat, weight_3d, output_flat,
    batch, H, W, OH, OW, M,
    IC: ConstInt, OC: ConstInt, KS: ConstInt,
    STRIDE: ConstInt, PADDING: ConstInt,
    NUM_C_BLOCKS: ConstInt,
    BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_C: ConstInt,
):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    offs_m = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    offs_n = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=np.int32)

    OHW = OH * OW
    HW = H * W
    ICHW = IC * HW

    b = offs_m // OHW
    rem_m = offs_m % OHW
    oh = rem_m // OW
    ow = rem_m % OW

    b_base = b * ICHW

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    valid_m = offs_m < M
    valid_n = offs_n < OC

    # (kh, kw) outer + c_blk inner. weight_3d viewed as (OC, KSKS, IC).
    for kh in range(0, KS):
        for kw in range(0, KS):
            ih = oh * STRIDE + kh - PADDING
            iw = ow * STRIDE + kw - PADDING
            in_bounds_m = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & valid_m
            ihw = ih * W + iw

            kk_idx = kh * KS + kw

            for c_blk in range(0, NUM_C_BLOCKS):
                ic = c_blk * BLOCK_C + ct.arange(BLOCK_C, dtype=np.int32)
                ic_valid = ic < IC

                input_idx = b_base[:, None] + ihw[:, None] + ic[None, :] * HW
                valid_full = in_bounds_m[:, None] & ic_valid[None, :]
                input_idx_safe = ct.where(valid_full, input_idx, -1)
                a = ct.gather(input_flat, input_idx_safe, padding_value=0.0)

                # weight_3d[pid_n*BN:..., kk_idx, c_blk*BC:...] → tile (BN, 1, BC)
                w_tile = ct.load(weight_3d, index=(pid_n, kk_idx, c_blk),
                                 shape=(BLOCK_N, 1, BLOCK_C),
                                 padding_mode=ct.PaddingMode.ZERO)
                w_tile = ct.reshape(w_tile, (BLOCK_N, BLOCK_C))
                w_t = ct.transpose(w_tile)  # (BLOCK_C, BLOCK_N)

                acc = ct.mma(a, w_t, acc)

    out_idx = (b[:, None] * (OC * OHW)
               + offs_n[None, :] * OHW
               + oh[:, None] * OW
               + ow[:, None])
    valid_out = valid_m[:, None] & valid_n[None, :]
    out_idx_safe = ct.where(valid_out, out_idx, -1)
    result = ct.astype(acc, output_flat.dtype)
    ct.scatter(output_flat, out_idx_safe, result)


def run(input, weight, stride=1, padding=1, groups=1, **kwargs):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    batch, IC, H, W = input.shape
    OC = weight.shape[0]
    KS = weight.shape[2]
    OH = (H + 2 * padding - KS) // stride + 1
    OW = (W + 2 * padding - KS) // stride + 1

    output = torch.empty((batch, OC, OH, OW), dtype=input.dtype, device=input.device)

    M = batch * OH * OW

    input_c = input.contiguous()
    weight_c = weight.contiguous()
    # weight (OC, IC, KH, KW) -> (OC, KH*KW, IC) so we can load tiles of (BN, 1, BC).
    weight_perm = weight_c.permute(0, 2, 3, 1).contiguous()
    weight_3d = weight_perm.view(OC, KS * KS, IC)

    input_flat = input_c.view(-1)
    output_flat = output.view(-1)

    # Single config across fp16/fp32. BC=64 fits fp32 shmem comfortably,
    # gives fp16 strong arithmetic intensity (BC=64 K-tile per dot).
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_C = 64
    NUM_C_BLOCKS = (IC + BLOCK_C - 1) // BLOCK_C

    grid = (ct.cdiv(M, BLOCK_M), ct.cdiv(OC, BLOCK_N), 1)
    stream = torch.cuda.current_stream()
    ct.launch(stream, grid, _conv2d_kernel, (
        input_flat, weight_3d, output_flat,
        batch, H, W, OH, OW, M,
        IC, OC, KS, stride, padding,
        NUM_C_BLOCKS,
        BLOCK_M, BLOCK_N, BLOCK_C,
    ))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_C": BLOCK_C,
        "NUM_C_BLOCKS": NUM_C_BLOCKS,
        "occupancy": 2, "path": "mma_v2_unified",
    })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
