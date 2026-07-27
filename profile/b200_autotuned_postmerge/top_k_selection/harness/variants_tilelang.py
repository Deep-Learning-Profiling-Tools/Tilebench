"""TileLang bitonic-step variants used to attribute the instruction excess.

v0  baseline           exactly benchmarks/operators/top_k_selection/impl_tilelang.py
v1  shift              runtime div/mod replaced by log2 shift/mask (stride, stage
                       are always powers of two in this pipeline)
v2  shift_novalid      v1 without the `< N` guards (padding_len == N whenever N
                       is a power of two, which holds for every benchmark case)
v3  shift_novalid_vec  v2 with an explicit vectorized inner axis
"""

import tilelang
import tilelang.language as T


@tilelang.jit
def v0_baseline(padding_len, dtype, BLOCK_SIZE: int = 512, threads: int = 256):
    @T.prim_func
    def main(
        input_padding: T.Tensor((padding_len,), dtype),
        N: T.int32,
        stage: T.int32,
        stride: T.int32,
    ):
        neg_inf = -T.infinity(dtype)
        with T.Kernel(T.ceildiv(padding_len, BLOCK_SIZE * 2), threads=threads) as pid:
            for local_idx in T.Parallel(BLOCK_SIZE):
                offset = pid * BLOCK_SIZE + local_idx
                slice_1_offset = (offset // stride) * (2 * stride) + (offset % stride)
                slice_2_offset = slice_1_offset + stride
                valid_1 = slice_1_offset < N
                valid_2 = slice_2_offset < N
                slice_1_t = T.if_then_else(valid_1, input_padding[slice_1_offset], neg_inf)
                slice_2_t = T.if_then_else(valid_2, input_padding[slice_2_offset], neg_inf)
                descend = ((slice_1_offset // stage) % 2) == 1
                greater = slice_1_t > slice_2_t
                swap = descend == greater
                new_slice_1_t = T.if_then_else(swap, slice_2_t, slice_1_t)
                new_slice_2_t = T.if_then_else(swap, slice_1_t, slice_2_t)
                if valid_1:
                    input_padding[slice_1_offset] = new_slice_1_t
                if valid_2:
                    input_padding[slice_2_offset] = new_slice_2_t

    return main


@tilelang.jit
def v1_shift(padding_len, dtype, BLOCK_SIZE: int = 512, threads: int = 256):
    @T.prim_func
    def main(
        input_padding: T.Tensor((padding_len,), dtype),
        N: T.int32,
        log_stage: T.int32,
        log_stride: T.int32,
    ):
        neg_inf = -T.infinity(dtype)
        with T.Kernel(T.ceildiv(padding_len, BLOCK_SIZE * 2), threads=threads) as pid:
            for local_idx in T.Parallel(BLOCK_SIZE):
                offset = pid * BLOCK_SIZE + local_idx
                stride = T.shift_left(1, log_stride)
                mask = stride - 1
                slice_1_offset = (
                    T.shift_left(T.shift_right(offset, log_stride), log_stride + 1)
                    + T.bitwise_and(offset, mask)
                )
                slice_2_offset = slice_1_offset + stride
                valid_1 = slice_1_offset < N
                valid_2 = slice_2_offset < N
                slice_1_t = T.if_then_else(valid_1, input_padding[slice_1_offset], neg_inf)
                slice_2_t = T.if_then_else(valid_2, input_padding[slice_2_offset], neg_inf)
                descend = (
                    T.bitwise_and(T.shift_right(slice_1_offset, log_stage), 1) == 1
                )
                greater = slice_1_t > slice_2_t
                swap = descend == greater
                new_slice_1_t = T.if_then_else(swap, slice_2_t, slice_1_t)
                new_slice_2_t = T.if_then_else(swap, slice_1_t, slice_2_t)
                if valid_1:
                    input_padding[slice_1_offset] = new_slice_1_t
                if valid_2:
                    input_padding[slice_2_offset] = new_slice_2_t

    return main


@tilelang.jit
def v2_shift_novalid(padding_len, dtype, BLOCK_SIZE: int = 512, threads: int = 256):
    @T.prim_func
    def main(
        input_padding: T.Tensor((padding_len,), dtype),
        N: T.int32,
        log_stage: T.int32,
        log_stride: T.int32,
    ):
        with T.Kernel(T.ceildiv(padding_len, BLOCK_SIZE * 2), threads=threads) as pid:
            for local_idx in T.Parallel(BLOCK_SIZE):
                offset = pid * BLOCK_SIZE + local_idx
                stride = T.shift_left(1, log_stride)
                mask = stride - 1
                slice_1_offset = (
                    T.shift_left(T.shift_right(offset, log_stride), log_stride + 1)
                    + T.bitwise_and(offset, mask)
                )
                slice_2_offset = slice_1_offset + stride
                slice_1_t = input_padding[slice_1_offset]
                slice_2_t = input_padding[slice_2_offset]
                descend = (
                    T.bitwise_and(T.shift_right(slice_1_offset, log_stage), 1) == 1
                )
                greater = slice_1_t > slice_2_t
                swap = descend == greater
                input_padding[slice_1_offset] = T.if_then_else(swap, slice_2_t, slice_1_t)
                input_padding[slice_2_offset] = T.if_then_else(swap, slice_1_t, slice_2_t)

    return main


@tilelang.jit
def v3_novalid_only(padding_len, dtype, BLOCK_SIZE: int = 512, threads: int = 256):
    """Baseline div/mod indexing, guards removed — isolates the guard cost alone."""

    @T.prim_func
    def main(
        input_padding: T.Tensor((padding_len,), dtype),
        N: T.int32,
        stage: T.int32,
        stride: T.int32,
    ):
        with T.Kernel(T.ceildiv(padding_len, BLOCK_SIZE * 2), threads=threads) as pid:
            for local_idx in T.Parallel(BLOCK_SIZE):
                offset = pid * BLOCK_SIZE + local_idx
                slice_1_offset = (offset // stride) * (2 * stride) + (offset % stride)
                slice_2_offset = slice_1_offset + stride
                slice_1_t = input_padding[slice_1_offset]
                slice_2_t = input_padding[slice_2_offset]
                descend = ((slice_1_offset // stage) % 2) == 1
                greater = slice_1_t > slice_2_t
                swap = descend == greater
                input_padding[slice_1_offset] = T.if_then_else(swap, slice_2_t, slice_1_t)
                input_padding[slice_2_offset] = T.if_then_else(swap, slice_1_t, slice_2_t)

    return main


VARIANTS = {
    "v0_baseline": v0_baseline,
    "v1_shift": v1_shift,
    "v2_shift_novalid": v2_shift_novalid,
    "v3_novalid_only": v3_novalid_only,
}

# Which variants take log2 arguments instead of raw stage/stride.
LOG_ARGS = {"v1_shift", "v2_shift_novalid"}
