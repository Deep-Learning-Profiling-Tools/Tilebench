#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(float* __restrict__ input_padding, int stage, int stride);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(float* __restrict__ input_padding, int stage, int stride) {
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    int rmod = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_1 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rmod_2 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rmod_3 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_1 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    float condval;
    if (((0 <= (((((((0 <= stride) && (0 <= rmod)) || ((stride < 0) && (rmod <= 0))) ? rdiv : (rdiv - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_1)) || ((stride < 0) && (rmod_1 <= 0))) ? rmod_1 : (rmod_1 + stride)))) && (((((((0 <= stride) && (0 <= rmod_2)) || ((stride < 0) && (rmod_2 <= 0))) ? rmod_2 : (rmod_2 + stride)) >> 1) + (((((0 <= stride) && (0 <= rmod_3)) || ((stride < 0) && (rmod_3 <= 0))) ? rdiv_1 : (rdiv_1 - 1)) * stride)) < 524288))) {
      int64_t rmod_4 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
      int64_t rdiv_2 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) / ((int64_t)stride));
      int64_t rmod_5 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
      int64_t rmod_6 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
      int64_t rdiv_3 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) / ((int64_t)stride));
      int64_t rmod_7 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
      condval = input_padding[((((((((int64_t)0 <= ((int64_t)stride)) && ((int64_t)0 <= rmod_6)) || ((((int64_t)stride) < (int64_t)0) && (rmod_6 <= (int64_t)0))) ? rdiv_3 : (rdiv_3 - (int64_t)1)) * ((int64_t)stride)) * (int64_t)2) + (((((int64_t)0 <= ((int64_t)stride)) && ((int64_t)0 <= rmod_7)) || ((((int64_t)stride) < (int64_t)0) && (rmod_7 <= (int64_t)0))) ? rmod_7 : (rmod_7 + ((int64_t)stride))))];
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    float slice_1_t = condval;
    int rmod_8 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_4 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_9 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rmod_10 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rmod_11 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_5 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    float condval_1;
    if (((0 <= ((((((((0 <= stride) && (0 <= rmod_8)) || ((stride < 0) && (rmod_8 <= 0))) ? rdiv_4 : (rdiv_4 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_9)) || ((stride < 0) && (rmod_9 <= 0))) ? rmod_9 : (rmod_9 + stride))) + stride)) && ((((((((0 <= stride) && (0 <= rmod_10)) || ((stride < 0) && (rmod_10 <= 0))) ? rmod_10 : (rmod_10 + stride)) + stride) >> 1) + (((((0 <= stride) && (0 <= rmod_11)) || ((stride < 0) && (rmod_11 <= 0))) ? rdiv_5 : (rdiv_5 - 1)) * stride)) < 524288))) {
      int64_t rmod_12 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
      int64_t rdiv_6 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) / ((int64_t)stride));
      int64_t rmod_13 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
      int64_t rmod_14 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
      int64_t rdiv_7 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) / ((int64_t)stride));
      int64_t rmod_15 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
      condval_1 = input_padding[(((((((((int64_t)0 <= ((int64_t)stride)) && ((int64_t)0 <= rmod_14)) || ((((int64_t)stride) < (int64_t)0) && (rmod_14 <= (int64_t)0))) ? rdiv_7 : (rdiv_7 - (int64_t)1)) * ((int64_t)stride)) * (int64_t)2) + (((((int64_t)0 <= ((int64_t)stride)) && ((int64_t)0 <= rmod_15)) || ((((int64_t)stride) < (int64_t)0) && (rmod_15 <= (int64_t)0))) ? rmod_15 : (rmod_15 + ((int64_t)stride)))) + ((int64_t)stride))];
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    float slice_2_t = condval_1;
    int rmod_16 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_8 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_17 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rmod_18 = ((((((((0 <= stride) && (0 <= rmod_16)) || ((stride < 0) && (rmod_16 <= 0))) ? rdiv_8 : (rdiv_8 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_17)) || ((stride < 0) && (rmod_17 <= 0))) ? rmod_17 : (rmod_17 + stride))) % stage);
    int rmod_19 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_9 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_20 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_10 = ((((((((0 <= stride) && (0 <= rmod_19)) || ((stride < 0) && (rmod_19 <= 0))) ? rdiv_9 : (rdiv_9 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_20)) || ((stride < 0) && (rmod_20 <= 0))) ? rmod_20 : (rmod_20 + stride))) / stage);
    bool descend = ((((((0 <= stage) && (0 <= rmod_18)) || ((stage < 0) && (rmod_18 <= 0))) ? rdiv_10 : (rdiv_10 - 1)) & 1) == 1);
    bool greater = (slice_2_t < slice_1_t);
    int rmod_21 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_11 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_22 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rmod_23 = ((((((((0 <= stride) && (0 <= rmod_21)) || ((stride < 0) && (rmod_21 <= 0))) ? rdiv_11 : (rdiv_11 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_22)) || ((stride < 0) && (rmod_22 <= 0))) ? rmod_22 : (rmod_22 + stride))) % stage);
    int rmod_24 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_12 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_25 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_13 = ((((((((0 <= stride) && (0 <= rmod_24)) || ((stride < 0) && (rmod_24 <= 0))) ? rdiv_12 : (rdiv_12 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_25)) || ((stride < 0) && (rmod_25 <= 0))) ? rmod_25 : (rmod_25 + stride))) / stage);
    bool swap = (((((((0 <= stage) && (0 <= rmod_23)) || ((stage < 0) && (rmod_23 <= 0))) ? rdiv_13 : (rdiv_13 - 1)) & 1) == 1) == (slice_2_t < slice_1_t));
    int rmod_26 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_14 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_27 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    if (0 <= (((((((0 <= stride) && (0 <= rmod_26)) || ((stride < 0) && (rmod_26 <= 0))) ? rdiv_14 : (rdiv_14 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_27)) || ((stride < 0) && (rmod_27 <= 0))) ? rmod_27 : (rmod_27 + stride)))) {
      int rmod_28 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      int rmod_29 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      int rdiv_15 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
      if (((((((0 <= stride) && (0 <= rmod_28)) || ((stride < 0) && (rmod_28 <= 0))) ? rmod_28 : (rmod_28 + stride)) >> 1) + (((((0 <= stride) && (0 <= rmod_29)) || ((stride < 0) && (rmod_29 <= 0))) ? rdiv_15 : (rdiv_15 - 1)) * stride)) < 524288) {
        int rmod_30 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rdiv_16 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
        int rmod_31 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rmod_32 = ((((((((0 <= stride) && (0 <= rmod_30)) || ((stride < 0) && (rmod_30 <= 0))) ? rdiv_16 : (rdiv_16 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_31)) || ((stride < 0) && (rmod_31 <= 0))) ? rmod_31 : (rmod_31 + stride))) % stage);
        int rmod_33 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rdiv_17 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
        int rmod_34 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rdiv_18 = ((((((((0 <= stride) && (0 <= rmod_33)) || ((stride < 0) && (rmod_33 <= 0))) ? rdiv_17 : (rdiv_17 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_34)) || ((stride < 0) && (rmod_34 <= 0))) ? rmod_34 : (rmod_34 + stride))) / stage);
        float condval_2;
        if ((((((((0 <= stage) && (0 <= rmod_32)) || ((stage < 0) && (rmod_32 <= 0))) ? rdiv_18 : (rdiv_18 - 1)) & 1) == 1) == (slice_2_t < slice_1_t))) {
          condval_2 = slice_2_t;
        } else {
          condval_2 = slice_1_t;
        }
        int rmod_35 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rdiv_19 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
        int rmod_36 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rmod_37 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rdiv_20 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
        int rmod_38 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        input_padding[(((((((0 <= stride) && (0 <= rmod_37)) || ((stride < 0) && (rmod_37 <= 0))) ? rdiv_20 : (rdiv_20 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_38)) || ((stride < 0) && (rmod_38 <= 0))) ? rmod_38 : (rmod_38 + stride)))] = condval_2;
      }
    }
    int rmod_39 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_21 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_40 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    if (0 <= ((((((((0 <= stride) && (0 <= rmod_39)) || ((stride < 0) && (rmod_39 <= 0))) ? rdiv_21 : (rdiv_21 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_40)) || ((stride < 0) && (rmod_40 <= 0))) ? rmod_40 : (rmod_40 + stride))) + stride)) {
      int rmod_41 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      int rmod_42 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      int rdiv_22 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
      if ((((((((0 <= stride) && (0 <= rmod_41)) || ((stride < 0) && (rmod_41 <= 0))) ? rmod_41 : (rmod_41 + stride)) + stride) >> 1) + (((((0 <= stride) && (0 <= rmod_42)) || ((stride < 0) && (rmod_42 <= 0))) ? rdiv_22 : (rdiv_22 - 1)) * stride)) < 524288) {
        int rmod_43 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rdiv_23 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
        int rmod_44 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rmod_45 = ((((((((0 <= stride) && (0 <= rmod_43)) || ((stride < 0) && (rmod_43 <= 0))) ? rdiv_23 : (rdiv_23 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_44)) || ((stride < 0) && (rmod_44 <= 0))) ? rmod_44 : (rmod_44 + stride))) % stage);
        int rmod_46 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rdiv_24 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
        int rmod_47 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rdiv_25 = ((((((((0 <= stride) && (0 <= rmod_46)) || ((stride < 0) && (rmod_46 <= 0))) ? rdiv_24 : (rdiv_24 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_47)) || ((stride < 0) && (rmod_47 <= 0))) ? rmod_47 : (rmod_47 + stride))) / stage);
        float condval_3;
        if ((((((((0 <= stage) && (0 <= rmod_45)) || ((stage < 0) && (rmod_45 <= 0))) ? rdiv_25 : (rdiv_25 - 1)) & 1) == 1) == (slice_2_t < slice_1_t))) {
          condval_3 = slice_1_t;
        } else {
          condval_3 = slice_2_t;
        }
        int rmod_48 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rdiv_26 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
        int rmod_49 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rmod_50 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rdiv_27 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
        int rmod_51 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        input_padding[((((((((0 <= stride) && (0 <= rmod_50)) || ((stride < 0) && (rmod_50 <= 0))) ? rdiv_27 : (rdiv_27 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_51)) || ((stride < 0) && (rmod_51 <= 0))) ? rmod_51 : (rmod_51 + stride))) + stride)] = condval_3;
      }
    }
  }
}

