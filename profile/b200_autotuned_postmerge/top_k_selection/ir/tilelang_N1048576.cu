#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <math_constants.h>
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

extern "C" __global__ void main_kernel(float* __restrict__ input_padding, int N, int stage, int stride);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(float* __restrict__ input_padding, int N, int stage, int stride) {
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    int rmod = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_1 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    bool valid_1 = ((((((((0 <= stride) && (0 <= rmod)) || ((stride < 0) && (rmod <= 0))) ? rdiv : (rdiv - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_1)) || ((stride < 0) && (rmod_1 <= 0))) ? rmod_1 : (rmod_1 + stride))) < N);
    int rmod_2 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_1 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_3 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    bool valid_2 = (((((((((0 <= stride) && (0 <= rmod_2)) || ((stride < 0) && (rmod_2 <= 0))) ? rdiv_1 : (rdiv_1 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_3)) || ((stride < 0) && (rmod_3 <= 0))) ? rmod_3 : (rmod_3 + stride))) + stride) < N);
    int rmod_4 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_2 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_5 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    float condval;
    if (((((((((0 <= stride) && (0 <= rmod_4)) || ((stride < 0) && (rmod_4 <= 0))) ? rdiv_2 : (rdiv_2 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_5)) || ((stride < 0) && (rmod_5 <= 0))) ? rmod_5 : (rmod_5 + stride))) < N)) {
      int rmod_6 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      int rdiv_3 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
      int rmod_7 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      int rmod_8 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      int rmod_9 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      int rdiv_4 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
      float condval_1;
      if (((0 <= (((((((0 <= stride) && (0 <= rmod_6)) || ((stride < 0) && (rmod_6 <= 0))) ? rdiv_3 : (rdiv_3 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_7)) || ((stride < 0) && (rmod_7 <= 0))) ? rmod_7 : (rmod_7 + stride)))) && (((((((0 <= stride) && (0 <= rmod_8)) || ((stride < 0) && (rmod_8 <= 0))) ? rmod_8 : (rmod_8 + stride)) >> 1) + (((((0 <= stride) && (0 <= rmod_9)) || ((stride < 0) && (rmod_9 <= 0))) ? rdiv_4 : (rdiv_4 - 1)) * stride)) < 524288))) {
        int64_t rmod_10 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
        int64_t rdiv_5 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) / ((int64_t)stride));
        int64_t rmod_11 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
        int64_t rmod_12 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
        int64_t rdiv_6 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) / ((int64_t)stride));
        int64_t rmod_13 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
        condval_1 = input_padding[((((((((int64_t)0 <= ((int64_t)stride)) && ((int64_t)0 <= rmod_12)) || ((((int64_t)stride) < (int64_t)0) && (rmod_12 <= (int64_t)0))) ? rdiv_6 : (rdiv_6 - (int64_t)1)) * ((int64_t)stride)) * (int64_t)2) + (((((int64_t)0 <= ((int64_t)stride)) && ((int64_t)0 <= rmod_13)) || ((((int64_t)stride) < (int64_t)0) && (rmod_13 <= (int64_t)0))) ? rmod_13 : (rmod_13 + ((int64_t)stride))))];
      } else {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      }
      condval = condval_1;
    } else {
      condval = -CUDART_INF_F;
    }
    float slice_1_t = condval;
    int rmod_14 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_7 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_15 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    float condval_2;
    if ((((((((((0 <= stride) && (0 <= rmod_14)) || ((stride < 0) && (rmod_14 <= 0))) ? rdiv_7 : (rdiv_7 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_15)) || ((stride < 0) && (rmod_15 <= 0))) ? rmod_15 : (rmod_15 + stride))) + stride) < N)) {
      int rmod_16 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      int rdiv_8 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
      int rmod_17 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      int rmod_18 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      int rmod_19 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      int rdiv_9 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
      float condval_3;
      if (((0 <= ((((((((0 <= stride) && (0 <= rmod_16)) || ((stride < 0) && (rmod_16 <= 0))) ? rdiv_8 : (rdiv_8 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_17)) || ((stride < 0) && (rmod_17 <= 0))) ? rmod_17 : (rmod_17 + stride))) + stride)) && ((((((((0 <= stride) && (0 <= rmod_18)) || ((stride < 0) && (rmod_18 <= 0))) ? rmod_18 : (rmod_18 + stride)) + stride) >> 1) + (((((0 <= stride) && (0 <= rmod_19)) || ((stride < 0) && (rmod_19 <= 0))) ? rdiv_9 : (rdiv_9 - 1)) * stride)) < 524288))) {
        int64_t rmod_20 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
        int64_t rdiv_10 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) / ((int64_t)stride));
        int64_t rmod_21 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
        int64_t rmod_22 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
        int64_t rdiv_11 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) / ((int64_t)stride));
        int64_t rmod_23 = ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) % ((int64_t)stride));
        condval_3 = input_padding[(((((((((int64_t)0 <= ((int64_t)stride)) && ((int64_t)0 <= rmod_22)) || ((((int64_t)stride) < (int64_t)0) && (rmod_22 <= (int64_t)0))) ? rdiv_11 : (rdiv_11 - (int64_t)1)) * ((int64_t)stride)) * (int64_t)2) + (((((int64_t)0 <= ((int64_t)stride)) && ((int64_t)0 <= rmod_23)) || ((((int64_t)stride) < (int64_t)0) && (rmod_23 <= (int64_t)0))) ? rmod_23 : (rmod_23 + ((int64_t)stride)))) + ((int64_t)stride))];
      } else {
        condval_3 = 0x0p+0f/*0.000000e+00*/;
      }
      condval_2 = condval_3;
    } else {
      condval_2 = -CUDART_INF_F;
    }
    float slice_2_t = condval_2;
    int rmod_24 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_12 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_25 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rmod_26 = ((((((((0 <= stride) && (0 <= rmod_24)) || ((stride < 0) && (rmod_24 <= 0))) ? rdiv_12 : (rdiv_12 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_25)) || ((stride < 0) && (rmod_25 <= 0))) ? rmod_25 : (rmod_25 + stride))) % stage);
    int rmod_27 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_13 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_28 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_14 = ((((((((0 <= stride) && (0 <= rmod_27)) || ((stride < 0) && (rmod_27 <= 0))) ? rdiv_13 : (rdiv_13 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_28)) || ((stride < 0) && (rmod_28 <= 0))) ? rmod_28 : (rmod_28 + stride))) / stage);
    bool descend = ((((((0 <= stage) && (0 <= rmod_26)) || ((stage < 0) && (rmod_26 <= 0))) ? rdiv_14 : (rdiv_14 - 1)) & 1) == 1);
    bool greater = (slice_2_t < slice_1_t);
    int rmod_29 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_15 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_30 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rmod_31 = ((((((((0 <= stride) && (0 <= rmod_29)) || ((stride < 0) && (rmod_29 <= 0))) ? rdiv_15 : (rdiv_15 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_30)) || ((stride < 0) && (rmod_30 <= 0))) ? rmod_30 : (rmod_30 + stride))) % stage);
    int rmod_32 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_16 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_33 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_17 = ((((((((0 <= stride) && (0 <= rmod_32)) || ((stride < 0) && (rmod_32 <= 0))) ? rdiv_16 : (rdiv_16 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_33)) || ((stride < 0) && (rmod_33 <= 0))) ? rmod_33 : (rmod_33 + stride))) / stage);
    bool swap = (((((((0 <= stage) && (0 <= rmod_31)) || ((stage < 0) && (rmod_31 <= 0))) ? rdiv_17 : (rdiv_17 - 1)) & 1) == 1) == (slice_2_t < slice_1_t));
    int rmod_34 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_18 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_35 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rmod_36 = ((((((((0 <= stride) && (0 <= rmod_34)) || ((stride < 0) && (rmod_34 <= 0))) ? rdiv_18 : (rdiv_18 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_35)) || ((stride < 0) && (rmod_35 <= 0))) ? rmod_35 : (rmod_35 + stride))) % stage);
    int rmod_37 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_19 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_38 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_20 = ((((((((0 <= stride) && (0 <= rmod_37)) || ((stride < 0) && (rmod_37 <= 0))) ? rdiv_19 : (rdiv_19 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_38)) || ((stride < 0) && (rmod_38 <= 0))) ? rmod_38 : (rmod_38 + stride))) / stage);
    float condval_4;
    if ((((((((0 <= stage) && (0 <= rmod_36)) || ((stage < 0) && (rmod_36 <= 0))) ? rdiv_20 : (rdiv_20 - 1)) & 1) == 1) == (slice_2_t < slice_1_t))) {
      condval_4 = slice_2_t;
    } else {
      condval_4 = slice_1_t;
    }
    float new_slice_1_t = condval_4;
    int rmod_39 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_21 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_40 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rmod_41 = ((((((((0 <= stride) && (0 <= rmod_39)) || ((stride < 0) && (rmod_39 <= 0))) ? rdiv_21 : (rdiv_21 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_40)) || ((stride < 0) && (rmod_40 <= 0))) ? rmod_40 : (rmod_40 + stride))) % stage);
    int rmod_42 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_22 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_43 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_23 = ((((((((0 <= stride) && (0 <= rmod_42)) || ((stride < 0) && (rmod_42 <= 0))) ? rdiv_22 : (rdiv_22 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_43)) || ((stride < 0) && (rmod_43 <= 0))) ? rmod_43 : (rmod_43 + stride))) / stage);
    float condval_5;
    if ((((((((0 <= stage) && (0 <= rmod_41)) || ((stage < 0) && (rmod_41 <= 0))) ? rdiv_23 : (rdiv_23 - 1)) & 1) == 1) == (slice_2_t < slice_1_t))) {
      condval_5 = slice_1_t;
    } else {
      condval_5 = slice_2_t;
    }
    float new_slice_2_t = condval_5;
    int rmod_44 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_24 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_45 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    if ((((((((0 <= stride) && (0 <= rmod_44)) || ((stride < 0) && (rmod_44 <= 0))) ? rdiv_24 : (rdiv_24 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_45)) || ((stride < 0) && (rmod_45 <= 0))) ? rmod_45 : (rmod_45 + stride))) < N) {
      int rmod_46 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      int rdiv_25 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
      int rmod_47 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      if (0 <= (((((((0 <= stride) && (0 <= rmod_46)) || ((stride < 0) && (rmod_46 <= 0))) ? rdiv_25 : (rdiv_25 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_47)) || ((stride < 0) && (rmod_47 <= 0))) ? rmod_47 : (rmod_47 + stride)))) {
        int rmod_48 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rmod_49 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rdiv_26 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
        if (((((((0 <= stride) && (0 <= rmod_48)) || ((stride < 0) && (rmod_48 <= 0))) ? rmod_48 : (rmod_48 + stride)) >> 1) + (((((0 <= stride) && (0 <= rmod_49)) || ((stride < 0) && (rmod_49 <= 0))) ? rdiv_26 : (rdiv_26 - 1)) * stride)) < 524288) {
          int rmod_50 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          int rdiv_27 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
          int rmod_51 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          int rmod_52 = ((((((((0 <= stride) && (0 <= rmod_50)) || ((stride < 0) && (rmod_50 <= 0))) ? rdiv_27 : (rdiv_27 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_51)) || ((stride < 0) && (rmod_51 <= 0))) ? rmod_51 : (rmod_51 + stride))) % stage);
          int rmod_53 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          int rdiv_28 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
          int rmod_54 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          int rdiv_29 = ((((((((0 <= stride) && (0 <= rmod_53)) || ((stride < 0) && (rmod_53 <= 0))) ? rdiv_28 : (rdiv_28 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_54)) || ((stride < 0) && (rmod_54 <= 0))) ? rmod_54 : (rmod_54 + stride))) / stage);
          float condval_6;
          if ((((((((0 <= stage) && (0 <= rmod_52)) || ((stage < 0) && (rmod_52 <= 0))) ? rdiv_29 : (rdiv_29 - 1)) & 1) == 1) == (slice_2_t < slice_1_t))) {
            condval_6 = slice_2_t;
          } else {
            condval_6 = slice_1_t;
          }
          int rmod_55 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          int rdiv_30 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
          int rmod_56 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          int rmod_57 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          int rdiv_31 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
          int rmod_58 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          input_padding[(((((((0 <= stride) && (0 <= rmod_57)) || ((stride < 0) && (rmod_57 <= 0))) ? rdiv_31 : (rdiv_31 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_58)) || ((stride < 0) && (rmod_58 <= 0))) ? rmod_58 : (rmod_58 + stride)))] = condval_6;
        }
      }
    }
    int rmod_59 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    int rdiv_32 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
    int rmod_60 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
    if (((((((((0 <= stride) && (0 <= rmod_59)) || ((stride < 0) && (rmod_59 <= 0))) ? rdiv_32 : (rdiv_32 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_60)) || ((stride < 0) && (rmod_60 <= 0))) ? rmod_60 : (rmod_60 + stride))) + stride) < N) {
      int rmod_61 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      int rdiv_33 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
      int rmod_62 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
      if (0 <= ((((((((0 <= stride) && (0 <= rmod_61)) || ((stride < 0) && (rmod_61 <= 0))) ? rdiv_33 : (rdiv_33 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_62)) || ((stride < 0) && (rmod_62 <= 0))) ? rmod_62 : (rmod_62 + stride))) + stride)) {
        int rmod_63 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rmod_64 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
        int rdiv_34 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
        if ((((((((0 <= stride) && (0 <= rmod_63)) || ((stride < 0) && (rmod_63 <= 0))) ? rmod_63 : (rmod_63 + stride)) + stride) >> 1) + (((((0 <= stride) && (0 <= rmod_64)) || ((stride < 0) && (rmod_64 <= 0))) ? rdiv_34 : (rdiv_34 - 1)) * stride)) < 524288) {
          int rmod_65 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          int rdiv_35 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
          int rmod_66 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          int rmod_67 = ((((((((0 <= stride) && (0 <= rmod_65)) || ((stride < 0) && (rmod_65 <= 0))) ? rdiv_35 : (rdiv_35 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_66)) || ((stride < 0) && (rmod_66 <= 0))) ? rmod_66 : (rmod_66 + stride))) % stage);
          int rmod_68 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          int rdiv_36 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
          int rmod_69 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          int rdiv_37 = ((((((((0 <= stride) && (0 <= rmod_68)) || ((stride < 0) && (rmod_68 <= 0))) ? rdiv_36 : (rdiv_36 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_69)) || ((stride < 0) && (rmod_69 <= 0))) ? rmod_69 : (rmod_69 + stride))) / stage);
          float condval_7;
          if ((((((((0 <= stage) && (0 <= rmod_67)) || ((stage < 0) && (rmod_67 <= 0))) ? rdiv_37 : (rdiv_37 - 1)) & 1) == 1) == (slice_2_t < slice_1_t))) {
            condval_7 = slice_1_t;
          } else {
            condval_7 = slice_2_t;
          }
          int rmod_70 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          int rdiv_38 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
          int rmod_71 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          int rmod_72 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          int rdiv_39 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) / stride);
          int rmod_73 = ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) % stride);
          input_padding[((((((((0 <= stride) && (0 <= rmod_72)) || ((stride < 0) && (rmod_72 <= 0))) ? rdiv_39 : (rdiv_39 - 1)) * stride) * 2) + ((((0 <= stride) && (0 <= rmod_73)) || ((stride < 0) && (rmod_73 <= 0))) ? rmod_73 : (rmod_73 + stride))) + stride)] = condval_7;
        }
      }
    }
  }
}

