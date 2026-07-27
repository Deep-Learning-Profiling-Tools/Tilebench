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

extern "C" __global__ void main_kernel(float* __restrict__ input_padding, int N, int log_stage, int log_stride);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(float* __restrict__ input_padding, int N, int log_stage, int log_stride) {
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    bool valid_1 = (((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) < N);
    bool valid_2 = ((((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) + (1 << log_stride)) < N);
    float condval;
    if ((((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) < N)) {
      float condval_1;
      if (((0 <= ((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1)))) && (((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) < 1048576))) {
        condval_1 = input_padding[((((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) >> ((int64_t)log_stride)) << (((int64_t)log_stride) + (int64_t)1)) + ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) & (((int64_t)1 << ((int64_t)log_stride)) - (int64_t)1)))];
      } else {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      }
      condval = condval_1;
    } else {
      condval = -CUDART_INF_F;
    }
    float slice_1_t = condval;
    float condval_2;
    if (((((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) + (1 << log_stride)) < N)) {
      float condval_3;
      if (((0 <= (((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) + (1 << log_stride))) && ((((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) + (1 << log_stride)) < 1048576))) {
        condval_3 = input_padding[(((((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) >> ((int64_t)log_stride)) << (((int64_t)log_stride) + (int64_t)1)) + ((((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)i) * (int64_t)256)) + ((int64_t)((int)threadIdx.x))) & (((int64_t)1 << ((int64_t)log_stride)) - (int64_t)1))) + ((int64_t)1 << ((int64_t)log_stride)))];
      } else {
        condval_3 = 0x0p+0f/*0.000000e+00*/;
      }
      condval_2 = condval_3;
    } else {
      condval_2 = -CUDART_INF_F;
    }
    float slice_2_t = condval_2;
    bool descend = (((((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) >> log_stage) & 1) == 1);
    bool greater = (slice_2_t < slice_1_t);
    bool swap = ((((((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) >> log_stage) & 1) == 1) == (slice_2_t < slice_1_t));
    float condval_4;
    if (((((((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) >> log_stage) & 1) == 1) == (slice_2_t < slice_1_t))) {
      condval_4 = slice_2_t;
    } else {
      condval_4 = slice_1_t;
    }
    float new_slice_1_t = condval_4;
    float condval_5;
    if (((((((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) >> log_stage) & 1) == 1) == (slice_2_t < slice_1_t))) {
      condval_5 = slice_1_t;
    } else {
      condval_5 = slice_2_t;
    }
    float new_slice_2_t = condval_5;
    if (((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) < N) {
      if (0 <= ((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1)))) {
        if (((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) < 1048576) {
          float condval_6;
          if (((((((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) >> log_stage) & 1) == 1) == (slice_2_t < slice_1_t))) {
            condval_6 = slice_2_t;
          } else {
            condval_6 = slice_1_t;
          }
          input_padding[((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1)))] = condval_6;
        }
      }
    }
    if ((((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) + (1 << log_stride)) < N) {
      if (0 <= (((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) + (1 << log_stride))) {
        if ((((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) + (1 << log_stride)) < 1048576) {
          float condval_7;
          if (((((((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) >> log_stage) & 1) == 1) == (slice_2_t < slice_1_t))) {
            condval_7 = slice_1_t;
          } else {
            condval_7 = slice_2_t;
          }
          input_padding[(((((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) >> log_stride) << (log_stride + 1)) + ((((((int)blockIdx.x) * 512) + (i * 256)) + ((int)threadIdx.x)) & ((1 << log_stride) - 1))) + (1 << log_stride))] = condval_7;
        }
      }
    }
  }
}

