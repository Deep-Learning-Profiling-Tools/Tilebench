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

extern "C" __global__ void max_pool2d_kernel_kernel(const bfloat16_t* __restrict__ input, bfloat16_t* __restrict__ output);
extern "C" __global__ void __launch_bounds__(128, 1) max_pool2d_kernel_kernel(const bfloat16_t* __restrict__ input, bfloat16_t* __restrict__ output) {
  bfloat16_t acc[4];
  bfloat16_t neg_inf = (std::numeric_limits<bfloat16_t>::infinity() * bfloat16_t(-0x1p+0f/*-1.000000e+00*/));
  *(uint2*)(acc + 0) = make_uint2(__pack_nv_bfloat162((std::numeric_limits<bfloat16_t>::infinity() * bfloat16_t(-0x1p+0f/*-1.000000e+00*/)), (std::numeric_limits<bfloat16_t>::infinity() * bfloat16_t(-0x1p+0f/*-1.000000e+00*/))), __pack_nv_bfloat162((std::numeric_limits<bfloat16_t>::infinity() * bfloat16_t(-0x1p+0f/*-1.000000e+00*/)), (std::numeric_limits<bfloat16_t>::infinity() * bfloat16_t(-0x1p+0f/*-1.000000e+00*/))));
  #pragma unroll
  for (int kh = 0; kh < 3; ++kh) {
    #pragma unroll
    for (int kw = 0; kw < 3; ++kw) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        bfloat16_t condval;
        if ((1 <= (((((int)blockIdx.z) * 128) + ((((int)threadIdx.x) & 63) * 2)) + kw))) {
          bfloat16_t condval_1;
          if ((1 <= ((((((int)blockIdx.y) * 16) + (i * 4)) + ((((int)threadIdx.x) >> 6) * 2)) + kh))) {
            condval_1 = input[(((((((((((int)blockIdx.x) * 409600) + (((int)blockIdx.y) * 10240)) + (i * 2560)) + ((((int)threadIdx.x) >> 6) * 1280)) + (kh * 640)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) & 63) * 2)) + kw) - 641)];
          } else {
            condval_1 = (std::numeric_limits<bfloat16_t>::infinity() * bfloat16_t(-0x1p+0f/*-1.000000e+00*/));
          }
          condval = condval_1;
        } else {
          condval = (std::numeric_limits<bfloat16_t>::infinity() * bfloat16_t(-0x1p+0f/*-1.000000e+00*/));
        }
        acc[i] = cutlass::bfloat16_t(__hmax((acc[i]).to_nv_bfloat16(), (condval).to_nv_bfloat16()));
      }
    }
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    output[((((((((int)blockIdx.x) * 102400) + (((int)blockIdx.y) * 2560)) + (i_1 * 640)) + ((((int)threadIdx.x) >> 6) * 320)) + (((int)blockIdx.z) * 64)) + (((int)threadIdx.x) & 63))] = acc[i_1];
  }
}

