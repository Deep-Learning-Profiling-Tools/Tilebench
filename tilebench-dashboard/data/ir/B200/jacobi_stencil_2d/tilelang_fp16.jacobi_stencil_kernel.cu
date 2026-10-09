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

extern "C" __global__ void jacobi_stencil_kernel_kernel(const half_t* __restrict__ input, half_t* __restrict__ output);
extern "C" __global__ void __launch_bounds__(256, 1) jacobi_stencil_kernel_kernel(const half_t* __restrict__ input, half_t* __restrict__ output) {
  half_t local_tile_out[4];
  *(uint2*)(local_tile_out + 0) = *(uint2*)(input + ((((((int)blockIdx.x) * 20480) + ((((int)threadIdx.x) >> 7) * 10240)) + (((int)blockIdx.y) * 512)) + ((((int)threadIdx.x) & 127) * 4)));
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    bool is_inner = ((((1 <= ((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 7))) && (1 <= (((((int)blockIdx.y) * 512) + ((((int)threadIdx.x) & 127) * 4)) + i))) && (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 7)) < 10239)) && ((((((int)blockIdx.y) * 512) + ((((int)threadIdx.x) & 127) * 4)) + i) < 10239));
    half_t condval;
    if ((1 <= ((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 7)))) {
      condval = input[((((((((int)blockIdx.x) * 20480) + ((((int)threadIdx.x) >> 7) * 10240)) + (((int)blockIdx.y) * 512)) + ((((int)threadIdx.x) & 127) * 4)) + i) - 10240)];
    } else {
      condval = half_t(0x0p+0f/*0.000000e+00*/);
    }
    half_t condval_1;
    if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 7)) < 10239)) {
      condval_1 = input[((((((((int)blockIdx.x) * 20480) + ((((int)threadIdx.x) >> 7) * 10240)) + (((int)blockIdx.y) * 512)) + ((((int)threadIdx.x) & 127) * 4)) + i) + 10240)];
    } else {
      condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
    }
    half_t condval_2;
    if ((1 <= (((((int)blockIdx.y) * 512) + ((((int)threadIdx.x) & 127) * 4)) + i))) {
      condval_2 = input[((((((((int)blockIdx.x) * 20480) + ((((int)threadIdx.x) >> 7) * 10240)) + (((int)blockIdx.y) * 512)) + ((((int)threadIdx.x) & 127) * 4)) + i) - 1)];
    } else {
      condval_2 = half_t(0x0p+0f/*0.000000e+00*/);
    }
    half_t condval_3;
    if (((((((int)blockIdx.y) * 512) + ((((int)threadIdx.x) & 127) * 4)) + i) < 10239)) {
      condval_3 = input[((((((((int)blockIdx.x) * 20480) + ((((int)threadIdx.x) >> 7) * 10240)) + (((int)blockIdx.y) * 512)) + ((((int)threadIdx.x) & 127) * 4)) + i) + 1)];
    } else {
      condval_3 = half_t(0x0p+0f/*0.000000e+00*/);
    }
    half_t avg = (half_t(0x1p-2f/*2.500000e-01*/) * (((condval + condval_1) + condval_2) + condval_3));
    local_tile_out[i] = (((((1 <= ((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 7))) && (1 <= (((((int)blockIdx.y) * 512) + ((((int)threadIdx.x) & 127) * 4)) + i))) && (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 7)) < 10239)) && ((((((int)blockIdx.y) * 512) + ((((int)threadIdx.x) & 127) * 4)) + i) < 10239)) ? avg : local_tile_out[i]);
  }
  *(uint2*)(output + ((((((int)blockIdx.x) * 20480) + ((((int)threadIdx.x) >> 7) * 10240)) + (((int)blockIdx.y) * 512)) + ((((int)threadIdx.x) & 127) * 4))) = *(uint2*)(local_tile_out + 0);
}

