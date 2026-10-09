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

extern "C" __global__ void matrix_copy_kernel_kernel(const float* __restrict__ A, float* __restrict__ B);
extern "C" __global__ void __launch_bounds__(256, 1) matrix_copy_kernel_kernel(const float* __restrict__ A, float* __restrict__ B) {
  *(float4*)(B + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4))) = *(float4*)(A + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)));
}

