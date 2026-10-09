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

extern "C" __global__ void fused_activation_kernel_kernel(const float* __restrict__ bias, const float* __restrict__ gate, float* __restrict__ output, const float* __restrict__ x);
extern "C" __global__ void __launch_bounds__(256, 1) fused_activation_kernel_kernel(const float* __restrict__ bias, const float* __restrict__ gate, float* __restrict__ output, const float* __restrict__ x) {
  float x_reg[8];
  float gate_reg[8];
  float bias_reg[8];
  float output_reg[8];
  *(ulonglong4*)(x_reg + 0) = tl::load_global_256(&(*(ulonglong4*)(x + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)))));
  *(ulonglong4*)(gate_reg + 0) = tl::load_global_256(&(*(ulonglong4*)(gate + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)))));
  *(ulonglong4*)(bias_reg + 0) = tl::load_global_256(&(*(ulonglong4*)(bias + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)))));
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float z = ((x_reg[i] * gate_reg[i]) + bias_reg[i]);
    output_reg[i] = (z * (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - z)))));
  }
  tl::store_global_256(&(*(ulonglong4*)(output + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)))), *(ulonglong4*)(output_reg + 0));
}

