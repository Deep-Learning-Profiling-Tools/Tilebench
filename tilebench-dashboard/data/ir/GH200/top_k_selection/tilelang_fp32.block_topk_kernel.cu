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

extern "C" __global__ void main_kernel(const float* __restrict__ input, float* __restrict__ output);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const float* __restrict__ input, float* __restrict__ output) {
  extern __shared__ __align__(1024) float workspace[];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(float4*)(workspace + ((i * 1024) + (((int)threadIdx.x) * 4))) = *(float4*)(input + (((((int)blockIdx.x) * 4096) + (i * 1024)) + (((int)threadIdx.x) * 4)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    float a = workspace[((i_1 * 512) + (((int)threadIdx.x) * 2))];
    float b = workspace[(((i_1 * 512) + (((int)threadIdx.x) * 2)) + 1)];
    float hi = ((b < a) ? a : b);
    float lo = ((b < a) ? b : a);
    bool left_desc = ((((i_1 * 512) + (((int)threadIdx.x) * 2)) & 2) == 0);
    workspace[((i_1 * 512) + (((int)threadIdx.x) * 2))] = (((((i_1 * 512) + (((int)threadIdx.x) * 2)) & 2) == 0) ? ((b < a) ? a : b) : ((b < a) ? b : a));
    workspace[(((i_1 * 512) + (((int)threadIdx.x) * 2)) + 1)] = (((((i_1 * 512) + (((int)threadIdx.x) * 2)) & 2) == 0) ? ((b < a) ? b : a) : ((b < a) ? a : b));
  }
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    float a_1 = workspace[(((i_2 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))];
    float b_1 = workspace[((((i_2 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)];
    float hi_1 = ((b_1 < a_1) ? a_1 : b_1);
    float lo_1 = ((b_1 < a_1) ? b_1 : a_1);
    bool left_desc_1 = (((((i_2 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 4) == 0);
    workspace[(((i_2 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))] = ((((((i_2 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 4) == 0) ? ((b_1 < a_1) ? a_1 : b_1) : ((b_1 < a_1) ? b_1 : a_1));
    workspace[((((i_2 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)] = ((((((i_2 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 4) == 0) ? ((b_1 < a_1) ? b_1 : a_1) : ((b_1 < a_1) ? a_1 : b_1));
  }
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    float a_2 = workspace[((i_3 * 512) + (((int)threadIdx.x) * 2))];
    float b_2 = workspace[(((i_3 * 512) + (((int)threadIdx.x) * 2)) + 1)];
    float hi_2 = ((b_2 < a_2) ? a_2 : b_2);
    float lo_2 = ((b_2 < a_2) ? b_2 : a_2);
    bool left_desc_2 = ((((i_3 * 512) + (((int)threadIdx.x) * 2)) & 4) == 0);
    workspace[((i_3 * 512) + (((int)threadIdx.x) * 2))] = (((((i_3 * 512) + (((int)threadIdx.x) * 2)) & 4) == 0) ? ((b_2 < a_2) ? a_2 : b_2) : ((b_2 < a_2) ? b_2 : a_2));
    workspace[(((i_3 * 512) + (((int)threadIdx.x) * 2)) + 1)] = (((((i_3 * 512) + (((int)threadIdx.x) * 2)) & 4) == 0) ? ((b_2 < a_2) ? b_2 : a_2) : ((b_2 < a_2) ? a_2 : b_2));
  }
  __syncthreads();
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    float a_3 = workspace[(((i_4 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))];
    float b_3 = workspace[((((i_4 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)];
    float hi_3 = ((b_3 < a_3) ? a_3 : b_3);
    float lo_3 = ((b_3 < a_3) ? b_3 : a_3);
    bool left_desc_3 = (((((i_4 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 8) == 0);
    workspace[(((i_4 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))] = ((((((i_4 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 8) == 0) ? ((b_3 < a_3) ? a_3 : b_3) : ((b_3 < a_3) ? b_3 : a_3));
    workspace[((((i_4 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)] = ((((((i_4 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 8) == 0) ? ((b_3 < a_3) ? b_3 : a_3) : ((b_3 < a_3) ? a_3 : b_3));
  }
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 8; ++i_5) {
    float a_4 = workspace[(((i_5 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))];
    float b_4 = workspace[((((i_5 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)];
    float hi_4 = ((b_4 < a_4) ? a_4 : b_4);
    float lo_4 = ((b_4 < a_4) ? b_4 : a_4);
    bool left_desc_4 = (((((i_5 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 8) == 0);
    workspace[(((i_5 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))] = ((((((i_5 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 8) == 0) ? ((b_4 < a_4) ? a_4 : b_4) : ((b_4 < a_4) ? b_4 : a_4));
    workspace[((((i_5 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)] = ((((((i_5 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 8) == 0) ? ((b_4 < a_4) ? b_4 : a_4) : ((b_4 < a_4) ? a_4 : b_4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_6 = 0; i_6 < 8; ++i_6) {
    float a_5 = workspace[((i_6 * 512) + (((int)threadIdx.x) * 2))];
    float b_5 = workspace[(((i_6 * 512) + (((int)threadIdx.x) * 2)) + 1)];
    float hi_5 = ((b_5 < a_5) ? a_5 : b_5);
    float lo_5 = ((b_5 < a_5) ? b_5 : a_5);
    bool left_desc_5 = ((((i_6 * 512) + (((int)threadIdx.x) * 2)) & 8) == 0);
    workspace[((i_6 * 512) + (((int)threadIdx.x) * 2))] = (((((i_6 * 512) + (((int)threadIdx.x) * 2)) & 8) == 0) ? ((b_5 < a_5) ? a_5 : b_5) : ((b_5 < a_5) ? b_5 : a_5));
    workspace[(((i_6 * 512) + (((int)threadIdx.x) * 2)) + 1)] = (((((i_6 * 512) + (((int)threadIdx.x) * 2)) & 8) == 0) ? ((b_5 < a_5) ? b_5 : a_5) : ((b_5 < a_5) ? a_5 : b_5));
  }
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 8; ++i_7) {
    float a_6 = workspace[(((i_7 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))];
    float b_6 = workspace[((((i_7 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)];
    float hi_6 = ((b_6 < a_6) ? a_6 : b_6);
    float lo_6 = ((b_6 < a_6) ? b_6 : a_6);
    bool left_desc_6 = (((((i_7 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 16) == 0);
    workspace[(((i_7 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))] = ((((((i_7 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 16) == 0) ? ((b_6 < a_6) ? a_6 : b_6) : ((b_6 < a_6) ? b_6 : a_6));
    workspace[((((i_7 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = ((((((i_7 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 16) == 0) ? ((b_6 < a_6) ? b_6 : a_6) : ((b_6 < a_6) ? a_6 : b_6));
  }
  __syncthreads();
  #pragma unroll
  for (int i_8 = 0; i_8 < 8; ++i_8) {
    float a_7 = workspace[(((i_8 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))];
    float b_7 = workspace[((((i_8 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)];
    float hi_7 = ((b_7 < a_7) ? a_7 : b_7);
    float lo_7 = ((b_7 < a_7) ? b_7 : a_7);
    bool left_desc_7 = (((((i_8 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 16) == 0);
    workspace[(((i_8 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))] = ((((((i_8 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 16) == 0) ? ((b_7 < a_7) ? a_7 : b_7) : ((b_7 < a_7) ? b_7 : a_7));
    workspace[((((i_8 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)] = ((((((i_8 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 16) == 0) ? ((b_7 < a_7) ? b_7 : a_7) : ((b_7 < a_7) ? a_7 : b_7));
  }
  __syncthreads();
  #pragma unroll
  for (int i_9 = 0; i_9 < 8; ++i_9) {
    float a_8 = workspace[(((i_9 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))];
    float b_8 = workspace[((((i_9 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)];
    float hi_8 = ((b_8 < a_8) ? a_8 : b_8);
    float lo_8 = ((b_8 < a_8) ? b_8 : a_8);
    bool left_desc_8 = (((((i_9 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 16) == 0);
    workspace[(((i_9 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))] = ((((((i_9 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 16) == 0) ? ((b_8 < a_8) ? a_8 : b_8) : ((b_8 < a_8) ? b_8 : a_8));
    workspace[((((i_9 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)] = ((((((i_9 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 16) == 0) ? ((b_8 < a_8) ? b_8 : a_8) : ((b_8 < a_8) ? a_8 : b_8));
  }
  __syncthreads();
  #pragma unroll
  for (int i_10 = 0; i_10 < 8; ++i_10) {
    float a_9 = workspace[((i_10 * 512) + (((int)threadIdx.x) * 2))];
    float b_9 = workspace[(((i_10 * 512) + (((int)threadIdx.x) * 2)) + 1)];
    float hi_9 = ((b_9 < a_9) ? a_9 : b_9);
    float lo_9 = ((b_9 < a_9) ? b_9 : a_9);
    bool left_desc_9 = ((((i_10 * 512) + (((int)threadIdx.x) * 2)) & 16) == 0);
    workspace[((i_10 * 512) + (((int)threadIdx.x) * 2))] = (((((i_10 * 512) + (((int)threadIdx.x) * 2)) & 16) == 0) ? ((b_9 < a_9) ? a_9 : b_9) : ((b_9 < a_9) ? b_9 : a_9));
    workspace[(((i_10 * 512) + (((int)threadIdx.x) * 2)) + 1)] = (((((i_10 * 512) + (((int)threadIdx.x) * 2)) & 16) == 0) ? ((b_9 < a_9) ? b_9 : a_9) : ((b_9 < a_9) ? a_9 : b_9));
  }
  __syncthreads();
  #pragma unroll
  for (int i_11 = 0; i_11 < 8; ++i_11) {
    float a_10 = workspace[(((i_11 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))];
    float b_10 = workspace[((((i_11 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)];
    float hi_10 = ((b_10 < a_10) ? a_10 : b_10);
    float lo_10 = ((b_10 < a_10) ? b_10 : a_10);
    bool left_desc_10 = (((((i_11 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 32) == 0);
    workspace[(((i_11 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))] = ((((((i_11 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 32) == 0) ? ((b_10 < a_10) ? a_10 : b_10) : ((b_10 < a_10) ? b_10 : a_10));
    workspace[((((i_11 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)] = ((((((i_11 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 32) == 0) ? ((b_10 < a_10) ? b_10 : a_10) : ((b_10 < a_10) ? a_10 : b_10));
  }
  __syncthreads();
  #pragma unroll
  for (int i_12 = 0; i_12 < 8; ++i_12) {
    float a_11 = workspace[(((i_12 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))];
    float b_11 = workspace[((((i_12 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)];
    float hi_11 = ((b_11 < a_11) ? a_11 : b_11);
    float lo_11 = ((b_11 < a_11) ? b_11 : a_11);
    bool left_desc_11 = (((((i_12 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 32) == 0);
    workspace[(((i_12 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))] = ((((((i_12 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 32) == 0) ? ((b_11 < a_11) ? a_11 : b_11) : ((b_11 < a_11) ? b_11 : a_11));
    workspace[((((i_12 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = ((((((i_12 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 32) == 0) ? ((b_11 < a_11) ? b_11 : a_11) : ((b_11 < a_11) ? a_11 : b_11));
  }
  __syncthreads();
  #pragma unroll
  for (int i_13 = 0; i_13 < 8; ++i_13) {
    float a_12 = workspace[(((i_13 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))];
    float b_12 = workspace[((((i_13 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)];
    float hi_12 = ((b_12 < a_12) ? a_12 : b_12);
    float lo_12 = ((b_12 < a_12) ? b_12 : a_12);
    bool left_desc_12 = (((((i_13 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 32) == 0);
    workspace[(((i_13 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))] = ((((((i_13 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 32) == 0) ? ((b_12 < a_12) ? a_12 : b_12) : ((b_12 < a_12) ? b_12 : a_12));
    workspace[((((i_13 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)] = ((((((i_13 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 32) == 0) ? ((b_12 < a_12) ? b_12 : a_12) : ((b_12 < a_12) ? a_12 : b_12));
  }
  __syncthreads();
  #pragma unroll
  for (int i_14 = 0; i_14 < 8; ++i_14) {
    float a_13 = workspace[(((i_14 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))];
    float b_13 = workspace[((((i_14 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)];
    float hi_13 = ((b_13 < a_13) ? a_13 : b_13);
    float lo_13 = ((b_13 < a_13) ? b_13 : a_13);
    bool left_desc_13 = (((((i_14 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 32) == 0);
    workspace[(((i_14 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))] = ((((((i_14 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 32) == 0) ? ((b_13 < a_13) ? a_13 : b_13) : ((b_13 < a_13) ? b_13 : a_13));
    workspace[((((i_14 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)] = ((((((i_14 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 32) == 0) ? ((b_13 < a_13) ? b_13 : a_13) : ((b_13 < a_13) ? a_13 : b_13));
  }
  __syncthreads();
  #pragma unroll
  for (int i_15 = 0; i_15 < 8; ++i_15) {
    float a_14 = workspace[((i_15 * 512) + (((int)threadIdx.x) * 2))];
    float b_14 = workspace[(((i_15 * 512) + (((int)threadIdx.x) * 2)) + 1)];
    float hi_14 = ((b_14 < a_14) ? a_14 : b_14);
    float lo_14 = ((b_14 < a_14) ? b_14 : a_14);
    bool left_desc_14 = ((((i_15 * 512) + (((int)threadIdx.x) * 2)) & 32) == 0);
    workspace[((i_15 * 512) + (((int)threadIdx.x) * 2))] = (((((i_15 * 512) + (((int)threadIdx.x) * 2)) & 32) == 0) ? ((b_14 < a_14) ? a_14 : b_14) : ((b_14 < a_14) ? b_14 : a_14));
    workspace[(((i_15 * 512) + (((int)threadIdx.x) * 2)) + 1)] = (((((i_15 * 512) + (((int)threadIdx.x) * 2)) & 32) == 0) ? ((b_14 < a_14) ? b_14 : a_14) : ((b_14 < a_14) ? a_14 : b_14));
  }
  __syncthreads();
  #pragma unroll
  for (int i_16 = 0; i_16 < 8; ++i_16) {
    float a_15 = workspace[(((i_16 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))];
    float b_15 = workspace[((((i_16 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)];
    float hi_15 = ((b_15 < a_15) ? a_15 : b_15);
    float lo_15 = ((b_15 < a_15) ? b_15 : a_15);
    bool left_desc_15 = (((((i_16 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 64) == 0);
    workspace[(((i_16 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))] = ((((((i_16 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 64) == 0) ? ((b_15 < a_15) ? a_15 : b_15) : ((b_15 < a_15) ? b_15 : a_15));
    workspace[((((i_16 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)] = ((((((i_16 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 64) == 0) ? ((b_15 < a_15) ? b_15 : a_15) : ((b_15 < a_15) ? a_15 : b_15));
  }
  __syncthreads();
  #pragma unroll
  for (int i_17 = 0; i_17 < 8; ++i_17) {
    float a_16 = workspace[(((i_17 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))];
    float b_16 = workspace[((((i_17 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)];
    float hi_16 = ((b_16 < a_16) ? a_16 : b_16);
    float lo_16 = ((b_16 < a_16) ? b_16 : a_16);
    bool left_desc_16 = (((((i_17 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 64) == 0);
    workspace[(((i_17 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))] = ((((((i_17 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 64) == 0) ? ((b_16 < a_16) ? a_16 : b_16) : ((b_16 < a_16) ? b_16 : a_16));
    workspace[((((i_17 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)] = ((((((i_17 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 64) == 0) ? ((b_16 < a_16) ? b_16 : a_16) : ((b_16 < a_16) ? a_16 : b_16));
  }
  __syncthreads();
  #pragma unroll
  for (int i_18 = 0; i_18 < 8; ++i_18) {
    float a_17 = workspace[(((i_18 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))];
    float b_17 = workspace[((((i_18 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)];
    float hi_17 = ((b_17 < a_17) ? a_17 : b_17);
    float lo_17 = ((b_17 < a_17) ? b_17 : a_17);
    bool left_desc_17 = (((((i_18 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 64) == 0);
    workspace[(((i_18 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))] = ((((((i_18 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 64) == 0) ? ((b_17 < a_17) ? a_17 : b_17) : ((b_17 < a_17) ? b_17 : a_17));
    workspace[((((i_18 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = ((((((i_18 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 64) == 0) ? ((b_17 < a_17) ? b_17 : a_17) : ((b_17 < a_17) ? a_17 : b_17));
  }
  __syncthreads();
  #pragma unroll
  for (int i_19 = 0; i_19 < 8; ++i_19) {
    float a_18 = workspace[(((i_19 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))];
    float b_18 = workspace[((((i_19 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)];
    float hi_18 = ((b_18 < a_18) ? a_18 : b_18);
    float lo_18 = ((b_18 < a_18) ? b_18 : a_18);
    bool left_desc_18 = (((((i_19 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 64) == 0);
    workspace[(((i_19 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))] = ((((((i_19 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 64) == 0) ? ((b_18 < a_18) ? a_18 : b_18) : ((b_18 < a_18) ? b_18 : a_18));
    workspace[((((i_19 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)] = ((((((i_19 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 64) == 0) ? ((b_18 < a_18) ? b_18 : a_18) : ((b_18 < a_18) ? a_18 : b_18));
  }
  __syncthreads();
  #pragma unroll
  for (int i_20 = 0; i_20 < 8; ++i_20) {
    float a_19 = workspace[(((i_20 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))];
    float b_19 = workspace[((((i_20 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)];
    float hi_19 = ((b_19 < a_19) ? a_19 : b_19);
    float lo_19 = ((b_19 < a_19) ? b_19 : a_19);
    bool left_desc_19 = (((((i_20 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 64) == 0);
    workspace[(((i_20 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))] = ((((((i_20 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 64) == 0) ? ((b_19 < a_19) ? a_19 : b_19) : ((b_19 < a_19) ? b_19 : a_19));
    workspace[((((i_20 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)] = ((((((i_20 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 64) == 0) ? ((b_19 < a_19) ? b_19 : a_19) : ((b_19 < a_19) ? a_19 : b_19));
  }
  __syncthreads();
  #pragma unroll
  for (int i_21 = 0; i_21 < 8; ++i_21) {
    float a_20 = workspace[((i_21 * 512) + (((int)threadIdx.x) * 2))];
    float b_20 = workspace[(((i_21 * 512) + (((int)threadIdx.x) * 2)) + 1)];
    float hi_20 = ((b_20 < a_20) ? a_20 : b_20);
    float lo_20 = ((b_20 < a_20) ? b_20 : a_20);
    bool left_desc_20 = ((((i_21 * 512) + (((int)threadIdx.x) * 2)) & 64) == 0);
    workspace[((i_21 * 512) + (((int)threadIdx.x) * 2))] = (((((i_21 * 512) + (((int)threadIdx.x) * 2)) & 64) == 0) ? ((b_20 < a_20) ? a_20 : b_20) : ((b_20 < a_20) ? b_20 : a_20));
    workspace[(((i_21 * 512) + (((int)threadIdx.x) * 2)) + 1)] = (((((i_21 * 512) + (((int)threadIdx.x) * 2)) & 64) == 0) ? ((b_20 < a_20) ? b_20 : a_20) : ((b_20 < a_20) ? a_20 : b_20));
  }
  __syncthreads();
  #pragma unroll
  for (int i_22 = 0; i_22 < 8; ++i_22) {
    float a_21 = workspace[(((i_22 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63))];
    float b_21 = workspace[((((i_22 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) + 64)];
    float hi_21 = ((b_21 < a_21) ? a_21 : b_21);
    float lo_21 = ((b_21 < a_21) ? b_21 : a_21);
    bool left_desc_21 = (((((i_22 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 128) == 0);
    workspace[(((i_22 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63))] = ((((((i_22 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 128) == 0) ? ((b_21 < a_21) ? a_21 : b_21) : ((b_21 < a_21) ? b_21 : a_21));
    workspace[((((i_22 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) + 64)] = ((((((i_22 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 128) == 0) ? ((b_21 < a_21) ? b_21 : a_21) : ((b_21 < a_21) ? a_21 : b_21));
  }
  __syncthreads();
  #pragma unroll
  for (int i_23 = 0; i_23 < 8; ++i_23) {
    float a_22 = workspace[(((i_23 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))];
    float b_22 = workspace[((((i_23 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)];
    float hi_22 = ((b_22 < a_22) ? a_22 : b_22);
    float lo_22 = ((b_22 < a_22) ? b_22 : a_22);
    bool left_desc_22 = (((((i_23 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 128) == 0);
    workspace[(((i_23 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))] = ((((((i_23 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 128) == 0) ? ((b_22 < a_22) ? a_22 : b_22) : ((b_22 < a_22) ? b_22 : a_22));
    workspace[((((i_23 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)] = ((((((i_23 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 128) == 0) ? ((b_22 < a_22) ? b_22 : a_22) : ((b_22 < a_22) ? a_22 : b_22));
  }
  __syncthreads();
  #pragma unroll
  for (int i_24 = 0; i_24 < 8; ++i_24) {
    float a_23 = workspace[(((i_24 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))];
    float b_23 = workspace[((((i_24 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)];
    float hi_23 = ((b_23 < a_23) ? a_23 : b_23);
    float lo_23 = ((b_23 < a_23) ? b_23 : a_23);
    bool left_desc_23 = (((((i_24 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 128) == 0);
    workspace[(((i_24 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))] = ((((((i_24 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 128) == 0) ? ((b_23 < a_23) ? a_23 : b_23) : ((b_23 < a_23) ? b_23 : a_23));
    workspace[((((i_24 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)] = ((((((i_24 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 128) == 0) ? ((b_23 < a_23) ? b_23 : a_23) : ((b_23 < a_23) ? a_23 : b_23));
  }
  __syncthreads();
  #pragma unroll
  for (int i_25 = 0; i_25 < 8; ++i_25) {
    float a_24 = workspace[(((i_25 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))];
    float b_24 = workspace[((((i_25 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)];
    float hi_24 = ((b_24 < a_24) ? a_24 : b_24);
    float lo_24 = ((b_24 < a_24) ? b_24 : a_24);
    bool left_desc_24 = (((((i_25 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 128) == 0);
    workspace[(((i_25 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))] = ((((((i_25 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 128) == 0) ? ((b_24 < a_24) ? a_24 : b_24) : ((b_24 < a_24) ? b_24 : a_24));
    workspace[((((i_25 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = ((((((i_25 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 128) == 0) ? ((b_24 < a_24) ? b_24 : a_24) : ((b_24 < a_24) ? a_24 : b_24));
  }
  __syncthreads();
  #pragma unroll
  for (int i_26 = 0; i_26 < 8; ++i_26) {
    float a_25 = workspace[(((i_26 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))];
    float b_25 = workspace[((((i_26 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)];
    float hi_25 = ((b_25 < a_25) ? a_25 : b_25);
    float lo_25 = ((b_25 < a_25) ? b_25 : a_25);
    bool left_desc_25 = (((((i_26 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 128) == 0);
    workspace[(((i_26 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))] = ((((((i_26 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 128) == 0) ? ((b_25 < a_25) ? a_25 : b_25) : ((b_25 < a_25) ? b_25 : a_25));
    workspace[((((i_26 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)] = ((((((i_26 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 128) == 0) ? ((b_25 < a_25) ? b_25 : a_25) : ((b_25 < a_25) ? a_25 : b_25));
  }
  __syncthreads();
  #pragma unroll
  for (int i_27 = 0; i_27 < 8; ++i_27) {
    float a_26 = workspace[(((i_27 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))];
    float b_26 = workspace[((((i_27 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)];
    float hi_26 = ((b_26 < a_26) ? a_26 : b_26);
    float lo_26 = ((b_26 < a_26) ? b_26 : a_26);
    bool left_desc_26 = (((((i_27 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 128) == 0);
    workspace[(((i_27 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))] = ((((((i_27 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 128) == 0) ? ((b_26 < a_26) ? a_26 : b_26) : ((b_26 < a_26) ? b_26 : a_26));
    workspace[((((i_27 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)] = ((((((i_27 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 128) == 0) ? ((b_26 < a_26) ? b_26 : a_26) : ((b_26 < a_26) ? a_26 : b_26));
  }
  __syncthreads();
  #pragma unroll
  for (int i_28 = 0; i_28 < 8; ++i_28) {
    float a_27 = workspace[((i_28 * 512) + (((int)threadIdx.x) * 2))];
    float b_27 = workspace[(((i_28 * 512) + (((int)threadIdx.x) * 2)) + 1)];
    float hi_27 = ((b_27 < a_27) ? a_27 : b_27);
    float lo_27 = ((b_27 < a_27) ? b_27 : a_27);
    bool left_desc_27 = ((((i_28 * 512) + (((int)threadIdx.x) * 2)) & 128) == 0);
    workspace[((i_28 * 512) + (((int)threadIdx.x) * 2))] = (((((i_28 * 512) + (((int)threadIdx.x) * 2)) & 128) == 0) ? ((b_27 < a_27) ? a_27 : b_27) : ((b_27 < a_27) ? b_27 : a_27));
    workspace[(((i_28 * 512) + (((int)threadIdx.x) * 2)) + 1)] = (((((i_28 * 512) + (((int)threadIdx.x) * 2)) & 128) == 0) ? ((b_27 < a_27) ? b_27 : a_27) : ((b_27 < a_27) ? a_27 : b_27));
  }
  __syncthreads();
  #pragma unroll
  for (int i_29 = 0; i_29 < 8; ++i_29) {
    float a_28 = workspace[(((i_29 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127))];
    float b_28 = workspace[((((i_29 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) + 128)];
    float hi_28 = ((b_28 < a_28) ? a_28 : b_28);
    float lo_28 = ((b_28 < a_28) ? b_28 : a_28);
    bool left_desc_28 = (((((i_29 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 256) == 0);
    workspace[(((i_29 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127))] = ((((((i_29 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 256) == 0) ? ((b_28 < a_28) ? a_28 : b_28) : ((b_28 < a_28) ? b_28 : a_28));
    workspace[((((i_29 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) + 128)] = ((((((i_29 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 256) == 0) ? ((b_28 < a_28) ? b_28 : a_28) : ((b_28 < a_28) ? a_28 : b_28));
  }
  __syncthreads();
  #pragma unroll
  for (int i_30 = 0; i_30 < 8; ++i_30) {
    float a_29 = workspace[(((i_30 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63))];
    float b_29 = workspace[((((i_30 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) + 64)];
    float hi_29 = ((b_29 < a_29) ? a_29 : b_29);
    float lo_29 = ((b_29 < a_29) ? b_29 : a_29);
    bool left_desc_29 = (((((i_30 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 256) == 0);
    workspace[(((i_30 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63))] = ((((((i_30 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 256) == 0) ? ((b_29 < a_29) ? a_29 : b_29) : ((b_29 < a_29) ? b_29 : a_29));
    workspace[((((i_30 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) + 64)] = ((((((i_30 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 256) == 0) ? ((b_29 < a_29) ? b_29 : a_29) : ((b_29 < a_29) ? a_29 : b_29));
  }
  __syncthreads();
  #pragma unroll
  for (int i_31 = 0; i_31 < 8; ++i_31) {
    float a_30 = workspace[(((i_31 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))];
    float b_30 = workspace[((((i_31 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)];
    float hi_30 = ((b_30 < a_30) ? a_30 : b_30);
    float lo_30 = ((b_30 < a_30) ? b_30 : a_30);
    bool left_desc_30 = (((((i_31 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 256) == 0);
    workspace[(((i_31 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))] = ((((((i_31 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 256) == 0) ? ((b_30 < a_30) ? a_30 : b_30) : ((b_30 < a_30) ? b_30 : a_30));
    workspace[((((i_31 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)] = ((((((i_31 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 256) == 0) ? ((b_30 < a_30) ? b_30 : a_30) : ((b_30 < a_30) ? a_30 : b_30));
  }
  __syncthreads();
  #pragma unroll
  for (int i_32 = 0; i_32 < 8; ++i_32) {
    float a_31 = workspace[(((i_32 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))];
    float b_31 = workspace[((((i_32 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)];
    float hi_31 = ((b_31 < a_31) ? a_31 : b_31);
    float lo_31 = ((b_31 < a_31) ? b_31 : a_31);
    bool left_desc_31 = (((((i_32 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 256) == 0);
    workspace[(((i_32 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))] = ((((((i_32 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 256) == 0) ? ((b_31 < a_31) ? a_31 : b_31) : ((b_31 < a_31) ? b_31 : a_31));
    workspace[((((i_32 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)] = ((((((i_32 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 256) == 0) ? ((b_31 < a_31) ? b_31 : a_31) : ((b_31 < a_31) ? a_31 : b_31));
  }
  __syncthreads();
  #pragma unroll
  for (int i_33 = 0; i_33 < 8; ++i_33) {
    float a_32 = workspace[(((i_33 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))];
    float b_32 = workspace[((((i_33 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)];
    float hi_32 = ((b_32 < a_32) ? a_32 : b_32);
    float lo_32 = ((b_32 < a_32) ? b_32 : a_32);
    bool left_desc_32 = (((((i_33 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 256) == 0);
    workspace[(((i_33 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))] = ((((((i_33 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 256) == 0) ? ((b_32 < a_32) ? a_32 : b_32) : ((b_32 < a_32) ? b_32 : a_32));
    workspace[((((i_33 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = ((((((i_33 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 256) == 0) ? ((b_32 < a_32) ? b_32 : a_32) : ((b_32 < a_32) ? a_32 : b_32));
  }
  __syncthreads();
  #pragma unroll
  for (int i_34 = 0; i_34 < 8; ++i_34) {
    float a_33 = workspace[(((i_34 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))];
    float b_33 = workspace[((((i_34 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)];
    float hi_33 = ((b_33 < a_33) ? a_33 : b_33);
    float lo_33 = ((b_33 < a_33) ? b_33 : a_33);
    bool left_desc_33 = (((((i_34 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 256) == 0);
    workspace[(((i_34 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))] = ((((((i_34 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 256) == 0) ? ((b_33 < a_33) ? a_33 : b_33) : ((b_33 < a_33) ? b_33 : a_33));
    workspace[((((i_34 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)] = ((((((i_34 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 256) == 0) ? ((b_33 < a_33) ? b_33 : a_33) : ((b_33 < a_33) ? a_33 : b_33));
  }
  __syncthreads();
  #pragma unroll
  for (int i_35 = 0; i_35 < 8; ++i_35) {
    float a_34 = workspace[(((i_35 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))];
    float b_34 = workspace[((((i_35 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)];
    float hi_34 = ((b_34 < a_34) ? a_34 : b_34);
    float lo_34 = ((b_34 < a_34) ? b_34 : a_34);
    bool left_desc_34 = (((((i_35 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 256) == 0);
    workspace[(((i_35 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))] = ((((((i_35 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 256) == 0) ? ((b_34 < a_34) ? a_34 : b_34) : ((b_34 < a_34) ? b_34 : a_34));
    workspace[((((i_35 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)] = ((((((i_35 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 256) == 0) ? ((b_34 < a_34) ? b_34 : a_34) : ((b_34 < a_34) ? a_34 : b_34));
  }
  __syncthreads();
  #pragma unroll
  for (int i_36 = 0; i_36 < 8; ++i_36) {
    float a_35 = workspace[((i_36 * 512) + (((int)threadIdx.x) * 2))];
    float b_35 = workspace[(((i_36 * 512) + (((int)threadIdx.x) * 2)) + 1)];
    float hi_35 = ((b_35 < a_35) ? a_35 : b_35);
    float lo_35 = ((b_35 < a_35) ? b_35 : a_35);
    bool left_desc_35 = ((((i_36 * 512) + (((int)threadIdx.x) * 2)) & 256) == 0);
    workspace[((i_36 * 512) + (((int)threadIdx.x) * 2))] = (((((i_36 * 512) + (((int)threadIdx.x) * 2)) & 256) == 0) ? ((b_35 < a_35) ? a_35 : b_35) : ((b_35 < a_35) ? b_35 : a_35));
    workspace[(((i_36 * 512) + (((int)threadIdx.x) * 2)) + 1)] = (((((i_36 * 512) + (((int)threadIdx.x) * 2)) & 256) == 0) ? ((b_35 < a_35) ? b_35 : a_35) : ((b_35 < a_35) ? a_35 : b_35));
  }
  __syncthreads();
  #pragma unroll
  for (int i_37 = 0; i_37 < 8; ++i_37) {
    float a_36 = workspace[((i_37 * 512) + ((int)threadIdx.x))];
    float b_36 = workspace[(((i_37 * 512) + ((int)threadIdx.x)) + 256)];
    float hi_36 = ((b_36 < a_36) ? a_36 : b_36);
    float lo_36 = ((b_36 < a_36) ? b_36 : a_36);
    bool left_desc_36 = ((((i_37 * 512) + ((int)threadIdx.x)) & 512) == 0);
    workspace[((i_37 * 512) + ((int)threadIdx.x))] = (((((i_37 * 512) + ((int)threadIdx.x)) & 512) == 0) ? ((b_36 < a_36) ? a_36 : b_36) : ((b_36 < a_36) ? b_36 : a_36));
    workspace[(((i_37 * 512) + ((int)threadIdx.x)) + 256)] = (((((i_37 * 512) + ((int)threadIdx.x)) & 512) == 0) ? ((b_36 < a_36) ? b_36 : a_36) : ((b_36 < a_36) ? a_36 : b_36));
  }
  __syncthreads();
  #pragma unroll
  for (int i_38 = 0; i_38 < 8; ++i_38) {
    float a_37 = workspace[(((i_38 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127))];
    float b_37 = workspace[((((i_38 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) + 128)];
    float hi_37 = ((b_37 < a_37) ? a_37 : b_37);
    float lo_37 = ((b_37 < a_37) ? b_37 : a_37);
    bool left_desc_37 = (((((i_38 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 512) == 0);
    workspace[(((i_38 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127))] = ((((((i_38 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 512) == 0) ? ((b_37 < a_37) ? a_37 : b_37) : ((b_37 < a_37) ? b_37 : a_37));
    workspace[((((i_38 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) + 128)] = ((((((i_38 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 512) == 0) ? ((b_37 < a_37) ? b_37 : a_37) : ((b_37 < a_37) ? a_37 : b_37));
  }
  __syncthreads();
  #pragma unroll
  for (int i_39 = 0; i_39 < 8; ++i_39) {
    float a_38 = workspace[(((i_39 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63))];
    float b_38 = workspace[((((i_39 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) + 64)];
    float hi_38 = ((b_38 < a_38) ? a_38 : b_38);
    float lo_38 = ((b_38 < a_38) ? b_38 : a_38);
    bool left_desc_38 = (((((i_39 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 512) == 0);
    workspace[(((i_39 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63))] = ((((((i_39 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 512) == 0) ? ((b_38 < a_38) ? a_38 : b_38) : ((b_38 < a_38) ? b_38 : a_38));
    workspace[((((i_39 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) + 64)] = ((((((i_39 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 512) == 0) ? ((b_38 < a_38) ? b_38 : a_38) : ((b_38 < a_38) ? a_38 : b_38));
  }
  __syncthreads();
  #pragma unroll
  for (int i_40 = 0; i_40 < 8; ++i_40) {
    float a_39 = workspace[(((i_40 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))];
    float b_39 = workspace[((((i_40 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)];
    float hi_39 = ((b_39 < a_39) ? a_39 : b_39);
    float lo_39 = ((b_39 < a_39) ? b_39 : a_39);
    bool left_desc_39 = (((((i_40 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 512) == 0);
    workspace[(((i_40 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))] = ((((((i_40 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 512) == 0) ? ((b_39 < a_39) ? a_39 : b_39) : ((b_39 < a_39) ? b_39 : a_39));
    workspace[((((i_40 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)] = ((((((i_40 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 512) == 0) ? ((b_39 < a_39) ? b_39 : a_39) : ((b_39 < a_39) ? a_39 : b_39));
  }
  __syncthreads();
  #pragma unroll
  for (int i_41 = 0; i_41 < 8; ++i_41) {
    float a_40 = workspace[(((i_41 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))];
    float b_40 = workspace[((((i_41 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)];
    float hi_40 = ((b_40 < a_40) ? a_40 : b_40);
    float lo_40 = ((b_40 < a_40) ? b_40 : a_40);
    bool left_desc_40 = (((((i_41 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 512) == 0);
    workspace[(((i_41 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))] = ((((((i_41 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 512) == 0) ? ((b_40 < a_40) ? a_40 : b_40) : ((b_40 < a_40) ? b_40 : a_40));
    workspace[((((i_41 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)] = ((((((i_41 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 512) == 0) ? ((b_40 < a_40) ? b_40 : a_40) : ((b_40 < a_40) ? a_40 : b_40));
  }
  __syncthreads();
  #pragma unroll
  for (int i_42 = 0; i_42 < 8; ++i_42) {
    float a_41 = workspace[(((i_42 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))];
    float b_41 = workspace[((((i_42 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)];
    float hi_41 = ((b_41 < a_41) ? a_41 : b_41);
    float lo_41 = ((b_41 < a_41) ? b_41 : a_41);
    bool left_desc_41 = (((((i_42 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 512) == 0);
    workspace[(((i_42 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))] = ((((((i_42 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 512) == 0) ? ((b_41 < a_41) ? a_41 : b_41) : ((b_41 < a_41) ? b_41 : a_41));
    workspace[((((i_42 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = ((((((i_42 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 512) == 0) ? ((b_41 < a_41) ? b_41 : a_41) : ((b_41 < a_41) ? a_41 : b_41));
  }
  __syncthreads();
  #pragma unroll
  for (int i_43 = 0; i_43 < 8; ++i_43) {
    float a_42 = workspace[(((i_43 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))];
    float b_42 = workspace[((((i_43 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)];
    float hi_42 = ((b_42 < a_42) ? a_42 : b_42);
    float lo_42 = ((b_42 < a_42) ? b_42 : a_42);
    bool left_desc_42 = (((((i_43 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 512) == 0);
    workspace[(((i_43 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))] = ((((((i_43 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 512) == 0) ? ((b_42 < a_42) ? a_42 : b_42) : ((b_42 < a_42) ? b_42 : a_42));
    workspace[((((i_43 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)] = ((((((i_43 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 512) == 0) ? ((b_42 < a_42) ? b_42 : a_42) : ((b_42 < a_42) ? a_42 : b_42));
  }
  __syncthreads();
  #pragma unroll
  for (int i_44 = 0; i_44 < 8; ++i_44) {
    float a_43 = workspace[(((i_44 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))];
    float b_43 = workspace[((((i_44 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)];
    float hi_43 = ((b_43 < a_43) ? a_43 : b_43);
    float lo_43 = ((b_43 < a_43) ? b_43 : a_43);
    bool left_desc_43 = (((((i_44 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 512) == 0);
    workspace[(((i_44 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))] = ((((((i_44 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 512) == 0) ? ((b_43 < a_43) ? a_43 : b_43) : ((b_43 < a_43) ? b_43 : a_43));
    workspace[((((i_44 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)] = ((((((i_44 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 512) == 0) ? ((b_43 < a_43) ? b_43 : a_43) : ((b_43 < a_43) ? a_43 : b_43));
  }
  __syncthreads();
  #pragma unroll
  for (int i_45 = 0; i_45 < 8; ++i_45) {
    float a_44 = workspace[((i_45 * 512) + (((int)threadIdx.x) * 2))];
    float b_44 = workspace[(((i_45 * 512) + (((int)threadIdx.x) * 2)) + 1)];
    float hi_44 = ((b_44 < a_44) ? a_44 : b_44);
    float lo_44 = ((b_44 < a_44) ? b_44 : a_44);
    bool left_desc_44 = ((((i_45 * 512) + (((int)threadIdx.x) * 2)) & 512) == 0);
    workspace[((i_45 * 512) + (((int)threadIdx.x) * 2))] = (((((i_45 * 512) + (((int)threadIdx.x) * 2)) & 512) == 0) ? ((b_44 < a_44) ? a_44 : b_44) : ((b_44 < a_44) ? b_44 : a_44));
    workspace[(((i_45 * 512) + (((int)threadIdx.x) * 2)) + 1)] = (((((i_45 * 512) + (((int)threadIdx.x) * 2)) & 512) == 0) ? ((b_44 < a_44) ? b_44 : a_44) : ((b_44 < a_44) ? a_44 : b_44));
  }
  __syncthreads();
  #pragma unroll
  for (int i_46 = 0; i_46 < 8; ++i_46) {
    float a_45 = workspace[((((i_46 >> 1) * 1024) + ((i_46 & 1) * 256)) + ((int)threadIdx.x))];
    float b_45 = workspace[(((((i_46 >> 1) * 1024) + ((i_46 & 1) * 256)) + ((int)threadIdx.x)) + 512)];
    float hi_45 = ((b_45 < a_45) ? a_45 : b_45);
    float lo_45 = ((b_45 < a_45) ? b_45 : a_45);
    bool left_desc_45 = ((((((i_46 >> 1) * 1024) + ((i_46 & 1) * 256)) + ((int)threadIdx.x)) & 1024) == 0);
    workspace[((((i_46 >> 1) * 1024) + ((i_46 & 1) * 256)) + ((int)threadIdx.x))] = (((((((i_46 >> 1) * 1024) + ((i_46 & 1) * 256)) + ((int)threadIdx.x)) & 1024) == 0) ? ((b_45 < a_45) ? a_45 : b_45) : ((b_45 < a_45) ? b_45 : a_45));
    workspace[(((((i_46 >> 1) * 1024) + ((i_46 & 1) * 256)) + ((int)threadIdx.x)) + 512)] = (((((((i_46 >> 1) * 1024) + ((i_46 & 1) * 256)) + ((int)threadIdx.x)) & 1024) == 0) ? ((b_45 < a_45) ? b_45 : a_45) : ((b_45 < a_45) ? a_45 : b_45));
  }
  __syncthreads();
  #pragma unroll
  for (int i_47 = 0; i_47 < 8; ++i_47) {
    float a_46 = workspace[((i_47 * 512) + ((int)threadIdx.x))];
    float b_46 = workspace[(((i_47 * 512) + ((int)threadIdx.x)) + 256)];
    float hi_46 = ((b_46 < a_46) ? a_46 : b_46);
    float lo_46 = ((b_46 < a_46) ? b_46 : a_46);
    bool left_desc_46 = ((((i_47 * 512) + ((int)threadIdx.x)) & 1024) == 0);
    workspace[((i_47 * 512) + ((int)threadIdx.x))] = (((((i_47 * 512) + ((int)threadIdx.x)) & 1024) == 0) ? ((b_46 < a_46) ? a_46 : b_46) : ((b_46 < a_46) ? b_46 : a_46));
    workspace[(((i_47 * 512) + ((int)threadIdx.x)) + 256)] = (((((i_47 * 512) + ((int)threadIdx.x)) & 1024) == 0) ? ((b_46 < a_46) ? b_46 : a_46) : ((b_46 < a_46) ? a_46 : b_46));
  }
  __syncthreads();
  #pragma unroll
  for (int i_48 = 0; i_48 < 8; ++i_48) {
    float a_47 = workspace[(((i_48 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127))];
    float b_47 = workspace[((((i_48 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) + 128)];
    float hi_47 = ((b_47 < a_47) ? a_47 : b_47);
    float lo_47 = ((b_47 < a_47) ? b_47 : a_47);
    bool left_desc_47 = (((((i_48 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 1024) == 0);
    workspace[(((i_48 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127))] = ((((((i_48 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 1024) == 0) ? ((b_47 < a_47) ? a_47 : b_47) : ((b_47 < a_47) ? b_47 : a_47));
    workspace[((((i_48 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) + 128)] = ((((((i_48 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 1024) == 0) ? ((b_47 < a_47) ? b_47 : a_47) : ((b_47 < a_47) ? a_47 : b_47));
  }
  __syncthreads();
  #pragma unroll
  for (int i_49 = 0; i_49 < 8; ++i_49) {
    float a_48 = workspace[(((i_49 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63))];
    float b_48 = workspace[((((i_49 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) + 64)];
    float hi_48 = ((b_48 < a_48) ? a_48 : b_48);
    float lo_48 = ((b_48 < a_48) ? b_48 : a_48);
    bool left_desc_48 = (((((i_49 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 1024) == 0);
    workspace[(((i_49 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63))] = ((((((i_49 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 1024) == 0) ? ((b_48 < a_48) ? a_48 : b_48) : ((b_48 < a_48) ? b_48 : a_48));
    workspace[((((i_49 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) + 64)] = ((((((i_49 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 1024) == 0) ? ((b_48 < a_48) ? b_48 : a_48) : ((b_48 < a_48) ? a_48 : b_48));
  }
  __syncthreads();
  #pragma unroll
  for (int i_50 = 0; i_50 < 8; ++i_50) {
    float a_49 = workspace[(((i_50 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))];
    float b_49 = workspace[((((i_50 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)];
    float hi_49 = ((b_49 < a_49) ? a_49 : b_49);
    float lo_49 = ((b_49 < a_49) ? b_49 : a_49);
    bool left_desc_49 = (((((i_50 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 1024) == 0);
    workspace[(((i_50 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))] = ((((((i_50 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 1024) == 0) ? ((b_49 < a_49) ? a_49 : b_49) : ((b_49 < a_49) ? b_49 : a_49));
    workspace[((((i_50 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)] = ((((((i_50 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 1024) == 0) ? ((b_49 < a_49) ? b_49 : a_49) : ((b_49 < a_49) ? a_49 : b_49));
  }
  __syncthreads();
  #pragma unroll
  for (int i_51 = 0; i_51 < 8; ++i_51) {
    float a_50 = workspace[(((i_51 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))];
    float b_50 = workspace[((((i_51 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)];
    float hi_50 = ((b_50 < a_50) ? a_50 : b_50);
    float lo_50 = ((b_50 < a_50) ? b_50 : a_50);
    bool left_desc_50 = (((((i_51 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 1024) == 0);
    workspace[(((i_51 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))] = ((((((i_51 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 1024) == 0) ? ((b_50 < a_50) ? a_50 : b_50) : ((b_50 < a_50) ? b_50 : a_50));
    workspace[((((i_51 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)] = ((((((i_51 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 1024) == 0) ? ((b_50 < a_50) ? b_50 : a_50) : ((b_50 < a_50) ? a_50 : b_50));
  }
  __syncthreads();
  #pragma unroll
  for (int i_52 = 0; i_52 < 8; ++i_52) {
    float a_51 = workspace[(((i_52 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))];
    float b_51 = workspace[((((i_52 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)];
    float hi_51 = ((b_51 < a_51) ? a_51 : b_51);
    float lo_51 = ((b_51 < a_51) ? b_51 : a_51);
    bool left_desc_51 = (((((i_52 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 1024) == 0);
    workspace[(((i_52 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))] = ((((((i_52 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 1024) == 0) ? ((b_51 < a_51) ? a_51 : b_51) : ((b_51 < a_51) ? b_51 : a_51));
    workspace[((((i_52 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = ((((((i_52 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 1024) == 0) ? ((b_51 < a_51) ? b_51 : a_51) : ((b_51 < a_51) ? a_51 : b_51));
  }
  __syncthreads();
  #pragma unroll
  for (int i_53 = 0; i_53 < 8; ++i_53) {
    float a_52 = workspace[(((i_53 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))];
    float b_52 = workspace[((((i_53 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)];
    float hi_52 = ((b_52 < a_52) ? a_52 : b_52);
    float lo_52 = ((b_52 < a_52) ? b_52 : a_52);
    bool left_desc_52 = (((((i_53 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 1024) == 0);
    workspace[(((i_53 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))] = ((((((i_53 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 1024) == 0) ? ((b_52 < a_52) ? a_52 : b_52) : ((b_52 < a_52) ? b_52 : a_52));
    workspace[((((i_53 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)] = ((((((i_53 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 1024) == 0) ? ((b_52 < a_52) ? b_52 : a_52) : ((b_52 < a_52) ? a_52 : b_52));
  }
  __syncthreads();
  #pragma unroll
  for (int i_54 = 0; i_54 < 8; ++i_54) {
    float a_53 = workspace[(((i_54 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))];
    float b_53 = workspace[((((i_54 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)];
    float hi_53 = ((b_53 < a_53) ? a_53 : b_53);
    float lo_53 = ((b_53 < a_53) ? b_53 : a_53);
    bool left_desc_53 = (((((i_54 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 1024) == 0);
    workspace[(((i_54 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))] = ((((((i_54 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 1024) == 0) ? ((b_53 < a_53) ? a_53 : b_53) : ((b_53 < a_53) ? b_53 : a_53));
    workspace[((((i_54 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)] = ((((((i_54 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 1024) == 0) ? ((b_53 < a_53) ? b_53 : a_53) : ((b_53 < a_53) ? a_53 : b_53));
  }
  __syncthreads();
  #pragma unroll
  for (int i_55 = 0; i_55 < 8; ++i_55) {
    float a_54 = workspace[((i_55 * 512) + (((int)threadIdx.x) * 2))];
    float b_54 = workspace[(((i_55 * 512) + (((int)threadIdx.x) * 2)) + 1)];
    float hi_54 = ((b_54 < a_54) ? a_54 : b_54);
    float lo_54 = ((b_54 < a_54) ? b_54 : a_54);
    bool left_desc_54 = ((((i_55 * 512) + (((int)threadIdx.x) * 2)) & 1024) == 0);
    workspace[((i_55 * 512) + (((int)threadIdx.x) * 2))] = (((((i_55 * 512) + (((int)threadIdx.x) * 2)) & 1024) == 0) ? ((b_54 < a_54) ? a_54 : b_54) : ((b_54 < a_54) ? b_54 : a_54));
    workspace[(((i_55 * 512) + (((int)threadIdx.x) * 2)) + 1)] = (((((i_55 * 512) + (((int)threadIdx.x) * 2)) & 1024) == 0) ? ((b_54 < a_54) ? b_54 : a_54) : ((b_54 < a_54) ? a_54 : b_54));
  }
  __syncthreads();
  #pragma unroll
  for (int i_56 = 0; i_56 < 8; ++i_56) {
    float a_55 = workspace[((((i_56 >> 2) * 2048) + ((i_56 & 3) * 256)) + ((int)threadIdx.x))];
    float b_55 = workspace[(((((i_56 >> 2) * 2048) + ((i_56 & 3) * 256)) + ((int)threadIdx.x)) + 1024)];
    float hi_55 = ((b_55 < a_55) ? a_55 : b_55);
    float lo_55 = ((b_55 < a_55) ? b_55 : a_55);
    bool left_desc_55 = ((((((i_56 >> 2) * 2048) + ((i_56 & 3) * 256)) + ((int)threadIdx.x)) & 2048) == 0);
    workspace[((((i_56 >> 2) * 2048) + ((i_56 & 3) * 256)) + ((int)threadIdx.x))] = (((((((i_56 >> 2) * 2048) + ((i_56 & 3) * 256)) + ((int)threadIdx.x)) & 2048) == 0) ? ((b_55 < a_55) ? a_55 : b_55) : ((b_55 < a_55) ? b_55 : a_55));
    workspace[(((((i_56 >> 2) * 2048) + ((i_56 & 3) * 256)) + ((int)threadIdx.x)) + 1024)] = (((((((i_56 >> 2) * 2048) + ((i_56 & 3) * 256)) + ((int)threadIdx.x)) & 2048) == 0) ? ((b_55 < a_55) ? b_55 : a_55) : ((b_55 < a_55) ? a_55 : b_55));
  }
  __syncthreads();
  #pragma unroll
  for (int i_57 = 0; i_57 < 8; ++i_57) {
    float a_56 = workspace[((((i_57 >> 1) * 1024) + ((i_57 & 1) * 256)) + ((int)threadIdx.x))];
    float b_56 = workspace[(((((i_57 >> 1) * 1024) + ((i_57 & 1) * 256)) + ((int)threadIdx.x)) + 512)];
    float hi_56 = ((b_56 < a_56) ? a_56 : b_56);
    float lo_56 = ((b_56 < a_56) ? b_56 : a_56);
    bool left_desc_56 = ((((((i_57 >> 1) * 1024) + ((i_57 & 1) * 256)) + ((int)threadIdx.x)) & 2048) == 0);
    workspace[((((i_57 >> 1) * 1024) + ((i_57 & 1) * 256)) + ((int)threadIdx.x))] = (((((((i_57 >> 1) * 1024) + ((i_57 & 1) * 256)) + ((int)threadIdx.x)) & 2048) == 0) ? ((b_56 < a_56) ? a_56 : b_56) : ((b_56 < a_56) ? b_56 : a_56));
    workspace[(((((i_57 >> 1) * 1024) + ((i_57 & 1) * 256)) + ((int)threadIdx.x)) + 512)] = (((((((i_57 >> 1) * 1024) + ((i_57 & 1) * 256)) + ((int)threadIdx.x)) & 2048) == 0) ? ((b_56 < a_56) ? b_56 : a_56) : ((b_56 < a_56) ? a_56 : b_56));
  }
  __syncthreads();
  #pragma unroll
  for (int i_58 = 0; i_58 < 8; ++i_58) {
    float a_57 = workspace[((i_58 * 512) + ((int)threadIdx.x))];
    float b_57 = workspace[(((i_58 * 512) + ((int)threadIdx.x)) + 256)];
    float hi_57 = ((b_57 < a_57) ? a_57 : b_57);
    float lo_57 = ((b_57 < a_57) ? b_57 : a_57);
    bool left_desc_57 = ((((i_58 * 512) + ((int)threadIdx.x)) & 2048) == 0);
    workspace[((i_58 * 512) + ((int)threadIdx.x))] = (((((i_58 * 512) + ((int)threadIdx.x)) & 2048) == 0) ? ((b_57 < a_57) ? a_57 : b_57) : ((b_57 < a_57) ? b_57 : a_57));
    workspace[(((i_58 * 512) + ((int)threadIdx.x)) + 256)] = (((((i_58 * 512) + ((int)threadIdx.x)) & 2048) == 0) ? ((b_57 < a_57) ? b_57 : a_57) : ((b_57 < a_57) ? a_57 : b_57));
  }
  __syncthreads();
  #pragma unroll
  for (int i_59 = 0; i_59 < 8; ++i_59) {
    float a_58 = workspace[(((i_59 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127))];
    float b_58 = workspace[((((i_59 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) + 128)];
    float hi_58 = ((b_58 < a_58) ? a_58 : b_58);
    float lo_58 = ((b_58 < a_58) ? b_58 : a_58);
    bool left_desc_58 = (((((i_59 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 2048) == 0);
    workspace[(((i_59 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127))] = ((((((i_59 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 2048) == 0) ? ((b_58 < a_58) ? a_58 : b_58) : ((b_58 < a_58) ? b_58 : a_58));
    workspace[((((i_59 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) + 128)] = ((((((i_59 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 2048) == 0) ? ((b_58 < a_58) ? b_58 : a_58) : ((b_58 < a_58) ? a_58 : b_58));
  }
  __syncthreads();
  #pragma unroll
  for (int i_60 = 0; i_60 < 8; ++i_60) {
    float a_59 = workspace[(((i_60 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63))];
    float b_59 = workspace[((((i_60 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) + 64)];
    float hi_59 = ((b_59 < a_59) ? a_59 : b_59);
    float lo_59 = ((b_59 < a_59) ? b_59 : a_59);
    bool left_desc_59 = (((((i_60 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 2048) == 0);
    workspace[(((i_60 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63))] = ((((((i_60 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 2048) == 0) ? ((b_59 < a_59) ? a_59 : b_59) : ((b_59 < a_59) ? b_59 : a_59));
    workspace[((((i_60 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) + 64)] = ((((((i_60 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 2048) == 0) ? ((b_59 < a_59) ? b_59 : a_59) : ((b_59 < a_59) ? a_59 : b_59));
  }
  __syncthreads();
  #pragma unroll
  for (int i_61 = 0; i_61 < 8; ++i_61) {
    float a_60 = workspace[(((i_61 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))];
    float b_60 = workspace[((((i_61 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)];
    float hi_60 = ((b_60 < a_60) ? a_60 : b_60);
    float lo_60 = ((b_60 < a_60) ? b_60 : a_60);
    bool left_desc_60 = (((((i_61 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 2048) == 0);
    workspace[(((i_61 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))] = ((((((i_61 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 2048) == 0) ? ((b_60 < a_60) ? a_60 : b_60) : ((b_60 < a_60) ? b_60 : a_60));
    workspace[((((i_61 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)] = ((((((i_61 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 2048) == 0) ? ((b_60 < a_60) ? b_60 : a_60) : ((b_60 < a_60) ? a_60 : b_60));
  }
  __syncthreads();
  #pragma unroll
  for (int i_62 = 0; i_62 < 8; ++i_62) {
    float a_61 = workspace[(((i_62 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))];
    float b_61 = workspace[((((i_62 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)];
    float hi_61 = ((b_61 < a_61) ? a_61 : b_61);
    float lo_61 = ((b_61 < a_61) ? b_61 : a_61);
    bool left_desc_61 = (((((i_62 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 2048) == 0);
    workspace[(((i_62 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))] = ((((((i_62 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 2048) == 0) ? ((b_61 < a_61) ? a_61 : b_61) : ((b_61 < a_61) ? b_61 : a_61));
    workspace[((((i_62 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)] = ((((((i_62 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 2048) == 0) ? ((b_61 < a_61) ? b_61 : a_61) : ((b_61 < a_61) ? a_61 : b_61));
  }
  __syncthreads();
  #pragma unroll
  for (int i_63 = 0; i_63 < 8; ++i_63) {
    float a_62 = workspace[(((i_63 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))];
    float b_62 = workspace[((((i_63 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)];
    float hi_62 = ((b_62 < a_62) ? a_62 : b_62);
    float lo_62 = ((b_62 < a_62) ? b_62 : a_62);
    bool left_desc_62 = (((((i_63 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 2048) == 0);
    workspace[(((i_63 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))] = ((((((i_63 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 2048) == 0) ? ((b_62 < a_62) ? a_62 : b_62) : ((b_62 < a_62) ? b_62 : a_62));
    workspace[((((i_63 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = ((((((i_63 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 2048) == 0) ? ((b_62 < a_62) ? b_62 : a_62) : ((b_62 < a_62) ? a_62 : b_62));
  }
  __syncthreads();
  #pragma unroll
  for (int i_64 = 0; i_64 < 8; ++i_64) {
    float a_63 = workspace[(((i_64 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))];
    float b_63 = workspace[((((i_64 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)];
    float hi_63 = ((b_63 < a_63) ? a_63 : b_63);
    float lo_63 = ((b_63 < a_63) ? b_63 : a_63);
    bool left_desc_63 = (((((i_64 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 2048) == 0);
    workspace[(((i_64 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))] = ((((((i_64 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 2048) == 0) ? ((b_63 < a_63) ? a_63 : b_63) : ((b_63 < a_63) ? b_63 : a_63));
    workspace[((((i_64 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)] = ((((((i_64 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 2048) == 0) ? ((b_63 < a_63) ? b_63 : a_63) : ((b_63 < a_63) ? a_63 : b_63));
  }
  __syncthreads();
  #pragma unroll
  for (int i_65 = 0; i_65 < 8; ++i_65) {
    float a_64 = workspace[(((i_65 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))];
    float b_64 = workspace[((((i_65 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)];
    float hi_64 = ((b_64 < a_64) ? a_64 : b_64);
    float lo_64 = ((b_64 < a_64) ? b_64 : a_64);
    bool left_desc_64 = (((((i_65 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 2048) == 0);
    workspace[(((i_65 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))] = ((((((i_65 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 2048) == 0) ? ((b_64 < a_64) ? a_64 : b_64) : ((b_64 < a_64) ? b_64 : a_64));
    workspace[((((i_65 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)] = ((((((i_65 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 2048) == 0) ? ((b_64 < a_64) ? b_64 : a_64) : ((b_64 < a_64) ? a_64 : b_64));
  }
  __syncthreads();
  #pragma unroll
  for (int i_66 = 0; i_66 < 8; ++i_66) {
    float a_65 = workspace[((i_66 * 512) + (((int)threadIdx.x) * 2))];
    float b_65 = workspace[(((i_66 * 512) + (((int)threadIdx.x) * 2)) + 1)];
    float hi_65 = ((b_65 < a_65) ? a_65 : b_65);
    float lo_65 = ((b_65 < a_65) ? b_65 : a_65);
    bool left_desc_65 = ((((i_66 * 512) + (((int)threadIdx.x) * 2)) & 2048) == 0);
    workspace[((i_66 * 512) + (((int)threadIdx.x) * 2))] = (((((i_66 * 512) + (((int)threadIdx.x) * 2)) & 2048) == 0) ? ((b_65 < a_65) ? a_65 : b_65) : ((b_65 < a_65) ? b_65 : a_65));
    workspace[(((i_66 * 512) + (((int)threadIdx.x) * 2)) + 1)] = (((((i_66 * 512) + (((int)threadIdx.x) * 2)) & 2048) == 0) ? ((b_65 < a_65) ? b_65 : a_65) : ((b_65 < a_65) ? a_65 : b_65));
  }
  __syncthreads();
  #pragma unroll
  for (int i_67 = 0; i_67 < 8; ++i_67) {
    float a_66 = workspace[((i_67 * 256) + ((int)threadIdx.x))];
    float b_66 = workspace[(((i_67 * 256) + ((int)threadIdx.x)) + 2048)];
    float hi_66 = ((b_66 < a_66) ? a_66 : b_66);
    float lo_66 = ((b_66 < a_66) ? b_66 : a_66);
    bool left_desc_66 = ((((i_67 * 256) + ((int)threadIdx.x)) & 4096) == 0);
    workspace[((i_67 * 256) + ((int)threadIdx.x))] = (((((i_67 * 256) + ((int)threadIdx.x)) & 4096) == 0) ? ((b_66 < a_66) ? a_66 : b_66) : ((b_66 < a_66) ? b_66 : a_66));
    workspace[(((i_67 * 256) + ((int)threadIdx.x)) + 2048)] = (((((i_67 * 256) + ((int)threadIdx.x)) & 4096) == 0) ? ((b_66 < a_66) ? b_66 : a_66) : ((b_66 < a_66) ? a_66 : b_66));
  }
  __syncthreads();
  #pragma unroll
  for (int i_68 = 0; i_68 < 8; ++i_68) {
    float a_67 = workspace[((((i_68 >> 2) * 2048) + ((i_68 & 3) * 256)) + ((int)threadIdx.x))];
    float b_67 = workspace[(((((i_68 >> 2) * 2048) + ((i_68 & 3) * 256)) + ((int)threadIdx.x)) + 1024)];
    float hi_67 = ((b_67 < a_67) ? a_67 : b_67);
    float lo_67 = ((b_67 < a_67) ? b_67 : a_67);
    bool left_desc_67 = ((((((i_68 >> 2) * 2048) + ((i_68 & 3) * 256)) + ((int)threadIdx.x)) & 4096) == 0);
    workspace[((((i_68 >> 2) * 2048) + ((i_68 & 3) * 256)) + ((int)threadIdx.x))] = (((((((i_68 >> 2) * 2048) + ((i_68 & 3) * 256)) + ((int)threadIdx.x)) & 4096) == 0) ? ((b_67 < a_67) ? a_67 : b_67) : ((b_67 < a_67) ? b_67 : a_67));
    workspace[(((((i_68 >> 2) * 2048) + ((i_68 & 3) * 256)) + ((int)threadIdx.x)) + 1024)] = (((((((i_68 >> 2) * 2048) + ((i_68 & 3) * 256)) + ((int)threadIdx.x)) & 4096) == 0) ? ((b_67 < a_67) ? b_67 : a_67) : ((b_67 < a_67) ? a_67 : b_67));
  }
  __syncthreads();
  #pragma unroll
  for (int i_69 = 0; i_69 < 8; ++i_69) {
    float a_68 = workspace[((((i_69 >> 1) * 1024) + ((i_69 & 1) * 256)) + ((int)threadIdx.x))];
    float b_68 = workspace[(((((i_69 >> 1) * 1024) + ((i_69 & 1) * 256)) + ((int)threadIdx.x)) + 512)];
    float hi_68 = ((b_68 < a_68) ? a_68 : b_68);
    float lo_68 = ((b_68 < a_68) ? b_68 : a_68);
    bool left_desc_68 = ((((((i_69 >> 1) * 1024) + ((i_69 & 1) * 256)) + ((int)threadIdx.x)) & 4096) == 0);
    workspace[((((i_69 >> 1) * 1024) + ((i_69 & 1) * 256)) + ((int)threadIdx.x))] = (((((((i_69 >> 1) * 1024) + ((i_69 & 1) * 256)) + ((int)threadIdx.x)) & 4096) == 0) ? ((b_68 < a_68) ? a_68 : b_68) : ((b_68 < a_68) ? b_68 : a_68));
    workspace[(((((i_69 >> 1) * 1024) + ((i_69 & 1) * 256)) + ((int)threadIdx.x)) + 512)] = (((((((i_69 >> 1) * 1024) + ((i_69 & 1) * 256)) + ((int)threadIdx.x)) & 4096) == 0) ? ((b_68 < a_68) ? b_68 : a_68) : ((b_68 < a_68) ? a_68 : b_68));
  }
  __syncthreads();
  #pragma unroll
  for (int i_70 = 0; i_70 < 8; ++i_70) {
    float a_69 = workspace[((i_70 * 512) + ((int)threadIdx.x))];
    float b_69 = workspace[(((i_70 * 512) + ((int)threadIdx.x)) + 256)];
    float hi_69 = ((b_69 < a_69) ? a_69 : b_69);
    float lo_69 = ((b_69 < a_69) ? b_69 : a_69);
    bool left_desc_69 = ((((i_70 * 512) + ((int)threadIdx.x)) & 4096) == 0);
    workspace[((i_70 * 512) + ((int)threadIdx.x))] = (((((i_70 * 512) + ((int)threadIdx.x)) & 4096) == 0) ? ((b_69 < a_69) ? a_69 : b_69) : ((b_69 < a_69) ? b_69 : a_69));
    workspace[(((i_70 * 512) + ((int)threadIdx.x)) + 256)] = (((((i_70 * 512) + ((int)threadIdx.x)) & 4096) == 0) ? ((b_69 < a_69) ? b_69 : a_69) : ((b_69 < a_69) ? a_69 : b_69));
  }
  __syncthreads();
  #pragma unroll
  for (int i_71 = 0; i_71 < 8; ++i_71) {
    float a_70 = workspace[(((i_71 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127))];
    float b_70 = workspace[((((i_71 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) + 128)];
    float hi_70 = ((b_70 < a_70) ? a_70 : b_70);
    float lo_70 = ((b_70 < a_70) ? b_70 : a_70);
    bool left_desc_70 = (((((i_71 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 4096) == 0);
    workspace[(((i_71 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127))] = ((((((i_71 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 4096) == 0) ? ((b_70 < a_70) ? a_70 : b_70) : ((b_70 < a_70) ? b_70 : a_70));
    workspace[((((i_71 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) + 128)] = ((((((i_71 * 512) + ((((int)threadIdx.x) >> 7) * 256)) + (((int)threadIdx.x) & 127)) & 4096) == 0) ? ((b_70 < a_70) ? b_70 : a_70) : ((b_70 < a_70) ? a_70 : b_70));
  }
  __syncthreads();
  #pragma unroll
  for (int i_72 = 0; i_72 < 8; ++i_72) {
    float a_71 = workspace[(((i_72 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63))];
    float b_71 = workspace[((((i_72 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) + 64)];
    float hi_71 = ((b_71 < a_71) ? a_71 : b_71);
    float lo_71 = ((b_71 < a_71) ? b_71 : a_71);
    bool left_desc_71 = (((((i_72 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 4096) == 0);
    workspace[(((i_72 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63))] = ((((((i_72 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 4096) == 0) ? ((b_71 < a_71) ? a_71 : b_71) : ((b_71 < a_71) ? b_71 : a_71));
    workspace[((((i_72 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) + 64)] = ((((((i_72 * 512) + ((((int)threadIdx.x) >> 6) * 128)) + (((int)threadIdx.x) & 63)) & 4096) == 0) ? ((b_71 < a_71) ? b_71 : a_71) : ((b_71 < a_71) ? a_71 : b_71));
  }
  __syncthreads();
  #pragma unroll
  for (int i_73 = 0; i_73 < 8; ++i_73) {
    float a_72 = workspace[(((i_73 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))];
    float b_72 = workspace[((((i_73 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)];
    float hi_72 = ((b_72 < a_72) ? a_72 : b_72);
    float lo_72 = ((b_72 < a_72) ? b_72 : a_72);
    bool left_desc_72 = (((((i_73 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 4096) == 0);
    workspace[(((i_73 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))] = ((((((i_73 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 4096) == 0) ? ((b_72 < a_72) ? a_72 : b_72) : ((b_72 < a_72) ? b_72 : a_72));
    workspace[((((i_73 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)] = ((((((i_73 * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) & 4096) == 0) ? ((b_72 < a_72) ? b_72 : a_72) : ((b_72 < a_72) ? a_72 : b_72));
  }
  __syncthreads();
  #pragma unroll
  for (int i_74 = 0; i_74 < 8; ++i_74) {
    float a_73 = workspace[(((i_74 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))];
    float b_73 = workspace[((((i_74 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)];
    float hi_73 = ((b_73 < a_73) ? a_73 : b_73);
    float lo_73 = ((b_73 < a_73) ? b_73 : a_73);
    bool left_desc_73 = (((((i_74 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 4096) == 0);
    workspace[(((i_74 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15))] = ((((((i_74 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 4096) == 0) ? ((b_73 < a_73) ? a_73 : b_73) : ((b_73 < a_73) ? b_73 : a_73));
    workspace[((((i_74 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) + 16)] = ((((((i_74 * 512) + ((((int)threadIdx.x) >> 4) * 32)) + (((int)threadIdx.x) & 15)) & 4096) == 0) ? ((b_73 < a_73) ? b_73 : a_73) : ((b_73 < a_73) ? a_73 : b_73));
  }
  __syncthreads();
  #pragma unroll
  for (int i_75 = 0; i_75 < 8; ++i_75) {
    float a_74 = workspace[(((i_75 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))];
    float b_74 = workspace[((((i_75 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)];
    float hi_74 = ((b_74 < a_74) ? a_74 : b_74);
    float lo_74 = ((b_74 < a_74) ? b_74 : a_74);
    bool left_desc_74 = (((((i_75 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 4096) == 0);
    workspace[(((i_75 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7))] = ((((((i_75 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 4096) == 0) ? ((b_74 < a_74) ? a_74 : b_74) : ((b_74 < a_74) ? b_74 : a_74));
    workspace[((((i_75 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = ((((((i_75 * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((int)threadIdx.x) & 7)) & 4096) == 0) ? ((b_74 < a_74) ? b_74 : a_74) : ((b_74 < a_74) ? a_74 : b_74));
  }
  __syncthreads();
  #pragma unroll
  for (int i_76 = 0; i_76 < 8; ++i_76) {
    float a_75 = workspace[(((i_76 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))];
    float b_75 = workspace[((((i_76 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)];
    float hi_75 = ((b_75 < a_75) ? a_75 : b_75);
    float lo_75 = ((b_75 < a_75) ? b_75 : a_75);
    bool left_desc_75 = (((((i_76 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 4096) == 0);
    workspace[(((i_76 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3))] = ((((((i_76 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 4096) == 0) ? ((b_75 < a_75) ? a_75 : b_75) : ((b_75 < a_75) ? b_75 : a_75));
    workspace[((((i_76 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) + 4)] = ((((((i_76 * 512) + ((((int)threadIdx.x) >> 2) * 8)) + (((int)threadIdx.x) & 3)) & 4096) == 0) ? ((b_75 < a_75) ? b_75 : a_75) : ((b_75 < a_75) ? a_75 : b_75));
  }
  __syncthreads();
  #pragma unroll
  for (int i_77 = 0; i_77 < 8; ++i_77) {
    float a_76 = workspace[(((i_77 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))];
    float b_76 = workspace[((((i_77 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)];
    float hi_76 = ((b_76 < a_76) ? a_76 : b_76);
    float lo_76 = ((b_76 < a_76) ? b_76 : a_76);
    bool left_desc_76 = (((((i_77 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 4096) == 0);
    workspace[(((i_77 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1))] = ((((((i_77 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 4096) == 0) ? ((b_76 < a_76) ? a_76 : b_76) : ((b_76 < a_76) ? b_76 : a_76));
    workspace[((((i_77 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) + 2)] = ((((((i_77 * 512) + ((((int)threadIdx.x) >> 1) * 4)) + (((int)threadIdx.x) & 1)) & 4096) == 0) ? ((b_76 < a_76) ? b_76 : a_76) : ((b_76 < a_76) ? a_76 : b_76));
  }
  __syncthreads();
  #pragma unroll
  for (int i_78 = 0; i_78 < 8; ++i_78) {
    float a_77 = workspace[((i_78 * 512) + (((int)threadIdx.x) * 2))];
    float b_77 = workspace[(((i_78 * 512) + (((int)threadIdx.x) * 2)) + 1)];
    float hi_77 = ((b_77 < a_77) ? a_77 : b_77);
    float lo_77 = ((b_77 < a_77) ? b_77 : a_77);
    bool left_desc_77 = ((((i_78 * 512) + (((int)threadIdx.x) * 2)) & 4096) == 0);
    workspace[((i_78 * 512) + (((int)threadIdx.x) * 2))] = (((((i_78 * 512) + (((int)threadIdx.x) * 2)) & 4096) == 0) ? ((b_77 < a_77) ? a_77 : b_77) : ((b_77 < a_77) ? b_77 : a_77));
    workspace[(((i_78 * 512) + (((int)threadIdx.x) * 2)) + 1)] = (((((i_78 * 512) + (((int)threadIdx.x) * 2)) & 4096) == 0) ? ((b_77 < a_77) ? b_77 : a_77) : ((b_77 < a_77) ? a_77 : b_77));
  }
  __syncthreads();
  *(float4*)(output + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4))) = *(float4*)(workspace + (((int)threadIdx.x) * 4));
}

