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

extern "C" __global__ void cross_entropy_kernel_kernel(const float* __restrict__ logits, float* __restrict__ output, const int64_t* __restrict__ targets, int M);
extern "C" __global__ void __launch_bounds__(64, 1) cross_entropy_kernel_kernel(const float* __restrict__ logits, float* __restrict__ output, const int64_t* __restrict__ targets, int M) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  float local_logits[8];
  float row_max[1];
  float exp_shifted[8];
  float row_sum[1];
  float loss[1];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = -CUDART_INF_F;
    *(float4*)(local_logits + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  *(ulonglong4*)(local_logits + 0) = tl::load_global_256(&(*(ulonglong4*)(logits + ((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)))));
  row_max[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv = 0; rv < 8; ++rv) {
    row_max[0] = max(row_max[0], local_logits[rv]);
  }
  row_max[0] = tl::AllReduce<tl::MaxOp, 64, 1, 0, tl::NamedBarrier<64>>::run(row_max[0], (&(((float*)workspace_1)[0])));
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    exp_shifted[i_1] = expf((local_logits[i_1] - row_max[0]));
  }
  row_sum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
    row_sum[0] = (row_sum[0] + exp_shifted[rv_1]);
  }
  __syncthreads();
  row_sum[0] = tl::AllReduce<tl::SumOp, 64, 1, 0, tl::NamedBarrier<64>>::run(row_sum[0], (&(((float*)workspace)[0])));
  int target_lw = ((int)targets[((int64_t)((int)blockIdx.x))]);
  float condval;
  if (((0 <= target_lw) && (target_lw < 512))) {
    condval = logits[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)target_lw))];
  } else {
    condval = 0x0p+0f/*0.000000e+00*/;
  }
  loss[0] = ((logf(row_sum[0]) + row_max[0]) - condval);
  if (((int)threadIdx.x) == 0) {
    output[((int64_t)((int)blockIdx.x))] = loss[0];
  }
}

