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

extern "C" __global__ void compute_mean_invstd_kernel_kernel(const float* __restrict__ block_sq_sum, const float* __restrict__ block_sum, float* __restrict__ inv_std, float* __restrict__ mean);
extern "C" __global__ void __launch_bounds__(128, 1) compute_mean_invstd_kernel_kernel(const float* __restrict__ block_sq_sum, const float* __restrict__ block_sum, float* __restrict__ inv_std, float* __restrict__ mean) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  float sums[4];
  float sq_sums[4];
  float total_sum[1];
  float total_sq_sum[1];
  float mean_local[1];
  float inv_std_local[1];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float condval;
    if ((((i * 128) + ((int)threadIdx.x)) < 313)) {
      condval = block_sum[(((i * 131072) + (((int)threadIdx.x) * 1024)) + ((int)blockIdx.x))];
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    sums[i] = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    float condval_1;
    if ((((i_1 * 128) + ((int)threadIdx.x)) < 313)) {
      condval_1 = block_sq_sum[(((i_1 * 131072) + (((int)threadIdx.x) * 1024)) + ((int)blockIdx.x))];
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    sq_sums[i_1] = condval_1;
  }
  total_sum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 4; ++rv) {
    total_sum[0] = (total_sum[0] + sums[rv]);
  }
  total_sum[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(total_sum[0], (&(((float*)workspace_1)[0])));
  total_sq_sum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
    total_sq_sum[0] = (total_sq_sum[0] + sq_sums[rv_1]);
  }
  __syncthreads();
  total_sq_sum[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(total_sq_sum[0], (&(((float*)workspace)[0])));
  mean_local[0] = (total_sum[0] / 0x1.388p+14f/*2.000000e+04*/);
  float var_raw = ((total_sq_sum[0] / 0x1.388p+14f/*2.000000e+04*/) - (mean_local[0] * mean_local[0]));
  float var_clamped = max(var_raw, 0x0p+0f/*0.000000e+00*/);
  inv_std_local[0] = rsqrtf((max(var_raw, 0x0p+0f/*0.000000e+00*/) + 0x1.4f8b588e368f1p-17f/*1.000000e-05*/));
  if (((int)threadIdx.x) == 0) {
    mean[((int)blockIdx.x)] = mean_local[0];
    inv_std[((int)blockIdx.x)] = inv_std_local[0];
  }
}

