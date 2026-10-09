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

extern "C" __global__ void kl_divergence_kernel_kernel(const float* __restrict__ log_y_pred, float* __restrict__ loss, const float* __restrict__ y_true, int M);
extern "C" __global__ void __launch_bounds__(256, 1) kl_divergence_kernel_kernel(const float* __restrict__ log_y_pred, float* __restrict__ loss, const float* __restrict__ y_true, int M) {
  float loss_global[1];
  float y_true_local[16];
  float local_log_y_pred[16];
  float loss_local_arr[16];
  float loss_local[1];
  extern __shared__ __align__(1024) float workspace[];
  loss_global[0] = 0x0p+0f/*0.000000e+00*/;
  __syncthreads();
  for (int tile = 0; tile < 4; ++tile) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      *(float4*)(y_true_local + (i * 4)) = *(float4*)(y_true + ((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + (((int64_t)tile) * (int64_t)4096)) + (((int64_t)i) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      *(float4*)(local_log_y_pred + (i_1 * 4)) = *(float4*)(log_y_pred + ((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + (((int64_t)tile) * (int64_t)4096)) + (((int64_t)i_1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(loss_local_arr + (i_2 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      float yt = y_true_local[i_3];
      loss_local_arr[i_3] = ((0x0p+0f/*0.000000e+00*/ < yt) ? (yt * (logf(yt) - local_log_y_pred[i_3])) : 0x0p+0f/*0.000000e+00*/);
    }
    loss_local[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv = 0; rv < 16; ++rv) {
      loss_local[0] = (loss_local[0] + loss_local_arr[(((rv & 3) * 4) + (rv >> 2))]);
    }
    loss_local[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(loss_local[0], (&(workspace[0])));
    loss_global[0] = (loss_global[0] + loss_local[0]);
  }
  if (((int)threadIdx.x) == 0) {
    loss[((int64_t)((int)blockIdx.x))] = loss_global[0];
  }
}

