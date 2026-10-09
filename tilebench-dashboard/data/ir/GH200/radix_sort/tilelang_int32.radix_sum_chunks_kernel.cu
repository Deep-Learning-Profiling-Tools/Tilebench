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

extern "C" __global__ void main_kernel(int* __restrict__ dst, const int* __restrict__ src);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(int* __restrict__ dst, const int* __restrict__ src) {
  int vals[8];
  int total[1];
  extern __shared__ __align__(1024) int workspace[];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    int broadcast_var = 0;
    int4 condval;
    if (((((((int)blockIdx.x) * 64) + (i * 32)) + (((int)threadIdx.x) >> 2)) < 4883)) {
      condval = *(int4*)(src + (((((int)blockIdx.x) * 1024) + (i * 512)) + (((int)threadIdx.x) * 4)));
    } else {
      condval = make_int4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    *(int4*)(vals + (i * 4)) = condval;
  }
  total[0] = 0;
  #pragma unroll
  for (int rv = 0; rv < 8; ++rv) {
    total[0] = (total[0] + vals[(((rv & 1) * 4) + (rv >> 1))]);
  }
  total[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(total[0], (&(workspace[0])));
  dst[((int)blockIdx.x)] = total[0];
}

