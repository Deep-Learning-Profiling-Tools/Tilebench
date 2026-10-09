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

extern "C" __global__ void main_kernel(int* __restrict__ hist, const int* __restrict__ input, int shift);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(int* __restrict__ hist, const int* __restrict__ input, int shift) {
  int64_t packed[8];
  int input_local_cast[4];
  int64_t total[1];
  extern __shared__ __align__(1024) int64_t workspace[];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    int broadcast_var = 0;
    int4 condval;
    if (((((((int)blockIdx.x) * 4) + (i * 2)) + (((int)threadIdx.x) >> 6)) < 78125)) {
      condval = *(int4*)(input + (((((int)blockIdx.x) * 1024) + (i * 512)) + (((int)threadIdx.x) * 4)));
    } else {
      condval = make_int4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    *(int4*)(input_local_cast + 0) = condval;
    for (int vec = 0; vec < 2; ++vec) {
      int broadcast_var_1 = 3;
      int64_t broadcast_var_2 = (int64_t)16;
      longlong2 __1;
        longlong2 v_ = make_longlong2(((int64_t)((((((int)blockIdx.x) * 4) + (i * 2)) + (((int)threadIdx.x) >> 6)) < 78125)), ((int64_t)((((((int)blockIdx.x) * 4) + (i * 2)) + (((int)threadIdx.x) >> 6)) < 78125)));
        longlong2 __2;
          longlong2 __3;
          int2 __4;
            int2 __5;
              int2 v__1 = *(int2*)(input_local_cast + (vec * 2));
              int2 v__2 = make_int2(shift, shift);
              __5.x = (v__1.x >> v__2.x);
              __5.y = (v__1.y >> v__2.y);
            int2 v__3 = make_int2(broadcast_var_1, broadcast_var_1);
            __4.x = (__5.x & v__3.x);
            __4.y = (__5.y & v__3.y);
          __3.x = (int64_t)(__4.x);
          __3.y = (int64_t)(__4.y);
          longlong2 v__4 = make_longlong2(broadcast_var_2, broadcast_var_2);
          __2.x = (__3.x*v__4.x);
          __2.y = (__3.y*v__4.y);
        __1.x = (v_.x << __2.x);
        __1.y = (v_.y << __2.y);
      *(longlong2*)(packed + ((i * 4) + (vec * 2))) = __1;
    }
  }
  total[0] = (int64_t)0;
  #pragma unroll
  for (int rv = 0; rv < 8; ++rv) {
    total[0] = (total[0] + packed[(((rv & 1) * 4) + (rv >> 1))]);
  }
  total[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(total[0], (&(workspace[0])));
  if ((((int)threadIdx.x) >> 2) == 0) {
    hist[(((((int)threadIdx.x) & 3) * 19532) + ((int)blockIdx.x))] = ((int)((total[0] >> ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)16)) & (int64_t)65535));
  }
}

