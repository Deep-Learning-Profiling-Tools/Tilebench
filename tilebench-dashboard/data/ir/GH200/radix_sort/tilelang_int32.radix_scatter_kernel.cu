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

extern "C" __global__ void main_kernel(const int* __restrict__ hist, const int* __restrict__ input, int* __restrict__ output, int shift);
extern "C" __global__ void __launch_bounds__(64, 1) main_kernel(const int* __restrict__ hist, const int* __restrict__ input, int* __restrict__ output, int shift) {
  int block[16];
  int digit_vals[16];
  int64_t src_buffer[16];
  extern __shared__ __align__(1024) int64_t scan_smem[];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    bool valid = (((((int)blockIdx.x) * 4) + (i >> 1)) < 78125);
    int broadcast_var = 0;
    int2 condval;
    if ((((((int)blockIdx.x) * 4) + (i >> 1)) < 78125)) {
      condval = *(int2*)(input + (((((int)blockIdx.x) * 1024) + (i * 128)) + (((int)threadIdx.x) * 2)));
    } else {
      condval = make_int2(broadcast_var, broadcast_var);
    }
    int2 val = condval;
    *(int2*)(block + (i * 2)) = val;
    int broadcast_var_1 = 3;
    int2 __1;
      int2 __2;
        int2 v_ = make_int2(shift, shift);
        __2.x = (val.x >> v_.x);
        __2.y = (val.y >> v_.y);
      int2 v__1 = make_int2(broadcast_var_1, broadcast_var_1);
      __1.x = (__2.x & v__1.x);
      __1.y = (__2.y & v__1.y);
    *(int2*)(digit_vals + (i * 2)) = __1;
    int broadcast_var_2 = 3;
    int64_t broadcast_var_3 = (int64_t)16;
    longlong2 __3;
      longlong2 v__2 = make_longlong2(((int64_t)(((((int)blockIdx.x) * 4) + (i >> 1)) < 78125)), ((int64_t)(((((int)blockIdx.x) * 4) + (i >> 1)) < 78125)));
      longlong2 __4;
        longlong2 __5;
        int2 __6;
          int2 __7;
            int2 v__3 = make_int2(shift, shift);
            __7.x = (val.x >> v__3.x);
            __7.y = (val.y >> v__3.y);
          int2 v__4 = make_int2(broadcast_var_2, broadcast_var_2);
          __6.x = (__7.x & v__4.x);
          __6.y = (__7.y & v__4.y);
        __5.x = (int64_t)(__6.x);
        __5.y = (int64_t)(__6.y);
        longlong2 v__5 = make_longlong2(broadcast_var_3, broadcast_var_3);
        __4.x = (__5.x*v__5.x);
        __4.y = (__5.y*v__5.y);
      __3.x = (v__2.x << __4.x);
      __3.y = (v__2.y << __4.y);
    *(longlong2*)(src_buffer + (i * 2)) = __3;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    *(longlong2*)(scan_smem + ((i_1 * 128) + (((int)threadIdx.x) * 2))) = *(longlong2*)(src_buffer + (i_1 * 2));
  }
  __syncthreads();
  tl::CumSum1D<64, false>::run((&(scan_smem[0])), (&(scan_smem[0])), 1024);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    *(longlong2*)(src_buffer + (i_2 * 2)) = *(longlong2*)(scan_smem + ((i_2 * 128) + (((int)threadIdx.x) * 2)));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    if (((((int)blockIdx.x) * 4) + (i_3 >> 2)) < 78125) {
      int condval_1;
      if (((0 <= digit_vals[i_3]) && (digit_vals[i_3] < 4))) {
        condval_1 = hist[((((int64_t)digit_vals[((int64_t)i_3)]) * (int64_t)19532) + ((int64_t)((int)blockIdx.x)))];
      } else {
        condval_1 = 0;
      }
      if (0 <= (condval_1 + ((int)(((src_buffer[i_3] - ((int64_t)1 << (((int64_t)digit_vals[i_3]) * (int64_t)16))) >> (((int64_t)digit_vals[i_3]) * (int64_t)16)) & (int64_t)65535)))) {
        int condval_2;
        if (((0 <= digit_vals[i_3]) && (digit_vals[i_3] < 4))) {
          condval_2 = hist[((((int64_t)digit_vals[((int64_t)i_3)]) * (int64_t)19532) + ((int64_t)((int)blockIdx.x)))];
        } else {
          condval_2 = 0;
        }
        if ((condval_2 + ((int)(((src_buffer[i_3] - ((int64_t)1 << (((int64_t)digit_vals[i_3]) * (int64_t)16))) >> (((int64_t)digit_vals[i_3]) * (int64_t)16)) & (int64_t)65535))) < 20000000) {
          int64_t condval_3;
          if ((((int64_t)0 <= ((int64_t)digit_vals[((int64_t)i_3)])) && (((int64_t)digit_vals[((int64_t)i_3)]) < (int64_t)4))) {
            condval_3 = ((int64_t)hist[((((int64_t)digit_vals[((int64_t)i_3)]) * (int64_t)19532) + ((int64_t)((int)blockIdx.x)))]);
          } else {
            condval_3 = (int64_t)0;
          }
          int64_t condval_4;
          if ((((int64_t)0 <= ((int64_t)digit_vals[((int64_t)i_3)])) && (((int64_t)digit_vals[((int64_t)i_3)]) < (int64_t)4))) {
            condval_4 = ((int64_t)hist[((((int64_t)digit_vals[((int64_t)i_3)]) * (int64_t)19532) + ((int64_t)((int)blockIdx.x)))]);
          } else {
            condval_4 = (int64_t)0;
          }
          output[(condval_4 + (((src_buffer[i_3] - ((int64_t)1 << (((int64_t)digit_vals[i_3]) * (int64_t)16))) >> (((int64_t)digit_vals[i_3]) * (int64_t)16)) & (int64_t)65535))] = block[i_3];
        }
      }
    }
  }
}

