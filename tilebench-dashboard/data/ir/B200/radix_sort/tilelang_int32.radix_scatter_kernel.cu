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
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const int* __restrict__ hist, const int* __restrict__ input, int* __restrict__ output, int shift) {
  int block[8];
  int digit_vals[8];
  int64_t src_buffer[8];
  extern __shared__ __align__(1024) int64_t scan_smem[];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    bool valid = ((((((int)blockIdx.x) * 4) + (i * 2)) + (((int)threadIdx.x) >> 6)) < 78125);
    int broadcast_var = 0;
    int4 condval;
    if (((((((int)blockIdx.x) * 4) + (i * 2)) + (((int)threadIdx.x) >> 6)) < 78125)) {
      condval = *(int4*)(input + (((((int)blockIdx.x) * 1024) + (i * 512)) + (((int)threadIdx.x) * 4)));
    } else {
      condval = make_int4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    int4 val = condval;
    *(int4*)(block + (i * 4)) = val;
    int broadcast_var_1 = 3;
    int4 __1;
      int4 __2;
        int4 v_ = make_int4(shift, shift, shift, shift);
        __2.x = (val.x >> v_.x);
        __2.y = (val.y >> v_.y);
        __2.z = (val.z >> v_.z);
        __2.w = (val.w >> v_.w);
      int4 v__1 = make_int4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
      __1.x = (__2.x & v__1.x);
      __1.y = (__2.y & v__1.y);
      __1.z = (__2.z & v__1.z);
      __1.w = (__2.w & v__1.w);
    *(int4*)(digit_vals + (i * 4)) = __1;
    int broadcast_var_2 = 3;
    int64_t broadcast_var_3 = (int64_t)16;
    longlong4 __3;
      longlong4 v__2 = make_longlong4(((int64_t)((((((int)blockIdx.x) * 4) + (i * 2)) + (((int)threadIdx.x) >> 6)) < 78125)), ((int64_t)((((((int)blockIdx.x) * 4) + (i * 2)) + (((int)threadIdx.x) >> 6)) < 78125)), ((int64_t)((((((int)blockIdx.x) * 4) + (i * 2)) + (((int)threadIdx.x) >> 6)) < 78125)), ((int64_t)((((((int)blockIdx.x) * 4) + (i * 2)) + (((int)threadIdx.x) >> 6)) < 78125)));
      longlong4 __4;
        longlong4 __5;
        int4 __6;
          int4 __7;
            int4 v__3 = make_int4(shift, shift, shift, shift);
            __7.x = (val.x >> v__3.x);
            __7.y = (val.y >> v__3.y);
            __7.z = (val.z >> v__3.z);
            __7.w = (val.w >> v__3.w);
          int4 v__4 = make_int4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
          __6.x = (__7.x & v__4.x);
          __6.y = (__7.y & v__4.y);
          __6.z = (__7.z & v__4.z);
          __6.w = (__7.w & v__4.w);
        __5.x = (int64_t)(__6.x);
        __5.y = (int64_t)(__6.y);
        __5.z = (int64_t)(__6.z);
        __5.w = (int64_t)(__6.w);
        longlong4 v__5 = make_longlong4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
        __4.x = (__5.x*v__5.x);
        __4.y = (__5.y*v__5.y);
        __4.z = (__5.z*v__5.z);
        __4.w = (__5.w*v__5.w);
      __3.x = (v__2.x << __4.x);
      __3.y = (v__2.y << __4.y);
      __3.z = (v__2.z << __4.z);
      __3.w = (v__2.w << __4.w);
    *(longlong4*)(src_buffer + (i * 4)) = __3;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(longlong2*)(scan_smem + ((((i_1 >> 1) * 512) + (((int)threadIdx.x) * 4)) + ((i_1 & 1) * 2))) = *(longlong2*)(src_buffer + (i_1 * 2));
  }
  __syncthreads();
  tl::CumSum1D<128, false>::run((&(scan_smem[0])), (&(scan_smem[0])), 1024);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    *(longlong2*)(src_buffer + (i_2 * 2)) = *(longlong2*)(scan_smem + ((((i_2 >> 1) * 512) + (((int)threadIdx.x) * 4)) + ((i_2 & 1) * 2)));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    if ((((((int)blockIdx.x) * 4) + ((i_3 >> 2) * 2)) + (((int)threadIdx.x) >> 6)) < 78125) {
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

