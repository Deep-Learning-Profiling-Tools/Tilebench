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

extern "C" __global__ void layernorm_kernel_kernel(const bfloat16_t* __restrict__ X, bfloat16_t* __restrict__ Y, const bfloat16_t* __restrict__ bias, const bfloat16_t* __restrict__ weight, int M);
extern "C" __global__ void __launch_bounds__(256, 1) layernorm_kernel_kernel(const bfloat16_t* __restrict__ X, bfloat16_t* __restrict__ Y, const bfloat16_t* __restrict__ bias, const bfloat16_t* __restrict__ weight, int M) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  float sum_local[16];
  float sumsq_local[16];
  float sum_row[1];
  float sumsq_row[1];
  float mean_row[1];
  float rstd_row[1];
  bfloat16_t Y_local[16];
  bfloat16_t X_local_cast[8];
  bfloat16_t weight_local_cast_1[8];
  bfloat16_t bias_local_cast_2[8];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(sum_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(sumsq_local + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  for (int tile = 0; tile < 3; ++tile) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      bfloat16_t broadcast_var_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      ulonglong4 __1;
      uint4 condval;
      if ((((tile * 2) + i_2) < 5)) {
        condval = *(uint4*)(X + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile) * (int64_t)4096)) + (((int64_t)i_2) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
      } else {
        condval = make_uint4(__pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2));
      }
      ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&condval))[0]);
      ((float2*)(&__1))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&condval))[1]);
      ((float2*)(&__1))[2] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&condval))[2]);
      ((float2*)(&__1))[3] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&condval))[3]);
      ulonglong4 x_val = __1;
      ulonglong4 __2;
        ulonglong4 v_ = *(ulonglong4*)(sum_local + (i_2 * 8));
        *(float2*)(&(__2.x)) = tl::add2(*(float2*)(&(v_.x)), *(float2*)(&(x_val.x)));
        *(float2*)(&(__2.y)) = tl::add2(*(float2*)(&(v_.y)), *(float2*)(&(x_val.y)));
        *(float2*)(&(__2.z)) = tl::add2(*(float2*)(&(v_.z)), *(float2*)(&(x_val.z)));
        *(float2*)(&(__2.w)) = tl::add2(*(float2*)(&(v_.w)), *(float2*)(&(x_val.w)));
      *(ulonglong4*)(sum_local + (i_2 * 8)) = __2;
      ulonglong4 __3;
        ulonglong4 v__1 = *(ulonglong4*)(sumsq_local + (i_2 * 8));
        *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(x_val.x)), *(float2*)(&(x_val.x)), *(float2*)(&(v__1.x)));
        *(float2*)(&(__3.y)) = tl::fma2(*(float2*)(&(x_val.y)), *(float2*)(&(x_val.y)), *(float2*)(&(v__1.y)));
        *(float2*)(&(__3.z)) = tl::fma2(*(float2*)(&(x_val.z)), *(float2*)(&(x_val.z)), *(float2*)(&(v__1.z)));
        *(float2*)(&(__3.w)) = tl::fma2(*(float2*)(&(x_val.w)), *(float2*)(&(x_val.w)), *(float2*)(&(v__1.w)));
      *(ulonglong4*)(sumsq_local + (i_2 * 8)) = __3;
    }
  }
  sum_row[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 16; ++rv) {
    sum_row[0] = (sum_row[0] + sum_local[(((rv & 1) * 8) + (rv >> 1))]);
  }
  sum_row[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(sum_row[0], (&(((float*)workspace_1)[0])));
  sumsq_row[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
    sumsq_row[0] = (sumsq_row[0] + sumsq_local[(((rv_1 & 1) * 8) + (rv_1 >> 1))]);
  }
  __syncthreads();
  sumsq_row[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(sumsq_row[0], (&(((float*)workspace)[0])));
  mean_row[0] = (sum_row[0] / 0x1.4p+13f/*1.024000e+04*/);
  float variance = ((sumsq_row[0] / 0x1.4p+13f/*1.024000e+04*/) - (mean_row[0] * mean_row[0]));
  rstd_row[0] = rsqrtf((variance + 0x1.4f8b588e368f1p-17f/*1.000000e-05*/));
  for (int tile_1 = 0; tile_1 < 3; ++tile_1) {
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_1;
      if ((((tile_1 * 2) + i_3) < 5)) {
        condval_1 = *(uint4*)(X + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)4096)) + (((int64_t)i_3) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
      } else {
        condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
      }
      *(uint4*)(X_local_cast + 0) = condval_1;
      bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_2;
      if ((((tile_1 * 2) + i_3) < 5)) {
        condval_2 = *(uint4*)(weight + (((tile_1 * 4096) + (i_3 * 2048)) + (((int)threadIdx.x) * 8)));
      } else {
        condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
      }
      *(uint4*)(weight_local_cast_1 + 0) = condval_2;
      bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_3;
      if ((((tile_1 * 2) + i_3) < 5)) {
        condval_3 = *(uint4*)(bias + (((tile_1 * 4096) + (i_3 * 2048)) + (((int)threadIdx.x) * 8)));
      } else {
        condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
      }
      *(uint4*)(bias_local_cast_2 + 0) = condval_3;
      for (int vec = 0; vec < 2; ++vec) {
        uint2 __4;
        float4 __5;
          float4 __6;
            float4 __7;
              float4 __8;
              uint2 v__2 = *(uint2*)(X_local_cast + (vec * 4));
              ((float2*)(&__8))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__2))[0]);
              ((float2*)(&__8))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__2))[1]);
              float4 v__3 = make_float4(mean_row[0], mean_row[0], mean_row[0], mean_row[0]);
              *(float2*)(&(__7.x)) = tl::sub2(*(float2*)(&(__8.x)), *(float2*)(&(v__3.x)));
              *(float2*)(&(__7.z)) = tl::sub2(*(float2*)(&(__8.z)), *(float2*)(&(v__3.z)));
            float4 v__4 = make_float4(rstd_row[0], rstd_row[0], rstd_row[0], rstd_row[0]);
            *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(__7.x)), *(float2*)(&(v__4.x)));
            *(float2*)(&(__6.z)) = tl::mul2(*(float2*)(&(__7.z)), *(float2*)(&(v__4.z)));
          float4 __9;
          uint2 v__5 = *(uint2*)(weight_local_cast_1 + (vec * 4));
          ((float2*)(&__9))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__5))[0]);
          ((float2*)(&__9))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__5))[1]);
          float4 __10;
          uint2 v__6 = *(uint2*)(bias_local_cast_2 + (vec * 4));
          ((float2*)(&__10))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__6))[0]);
          ((float2*)(&__10))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__6))[1]);
          *(float2*)(&(__5.x)) = tl::fma2(*(float2*)(&(__6.x)), *(float2*)(&(__9.x)), *(float2*)(&(__10.x)));
          *(float2*)(&(__5.z)) = tl::fma2(*(float2*)(&(__6.z)), *(float2*)(&(__9.z)), *(float2*)(&(__10.z)));
        (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&__5))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__4))[1] = __float22bfloat162_rn(((float2*)(&__5))[1]);
        *(uint2*)(Y_local + ((i_3 * 8) + (vec * 4))) = __4;
      }
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      if (((tile_1 * 2) + i_4) < 5) {
        *(uint4*)(Y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)4096)) + (((int64_t)i_4) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(Y_local + (i_4 * 8));
      }
    }
  }
}

