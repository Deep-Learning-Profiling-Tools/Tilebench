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

extern "C" __global__ void layernorm_kernel_kernel(const float* __restrict__ X, float* __restrict__ Y, const float* __restrict__ bias, const float* __restrict__ weight, int M);
extern "C" __global__ void __launch_bounds__(128, 1) layernorm_kernel_kernel(const float* __restrict__ X, float* __restrict__ Y, const float* __restrict__ bias, const float* __restrict__ weight, int M) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  float sum_local[32];
  float sumsq_local[32];
  float sum_row[1];
  float sumsq_row[1];
  float mean_row[1];
  float rstd_row[1];
  float Y_local[32];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(sum_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(sumsq_local + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  for (int tile = 0; tile < 3; ++tile) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      ulonglong4 condval;
      if ((((tile * 2) + (i_2 >> 1)) < 5)) {
        condval = tl::load_global_256(&(*(ulonglong4*)(X + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile) * (int64_t)4096)) + (((int64_t)i_2) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)))));
      } else {
        condval = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2), *(unsigned long long*)&make_float2(broadcast_var_2, broadcast_var_2));
      }
      ulonglong4 x_val = condval;
      ulonglong4 __1;
        ulonglong4 v_ = *(ulonglong4*)(sum_local + (i_2 * 8));
        *(float2*)(&(__1.x)) = tl::add2(*(float2*)(&(v_.x)), *(float2*)(&(x_val.x)));
        *(float2*)(&(__1.y)) = tl::add2(*(float2*)(&(v_.y)), *(float2*)(&(x_val.y)));
        *(float2*)(&(__1.z)) = tl::add2(*(float2*)(&(v_.z)), *(float2*)(&(x_val.z)));
        *(float2*)(&(__1.w)) = tl::add2(*(float2*)(&(v_.w)), *(float2*)(&(x_val.w)));
      *(ulonglong4*)(sum_local + (i_2 * 8)) = __1;
      ulonglong4 __2;
        ulonglong4 v__1 = *(ulonglong4*)(sumsq_local + (i_2 * 8));
        *(float2*)(&(__2.x)) = tl::fma2(*(float2*)(&(x_val.x)), *(float2*)(&(x_val.x)), *(float2*)(&(v__1.x)));
        *(float2*)(&(__2.y)) = tl::fma2(*(float2*)(&(x_val.y)), *(float2*)(&(x_val.y)), *(float2*)(&(v__1.y)));
        *(float2*)(&(__2.z)) = tl::fma2(*(float2*)(&(x_val.z)), *(float2*)(&(x_val.z)), *(float2*)(&(v__1.z)));
        *(float2*)(&(__2.w)) = tl::fma2(*(float2*)(&(x_val.w)), *(float2*)(&(x_val.w)), *(float2*)(&(v__1.w)));
      *(ulonglong4*)(sumsq_local + (i_2 * 8)) = __2;
    }
  }
  sum_row[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 32; ++rv) {
    sum_row[0] = (sum_row[0] + sum_local[(((rv & 3) * 8) + (rv >> 2))]);
  }
  sum_row[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(sum_row[0], (&(((float*)workspace_1)[0])));
  sumsq_row[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
    sumsq_row[0] = (sumsq_row[0] + sumsq_local[(((rv_1 & 3) * 8) + (rv_1 >> 2))]);
  }
  __syncthreads();
  sumsq_row[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(sumsq_row[0], (&(((float*)workspace)[0])));
  mean_row[0] = (sum_row[0] / 0x1.4p+13f/*1.024000e+04*/);
  float variance = ((sumsq_row[0] / 0x1.4p+13f/*1.024000e+04*/) - (mean_row[0] * mean_row[0]));
  rstd_row[0] = rsqrtf((variance + 0x1.4f8b588e368f1p-17f/*1.000000e-05*/));
  for (int tile_1 = 0; tile_1 < 3; ++tile_1) {
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
      ulonglong4 condval_1;
      if ((((tile_1 * 2) + (i_3 >> 1)) < 5)) {
        condval_1 = tl::load_global_256(&(*(ulonglong4*)(X + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)4096)) + (((int64_t)i_3) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)))));
      } else {
        condval_1 = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var_3, broadcast_var_3), *(unsigned long long*)&make_float2(broadcast_var_3, broadcast_var_3), *(unsigned long long*)&make_float2(broadcast_var_3, broadcast_var_3), *(unsigned long long*)&make_float2(broadcast_var_3, broadcast_var_3));
      }
      ulonglong4 x_val_1 = condval_1;
      ulonglong4 __3;
        ulonglong4 __4;
          ulonglong4 v__2 = make_ulonglong4(*(unsigned long long*)&make_float2(mean_row[0], mean_row[0]), *(unsigned long long*)&make_float2(mean_row[0], mean_row[0]), *(unsigned long long*)&make_float2(mean_row[0], mean_row[0]), *(unsigned long long*)&make_float2(mean_row[0], mean_row[0]));
          *(float2*)(&(__4.x)) = tl::sub2(*(float2*)(&(x_val_1.x)), *(float2*)(&(v__2.x)));
          *(float2*)(&(__4.y)) = tl::sub2(*(float2*)(&(x_val_1.y)), *(float2*)(&(v__2.y)));
          *(float2*)(&(__4.z)) = tl::sub2(*(float2*)(&(x_val_1.z)), *(float2*)(&(v__2.z)));
          *(float2*)(&(__4.w)) = tl::sub2(*(float2*)(&(x_val_1.w)), *(float2*)(&(v__2.w)));
        ulonglong4 v__3 = make_ulonglong4(*(unsigned long long*)&make_float2(rstd_row[0], rstd_row[0]), *(unsigned long long*)&make_float2(rstd_row[0], rstd_row[0]), *(unsigned long long*)&make_float2(rstd_row[0], rstd_row[0]), *(unsigned long long*)&make_float2(rstd_row[0], rstd_row[0]));
        *(float2*)(&(__3.x)) = tl::mul2(*(float2*)(&(__4.x)), *(float2*)(&(v__3.x)));
        *(float2*)(&(__3.y)) = tl::mul2(*(float2*)(&(__4.y)), *(float2*)(&(v__3.y)));
        *(float2*)(&(__3.z)) = tl::mul2(*(float2*)(&(__4.z)), *(float2*)(&(v__3.z)));
        *(float2*)(&(__3.w)) = tl::mul2(*(float2*)(&(__4.w)), *(float2*)(&(v__3.w)));
      ulonglong4 norm_1 = __3;
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
      ulonglong4 __5;
        ulonglong4 condval_2;
        if ((((tile_1 * 2) + (i_3 >> 1)) < 5)) {
          condval_2 = tl::load_global_256(&(*(ulonglong4*)(weight + (((tile_1 * 4096) + (i_3 * 1024)) + (((int)threadIdx.x) * 8)))));
        } else {
          condval_2 = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var_4, broadcast_var_4), *(unsigned long long*)&make_float2(broadcast_var_4, broadcast_var_4), *(unsigned long long*)&make_float2(broadcast_var_4, broadcast_var_4), *(unsigned long long*)&make_float2(broadcast_var_4, broadcast_var_4));
        }
        ulonglong4 condval_3;
        if ((((tile_1 * 2) + (i_3 >> 1)) < 5)) {
          condval_3 = tl::load_global_256(&(*(ulonglong4*)(bias + (((tile_1 * 4096) + (i_3 * 1024)) + (((int)threadIdx.x) * 8)))));
        } else {
          condval_3 = make_ulonglong4(*(unsigned long long*)&make_float2(broadcast_var_5, broadcast_var_5), *(unsigned long long*)&make_float2(broadcast_var_5, broadcast_var_5), *(unsigned long long*)&make_float2(broadcast_var_5, broadcast_var_5), *(unsigned long long*)&make_float2(broadcast_var_5, broadcast_var_5));
        }
        *(float2*)(&(__5.x)) = tl::fma2(*(float2*)(&(norm_1.x)), *(float2*)(&(condval_2.x)), *(float2*)(&(condval_3.x)));
        *(float2*)(&(__5.y)) = tl::fma2(*(float2*)(&(norm_1.y)), *(float2*)(&(condval_2.y)), *(float2*)(&(condval_3.y)));
        *(float2*)(&(__5.z)) = tl::fma2(*(float2*)(&(norm_1.z)), *(float2*)(&(condval_2.z)), *(float2*)(&(condval_3.z)));
        *(float2*)(&(__5.w)) = tl::fma2(*(float2*)(&(norm_1.w)), *(float2*)(&(condval_2.w)), *(float2*)(&(condval_3.w)));
      *(ulonglong4*)(Y_local + (i_3 * 8)) = __5;
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      if (((tile_1 * 2) + (i_4 >> 1)) < 5) {
        tl::store_global_256(&(*(ulonglong4*)(Y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)10240) + (((int64_t)tile_1) * (int64_t)4096)) + (((int64_t)i_4) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)))), *(ulonglong4*)(Y_local + (i_4 * 8)));
      }
    }
  }
}

