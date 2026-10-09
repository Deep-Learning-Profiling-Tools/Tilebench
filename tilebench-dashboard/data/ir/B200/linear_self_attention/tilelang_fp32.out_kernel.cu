#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
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

extern "C" __global__ void out_kernel_kernel(float* __restrict__ O, const float* __restrict__ PhiQ, const float* __restrict__ S, const float* __restrict__ Z);
extern "C" __global__ void __launch_bounds__(128, 1) out_kernel_kernel(float* __restrict__ O, const float* __restrict__ PhiQ, const float* __restrict__ S, const float* __restrict__ Z) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* s_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* q_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  float numer[64];
  float denom[4];
  float q_frag[64];
  float z_frag[1];
  float denom_prod[64];
  float denom_delta[4];
  float denom_delta_clear[64];
  float denom_delta_clear_1[64];
  float denom_delta_clear_2[64];
  float denom_delta_clear_3[64];
  float out[64];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(numer + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(denom + 0) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    ((tfloat32_t*)s_shared)[((((((i_1 * 128) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_1 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)S[(((((((int)threadIdx.x) & 31) * 256) + (((int)blockIdx.y) * 128)) + (i_1 * 4)) + (((int)threadIdx.x) >> 5))]);
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 32; ++i_2) {
    ((tfloat32_t*)s_shared)[(((((((i_2 * 128) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_2 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3)) + 4096)] = ((tfloat32_t)S[((((((((int)threadIdx.x) & 31) * 256) + (((int)blockIdx.y) * 128)) + (i_2 * 4)) + (((int)threadIdx.x) >> 5)) + 8192)]);
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 32; ++i_3) {
    ((tfloat32_t*)s_shared)[(((((((i_3 * 128) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_3 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3)) + 8192)] = ((tfloat32_t)S[((((((((int)threadIdx.x) & 31) * 256) + (((int)blockIdx.y) * 128)) + (i_3 * 4)) + (((int)threadIdx.x) >> 5)) + 16384)]);
  }
  tl::cp_async_commit();
  for (int k_tile = 0; k_tile < 5; ++k_tile) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      float condval;
      if ((((((int)blockIdx.x) * 4) + (i_4 >> 4)) < 625)) {
        condval = PhiQ[((((((int)blockIdx.x) * 16384) + (i_4 * 256)) + (k_tile * 32)) + (((int)threadIdx.x) & 31))];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      q_frag[i_4] = condval;
    }
    if ((((int)threadIdx.x) >> 5) == 0) {
      #pragma unroll
      for (int i_5 = 0; i_5 < 64; ++i_5) {
        ((tfloat32_t*)q_shared)[(((((i_5 * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_5 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_5 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_5 & 1)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)q_frag[i_5]);
      }
    }
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      tfloat32_t A_local[8];
      tfloat32_t B_local[16];
      for (int ki = 0; ki < 4; ++ki) {
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          tl::ptx_ldmatrix_x4((&(((tfloat32_t*)q_shared)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (i_6 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255))])), (&(A_local[(i_6 * 4)])));
        }
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          tl::ptx_ldmatrix_x4((&(((tfloat32_t*)s_shared)[(((((((((k_tile % 3) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (i_7 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(B_local[(i_7 * 4)])));
        }
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(numer + ((i_8 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_8 * 4)), reinterpret_cast<const unsigned*>(B_local + (j * 4)));
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(numer + (((i_8 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_8 * 4)), reinterpret_cast<const unsigned*>(B_local + ((j * 4) + 2)));
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 32; ++i_9) {
      ((tfloat32_t*)s_shared)[((((((((k_tile % 3) * 4096) + (i_9 * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_9 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)S[((((((k_tile * 8192) + ((((int)threadIdx.x) & 31) * 256)) + (((int)blockIdx.y) * 128)) + (i_9 * 4)) + (((int)threadIdx.x) >> 5)) + 24576)]);
    }
    tl::cp_async_commit();
    z_frag[0] = Z[((k_tile * 32) + (((int)threadIdx.x) & 31))];
    #pragma unroll
    for (int i_10 = 0; i_10 < 64; ++i_10) {
      denom_prod[i_10] = (q_frag[i_10] * z_frag[0]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 64; ++i_11) {
      denom_delta_clear[i_11] = 0x0p+0f/*0.000000e+00*/;
      denom_delta_clear[i_11] = (denom_delta_clear[i_11] + denom_prod[i_11]);
      denom_delta_clear[i_11] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<128>>::run(denom_delta_clear[i_11]);
      if ((((((((int)threadIdx.x) & 63) >> 5) * 32) + (((i_11 & 31) >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == i_11) {
        denom_delta[((i_11 & 31) >> 3)] = denom_delta_clear[i_11];
      }
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      denom[i_12] = (denom[i_12] + denom_delta[i_12]);
    }
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 64; ++i_13) {
    float condval_1;
    if ((((((int)blockIdx.x) * 4) + (i_13 >> 4)) < 625)) {
      condval_1 = PhiQ[((((((int)blockIdx.x) * 16384) + (i_13 * 256)) + (((int)threadIdx.x) & 31)) + 160)];
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    q_frag[i_13] = condval_1;
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    #pragma unroll
    for (int i_14 = 0; i_14 < 64; ++i_14) {
      ((tfloat32_t*)q_shared)[(((((i_14 * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_14 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_14 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_14 & 1)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)q_frag[i_14]);
    }
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  {
    tfloat32_t A_local_1[8];
    tfloat32_t B_local_1[16];
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        tl::ptx_ldmatrix_x4((&(((tfloat32_t*)q_shared)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (i_15 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255))])), (&(A_local_1[(i_15 * 4)])));
      }
      for (int i_16 = 0; i_16 < 4; ++i_16) {
        tl::ptx_ldmatrix_x4((&(((tfloat32_t*)s_shared)[(((((((((((int)threadIdx.x) >> 6) * 2048) + (i_16 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4)) + 8192)])), (&(B_local_1[(i_16 * 4)])));
      }
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(numer + ((i_17 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_17 * 4)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 4)));
          tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(numer + (((i_17 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_17 * 4)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 4) + 2)));
        }
      }
    }
  }
  z_frag[0] = Z[((((int)threadIdx.x) & 31) + 160)];
  #pragma unroll
  for (int i_18 = 0; i_18 < 64; ++i_18) {
    denom_prod[i_18] = (q_frag[i_18] * z_frag[0]);
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 64; ++i_19) {
    denom_delta_clear_1[i_19] = 0x0p+0f/*0.000000e+00*/;
    denom_delta_clear_1[i_19] = (denom_delta_clear_1[i_19] + denom_prod[i_19]);
    denom_delta_clear_1[i_19] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<128>>::run(denom_delta_clear_1[i_19]);
    if ((((((((int)threadIdx.x) & 63) >> 5) * 32) + (((i_19 & 31) >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == i_19) {
      denom_delta[((i_19 & 31) >> 3)] = denom_delta_clear_1[i_19];
    }
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 4; ++i_20) {
    denom[i_20] = (denom[i_20] + denom_delta[i_20]);
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 64; ++i_21) {
    float condval_2;
    if ((((((int)blockIdx.x) * 4) + (i_21 >> 4)) < 625)) {
      condval_2 = PhiQ[((((((int)blockIdx.x) * 16384) + (i_21 * 256)) + (((int)threadIdx.x) & 31)) + 192)];
    } else {
      condval_2 = 0x0p+0f/*0.000000e+00*/;
    }
    q_frag[i_21] = condval_2;
  }
  __syncthreads();
  if ((((int)threadIdx.x) >> 5) == 0) {
    #pragma unroll
    for (int i_22 = 0; i_22 < 64; ++i_22) {
      ((tfloat32_t*)q_shared)[(((((i_22 * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_22 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_22 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_22 & 1)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)q_frag[i_22]);
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    tfloat32_t A_local_2[8];
    tfloat32_t B_local_2[16];
    for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
      for (int i_23 = 0; i_23 < 2; ++i_23) {
        tl::ptx_ldmatrix_x4((&(((tfloat32_t*)q_shared)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (i_23 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255))])), (&(A_local_2[(i_23 * 4)])));
      }
      for (int i_24 = 0; i_24 < 4; ++i_24) {
        tl::ptx_ldmatrix_x4((&(((tfloat32_t*)s_shared)[((((((((((int)threadIdx.x) >> 6) * 2048) + (i_24 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(B_local_2[(i_24 * 4)])));
      }
      for (int i_25 = 0; i_25 < 2; ++i_25) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(numer + ((i_25 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_25 * 4)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 4)));
          tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(numer + (((i_25 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_25 * 4)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 4) + 2)));
        }
      }
    }
  }
  z_frag[0] = Z[((((int)threadIdx.x) & 31) + 192)];
  #pragma unroll
  for (int i_26 = 0; i_26 < 64; ++i_26) {
    denom_prod[i_26] = (q_frag[i_26] * z_frag[0]);
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 64; ++i_27) {
    denom_delta_clear_2[i_27] = 0x0p+0f/*0.000000e+00*/;
    denom_delta_clear_2[i_27] = (denom_delta_clear_2[i_27] + denom_prod[i_27]);
    denom_delta_clear_2[i_27] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<128>>::run(denom_delta_clear_2[i_27]);
    if ((((((((int)threadIdx.x) & 63) >> 5) * 32) + (((i_27 & 31) >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == i_27) {
      denom_delta[((i_27 & 31) >> 3)] = denom_delta_clear_2[i_27];
    }
  }
  #pragma unroll
  for (int i_28 = 0; i_28 < 4; ++i_28) {
    denom[i_28] = (denom[i_28] + denom_delta[i_28]);
  }
  #pragma unroll
  for (int i_29 = 0; i_29 < 64; ++i_29) {
    float condval_3;
    if ((((((int)blockIdx.x) * 4) + (i_29 >> 4)) < 625)) {
      condval_3 = PhiQ[((((((int)blockIdx.x) * 16384) + (i_29 * 256)) + (((int)threadIdx.x) & 31)) + 224)];
    } else {
      condval_3 = 0x0p+0f/*0.000000e+00*/;
    }
    q_frag[i_29] = condval_3;
  }
  __syncthreads();
  if ((((int)threadIdx.x) >> 5) == 0) {
    #pragma unroll
    for (int i_30 = 0; i_30 < 64; ++i_30) {
      ((tfloat32_t*)q_shared)[(((((i_30 * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_30 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_30 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_30 & 1)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)q_frag[i_30]);
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    tfloat32_t A_local_3[8];
    tfloat32_t B_local_3[16];
    for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
      for (int i_31 = 0; i_31 < 2; ++i_31) {
        tl::ptx_ldmatrix_x4((&(((tfloat32_t*)q_shared)[((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (i_31 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255))])), (&(A_local_3[(i_31 * 4)])));
      }
      for (int i_32 = 0; i_32 < 4; ++i_32) {
        tl::ptx_ldmatrix_x4((&(((tfloat32_t*)s_shared)[(((((((((((int)threadIdx.x) >> 6) * 2048) + (i_32 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4)) + 4096)])), (&(B_local_3[(i_32 * 4)])));
      }
      for (int i_33 = 0; i_33 < 2; ++i_33) {
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(numer + ((i_33 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_33 * 4)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 4)));
          tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(numer + (((i_33 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_33 * 4)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 4) + 2)));
        }
      }
    }
  }
  z_frag[0] = Z[((((int)threadIdx.x) & 31) + 224)];
  #pragma unroll
  for (int i_34 = 0; i_34 < 64; ++i_34) {
    denom_prod[i_34] = (q_frag[i_34] * z_frag[0]);
  }
  #pragma unroll
  for (int i_35 = 0; i_35 < 64; ++i_35) {
    denom_delta_clear_3[i_35] = 0x0p+0f/*0.000000e+00*/;
    denom_delta_clear_3[i_35] = (denom_delta_clear_3[i_35] + denom_prod[i_35]);
    denom_delta_clear_3[i_35] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<128>>::run(denom_delta_clear_3[i_35]);
    if ((((((((int)threadIdx.x) & 63) >> 5) * 32) + (((i_35 & 31) >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == i_35) {
      denom_delta[((i_35 & 31) >> 3)] = denom_delta_clear_3[i_35];
    }
  }
  #pragma unroll
  for (int i_36 = 0; i_36 < 4; ++i_36) {
    denom[i_36] = (denom[i_36] + denom_delta[i_36]);
  }
  #pragma unroll
  for (int i_37 = 0; i_37 < 64; ++i_37) {
    out[i_37] = (numer[i_37] / (denom[(((i_37 >> 5) * 2) + ((i_37 & 3) >> 1))] + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  }
  #pragma unroll
  for (int i_38 = 0; i_38 < 32; ++i_38) {
    if ((((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) & 63) >> 5) * 2)) + (i_38 >> 4)) < 625) {
      *(float2*)(O + (((((((((((int)blockIdx.x) * 16384) + (((((int)threadIdx.x) & 63) >> 5) * 8192)) + ((i_38 >> 4) * 4096)) + ((i_38 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_38 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(out + (i_38 * 2));
    }
  }
}

