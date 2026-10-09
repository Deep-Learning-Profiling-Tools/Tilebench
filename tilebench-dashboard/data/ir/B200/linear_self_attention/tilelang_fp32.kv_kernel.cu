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

extern "C" __global__ void kv_kernel_kernel(const float* __restrict__ PhiK, float* __restrict__ S, const float* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) kv_kernel_kernel(const float* __restrict__ PhiK, float* __restrict__ S, const float* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* k_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* v_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  float acc[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    ((tfloat32_t*)k_shared)[(((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (i_1 * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_1 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_1 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)PhiK[(((((((int)threadIdx.x) & 63) * 256) + (((int)blockIdx.x) * 32)) + (i_1 * 2)) + (((int)threadIdx.x) >> 6))]);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    ((tfloat32_t*)v_shared)[(((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (i_2 * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_2 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_2 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)V[(((((((int)threadIdx.x) & 63) * 256) + (((int)blockIdx.y) * 32)) + (i_2 * 2)) + (((int)threadIdx.x) >> 6))]);
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    ((tfloat32_t*)k_shared)[((((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (i_3 * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_3 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_3 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3)) + 2048)] = ((tfloat32_t)PhiK[((((((((int)threadIdx.x) & 63) * 256) + (((int)blockIdx.x) * 32)) + (i_3 * 2)) + (((int)threadIdx.x) >> 6)) + 16384)]);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 16; ++i_4) {
    ((tfloat32_t*)v_shared)[((((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (i_4 * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_4 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_4 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3)) + 2048)] = ((tfloat32_t)V[((((((((int)threadIdx.x) & 63) * 256) + (((int)blockIdx.y) * 32)) + (i_4 * 2)) + (((int)threadIdx.x) >> 6)) + 16384)]);
  }
  tl::cp_async_commit();
  for (int k_tile = 0; k_tile < 155; ++k_tile) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      float condval;
      if ((((k_tile * 4) + ((((int)threadIdx.x) & 63) >> 4)) < 617)) {
        condval = PhiK[((((((k_tile * 16384) + ((((int)threadIdx.x) & 63) * 256)) + (((int)blockIdx.x) * 32)) + (i_5 * 2)) + (((int)threadIdx.x) >> 6)) + 32768)];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      ((tfloat32_t*)k_shared)[((((((((((k_tile + 2) % 3) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (i_5 * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_5 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_5 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)condval);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      float condval_1;
      if ((((k_tile * 4) + ((((int)threadIdx.x) & 63) >> 4)) < 617)) {
        condval_1 = V[((((((k_tile * 16384) + ((((int)threadIdx.x) & 63) * 256)) + (((int)blockIdx.y) * 32)) + (i_6 * 2)) + (((int)threadIdx.x) >> 6)) + 32768)];
      } else {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      }
      ((tfloat32_t*)v_shared)[((((((((((k_tile + 2) % 3) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (i_6 * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_6 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_6 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((tfloat32_t)condval_1);
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      tfloat32_t A_local[4];
      tfloat32_t B_local[4];
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((tfloat32_t*)k_shared)[((((((k_tile % 3) * 2048) + ((ki >> 2) * 1024)) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4((&(((tfloat32_t*)v_shared)[(((((((((k_tile % 3) * 2048) + ((ki >> 2) * 1024)) + ((((int)threadIdx.x) >> 6) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 2));
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    tfloat32_t A_local_1[4];
    tfloat32_t B_local_1[4];
    for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
      tl::ptx_ldmatrix_x4((&(((tfloat32_t*)k_shared)[((((((ki_1 >> 2) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255)) + 4096)])), (&(A_local_1[0])));
      tl::ptx_ldmatrix_x4((&(((tfloat32_t*)v_shared)[(((((((((ki_1 >> 2) * 1024) + ((((int)threadIdx.x) >> 6) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4)) + 4096)])), (&(B_local_1[0])));
      tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
      tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + 4), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 2));
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    tfloat32_t A_local_2[4];
    tfloat32_t B_local_2[4];
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((tfloat32_t*)k_shared)[(((((ki_2 >> 2) * 1024) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255))])), (&(A_local_2[0])));
      tl::ptx_ldmatrix_x4((&(((tfloat32_t*)v_shared)[((((((((ki_2 >> 2) * 1024) + ((((int)threadIdx.x) >> 6) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(B_local_2[0])));
      tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + 0), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 0));
      tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + 4), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 2));
    }
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 4; ++i_7) {
    *(float2*)(S + ((((((((((int)blockIdx.x) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + ((i_7 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) >> 6) * 16)) + ((i_7 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_7 * 2));
  }
}

