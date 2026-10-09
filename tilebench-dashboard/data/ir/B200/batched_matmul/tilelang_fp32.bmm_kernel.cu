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

extern "C" __global__ void bmm_kernel_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) bmm_kernel_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_tile = ((void*)((char*)buf_dyn_shmem + 32768));
  float acc[128];
  const dim3 blockIdx = tl::rasterization2DRow<8>();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs<16>((&(((float*)A_tile)[(((((i_1 * 512) + ((((int)threadIdx.x) >> 3) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(A[(((((((int)blockIdx.z) * 409600) + (((int)blockIdx.x) * 81920)) + (i_1 * 10240)) + ((((int)threadIdx.x) >> 3) * 640)) + ((((int)threadIdx.x) & 7) * 4))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    tl::cp_async_gs<16>((&(((float*)B_tile)[((((((((((int)threadIdx.x) & 31) >> 3) * 1024) + (i_2 * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(B[(((((((int)blockIdx.z) * 409600) + (i_2 * 2560)) + ((((int)threadIdx.x) >> 5) * 640)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 31) * 4))])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 19; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      tl::cp_async_gs<16>((&(((float*)A_tile)[((((((((k + 1) & 1) * 4096) + (i_3 * 512)) + ((((int)threadIdx.x) >> 3) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(A[(((((((((int)blockIdx.z) * 409600) + (((int)blockIdx.x) * 81920)) + (i_3 * 10240)) + ((((int)threadIdx.x) >> 3) * 640)) + (k * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 32)])));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      tl::cp_async_gs<16>((&(((float*)B_tile)[(((((((((k + 1) & 1) * 4096) + (((((int)threadIdx.x) & 31) >> 3) * 1024)) + (i_4 * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_4 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(B[(((((((((int)blockIdx.z) * 409600) + (k * 20480)) + (i_4 * 2560)) + ((((int)threadIdx.x) >> 5) * 640)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 20480)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      float A_local[16];
      float B_local[16];
      for (int ki = 0; ki < 4; ++ki) {
        for (int i_5 = 0; i_5 < 4; ++i_5) {
          tl::ptx_ldmatrix_x4((&(((float*)A_tile)[((((((k & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_5 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255))])), (&(A_local[(i_5 * 4)])));
        }
        for (int i_6 = 0; i_6 < 4; ++i_6) {
          for (int j = 0; j < 4; ++j) {
            B_local[((i_6 * 4) + j)] = ((float*)B_tile)[(((((((((((k & 1) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + ((i_6 >> 1) * 1024)) + (ki * 256)) + ((j & 1) * 128)) + ((((int)threadIdx.x) & 3) * 32)) + ((((i_6 & 1) + (j & 1)) & 1) * 16)) + ((((j >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2))];
          }
        }
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + ((i_7 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_7 * 4)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 4)));
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + (((i_7 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_7 * 4)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 4) + 2)));
          }
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    float A_local_1[16];
    float B_local_1[16];
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        tl::ptx_ldmatrix_x4((&(((float*)A_tile)[(((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_8 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255)) + 4096)])), (&(A_local_1[(i_8 * 4)])));
      }
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          B_local_1[((i_9 * 4) + j_2)] = ((float*)B_tile)[(((((((((((((int)threadIdx.x) >> 6) * 2048) + ((i_9 >> 1) * 1024)) + (ki_1 * 256)) + ((j_2 & 1) * 128)) + ((((int)threadIdx.x) & 3) * 32)) + ((((i_9 & 1) + (j_2 & 1)) & 1) * 16)) + ((((j_2 >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2)) + 4096)];
        }
      }
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + ((i_10 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_10 * 4)), reinterpret_cast<const unsigned*>(B_local_1 + (j_3 * 4)));
          tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + (((i_10 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_10 * 4)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_3 * 4) + 2)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 64; ++i_11) {
    *(float2*)(C + ((((((((((((int)blockIdx.z) * 409600) + (((int)blockIdx.x) * 81920)) + (((((int)threadIdx.x) & 63) >> 5) * 40960)) + ((i_11 >> 4) * 10240)) + ((i_11 & 1) * 5120)) + (((((int)threadIdx.x) & 31) >> 2) * 640)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_11 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_11 * 2));
  }
}

