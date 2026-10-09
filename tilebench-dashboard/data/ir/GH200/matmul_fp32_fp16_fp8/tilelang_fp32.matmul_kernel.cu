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

extern "C" __global__ void matmul_kernel_kernel(const tfloat32_t* __restrict__ a, const tfloat32_t* __restrict__ b, tfloat32_t* __restrict__ c);
extern "C" __global__ void __launch_bounds__(128, 1) matmul_kernel_kernel(const tfloat32_t* __restrict__ a, const tfloat32_t* __restrict__ b, tfloat32_t* __restrict__ c) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* b_tile = ((void*)((char*)buf_dyn_shmem + 98304));
  float acc[128];
  tfloat32_t c_local_cast[2];
  const dim3 blockIdx = tl::rasterization2DRow<8>();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    tl::cp_async_gs<16>((&(((tfloat32_t*)a_tile)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_1 * 256)) + ((((int)threadIdx.x) >> 4) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(a[((((((int)blockIdx.x) * 2621440) + (i_1 * 163840)) + ((((int)threadIdx.x) >> 4) * 20480)) + ((((int)threadIdx.x) & 15) * 4))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    tl::cp_async_gs<16>((&(((tfloat32_t*)b_tile)[((((((((((int)threadIdx.x) & 31) >> 3) * 2048) + (i_2 * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(b[((((i_2 * 16384) + ((((int)threadIdx.x) >> 5) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 31) * 4))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    tl::cp_async_gs<16>((&(((tfloat32_t*)a_tile)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 256)) + ((((int)threadIdx.x) >> 4) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + 8192)])), (&(a[(((((((int)blockIdx.x) * 2621440) + (i_3 * 163840)) + ((((int)threadIdx.x) >> 4) * 20480)) + ((((int)threadIdx.x) & 15) * 4)) + 64)])));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 16; ++i_4) {
    tl::cp_async_gs<16>((&(((tfloat32_t*)b_tile)[(((((((((((int)threadIdx.x) & 31) >> 3) * 2048) + (i_4 * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_4 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 4)) + 8192)])), (&(b[(((((i_4 * 16384) + ((((int)threadIdx.x) >> 5) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 262144)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 318; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      tl::cp_async_gs<16>((&(((tfloat32_t*)a_tile)[(((((((((k + 2) % 3) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_5 * 256)) + ((((int)threadIdx.x) >> 4) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(a[((((((((int)blockIdx.x) * 2621440) + (i_5 * 163840)) + ((((int)threadIdx.x) >> 4) * 20480)) + (k * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 128)])));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      tl::cp_async_gs<16>((&(((tfloat32_t*)b_tile)[(((((((((k + 2) % 3) * 8192) + (((((int)threadIdx.x) & 31) >> 3) * 2048)) + (i_6 * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_6 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(b[((((((k * 262144) + (i_6 * 16384)) + ((((int)threadIdx.x) >> 5) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 524288)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      tfloat32_t A_local[16];
      tfloat32_t B_local[16];
      for (int ki = 0; ki < 8; ++ki) {
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          tl::ptx_ldmatrix_x4((&(((tfloat32_t*)a_tile)[(((((((k % 3) * 8192) + ((ki >> 2) * 4096)) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_7 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255))])), (&(A_local[(i_7 * 4)])));
        }
        for (int i_8 = 0; i_8 < 4; ++i_8) {
          for (int j = 0; j < 4; ++j) {
            B_local[((i_8 * 4) + j)] = ((tfloat32_t*)b_tile)[(((((((((((k % 3) * 8192) + ((((int)threadIdx.x) >> 6) * 4096)) + ((i_8 >> 1) * 2048)) + (ki * 256)) + ((j & 1) * 128)) + ((((int)threadIdx.x) & 3) * 32)) + ((((i_8 & 1) + (j & 1)) & 1) * 16)) + ((((j >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2))];
          }
        }
        for (int i_9 = 0; i_9 < 4; ++i_9) {
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + ((i_9 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_9 * 4)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 4)));
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + (((i_9 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_9 * 4)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 4) + 2)));
          }
        }
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    tfloat32_t A_local_1[16];
    tfloat32_t B_local_1[16];
    for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        tl::ptx_ldmatrix_x4((&(((tfloat32_t*)a_tile)[((((((ki_1 >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_10 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255))])), (&(A_local_1[(i_10 * 4)])));
      }
      for (int i_11 = 0; i_11 < 4; ++i_11) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          B_local_1[((i_11 * 4) + j_2)] = ((tfloat32_t*)b_tile)[((((((((((((int)threadIdx.x) >> 6) * 4096) + ((i_11 >> 1) * 2048)) + (ki_1 * 256)) + ((j_2 & 1) * 128)) + ((((int)threadIdx.x) & 3) * 32)) + ((((i_11 & 1) + (j_2 & 1)) & 1) * 16)) + ((((j_2 >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2))];
        }
      }
      for (int i_12 = 0; i_12 < 4; ++i_12) {
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + ((i_12 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_12 * 4)), reinterpret_cast<const unsigned*>(B_local_1 + (j_3 * 4)));
          tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + (((i_12 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_12 * 4)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_3 * 4) + 2)));
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    tfloat32_t A_local_2[16];
    tfloat32_t B_local_2[16];
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      for (int i_13 = 0; i_13 < 4; ++i_13) {
        tl::ptx_ldmatrix_x4((&(((tfloat32_t*)a_tile)[(((((((ki_2 >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_13 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255)) + 8192)])), (&(A_local_2[(i_13 * 4)])));
      }
      for (int i_14 = 0; i_14 < 4; ++i_14) {
        for (int j_4 = 0; j_4 < 4; ++j_4) {
          B_local_2[((i_14 * 4) + j_4)] = ((tfloat32_t*)b_tile)[(((((((((((((int)threadIdx.x) >> 6) * 4096) + ((i_14 >> 1) * 2048)) + (ki_2 * 256)) + ((j_4 & 1) * 128)) + ((((int)threadIdx.x) & 3) * 32)) + ((((i_14 & 1) + (j_4 & 1)) & 1) * 16)) + ((((j_4 >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2)) + 8192)];
        }
      }
      for (int i_15 = 0; i_15 < 4; ++i_15) {
        for (int j_5 = 0; j_5 < 4; ++j_5) {
          tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + ((i_15 * 32) + (j_5 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 4)), reinterpret_cast<const unsigned*>(B_local_2 + (j_5 * 4)));
          tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + (((i_15 * 32) + (j_5 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 4)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_5 * 4) + 2)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 64; ++i_16) {
    float2 __1;
    float2 v_ = *(float2*)(acc + (i_16 * 2));
    __1.x = (tfloat32_t)(v_.x);
    __1.y = (tfloat32_t)(v_.y);
    *(float2*)(c_local_cast + 0) = __1;
    *(float2*)(c + (((((((((((int)blockIdx.x) * 524288) + (((((int)threadIdx.x) & 63) >> 5) * 262144)) + ((i_16 >> 4) * 65536)) + ((i_16 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_16 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(c_local_cast + 0);
  }
}

