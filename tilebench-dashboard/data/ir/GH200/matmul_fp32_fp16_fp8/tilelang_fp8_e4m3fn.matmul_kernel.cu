#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/cuda_fp8.h>
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

extern "C" __global__ void matmul_kernel_kernel(const fp8_e4_t* __restrict__ a, const fp8_e4_t* __restrict__ b, fp8_e4_t* __restrict__ c);
extern "C" __global__ void __launch_bounds__(128, 1) matmul_kernel_kernel(const fp8_e4_t* __restrict__ a, const fp8_e4_t* __restrict__ b, fp8_e4_t* __restrict__ c) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* b_tile = ((void*)((char*)buf_dyn_shmem + 49152));
  float acc[128];
  fp8_e4_t c_local_cast[2];
  const dim3 blockIdx = tl::rasterization2DRow<8>();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)a_tile)[(((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(a[((((((int)blockIdx.x) * 2621440) + (i_1 * 327680)) + ((((int)threadIdx.x) >> 3) * 20480)) + ((((int)threadIdx.x) & 7) * 16))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)b_tile)[(((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(b[((((i_2 * 65536) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 7) * 16))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)a_tile)[((((((i_3 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384)])), (&(a[(((((((int)blockIdx.x) * 2621440) + (i_3 * 327680)) + ((((int)threadIdx.x) >> 3) * 20480)) + ((((int)threadIdx.x) & 7) * 16)) + 128)])));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)b_tile)[((((((i_4 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384)])), (&(b[(((((i_4 * 65536) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + 524288)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 158; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      tl::cp_async_gs<16>((&(((fp8_e4_t*)a_tile)[((((((((k + 2) % 3) * 16384) + (i_5 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(a[((((((((int)blockIdx.x) * 2621440) + (i_5 * 327680)) + ((((int)threadIdx.x) >> 3) * 20480)) + (k * 128)) + ((((int)threadIdx.x) & 7) * 16)) + 256)])));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      tl::cp_async_gs<16>((&(((fp8_e4_t*)b_tile)[((((((((k + 2) % 3) * 16384) + (i_6 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(b[((((((k * 524288) + (i_6 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + 1048576)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      fp8_e4_t A_local[64];
      fp8_e4_t B_local[64];
      for (int ki = 0; ki < 4; ++ki) {
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)a_tile)[((((((k % 3) * 16384) + (((((int)threadIdx.x) & 63) >> 5) * 8192)) + (i_7 * 2048)) + (((((int)threadIdx.x) & 15) >> 3) * 1024)) + ((((((((int)threadIdx.x) & 15) * 128) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) & 1023))])), (&(A_local[(i_7 * 16)])));
        }
        for (int i_8 = 0; i_8 < 4; ++i_8) {
          for (int j = 0; j < 16; ++j) {
            B_local[((i_8 * 16) + j)] = ((fp8_e4_t*)b_tile)[(((((((((((k % 3) * 16384) + (ki * 4096)) + (((j & 7) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 512)) + ((j & 3) * 128)) + ((((((int)threadIdx.x) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 64)) + ((((i_8 >> 1) + ((j & 3) >> 1)) & 1) * 32)) + ((((i_8 & 1) + (j & 1)) & 1) * 16)) + ((j >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          }
        }
        for (int i_9 = 0; i_9 < 4; ++i_9) {
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(acc + ((i_9 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_9 * 16)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 16)));
            tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(acc + (((i_9 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_9 * 16)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 16) + 8)));
          }
        }
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    fp8_e4_t A_local_1[64];
    fp8_e4_t B_local_1[64];
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)a_tile)[(((((((((int)threadIdx.x) & 63) >> 5) * 8192) + (i_10 * 2048)) + (((((int)threadIdx.x) & 15) >> 3) * 1024)) + ((((((((int)threadIdx.x) & 15) * 128) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) & 1023)) + 32768)])), (&(A_local_1[(i_10 * 16)])));
      }
      for (int i_11 = 0; i_11 < 4; ++i_11) {
        for (int j_2 = 0; j_2 < 16; ++j_2) {
          B_local_1[((i_11 * 16) + j_2)] = ((fp8_e4_t*)b_tile)[((((((((((ki_1 * 4096) + (((j_2 & 7) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 512)) + ((j_2 & 3) * 128)) + ((((((int)threadIdx.x) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 64)) + ((((i_11 >> 1) + ((j_2 & 3) >> 1)) & 1) * 32)) + ((((i_11 & 1) + (j_2 & 1)) & 1) * 16)) + ((j_2 >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 32768)];
        }
      }
      for (int i_12 = 0; i_12 < 4; ++i_12) {
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(acc + ((i_12 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_12 * 16)), reinterpret_cast<const unsigned*>(B_local_1 + (j_3 * 16)));
          tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(acc + (((i_12 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_12 * 16)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_3 * 16) + 8)));
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    fp8_e4_t A_local_2[64];
    fp8_e4_t B_local_2[64];
    for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
      for (int i_13 = 0; i_13 < 4; ++i_13) {
        tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)a_tile)[((((((((int)threadIdx.x) & 63) >> 5) * 8192) + (i_13 * 2048)) + (((((int)threadIdx.x) & 15) >> 3) * 1024)) + ((((((((int)threadIdx.x) & 15) * 128) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) & 1023))])), (&(A_local_2[(i_13 * 16)])));
      }
      for (int i_14 = 0; i_14 < 4; ++i_14) {
        for (int j_4 = 0; j_4 < 16; ++j_4) {
          B_local_2[((i_14 * 16) + j_4)] = ((fp8_e4_t*)b_tile)[(((((((((ki_2 * 4096) + (((j_4 & 7) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 512)) + ((j_4 & 3) * 128)) + ((((((int)threadIdx.x) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 64)) + ((((i_14 >> 1) + ((j_4 & 3) >> 1)) & 1) * 32)) + ((((i_14 & 1) + (j_4 & 1)) & 1) * 16)) + ((j_4 >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
        }
      }
      for (int i_15 = 0; i_15 < 4; ++i_15) {
        for (int j_5 = 0; j_5 < 4; ++j_5) {
          tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(acc + ((i_15 * 32) + (j_5 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 16)), reinterpret_cast<const unsigned*>(B_local_2 + (j_5 * 16)));
          tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(acc + (((i_15 * 32) + (j_5 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 16)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_5 * 16) + 8)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 64; ++i_16) {
    fp8_e4_2_t __1;
    float2 v_ = *(float2*)(acc + (i_16 * 2));
    (reinterpret_cast<__nv_fp8x2_storage_t*>(&__1))[0] = __nv_cvt_float2_to_fp8x2(((float2*)(&v_))[0], __NV_SATFINITE, __NV_E4M3);
    *(fp8_e4_2_t*)(c_local_cast + 0) = __1;
    *(fp8_e4_2_t*)(c + (((((((((((int)blockIdx.x) * 524288) + (((((int)threadIdx.x) & 63) >> 5) * 262144)) + ((i_16 >> 4) * 65536)) + ((i_16 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_16 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(fp8_e4_2_t*)(c_local_cast + 0);
  }
}

