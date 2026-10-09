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

extern "C" __global__ void first_wave_kernel_kernel(const tfloat32_t* __restrict__ A, const tfloat32_t* __restrict__ B, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) first_wave_kernel_kernel(const tfloat32_t* __restrict__ A, const tfloat32_t* __restrict__ B, float* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  int start_iter = 0;
  int last_iter = 0;
  int end_iter = 0;
  int tile_id = 0;
  int group_id = 0;
  int first_pid_m = 0;
  int group_size_m = 0;
  int pid_m = 0;
  int pid_n = 0;
  float acc[128];
  start_iter = ((((int)blockIdx.x) * 205) + min(((int)blockIdx.x), 76));
  last_iter = (((((int)blockIdx.x) * 205) + min((((int)blockIdx.x) + 1), 76)) + 205);
  while (1) {
    if (!((start_iter < last_iter))) { break; }
    end_iter = min(((start_iter + 128) - (start_iter & 127)), last_iter);
    tile_id = (start_iter >> 7);
    group_id = ((tile_id / 1792) + ((tile_id % 1792) >> 31));
    first_pid_m = (group_id * 8);
    group_size_m = (64 - max(first_pid_m, 56));
    int rmod = (tile_id % group_size_m);
    pid_m = (first_pid_m + ((((0 <= group_size_m) && (0 <= rmod)) || ((group_size_m < 0) && (rmod <= 0))) ? rmod : (rmod + group_size_m)));
    int rmod_1 = (((1792 & ((tile_id % 1792) >> 31)) + (tile_id % 1792)) % group_size_m);
    int rdiv = (((1792 & ((tile_id % 1792) >> 31)) + (tile_id % 1792)) / group_size_m);
    pid_n = ((((0 <= group_size_m) && (0 <= rmod_1)) || ((group_size_m < 0) && (rmod_1 <= 0))) ? rdiv : (rdiv - 1));
    if (start_iter < end_iter) {
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        tl::cp_async_gs_conditional<16>((&(((tfloat32_t*)a_shared)[(((((i * 512) + ((((int)threadIdx.x) >> 3) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(A[(((((((int64_t)pid_m) * (int64_t)524288) + (((int64_t)i) * (int64_t)65536)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)4096)) + ((((int64_t)start_iter) & (int64_t)127) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)4))])), ((((pid_m < 64) && (0 <= pid_m)) && (pid_m < 64)) && (0 <= pid_m)));
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        tl::cp_async_gs_conditional<16>((&(((tfloat32_t*)b_shared)[((((((((((int)threadIdx.x) & 31) >> 3) * 1024) + (i_1 * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_1 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(B[((((((((int64_t)start_iter) & (int64_t)127) * (int64_t)917504) + (((int64_t)i_1) * (int64_t)114688)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)28672)) + (((int64_t)pid_n) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4))])), ((((pid_n < 224) && (0 <= pid_n)) && (pid_n < 224)) && (0 <= pid_n)));
      }
      tl::cp_async_commit();
    }
    if ((start_iter < end_iter) & (start_iter < end_iter)) {
      tl::cp_async_wait<0>();
    }
    __syncthreads();
    if ((start_iter + 1) < end_iter) {
      #pragma unroll
      for (int i_2 = 0; i_2 < 8; ++i_2) {
        tl::cp_async_gs_conditional<16>((&(((tfloat32_t*)a_shared)[((((((i_2 * 512) + ((((int)threadIdx.x) >> 3) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4)) + 4096)])), (&(A[(((((((int64_t)pid_m) * (int64_t)524288) + (((int64_t)i_2) * (int64_t)65536)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)4096)) + (((((int64_t)start_iter) + (int64_t)1) & (int64_t)127) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)4))])), ((((pid_m < 64) && (0 <= pid_m)) && (pid_m < 64)) && (0 <= pid_m)));
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 8; ++i_3) {
        tl::cp_async_gs_conditional<16>((&(((tfloat32_t*)b_shared)[(((((((((((int)threadIdx.x) & 31) >> 3) * 1024) + (i_3 * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 4)) + 4096)])), (&(B[(((((((((int64_t)start_iter) + (int64_t)1) & (int64_t)127) * (int64_t)917504) + (((int64_t)i_3) * (int64_t)114688)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)28672)) + (((int64_t)pid_n) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4))])), ((((pid_n < 224) && (0 <= pid_n)) && (pid_n < 224)) && (0 <= pid_n)));
      }
      tl::cp_async_commit();
    }
    if (((start_iter + 1) < end_iter) & ((start_iter + 1) < end_iter)) {
      tl::cp_async_wait<0>();
    }
    __syncthreads();
    for (int current_iter = start_iter; current_iter < (end_iter - 2); ++current_iter) {
      __syncthreads();
      #pragma unroll
      for (int i_4 = 0; i_4 < 8; ++i_4) {
        tl::cp_async_gs_conditional<16>((&(((tfloat32_t*)a_shared)[((((((((3 & ((((current_iter + 2) - start_iter) % 3) >> 31)) * 4096) + ((((current_iter + 2) - start_iter) % 3) * 4096)) + (i_4 * 512)) + ((((int)threadIdx.x) >> 3) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(A[(((((((int64_t)pid_m) * (int64_t)524288) + (((int64_t)i_4) * (int64_t)65536)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)4096)) + (((((int64_t)current_iter) + (int64_t)2) & (int64_t)127) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)4))])), ((((pid_m < 64) && (0 <= pid_m)) && (pid_m < 64)) && (0 <= pid_m)));
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 8; ++i_5) {
        tl::cp_async_gs_conditional<16>((&(((tfloat32_t*)b_shared)[(((((((((3 & ((((current_iter + 2) - start_iter) % 3) >> 31)) * 4096) + ((((current_iter + 2) - start_iter) % 3) * 4096)) + (((((int)threadIdx.x) & 31) >> 3) * 1024)) + (i_5 * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_5 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(B[(((((((((int64_t)current_iter) + (int64_t)2) & (int64_t)127) * (int64_t)917504) + (((int64_t)i_5) * (int64_t)114688)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)28672)) + (((int64_t)pid_n) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4))])), ((((pid_n < 224) && (0 <= pid_n)) && (pid_n < 224)) && (0 <= pid_n)));
      }
      tl::cp_async_commit();
      if (current_iter == start_iter) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 32; ++i_6) {
          float broadcast_var = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(acc + (i_6 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
        }
      }
      tl::cp_async_wait<0>();
      __syncthreads();
      {
        tfloat32_t A_local[16];
        tfloat32_t B_local[16];
        for (int ki = 0; ki < 4; ++ki) {
          for (int i_7 = 0; i_7 < 4; ++i_7) {
            tl::ptx_ldmatrix_x4((&(((tfloat32_t*)a_shared)[(((((((3 & (((current_iter - start_iter) % 3) >> 31)) * 4096) + (((current_iter - start_iter) % 3) * 4096)) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_7 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255))])), (&(A_local[(i_7 * 4)])));
          }
          for (int i_8 = 0; i_8 < 4; ++i_8) {
            for (int j = 0; j < 4; ++j) {
              B_local[((i_8 * 4) + j)] = ((tfloat32_t*)b_shared)[((((((((((((3 & (((current_iter - start_iter) % 3) >> 31)) * 4096) + (((current_iter - start_iter) % 3) * 4096)) + ((((int)threadIdx.x) >> 6) * 2048)) + ((i_8 >> 1) * 1024)) + (ki * 256)) + ((j & 1) * 128)) + ((((int)threadIdx.x) & 3) * 32)) + ((((i_8 & 1) + (j & 1)) & 1) * 16)) + ((((j >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2))];
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
    if ((start_iter + 2) <= end_iter) {
      if ((end_iter - 2) == start_iter) {
        #pragma unroll
        for (int i_10 = 0; i_10 < 32; ++i_10) {
          float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(acc + (i_10 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
        }
      }
    }
    if ((bool)0 & (bool)0) {
      tl::cp_async_wait<0>();
    }
    __syncthreads();
    if ((start_iter + 2) <= end_iter) {
      {
        tfloat32_t A_local_1[16];
        tfloat32_t B_local_1[16];
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          for (int i_11 = 0; i_11 < 4; ++i_11) {
            tl::ptx_ldmatrix_x4((&(((tfloat32_t*)a_shared)[(((((((3 & ((((end_iter + 1) - start_iter) % 3) >> 31)) * 4096) + ((((end_iter + 1) - start_iter) % 3) * 4096)) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_11 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255))])), (&(A_local_1[(i_11 * 4)])));
          }
          for (int i_12 = 0; i_12 < 4; ++i_12) {
            for (int j_2 = 0; j_2 < 4; ++j_2) {
              B_local_1[((i_12 * 4) + j_2)] = ((tfloat32_t*)b_shared)[((((((((((((3 & ((((end_iter + 1) - start_iter) % 3) >> 31)) * 4096) + ((((end_iter + 1) - start_iter) % 3) * 4096)) + ((((int)threadIdx.x) >> 6) * 2048)) + ((i_12 >> 1) * 1024)) + (ki_1 * 256)) + ((j_2 & 1) * 128)) + ((((int)threadIdx.x) & 3) * 32)) + ((((i_12 & 1) + (j_2 & 1)) & 1) * 16)) + ((((j_2 >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2))];
            }
          }
          for (int i_13 = 0; i_13 < 4; ++i_13) {
            for (int j_3 = 0; j_3 < 4; ++j_3) {
              tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + ((i_13 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_13 * 4)), reinterpret_cast<const unsigned*>(B_local_1 + (j_3 * 4)));
              tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + (((i_13 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_13 * 4)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_3 * 4) + 2)));
            }
          }
        }
      }
    }
    if (start_iter < end_iter) {
      if ((end_iter - 1) == start_iter) {
        #pragma unroll
        for (int i_14 = 0; i_14 < 32; ++i_14) {
          float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(acc + (i_14 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
        }
      }
    }
    if ((bool)0 & (bool)0) {
      tl::cp_async_wait<0>();
    }
    __syncthreads();
    if (start_iter < end_iter) {
      {
        tfloat32_t A_local_2[16];
        tfloat32_t B_local_2[16];
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          for (int i_15 = 0; i_15 < 4; ++i_15) {
            tl::ptx_ldmatrix_x4((&(((tfloat32_t*)a_shared)[(((((((3 & ((((end_iter + 2) - start_iter) % 3) >> 31)) * 4096) + ((((end_iter + 2) - start_iter) % 3) * 4096)) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_15 * 512)) + (((((int)threadIdx.x) & 15) >> 3) * 256)) + ((((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) & 255))])), (&(A_local_2[(i_15 * 4)])));
          }
          for (int i_16 = 0; i_16 < 4; ++i_16) {
            for (int j_4 = 0; j_4 < 4; ++j_4) {
              B_local_2[((i_16 * 4) + j_4)] = ((tfloat32_t*)b_shared)[((((((((((((3 & ((((end_iter + 2) - start_iter) % 3) >> 31)) * 4096) + ((((end_iter + 2) - start_iter) % 3) * 4096)) + ((((int)threadIdx.x) >> 6) * 2048)) + ((i_16 >> 1) * 1024)) + (ki_2 * 256)) + ((j_4 & 1) * 128)) + ((((int)threadIdx.x) & 3) * 32)) + ((((i_16 & 1) + (j_4 & 1)) & 1) * 16)) + ((((j_4 >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2))];
            }
          }
          for (int i_17 = 0; i_17 < 4; ++i_17) {
            for (int j_5 = 0; j_5 < 4; ++j_5) {
              tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + ((i_17 * 32) + (j_5 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_17 * 4)), reinterpret_cast<const unsigned*>(B_local_2 + (j_5 * 4)));
              tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + (((i_17 * 32) + (j_5 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_17 * 4)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_5 * 4) + 2)));
            }
          }
        }
      }
    }
    if (0 <= pid_n) {
      #pragma unroll
      for (int i_18 = 0; i_18 < 64; ++i_18) {
        if (pid_n < 224) {
          if (pid_m < 64) {
            AtomicAddx2((&(C[(((((((((((int64_t)pid_m) * (int64_t)3670016) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)1835008)) + ((((int64_t)i_18) >> (int64_t)4) * (int64_t)458752)) + ((((int64_t)i_18) & (int64_t)1) * (int64_t)229376)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)28672)) + (((int64_t)pid_n) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + (((((int64_t)i_18) & (int64_t)15) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))])), *(float2*)(acc + (i_18 * 2)));
          }
        }
      }
    }
    __syncthreads();
    start_iter = end_iter;
  }
}

