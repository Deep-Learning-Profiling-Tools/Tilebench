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

extern "C" __global__ void matmul_kernel_kernel(const signed char* __restrict__ a, const uchar* __restrict__ b, int* __restrict__ c);
extern "C" __global__ void __launch_bounds__(128, 1) matmul_kernel_kernel(const signed char* __restrict__ a, const uchar* __restrict__ b, int* __restrict__ c) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* b_packed_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* b_unpacked_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  int acc[128];
  uchar b_packed_local[32];
  signed char b_unpacked_local[32];
  const dim3 blockIdx = tl::rasterization2DRow<8>();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    int broadcast_var = 0;
    *(int4*)(acc + (i * 4)) = make_int4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    tl::cp_async_gs<16>((&(((uchar*)b_packed_shared)[((i_1 * 2048) + (((int)threadIdx.x) * 16))])), (&(b[((((i_1 * 65536) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 3) * 16))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    tl::cp_async_gs<16>((&(((uchar*)b_packed_shared)[(((i_2 * 2048) + (((int)threadIdx.x) * 16)) + 4096)])), (&(b[(((((i_2 * 65536) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + 131072)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    tl::cp_async_gs<16>((&(((uchar*)b_packed_shared)[(((i_3 * 2048) + (((int)threadIdx.x) * 16)) + 8192)])), (&(b[(((((i_3 * 65536) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + 262144)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 2; ++i_4) {
    tl::cp_async_gs<16>((&(((uchar*)b_packed_shared)[(((i_4 * 2048) + (((int)threadIdx.x) * 16)) + 12288)])), (&(b[(((((i_4 * 65536) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + 393216)])));
  }
  tl::cp_async_commit();
  for (int kb_tile = 0; kb_tile < 76; ++kb_tile) {
    tl::cp_async_wait<3>();
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      *(uint4*)(b_packed_local + (i_5 * 16)) = *(uint4*)(((uchar*)b_packed_shared) + ((((kb_tile & 3) * 4096) + (i_5 * 2048)) + (((int)threadIdx.x) * 16)));
    }
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      tl::cp_async_gs<16>((&(((uchar*)b_packed_shared)[((((kb_tile & 3) * 4096) + (i_6 * 2048)) + (((int)threadIdx.x) * 16))])), (&(b[((((((kb_tile * 131072) + (i_6 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + 524288)])));
    }
    tl::cp_async_commit();
    __syncthreads();
    for (int field_i = 0; field_i < 4; ++field_i) {
      #pragma unroll
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        *(int4*)(((signed char*)a_shared) + ((((i_7 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(a + ((((((((int)blockIdx.x) * 5242880) + (i_7 * 655360)) + ((((int)threadIdx.x) >> 2) * 20480)) + (field_i * 5120)) + (kb_tile * 64)) + ((((int)threadIdx.x) & 3) * 16)));
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 32; ++i_8) {
        int field = ((((int)b_packed_local[i_8]) >> (field_i * 2)) & 3);
        b_unpacked_local[i_8] = (((signed char)field) - (signed char)1);
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        *(int4*)(((signed char*)b_unpacked_shared) + ((((i_9 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(b_unpacked_local + (i_9 * 16));
      }
      __syncthreads();
      {
        signed char A_local[128];
        signed char B_local[32];
        for (int ki = 0; ki < 2; ++ki) {
          for (int i_10 = 0; i_10 < 8; ++i_10) {
            tl::ptx_ldmatrix_x4((&(((signed char*)a_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 8192) + (i_10 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16))])), (&(A_local[(i_10 * 16)])));
          }
          for (int i_11 = 0; i_11 < 2; ++i_11) {
            for (int j = 0; j < 16; ++j) {
              B_local[((i_11 * 16) + j)] = ((signed char*)b_unpacked_shared)[((((((((ki * 2048) + (((j & 7) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + ((j & 3) * 64)) + ((((((int)threadIdx.x) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 32)) + (((((j & 3) >> 1) + i_11) & 1) * 16)) + ((j >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
            }
          }
          for (int i_12 = 0; i_12 < 8; ++i_12) {
            for (int j_1 = 0; j_1 < 2; ++j_1) {
              tl::mma_sync<tl::DataType::kInt8, tl::DataType::kInt8, tl::DataType::kInt32, 16, 8, 32, false, true>(reinterpret_cast<unsigned*>(acc + ((i_12 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_12 * 16)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 16)));
              tl::mma_sync<tl::DataType::kInt8, tl::DataType::kInt8, tl::DataType::kInt32, 16, 8, 32, false, true>(reinterpret_cast<unsigned*>(acc + (((i_12 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_12 * 16)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 16) + 8)));
            }
          }
        }
      }
      __syncthreads();
    }
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  #pragma unroll
  for (int i_13 = 0; i_13 < 2; ++i_13) {
    *(uint4*)(b_packed_local + (i_13 * 16)) = *(uint4*)(((uchar*)b_packed_shared) + ((i_13 * 2048) + (((int)threadIdx.x) * 16)));
  }
  for (int field_i_1 = 0; field_i_1 < 4; ++field_i_1) {
    #pragma unroll
    for (int i_14 = 0; i_14 < 8; ++i_14) {
      *(int4*)(((signed char*)a_shared) + ((((i_14 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(a + ((((((((int)blockIdx.x) * 5242880) + (i_14 * 655360)) + ((((int)threadIdx.x) >> 2) * 20480)) + (field_i_1 * 5120)) + ((((int)threadIdx.x) & 3) * 16)) + 4864));
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 32; ++i_15) {
      int field_1 = ((((int)b_packed_local[i_15]) >> (field_i_1 * 2)) & 3);
      b_unpacked_local[i_15] = (((signed char)field_1) - (signed char)1);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      *(int4*)(((signed char*)b_unpacked_shared) + ((((i_16 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(b_unpacked_local + (i_16 * 16));
    }
    __syncthreads();
    {
      signed char A_local_1[128];
      signed char B_local_1[32];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        for (int i_17 = 0; i_17 < 8; ++i_17) {
          tl::ptx_ldmatrix_x4((&(((signed char*)a_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 8192) + (i_17 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16))])), (&(A_local_1[(i_17 * 16)])));
        }
        for (int i_18 = 0; i_18 < 2; ++i_18) {
          for (int j_2 = 0; j_2 < 16; ++j_2) {
            B_local_1[((i_18 * 16) + j_2)] = ((signed char*)b_unpacked_shared)[((((((((ki_1 * 2048) + (((j_2 & 7) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + ((j_2 & 3) * 64)) + ((((((int)threadIdx.x) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 32)) + (((((j_2 & 3) >> 1) + i_18) & 1) * 16)) + ((j_2 >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          }
        }
        for (int i_19 = 0; i_19 < 8; ++i_19) {
          for (int j_3 = 0; j_3 < 2; ++j_3) {
            tl::mma_sync<tl::DataType::kInt8, tl::DataType::kInt8, tl::DataType::kInt32, 16, 8, 32, false, true>(reinterpret_cast<unsigned*>(acc + ((i_19 * 16) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_19 * 16)), reinterpret_cast<const unsigned*>(B_local_1 + (j_3 * 16)));
            tl::mma_sync<tl::DataType::kInt8, tl::DataType::kInt8, tl::DataType::kInt32, 16, 8, 32, false, true>(reinterpret_cast<unsigned*>(acc + (((i_19 * 16) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_19 * 16)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_3 * 16) + 8)));
          }
        }
      }
    }
    __syncthreads();
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  #pragma unroll
  for (int i_20 = 0; i_20 < 2; ++i_20) {
    *(uint4*)(b_packed_local + (i_20 * 16)) = *(uint4*)(((uchar*)b_packed_shared) + (((i_20 * 2048) + (((int)threadIdx.x) * 16)) + 4096));
  }
  for (int field_i_2 = 0; field_i_2 < 4; ++field_i_2) {
    #pragma unroll
    for (int i_21 = 0; i_21 < 8; ++i_21) {
      *(int4*)(((signed char*)a_shared) + ((((i_21 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(a + ((((((((int)blockIdx.x) * 5242880) + (i_21 * 655360)) + ((((int)threadIdx.x) >> 2) * 20480)) + (field_i_2 * 5120)) + ((((int)threadIdx.x) & 3) * 16)) + 4928));
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 32; ++i_22) {
      int field_2 = ((((int)b_packed_local[i_22]) >> (field_i_2 * 2)) & 3);
      b_unpacked_local[i_22] = (((signed char)field_2) - (signed char)1);
    }
    #pragma unroll
    for (int i_23 = 0; i_23 < 2; ++i_23) {
      *(int4*)(((signed char*)b_unpacked_shared) + ((((i_23 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(b_unpacked_local + (i_23 * 16));
    }
    __syncthreads();
    {
      signed char A_local_2[128];
      signed char B_local_2[32];
      for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
        for (int i_24 = 0; i_24 < 8; ++i_24) {
          tl::ptx_ldmatrix_x4((&(((signed char*)a_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 8192) + (i_24 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16))])), (&(A_local_2[(i_24 * 16)])));
        }
        for (int i_25 = 0; i_25 < 2; ++i_25) {
          for (int j_4 = 0; j_4 < 16; ++j_4) {
            B_local_2[((i_25 * 16) + j_4)] = ((signed char*)b_unpacked_shared)[((((((((ki_2 * 2048) + (((j_4 & 7) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + ((j_4 & 3) * 64)) + ((((((int)threadIdx.x) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 32)) + (((((j_4 & 3) >> 1) + i_25) & 1) * 16)) + ((j_4 >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          }
        }
        for (int i_26 = 0; i_26 < 8; ++i_26) {
          for (int j_5 = 0; j_5 < 2; ++j_5) {
            tl::mma_sync<tl::DataType::kInt8, tl::DataType::kInt8, tl::DataType::kInt32, 16, 8, 32, false, true>(reinterpret_cast<unsigned*>(acc + ((i_26 * 16) + (j_5 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_26 * 16)), reinterpret_cast<const unsigned*>(B_local_2 + (j_5 * 16)));
            tl::mma_sync<tl::DataType::kInt8, tl::DataType::kInt8, tl::DataType::kInt32, 16, 8, 32, false, true>(reinterpret_cast<unsigned*>(acc + (((i_26 * 16) + (j_5 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_26 * 16)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_5 * 16) + 8)));
          }
        }
      }
    }
    __syncthreads();
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int i_27 = 0; i_27 < 2; ++i_27) {
    *(uint4*)(b_packed_local + (i_27 * 16)) = *(uint4*)(((uchar*)b_packed_shared) + (((i_27 * 2048) + (((int)threadIdx.x) * 16)) + 8192));
  }
  for (int field_i_3 = 0; field_i_3 < 4; ++field_i_3) {
    #pragma unroll
    for (int i_28 = 0; i_28 < 8; ++i_28) {
      *(int4*)(((signed char*)a_shared) + ((((i_28 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(a + ((((((((int)blockIdx.x) * 5242880) + (i_28 * 655360)) + ((((int)threadIdx.x) >> 2) * 20480)) + (field_i_3 * 5120)) + ((((int)threadIdx.x) & 3) * 16)) + 4992));
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 32; ++i_29) {
      int field_3 = ((((int)b_packed_local[i_29]) >> (field_i_3 * 2)) & 3);
      b_unpacked_local[i_29] = (((signed char)field_3) - (signed char)1);
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 2; ++i_30) {
      *(int4*)(((signed char*)b_unpacked_shared) + ((((i_30 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(b_unpacked_local + (i_30 * 16));
    }
    __syncthreads();
    {
      signed char A_local_3[128];
      signed char B_local_3[32];
      for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
        for (int i_31 = 0; i_31 < 8; ++i_31) {
          tl::ptx_ldmatrix_x4((&(((signed char*)a_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 8192) + (i_31 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16))])), (&(A_local_3[(i_31 * 16)])));
        }
        for (int i_32 = 0; i_32 < 2; ++i_32) {
          for (int j_6 = 0; j_6 < 16; ++j_6) {
            B_local_3[((i_32 * 16) + j_6)] = ((signed char*)b_unpacked_shared)[((((((((ki_3 * 2048) + (((j_6 & 7) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + ((j_6 & 3) * 64)) + ((((((int)threadIdx.x) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 32)) + (((((j_6 & 3) >> 1) + i_32) & 1) * 16)) + ((j_6 >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          }
        }
        for (int i_33 = 0; i_33 < 8; ++i_33) {
          for (int j_7 = 0; j_7 < 2; ++j_7) {
            tl::mma_sync<tl::DataType::kInt8, tl::DataType::kInt8, tl::DataType::kInt32, 16, 8, 32, false, true>(reinterpret_cast<unsigned*>(acc + ((i_33 * 16) + (j_7 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_33 * 16)), reinterpret_cast<const unsigned*>(B_local_3 + (j_7 * 16)));
            tl::mma_sync<tl::DataType::kInt8, tl::DataType::kInt8, tl::DataType::kInt32, 16, 8, 32, false, true>(reinterpret_cast<unsigned*>(acc + (((i_33 * 16) + (j_7 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_33 * 16)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_7 * 16) + 8)));
          }
        }
      }
    }
    __syncthreads();
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  #pragma unroll
  for (int i_34 = 0; i_34 < 2; ++i_34) {
    *(uint4*)(b_packed_local + (i_34 * 16)) = *(uint4*)(((uchar*)b_packed_shared) + (((i_34 * 2048) + (((int)threadIdx.x) * 16)) + 12288));
  }
  for (int field_i_4 = 0; field_i_4 < 4; ++field_i_4) {
    #pragma unroll
    for (int i_35 = 0; i_35 < 8; ++i_35) {
      *(int4*)(((signed char*)a_shared) + ((((i_35 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(a + ((((((((int)blockIdx.x) * 5242880) + (i_35 * 655360)) + ((((int)threadIdx.x) >> 2) * 20480)) + (field_i_4 * 5120)) + ((((int)threadIdx.x) & 3) * 16)) + 5056));
    }
    #pragma unroll
    for (int i_36 = 0; i_36 < 32; ++i_36) {
      int field_4 = ((((int)b_packed_local[i_36]) >> (field_i_4 * 2)) & 3);
      b_unpacked_local[i_36] = (((signed char)field_4) - (signed char)1);
    }
    #pragma unroll
    for (int i_37 = 0; i_37 < 2; ++i_37) {
      *(int4*)(((signed char*)b_unpacked_shared) + ((((i_37 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(b_unpacked_local + (i_37 * 16));
    }
    __syncthreads();
    {
      signed char A_local_4[128];
      signed char B_local_4[32];
      for (int ki_4 = 0; ki_4 < 2; ++ki_4) {
        for (int i_38 = 0; i_38 < 8; ++i_38) {
          tl::ptx_ldmatrix_x4((&(((signed char*)a_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 8192) + (i_38 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_4) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16))])), (&(A_local_4[(i_38 * 16)])));
        }
        for (int i_39 = 0; i_39 < 2; ++i_39) {
          for (int j_8 = 0; j_8 < 16; ++j_8) {
            B_local_4[((i_39 * 16) + j_8)] = ((signed char*)b_unpacked_shared)[((((((((ki_4 * 2048) + (((j_8 & 7) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + ((j_8 & 3) * 64)) + ((((((int)threadIdx.x) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 32)) + (((((j_8 & 3) >> 1) + i_39) & 1) * 16)) + ((j_8 >> 3) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          }
        }
        for (int i_40 = 0; i_40 < 8; ++i_40) {
          for (int j_9 = 0; j_9 < 2; ++j_9) {
            tl::mma_sync<tl::DataType::kInt8, tl::DataType::kInt8, tl::DataType::kInt32, 16, 8, 32, false, true>(reinterpret_cast<unsigned*>(acc + ((i_40 * 16) + (j_9 * 8))), reinterpret_cast<const unsigned*>(A_local_4 + (i_40 * 16)), reinterpret_cast<const unsigned*>(B_local_4 + (j_9 * 16)));
            tl::mma_sync<tl::DataType::kInt8, tl::DataType::kInt8, tl::DataType::kInt32, 16, 8, 32, false, true>(reinterpret_cast<unsigned*>(acc + (((i_40 * 16) + (j_9 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + (i_40 * 16)), reinterpret_cast<const unsigned*>(B_local_4 + ((j_9 * 16) + 8)));
          }
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i_41 = 0; i_41 < 64; ++i_41) {
    *(int2*)(c + (((((((((((int)blockIdx.x) * 524288) + (((((int)threadIdx.x) & 63) >> 5) * 262144)) + ((i_41 >> 3) * 32768)) + ((i_41 & 1) * 16384)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_41 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(int2*)(acc + (i_41 * 2));
  }
}

