#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/tcgen05mma.h>
#include <tl_templates/cuda/tcgen_05.h>
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
extern "C" __global__ void __launch_bounds__(512, 1) matmul_kernel_kernel(const signed char* __restrict__ a, const uchar* __restrict__ b, int* __restrict__ c) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* b_packed_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  void* b_unpacked_shared = ((void*)((char*)buf_dyn_shmem + 40960));
  __shared__ __align__(16) uint64_t mbar_mem[1];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  __shared__ __align__(16) uint acc_tmem[1];
  uchar b_packed_local[16];
  signed char b_unpacked_local[16];
  int acc[64];
  if (tl::tl_shuffle_elect<0>()) {
    mbar[0].init(1);
  }
  tl::fence_barrier_init();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(acc_tmem[0])), 256);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  const dim3 blockIdx = tl::rasterization2DRow<8>();
  tl::cp_async_gs<16>((&(((uchar*)b_packed_shared)[(((int)threadIdx.x) * 16)])), (&(b[((((((int)threadIdx.x) >> 3) * 2048) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 7) * 16))])));
  tl::cp_async_commit();
  tl::cp_async_gs<16>((&(((uchar*)b_packed_shared)[((((int)threadIdx.x) * 16) + 8192)])), (&(b[(((((((int)threadIdx.x) >> 3) * 2048) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + 131072)])));
  tl::cp_async_commit();
  tl::cp_async_gs<16>((&(((uchar*)b_packed_shared)[((((int)threadIdx.x) * 16) + 16384)])), (&(b[(((((((int)threadIdx.x) >> 3) * 2048) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + 262144)])));
  tl::cp_async_commit();
  for (int kb_tile = 0; kb_tile < 77; ++kb_tile) {
    tl::cp_async_wait<2>();
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    *(uint4*)(b_packed_local + 0) = *(uint4*)(((uchar*)b_packed_shared) + (((kb_tile % 3) * 8192) + (((int)threadIdx.x) * 16)));
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    tl::cp_async_gs<16>((&(((uchar*)b_packed_shared)[(((kb_tile % 3) * 8192) + (((int)threadIdx.x) * 16))])), (&(b[(((((kb_tile * 131072) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + 393216)])));
    tl::cp_async_commit();
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    for (int field_i = 0; field_i < 4; ++field_i) {
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        *(int4*)(((signed char*)a_shared) + ((((i * 8192) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(a + ((((((((int)blockIdx.x) * 5242880) + (i * 2621440)) + ((((int)threadIdx.x) >> 2) * 20480)) + (field_i * 5120)) + (kb_tile * 64)) + ((((int)threadIdx.x) & 3) * 16)));
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        int field = ((((int)b_packed_local[i_1]) >> (field_i * 2)) & 3);
        b_unpacked_local[i_1] = (((signed char)field) - (signed char)1);
      }
      *(int4*)(((signed char*)b_unpacked_shared) + (((((((int)threadIdx.x) >> 3) * 128) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(b_unpacked_local + 0);
      tl::tcgen05_before_thread_sync();
      __syncthreads();
      tl::tcgen05_after_thread_sync();
      {
        tl::Tcgen05SMemDescriptor desc_a;
        tl::Tcgen05SMemDescriptor desc_b;
        if ((((int)threadIdx.x) >> 5) == 0) {
          tl::initialize_tcgen05_descriptor(desc_a, (&(((signed char*)a_shared)[0])), 1, 32, 0, 0, 4);
          tl::initialize_tcgen05_descriptor(desc_b, (&(((signed char*)b_unpacked_shared)[0])), 0, 64, 0, 0, 2);
          tl::fence_proxy_async();
          #pragma unroll
          for (int i_2 = 0; i_2 < 2; ++i_2) {
            #pragma unroll
            for (int ki = 0; ki < 2; ++ki) {
              tl::tcgen05mma_ws_ss<tl::DataType::kInt8>(uint64_t(desc_a + ((i_2 * 8192) + (ki * 32))), uint64_t(desc_b + (ki * 4096)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + (i_2 * 128), ((0 < ki) ? 1 : (((kb_tile + field_i) == 0) ? 0 : 1)), static_cast<uint32_t>(136381600), 0, 0, 0, 0);
            }
          }
          tl::tcgen05_mma_arrive((&(mbar[0])));
        }
        mbar[0].wait(((kb_tile % 6) / 3));
      }
      tl::tcgen05_before_thread_sync();
      __syncthreads();
      tl::tcgen05_after_thread_sync();
    }
  }
  tl::cp_async_wait<2>();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  *(uint4*)(b_packed_local + 0) = *(uint4*)(((uchar*)b_packed_shared) + ((((int)threadIdx.x) * 16) + 16384));
  for (int field_i_1 = 0; field_i_1 < 4; ++field_i_1) {
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      *(int4*)(((signed char*)a_shared) + ((((i_3 * 8192) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(a + ((((((((int)blockIdx.x) * 5242880) + (i_3 * 2621440)) + ((((int)threadIdx.x) >> 2) * 20480)) + (field_i_1 * 5120)) + ((((int)threadIdx.x) & 3) * 16)) + 4928));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      int field_1 = ((((int)b_packed_local[i_4]) >> (field_i_1 * 2)) & 3);
      b_unpacked_local[i_4] = (((signed char)field_1) - (signed char)1);
    }
    *(int4*)(((signed char*)b_unpacked_shared) + (((((((int)threadIdx.x) >> 3) * 128) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(b_unpacked_local + 0);
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    {
      tl::Tcgen05SMemDescriptor desc_a_1;
      tl::Tcgen05SMemDescriptor desc_b_1;
      if ((((int)threadIdx.x) >> 5) == 0) {
        tl::initialize_tcgen05_descriptor(desc_a_1, (&(((signed char*)a_shared)[0])), 1, 32, 0, 0, 4);
        tl::initialize_tcgen05_descriptor(desc_b_1, (&(((signed char*)b_unpacked_shared)[0])), 0, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int i_5 = 0; i_5 < 2; ++i_5) {
          #pragma unroll
          for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
            tl::tcgen05mma_ws_ss<tl::DataType::kInt8>(uint64_t(desc_a_1 + ((i_5 * 8192) + (ki_1 * 32))), uint64_t(desc_b_1 + (ki_1 * 4096)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + (i_5 * 128), 1, static_cast<uint32_t>(136381600), 0, 0, 0, 0);
          }
        }
        tl::tcgen05_mma_arrive((&(mbar[0])));
      }
      mbar[0].wait(1);
    }
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
  }
  tl::cp_async_wait<1>();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  *(uint4*)(b_packed_local + 0) = *(uint4*)(((uchar*)b_packed_shared) + (((int)threadIdx.x) * 16));
  for (int field_i_2 = 0; field_i_2 < 4; ++field_i_2) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      *(int4*)(((signed char*)a_shared) + ((((i_6 * 8192) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(a + ((((((((int)blockIdx.x) * 5242880) + (i_6 * 2621440)) + ((((int)threadIdx.x) >> 2) * 20480)) + (field_i_2 * 5120)) + ((((int)threadIdx.x) & 3) * 16)) + 4992));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 16; ++i_7) {
      int field_2 = ((((int)b_packed_local[i_7]) >> (field_i_2 * 2)) & 3);
      b_unpacked_local[i_7] = (((signed char)field_2) - (signed char)1);
    }
    *(int4*)(((signed char*)b_unpacked_shared) + (((((((int)threadIdx.x) >> 3) * 128) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(b_unpacked_local + 0);
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    {
      tl::Tcgen05SMemDescriptor desc_a_2;
      tl::Tcgen05SMemDescriptor desc_b_2;
      if ((((int)threadIdx.x) >> 5) == 0) {
        tl::initialize_tcgen05_descriptor(desc_a_2, (&(((signed char*)a_shared)[0])), 1, 32, 0, 0, 4);
        tl::initialize_tcgen05_descriptor(desc_b_2, (&(((signed char*)b_unpacked_shared)[0])), 0, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          #pragma unroll
          for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
            tl::tcgen05mma_ws_ss<tl::DataType::kInt8>(uint64_t(desc_a_2 + ((i_8 * 8192) + (ki_2 * 32))), uint64_t(desc_b_2 + (ki_2 * 4096)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + (i_8 * 128), 1, static_cast<uint32_t>(136381600), 0, 0, 0, 0);
          }
        }
        tl::tcgen05_mma_arrive((&(mbar[0])));
      }
      mbar[0].wait(0);
    }
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
  }
  tl::cp_async_wait<0>();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  *(uint4*)(b_packed_local + 0) = *(uint4*)(((uchar*)b_packed_shared) + ((((int)threadIdx.x) * 16) + 8192));
  for (int field_i_3 = 0; field_i_3 < 4; ++field_i_3) {
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      *(int4*)(((signed char*)a_shared) + ((((i_9 * 8192) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(a + ((((((((int)blockIdx.x) * 5242880) + (i_9 * 2621440)) + ((((int)threadIdx.x) >> 2) * 20480)) + (field_i_3 * 5120)) + ((((int)threadIdx.x) & 3) * 16)) + 5056));
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 16; ++i_10) {
      int field_3 = ((((int)b_packed_local[i_10]) >> (field_i_3 * 2)) & 3);
      b_unpacked_local[i_10] = (((signed char)field_3) - (signed char)1);
    }
    *(int4*)(((signed char*)b_unpacked_shared) + (((((((int)threadIdx.x) >> 3) * 128) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))) = *(int4*)(b_unpacked_local + 0);
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    {
      tl::Tcgen05SMemDescriptor desc_a_3;
      tl::Tcgen05SMemDescriptor desc_b_3;
      if ((((int)threadIdx.x) >> 5) == 0) {
        tl::initialize_tcgen05_descriptor(desc_a_3, (&(((signed char*)a_shared)[0])), 1, 32, 0, 0, 4);
        tl::initialize_tcgen05_descriptor(desc_b_3, (&(((signed char*)b_unpacked_shared)[0])), 0, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int i_11 = 0; i_11 < 2; ++i_11) {
          #pragma unroll
          for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
            tl::tcgen05mma_ws_ss<tl::DataType::kInt8>(uint64_t(desc_a_3 + ((i_11 * 8192) + (ki_3 * 32))), uint64_t(desc_b_3 + (ki_3 * 4096)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + (i_11 * 128), 1, static_cast<uint32_t>(136381600), 0, 0, 0, 0);
          }
        }
        tl::tcgen05_mma_arrive((&(mbar[0])));
      }
      mbar[0].wait(0);
    }
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
  }
  tl::tcgen05_ld_32dp32bNx<64, false>(acc_tmem[0], ((((int)threadIdx.x) >> 7) * 64), (&(acc[0])));
  #pragma unroll
  for (int i_12 = 0; i_12 < 8; ++i_12) {
    tl::store_global_256(&(*(longlong4*)(c + ((((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 8) * 262144)) + ((((int)threadIdx.x) & 127) * 2048)) + (((int)blockIdx.y) * 128)) + (((((int)threadIdx.x) & 255) >> 7) * 64)) + (i_12 * 8)))), *(longlong4*)(acc + (i_12 * 8)));
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(acc_tmem[0])), 256);
  }
}

