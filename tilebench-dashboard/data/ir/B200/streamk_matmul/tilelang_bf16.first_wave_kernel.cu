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

extern "C" __global__ void first_wave_kernel_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) first_wave_kernel_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, float* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* a_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  __shared__ __align__(16) uint64_t mbar_mem[1];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  __shared__ __align__(16) uint acc_tmem[1];
  int start_iter = 0;
  int last_iter = 0;
  int end_iter = 0;
  int tile_id = 0;
  int group_id = 0;
  int first_pid_m = 0;
  int group_size_m = 0;
  int pid_m = 0;
  int pid_n = 0;
  float acc[256];
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
  start_iter = ((((int)blockIdx.x) * 183) + min(((int)blockIdx.x), 52));
  last_iter = (((((int)blockIdx.x) * 183) + min((((int)blockIdx.x) + 1), 52)) + 183);
  while (1) {
    if (!((start_iter < last_iter))) { break; }
    end_iter = min(((start_iter + 128) - (start_iter & 127)), last_iter);
    tile_id = (start_iter >> 7);
    group_id = ((tile_id / 896) + ((tile_id % 896) >> 31));
    first_pid_m = (group_id * 8);
    group_size_m = (64 - max(first_pid_m, 56));
    int rmod = (tile_id % group_size_m);
    pid_m = (first_pid_m + ((((0 <= group_size_m) && (0 <= rmod)) || ((group_size_m < 0) && (rmod <= 0))) ? rmod : (rmod + group_size_m)));
    int rmod_1 = (((896 & ((tile_id % 896) >> 31)) + (tile_id % 896)) % group_size_m);
    int rdiv = (((896 & ((tile_id % 896) >> 31)) + (tile_id % 896)) / group_size_m);
    pid_n = ((((0 <= group_size_m) && (0 <= rmod_1)) || ((group_size_m < 0) && (rmod_1 <= 0))) ? rdiv : (rdiv - 1));
    if (start_iter < end_iter) {
      if (start_iter < end_iter) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)a_shared)[((((i * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((int64_t)pid_m) * (int64_t)524288) + (((int64_t)i) * (int64_t)131072)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)4096)) + ((((int64_t)start_iter) & (int64_t)127) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8))])), ((((pid_m < 64) && (0 <= pid_m)) && (pid_m < 64)) && (0 <= pid_m)));
        }
      }
      if (start_iter < end_iter) {
        #pragma unroll
        for (int i_1 = 0; i_1 < 8; ++i_1) {
          tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)b_shared)[((((((((((int)threadIdx.x) & 31) >> 3) * 2048) + (i_1 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_1 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((((int64_t)start_iter) & (int64_t)127) * (int64_t)917504) + (((int64_t)i_1) * (int64_t)114688)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)28672)) + (((int64_t)pid_n) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8))])), ((((pid_n < 112) && (0 <= pid_n)) && (pid_n < 112)) && (0 <= pid_n)));
        }
      }
      tl::cp_async_commit();
    }
    for (int current_iter = start_iter; current_iter < (end_iter - 1); ++current_iter) {
      tl::cp_async_wait<0>();
      tl::tcgen05_before_thread_sync();
      __syncthreads();
      tl::tcgen05_after_thread_sync();
      {
        tl::Tcgen05SMemDescriptor desc_a;
        tl::Tcgen05SMemDescriptor desc_b;
        if ((((int)threadIdx.x) >> 5) == 0) {
          tl::initialize_tcgen05_descriptor(desc_a, (&(((bfloat16_t*)a_shared)[0])), 1, 32, 0, 0, 4);
          tl::initialize_tcgen05_descriptor(desc_b, (&(((bfloat16_t*)b_shared)[0])), 256, 64, 0, 0, 2);
          tl::fence_proxy_async();
          #pragma unroll
          for (int ki = 0; ki < 2; ++ki) {
            tl::tcgen05mma_ss<tl::DataType::kBFloat16, false>(uint64_t(desc_a + (ki * 32)), uint64_t(desc_b + (ki * 2048)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + 0, ((0 < ki) ? 1 : ((current_iter == start_iter) ? 0 : 1)), static_cast<uint32_t>(138478736), 0, 0, 0, 0);
          }
          tl::tcgen05_mma_arrive((&(mbar[0])));
        }
        mbar[0].wait((current_iter & 1));
      }
      tl::tcgen05_before_thread_sync();
      __syncthreads();
      tl::tcgen05_after_thread_sync();
      #pragma unroll
      for (int i_2 = 0; i_2 < 4; ++i_2) {
        tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)a_shared)[((((i_2 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((int64_t)pid_m) * (int64_t)524288) + (((int64_t)i_2) * (int64_t)131072)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)4096)) + (((((int64_t)current_iter) + (int64_t)1) & (int64_t)127) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8))])), ((((pid_m < 64) && (0 <= pid_m)) && (pid_m < 64)) && (0 <= pid_m)));
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 8; ++i_3) {
        tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)b_shared)[((((((((((int)threadIdx.x) & 31) >> 3) * 2048) + (i_3 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((((((int64_t)current_iter) + (int64_t)1) & (int64_t)127) * (int64_t)917504) + (((int64_t)i_3) * (int64_t)114688)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)28672)) + (((int64_t)pid_n) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8))])), ((((pid_n < 112) && (0 <= pid_n)) && (pid_n < 112)) && (0 <= pid_n)));
      }
      tl::cp_async_commit();
      tl::tcgen05_before_thread_sync();
      __syncthreads();
      tl::tcgen05_after_thread_sync();
    }
    tl::cp_async_wait<0>();
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    if (start_iter < end_iter) {
      {
        tl::Tcgen05SMemDescriptor desc_a_1;
        tl::Tcgen05SMemDescriptor desc_b_1;
        if ((((int)threadIdx.x) >> 5) == 0) {
          tl::initialize_tcgen05_descriptor(desc_a_1, (&(((bfloat16_t*)a_shared)[0])), 1, 32, 0, 0, 4);
          tl::initialize_tcgen05_descriptor(desc_b_1, (&(((bfloat16_t*)b_shared)[0])), 256, 64, 0, 0, 2);
          tl::fence_proxy_async();
          #pragma unroll
          for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
            tl::tcgen05mma_ss<tl::DataType::kBFloat16, false>(uint64_t(desc_a_1 + (ki_1 * 32)), uint64_t(desc_b_1 + (ki_1 * 2048)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + 0, ((0 < ki_1) ? 1 : (((end_iter - 1) == start_iter) ? 0 : 1)), static_cast<uint32_t>(138478736), 0, 0, 0, 0);
          }
          tl::tcgen05_mma_arrive((&(mbar[0])));
        }
        mbar[0].wait(((end_iter + 1) & 1));
      }
      __syncthreads();
    }
    tl::tcgen05_ld_32dp32bNx<256, false>(acc_tmem[0], 0, (&(acc[0])));
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    if (0 <= pid_n) {
      #pragma unroll
      for (int i_4 = 0; i_4 < 64; ++i_4) {
        if (pid_n < 112) {
          if (pid_m < 64) {
            AtomicAddx4((&(C[((((((int64_t)pid_m) * (int64_t)3670016) + (((int64_t)((int)threadIdx.x)) * (int64_t)28672)) + (((int64_t)pid_n) * (int64_t)256)) + (((int64_t)i_4) * (int64_t)4))])), *(float4*)(acc + (i_4 * 4)));
          }
        }
      }
    }
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    start_iter = end_iter;
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(acc_tmem[0])), 256);
  }
}

