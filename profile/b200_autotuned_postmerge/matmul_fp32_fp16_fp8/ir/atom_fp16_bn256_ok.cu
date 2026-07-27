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

extern "C" __global__ void kern_kernel(__grid_constant__ const CUtensorMap a_desc, __grid_constant__ const CUtensorMap b_desc, half_t* __restrict__ c);
extern "C" __global__ void __launch_bounds__(384, 1) kern_kernel(__grid_constant__ const CUtensorMap a_desc, __grid_constant__ const CUtensorMap b_desc, half_t* __restrict__ c) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* b_tile = ((void*)((char*)buf_dyn_shmem + 98304));
  __shared__ __align__(16) uint64_t mbar_mem[3];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  __shared__ __align__(16) uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  __shared__ __align__(16) uint acc_tmem[1];
  float acc[256];
  half_t c_local_cast[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(a_desc);
    tl::prefetch_tma_descriptor(b_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbar[0].init(1);
    mbar[1].init(1);
    mbar[2].init(1);
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(256);
    mbarrier[4].init(256);
    mbarrier[5].init(256);
  }
  tl::fence_barrier_init();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(acc_tmem[0])), 512);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  const dim3 blockIdx = tl::rasterization2DRow<64>();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 16; ++k) {
      mbarrier[((k % 3) + 3)].wait((((k % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k % 3)].expect_transaction(32768);
        tl::tma_load(a_desc, mbarrier[(k % 3)], (&(((half_t*)a_tile)[((k % 3) * 16384)])), (k * 64), (((int)blockIdx.x) * 256));
        mbarrier[(k % 3)].arrive_and_expect_tx(32768);
        tl::tma_load(b_desc, mbarrier[(k % 3)], (&(((half_t*)b_tile)[((k % 3) * 16384)])), (((int)blockIdx.y) * 256), (k * 64));
        tl::tma_load(b_desc, mbarrier[(k % 3)], (&(((half_t*)b_tile)[(((k % 3) * 16384) + 4096)])), ((((int)blockIdx.y) * 256) + 64), (k * 64));
        tl::tma_load(b_desc, mbarrier[(k % 3)], (&(((half_t*)b_tile)[(((k % 3) * 16384) + 8192)])), ((((int)blockIdx.y) * 256) + 128), (k * 64));
        tl::tma_load(b_desc, mbarrier[(k % 3)], (&(((half_t*)b_tile)[(((k % 3) * 16384) + 12288)])), ((((int)blockIdx.y) * 256) + 192), (k * 64));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    for (int k_1 = 0; k_1 < 16; ++k_1) {
      mbarrier[(k_1 % 3)].wait(((k_1 % 6) / 3));
      tl::tcgen05_after_thread_sync();
      {
        tl::Tcgen05SMemDescriptor desc_a;
        tl::Tcgen05SMemDescriptor desc_b;
        if ((((int)threadIdx.x) >> 5) == 4) {
          tl::initialize_tcgen05_descriptor(desc_a, (&(((half_t*)a_tile)[((k_1 % 3) * 16384)])), 1, 64, 0, 0, 2);
          tl::initialize_tcgen05_descriptor(desc_b, (&(((half_t*)b_tile)[((k_1 % 3) * 16384)])), 512, 64, 0, 0, 2);
          tl::fence_proxy_async();
          #pragma unroll
          for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
              tl::tcgen05mma_ss<tl::DataType::kFloat16, false>(uint64_t(desc_a + ((i * 16384) + (ki * 32))), uint64_t(desc_b + (ki * 2048)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + (i * 256), ((0 < ki) ? 1 : ((k_1 == 0) ? 0 : 1)), static_cast<uint32_t>(138477584), 0, 0, 0, 0);
            }
          }
          tl::tcgen05_mma_arrive((&(mbar[(k_1 % 3)])));
        }
        mbar[(k_1 % 3)].wait(((k_1 % 6) / 3));
      }
      tl::tcgen05_before_thread_sync();
      mbarrier[((k_1 % 3) + 3)].arrive();
    }
    tl::tcgen05_ld_32dp32bNx<256, false>(acc_tmem[0], (((((int)threadIdx.x) >> 7) * 256) - 256), (&(acc[0])));
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      for (int vec = 0; vec < 4; ++vec) {
        uint2 __1;
        float4 v_ = *(float4*)(acc + ((i_1 * 16) + (vec * 4)));
        ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
        ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
        *(uint2*)(c_local_cast + (vec * 4)) = __1;
      }
      tl::store_global_256(&(*(ulonglong4*)(c + (((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 512)) + (((int)blockIdx.y) * 256)) + (i_1 * 16)) - 65536))), *(ulonglong4*)(c_local_cast + 0));
    }
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(acc_tmem[0])), 512);
  }
}

