#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/tcgen05mma.h>
#include <tl_templates/cuda/tcgen_05.h>
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
  void* b_tile = ((void*)((char*)buf_dyn_shmem + 98304));
  __shared__ __align__(16) uint64_t mbar_mem[1];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  __shared__ __align__(16) uint acc_tmem[1];
  float acc[512];
  fp8_e4_t c_local_cast[32];
  if (tl::tl_shuffle_elect<0>()) {
    mbar[0].init(1);
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
  const dim3 blockIdx = tl::rasterization2DRow<8>();
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)a_tile)[(((((i * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(a[((((((int)blockIdx.x) * 5242880) + (i * 327680)) + ((((int)threadIdx.x) >> 3) * 20480)) + ((((int)threadIdx.x) & 7) * 16))])));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)b_tile)[((((((((((int)threadIdx.x) & 15) >> 3) * 16384) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(b[((((i_1 * 32768) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 256)) + ((((int)threadIdx.x) & 15) * 16))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)a_tile)[((((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768)])), (&(a[(((((((int)blockIdx.x) * 5242880) + (i_2 * 327680)) + ((((int)threadIdx.x) >> 3) * 20480)) + ((((int)threadIdx.x) & 7) * 16)) + 128)])));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)b_tile)[(((((((((((int)threadIdx.x) & 15) >> 3) * 16384) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768)])), (&(b[(((((i_3 * 32768) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + 524288)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 158; ++k) {
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      tl::cp_async_gs<16>((&(((fp8_e4_t*)a_tile)[((((((((k + 2) % 3) * 32768) + (i_4 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(a[((((((((int)blockIdx.x) * 5242880) + (i_4 * 327680)) + ((((int)threadIdx.x) >> 3) * 20480)) + (k * 128)) + ((((int)threadIdx.x) & 7) * 16)) + 256)])));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      tl::cp_async_gs<16>((&(((fp8_e4_t*)b_tile)[(((((((((k + 2) % 3) * 32768) + (((((int)threadIdx.x) & 15) >> 3) * 16384)) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(b[((((((k * 524288) + (i_5 * 32768)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + 1048576)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    {
      tl::Tcgen05SMemDescriptor desc_a;
      tl::Tcgen05SMemDescriptor desc_b;
      if ((((int)threadIdx.x) >> 5) == 0) {
        tl::initialize_tcgen05_descriptor(desc_a, (&(((fp8_e4_t*)a_tile)[((k % 3) * 32768)])), 1, 64, 0, 0, 2);
        tl::initialize_tcgen05_descriptor(desc_b, (&(((fp8_e4_t*)b_tile)[((k % 3) * 32768)])), 512, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki) {
            tl::tcgen05mma_ws_ss<tl::DataType::kFloat8_e4m3>(uint64_t(desc_a + ((i_6 * 16384) + (ki * 32))), uint64_t(desc_b + (ki * 4096)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + (i_6 * 256), ((0 < ki) ? 1 : ((k == 0) ? 0 : 1)), static_cast<uint32_t>(138477584), 0, 0, 0, 0);
          }
        }
        tl::tcgen05_mma_arrive((&(mbar[0])));
      }
      mbar[0].wait(((k % 6) / 3));
    }
  }
  tl::cp_async_wait<1>();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  {
    tl::Tcgen05SMemDescriptor desc_a_1;
    tl::Tcgen05SMemDescriptor desc_b_1;
    if ((((int)threadIdx.x) >> 5) == 0) {
      tl::initialize_tcgen05_descriptor(desc_a_1, (&(((fp8_e4_t*)a_tile)[65536])), 1, 64, 0, 0, 2);
      tl::initialize_tcgen05_descriptor(desc_b_1, (&(((fp8_e4_t*)b_tile)[65536])), 512, 64, 0, 0, 2);
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        #pragma unroll
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          tl::tcgen05mma_ws_ss<tl::DataType::kFloat8_e4m3>(uint64_t(desc_a_1 + ((i_7 * 16384) + (ki_1 * 32))), uint64_t(desc_b_1 + (ki_1 * 4096)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + (i_7 * 256), 1, static_cast<uint32_t>(138477584), 0, 0, 0, 0);
        }
      }
      tl::tcgen05_mma_arrive((&(mbar[0])));
    }
    mbar[0].wait(0);
  }
  tl::cp_async_wait<0>();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  {
    tl::Tcgen05SMemDescriptor desc_a_2;
    tl::Tcgen05SMemDescriptor desc_b_2;
    if ((((int)threadIdx.x) >> 5) == 0) {
      tl::initialize_tcgen05_descriptor(desc_a_2, (&(((fp8_e4_t*)a_tile)[0])), 1, 64, 0, 0, 2);
      tl::initialize_tcgen05_descriptor(desc_b_2, (&(((fp8_e4_t*)b_tile)[0])), 512, 64, 0, 0, 2);
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        #pragma unroll
        for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
          tl::tcgen05mma_ws_ss<tl::DataType::kFloat8_e4m3>(uint64_t(desc_a_2 + ((i_8 * 16384) + (ki_2 * 32))), uint64_t(desc_b_2 + (ki_2 * 4096)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + (i_8 * 256), 1, static_cast<uint32_t>(138477584), 0, 0, 0, 0);
        }
      }
      tl::tcgen05_mma_arrive((&(mbar[0])));
    }
    mbar[0].wait(1);
  }
  tl::tcgen05_ld_32dp32bNx<512, false>(acc_tmem[0], 0, (&(acc[0])));
  #pragma unroll
  for (int i_9 = 0; i_9 < 16; ++i_9) {
    for (int vec = 0; vec < 8; ++vec) {
      fp8_e4_4_t __1;
      float4 v_ = *(float4*)(acc + ((i_9 * 32) + (vec * 4)));
      (reinterpret_cast<__nv_fp8x2_storage_t*>(&__1))[0] = __nv_cvt_float2_to_fp8x2(((float2*)(&v_))[0], __NV_SATFINITE, __NV_E4M3);
      (reinterpret_cast<__nv_fp8x2_storage_t*>(&__1))[1] = __nv_cvt_float2_to_fp8x2(((float2*)(&v_))[1], __NV_SATFINITE, __NV_E4M3);
      *(fp8_e4_4_t*)(c_local_cast + (vec * 4)) = __1;
    }
    tl::store_global_256(&(*(fp8_e4_32_t*)(c + (((((((int)blockIdx.x) * 1048576) + ((i_9 >> 3) * 524288)) + (((int)threadIdx.x) * 4096)) + (((int)blockIdx.y) * 256)) + ((i_9 & 7) * 32)))), *(fp8_e4_32_t*)(c_local_cast + 0));
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(acc_tmem[0])), 512);
  }
}

