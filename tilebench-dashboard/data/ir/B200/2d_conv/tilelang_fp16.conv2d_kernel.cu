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

extern "C" __global__ void conv2d_kernel_kernel(const half_t* __restrict__ input, half_t* __restrict__ output_flat, const half_t* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(128, 1) conv2d_kernel_kernel(const half_t* __restrict__ input, half_t* __restrict__ output_flat, const half_t* __restrict__ weight) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* input_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* weight_tile = ((void*)((char*)buf_dyn_shmem + 8192));
  __shared__ __align__(16) uint64_t mbar_mem[1];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  __shared__ __align__(16) uint acc_tmem[1];
  int bhw_offsets[1];
  int batch_ids[1];
  int out_rows[1];
  int out_cols[1];
  int feat_offsets[32];
  int in_channel_locals[32];
  int kernel_rows[32];
  int kernel_cols[32];
  float acc[128];
  if (tl::tl_shuffle_elect<0>()) {
    mbar[0].init(1);
  }
  tl::fence_barrier_init();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(acc_tmem[0])), 128);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  bhw_offsets[0] = ((((int)blockIdx.x) * 128) + ((int)threadIdx.x));
  batch_ids[0] = ((bhw_offsets[0] / 102400) + ((bhw_offsets[0] % 102400) >> 31));
  int hw_id = ((102400 & ((bhw_offsets[0] % 102400) >> 31)) + (bhw_offsets[0] % 102400));
  out_rows[0] = ((hw_id / 320) + ((hw_id % 320) >> 31));
  out_cols[0] = ((320 & ((hw_id % 320) >> 31)) + (hw_id % 320));
  for (int feat_block = 0; feat_block < 36; ++feat_block) {
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      feat_offsets[i] = ((feat_block * 32) + i);
      in_channel_locals[i] = ((feat_offsets[i] / 9) + ((feat_offsets[i] % 9) >> 31));
      int kernel_rem = ((9 & ((feat_offsets[i] % 9) >> 31)) + (feat_offsets[i] % 9));
      kernel_rows[i] = ((kernel_rem / 3) + ((kernel_rem % 3) >> 31));
      kernel_cols[i] = ((3 & ((kernel_rem % 3) >> 31)) + (kernel_rem % 3));
    }
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      for (int vec_s = 0; vec_s < 4; ++vec_s) {
        int in_row = ((out_rows[0] + kernel_rows[((i_1 * 4) + vec_s)]) - 1);
        int in_col = ((out_cols[0] + kernel_cols[((i_1 * 4) + vec_s)]) - 1);
        int in_channel = in_channel_locals[((i_1 * 4) + vec_s)];
        half_t condval;
        if ((((((((bhw_offsets[0] < 102400) && (feat_offsets[((i_1 * 4) + vec_s)] < 1152)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320))) && (((((0 <= in_channel) && (in_channel < 128)) && (0 <= batch_ids[0])) && (batch_ids[0] < 1)))) {
          condval = input[((((((int64_t)batch_ids[(int64_t)0]) * (int64_t)13107200) + (((int64_t)in_channel) * (int64_t)102400)) + (((int64_t)in_row) * (int64_t)320)) + ((int64_t)in_col))];
        } else if (((((((bhw_offsets[0] < 102400) && (feat_offsets[((i_1 * 4) + vec_s)] < 1152)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320))) {
          condval = half_t(0x0p+0f/*0.000000e+00*/);
        }
        ((half_t*)input_tile)[(((((((int)threadIdx.x) * 32) + ((((i_1 >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((i_1 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((i_1 & 1) * 4)) + vec_s)] = (((((((bhw_offsets[0] < 102400) && (feat_offsets[((i_1 * 4) + vec_s)] < 1152)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320)) ? condval : half_t(0x0p+0f/*0.000000e+00*/));
      }
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 32; ++i_2) {
      half_t condval_1;
      if (((feat_offsets[i_2] < 1152)) && (((((((0 <= kernel_cols[i_2]) && (kernel_cols[i_2] < 3)) && (0 <= kernel_rows[i_2])) && (kernel_rows[i_2] < 3)) && (0 <= in_channel_locals[i_2])) && (in_channel_locals[i_2] < 128)))) {
        condval_1 = weight[((((((int64_t)((int)threadIdx.x)) * (int64_t)1152) + (((int64_t)in_channel_locals[((int64_t)i_2)]) * (int64_t)9)) + (((int64_t)kernel_rows[((int64_t)i_2)]) * (int64_t)3)) + ((int64_t)kernel_cols[((int64_t)i_2)]))];
      } else if ((feat_offsets[i_2] < 1152)) {
        condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      ((half_t*)weight_tile)[(((((((((int)threadIdx.x) >> 6) * 2048) + (i_2 * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((i_2 & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_2 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_2 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))] = ((feat_offsets[i_2] < 1152) ? condval_1 : half_t(0x0p+0f/*0.000000e+00*/));
    }
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    {
      tl::Tcgen05SMemDescriptor desc_a;
      tl::Tcgen05SMemDescriptor desc_b;
      if ((((int)threadIdx.x) >> 5) == 0) {
        tl::initialize_tcgen05_descriptor(desc_a, (&(((half_t*)input_tile)[0])), 1, 32, 0, 0, 4);
        tl::initialize_tcgen05_descriptor(desc_b, (&(((half_t*)weight_tile)[0])), 256, 64, 0, 0, 2);
        tl::fence_proxy_async();
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
          tl::tcgen05mma_ss<tl::DataType::kFloat16, false>(uint64_t(desc_a + (ki * 32)), uint64_t(desc_b + (ki * 2048)), (*reinterpret_cast<uint32_t*>(acc_tmem)) + 0, ((0 < ki) ? 1 : ((feat_block == 0) ? 0 : 1)), static_cast<uint32_t>(136380432), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(mbar[0])));
      }
      mbar[0].wait((feat_block & 1));
    }
  }
  tl::tcgen05_ld_32dp32bNx<128, false>(acc_tmem[0], 0, (&(acc[0])));
  #pragma unroll
  for (int i_3 = 0; i_3 < 128; ++i_3) {
    int output_idx = ((((batch_ids[0] * 13107200) + (i_3 * 102400)) + (out_rows[0] * 320)) + out_cols[0]);
    if (0 <= output_idx) {
      if (output_idx < 13107200) {
        output_flat[output_idx] = ((half_t)acc[i_3]);
      }
    }
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(acc_tmem[0])), 128);
  }
}

