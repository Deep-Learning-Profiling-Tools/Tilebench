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

extern "C" __global__ void conv3d_kernel_kernel(const half_t* __restrict__ input, half_t* __restrict__ output_flat, const half_t* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(128, 1) conv3d_kernel_kernel(const half_t* __restrict__ input, half_t* __restrict__ output_flat, const half_t* __restrict__ weight) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* input_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* weight_tile = ((void*)((char*)buf_dyn_shmem + 4096));
  __shared__ __align__(16) uint64_t mbar_mem[1];
  auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  __shared__ __align__(16) uint acc_tmem[1];
  int bdhw_offsets[1];
  int batch_ids[1];
  int out_ds[1];
  int out_rows[1];
  int out_cols[1];
  int feat_offsets[16];
  int in_channel_locals[16];
  int kernel_ds[16];
  int kernel_rows[16];
  int kernel_cols[16];
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
  bdhw_offsets[0] = ((((int)blockIdx.x) * 128) + ((int)threadIdx.x));
  batch_ids[0] = ((bdhw_offsets[0] / 3276800) + ((bdhw_offsets[0] % 3276800) >> 31));
  int dhw_id = ((3276800 & ((bdhw_offsets[0] % 3276800) >> 31)) + (bdhw_offsets[0] % 3276800));
  out_ds[0] = ((dhw_id / 102400) + ((dhw_id % 102400) >> 31));
  out_rows[0] = (((102400 & ((dhw_id % 102400) >> 31)) + (dhw_id % 102400)) / 320);
  out_cols[0] = ((320 & ((dhw_id % 320) >> 31)) + (dhw_id % 320));
  for (int feat_block = 0; feat_block < 108; ++feat_block) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      feat_offsets[i] = ((feat_block * 16) + i);
      in_channel_locals[i] = ((feat_offsets[i] / 27) + ((feat_offsets[i] % 27) >> 31));
      int kernel_rem = ((27 & ((feat_offsets[i] % 27) >> 31)) + (feat_offsets[i] % 27));
      kernel_ds[i] = ((kernel_rem / 9) + ((kernel_rem % 9) >> 31));
      kernel_rows[i] = (((9 & ((kernel_rem % 9) >> 31)) + (kernel_rem % 9)) / 3);
      kernel_cols[i] = ((3 & ((kernel_rem % 3) >> 31)) + (kernel_rem % 3));
    }
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      for (int vec_s = 0; vec_s < 4; ++vec_s) {
        int in_d = ((out_ds[0] + kernel_ds[((i_1 * 4) + vec_s)]) - 1);
        int in_row = ((out_rows[0] + kernel_rows[((i_1 * 4) + vec_s)]) - 1);
        int in_col = ((out_cols[0] + kernel_cols[((i_1 * 4) + vec_s)]) - 1);
        int in_channel = in_channel_locals[((i_1 * 4) + vec_s)];
        half_t condval;
        if ((((((((((bdhw_offsets[0] < 3276800) && (feat_offsets[((i_1 * 4) + vec_s)] < 1728)) && (0 <= in_d)) && (in_d < 32)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320))) && (((((0 <= in_channel) && (in_channel < 64)) && (0 <= batch_ids[0])) && (batch_ids[0] < 1)))) {
          condval = input[(((((((int64_t)batch_ids[(int64_t)0]) * (int64_t)209715200) + (((int64_t)in_channel) * (int64_t)3276800)) + (((int64_t)in_d) * (int64_t)102400)) + (((int64_t)in_row) * (int64_t)320)) + ((int64_t)in_col))];
        } else if (((((((((bdhw_offsets[0] < 3276800) && (feat_offsets[((i_1 * 4) + vec_s)] < 1728)) && (0 <= in_d)) && (in_d < 32)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320))) {
          condval = half_t(0x0p+0f/*0.000000e+00*/);
        }
        ((half_t*)input_tile)[((((((int)threadIdx.x) * 16) + (((((((int)threadIdx.x) & 7) >> 2) + (i_1 >> 1)) & 1) * 8)) + ((i_1 & 1) * 4)) + vec_s)] = (((((((((bdhw_offsets[0] < 3276800) && (feat_offsets[((i_1 * 4) + vec_s)] < 1728)) && (0 <= in_d)) && (in_d < 32)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320)) ? condval : half_t(0x0p+0f/*0.000000e+00*/));
      }
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      half_t condval_1;
      if ((((feat_offsets[i_2] < 1728) && (((int)threadIdx.x) < 64))) && (((((((((0 <= kernel_cols[i_2]) && (kernel_cols[i_2] < 3)) && (0 <= kernel_rows[i_2])) && (kernel_rows[i_2] < 3)) && (0 <= kernel_ds[i_2])) && (kernel_ds[i_2] < 3)) && (0 <= in_channel_locals[i_2])) && (in_channel_locals[i_2] < 64)))) {
        condval_1 = weight[(((((((int64_t)((int)threadIdx.x)) * (int64_t)1728) + (((int64_t)in_channel_locals[((int64_t)i_2)]) * (int64_t)27)) + (((int64_t)kernel_ds[((int64_t)i_2)]) * (int64_t)9)) + (((int64_t)kernel_rows[((int64_t)i_2)]) * (int64_t)3)) + ((int64_t)kernel_cols[((int64_t)i_2)]))];
      } else if (((feat_offsets[i_2] < 1728) && (((int)threadIdx.x) < 64))) {
        condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      ((half_t*)weight_tile)[(((((((((int)threadIdx.x) >> 6) * 1024) + (i_2 * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((i_2 & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_2 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_2 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))] = (((feat_offsets[i_2] < 1728) && (((int)threadIdx.x) < 64)) ? condval_1 : half_t(0x0p+0f/*0.000000e+00*/));
    }
    tl::tcgen05_before_thread_sync();
    __syncthreads();
    tl::tcgen05_after_thread_sync();
    {
      tl::Tcgen05SMemDescriptor desc_a;
      tl::Tcgen05SMemDescriptor desc_b;
      if ((((int)threadIdx.x) >> 5) == 0) {
        tl::initialize_tcgen05_descriptor(desc_a, (&(((half_t*)input_tile)[0])), 1, 16, 0, 0, 6);
        tl::initialize_tcgen05_descriptor(desc_b, (&(((half_t*)weight_tile)[0])), 128, 64, 0, 0, 2);
        tl::fence_proxy_async();
        tl::tcgen05mma_ss<tl::DataType::kFloat16, false>(uint64_t(desc_a + 0), uint64_t(desc_b + 0), (*reinterpret_cast<uint32_t*>(acc_tmem)) + 0, ((feat_block == 0) ? 0 : 1), static_cast<uint32_t>(136380432), 0, 0, 0, 0);
        tl::tcgen05_mma_arrive((&(mbar[0])));
      }
      mbar[0].wait((feat_block & 1));
    }
  }
  tl::tcgen05_ld_32dp32bNx<128, false>(acc_tmem[0], 0, (&(acc[0])));
  #pragma unroll
  for (int i_3 = 0; i_3 < 128; ++i_3) {
    if (i_3 < 64) {
      if (0 <= (((((batch_ids[0] * 209715200) + (i_3 * 3276800)) + (out_ds[0] * 102400)) + (out_rows[0] * 320)) + out_cols[0])) {
        if ((((((((i_3 * 3276800) + (out_ds[0] * 102400)) + (out_rows[0] * 320)) + out_cols[0]) / 209715200) + ((((((i_3 * 3276800) + (out_ds[0] * 102400)) + (out_rows[0] * 320)) + out_cols[0]) % 209715200) >> 31)) + batch_ids[0]) < 1) {
          output_flat[(((((((int64_t)batch_ids[(int64_t)0]) * (int64_t)209715200) + (((int64_t)i_3) * (int64_t)3276800)) + (((int64_t)out_ds[(int64_t)0]) * (int64_t)102400)) + (((int64_t)out_rows[(int64_t)0]) * (int64_t)320)) + ((int64_t)out_cols[(int64_t)0]))] = ((half_t)acc[i_3]);
        }
      }
    }
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(acc_tmem[0])), 128);
  }
}

