#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/wgmma.h>
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
  float acc[128];
  int bdhw_offsets[4];
  int batch_ids[4];
  int out_ds[4];
  int out_rows[4];
  int out_cols[4];
  int feat_offsets[16];
  int in_channel_locals[16];
  int kernel_ds[16];
  int kernel_rows[16];
  int kernel_cols[16];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    bdhw_offsets[i_1] = (((((((int)blockIdx.x) * 128) + ((i_1 >> 1) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_1 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2));
    batch_ids[i_1] = ((bdhw_offsets[i_1] / 3276800) + ((bdhw_offsets[i_1] % 3276800) >> 31));
    int dhw_id = ((3276800 & ((bdhw_offsets[i_1] % 3276800) >> 31)) + (bdhw_offsets[i_1] % 3276800));
    out_ds[i_1] = ((dhw_id / 102400) + ((dhw_id % 102400) >> 31));
    out_rows[i_1] = (((102400 & ((dhw_id % 102400) >> 31)) + (dhw_id % 102400)) / 320);
    out_cols[i_1] = ((320 & ((dhw_id % 320) >> 31)) + (dhw_id % 320));
  }
  for (int feat_block = 0; feat_block < 108; ++feat_block) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      feat_offsets[i_2] = ((feat_block * 16) + i_2);
      in_channel_locals[i_2] = ((feat_offsets[i_2] / 27) + ((feat_offsets[i_2] % 27) >> 31));
      int kernel_rem = ((27 & ((feat_offsets[i_2] % 27) >> 31)) + (feat_offsets[i_2] % 27));
      kernel_ds[i_2] = ((kernel_rem / 9) + ((kernel_rem % 9) >> 31));
      kernel_rows[i_2] = (((9 & ((kernel_rem % 9) >> 31)) + (kernel_rem % 9)) / 3);
      kernel_cols[i_2] = ((3 & ((kernel_rem % 3) >> 31)) + (kernel_rem % 3));
    }
    __syncthreads();
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_3 = 0; i_3 < 16; ++i_3) {
        for (int vec_s = 0; vec_s < 4; ++vec_s) {
          int in_d = ((out_ds[(i_3 >> 2)] + kernel_ds[(((i_3 & 3) * 4) + vec_s)]) - 1);
          int in_row = ((out_rows[(i_3 >> 2)] + kernel_rows[(((i_3 & 3) * 4) + vec_s)]) - 1);
          int in_col = ((out_cols[(i_3 >> 2)] + kernel_cols[(((i_3 & 3) * 4) + vec_s)]) - 1);
          int in_channel = in_channel_locals[(((i_3 & 3) * 4) + vec_s)];
          half_t condval;
          if ((((((((((bdhw_offsets[(i_3 >> 2)] < 3276800) && (feat_offsets[(((i_3 & 3) * 4) + vec_s)] < 1728)) && (0 <= in_d)) && (in_d < 32)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320))) && (((((0 <= in_channel) && (in_channel < 64)) && (0 <= batch_ids[(i_3 >> 2)])) && (batch_ids[(i_3 >> 2)] < 1)))) {
            condval = input[(((((((int64_t)batch_ids[(((int64_t)i_3) >> (int64_t)2)]) * (int64_t)209715200) + (((int64_t)in_channel) * (int64_t)3276800)) + (((int64_t)in_d) * (int64_t)102400)) + (((int64_t)in_row) * (int64_t)320)) + ((int64_t)in_col))];
          } else if (((((((((bdhw_offsets[(i_3 >> 2)] < 3276800) && (feat_offsets[(((i_3 & 3) * 4) + vec_s)] < 1728)) && (0 <= in_d)) && (in_d < 32)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320))) {
            condval = half_t(0x0p+0f/*0.000000e+00*/);
          }
          ((half_t*)input_tile)[((((((((i_3 >> 3) * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + (((i_3 & 7) >> 2) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_3 & 3) >> 1)) & 1) * 8)) + ((i_3 & 1) * 4)) + vec_s)] = (((((((((bdhw_offsets[(i_3 >> 2)] < 3276800) && (feat_offsets[(((i_3 & 3) * 4) + vec_s)] < 1728)) && (0 <= in_d)) && (in_d < 32)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320)) ? condval : half_t(0x0p+0f/*0.000000e+00*/));
        }
      }
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      half_t condval_1;
      if ((((feat_offsets[i_4] < 1728) && (((int)threadIdx.x) < 64))) && (((((((((0 <= kernel_cols[i_4]) && (kernel_cols[i_4] < 3)) && (0 <= kernel_rows[i_4])) && (kernel_rows[i_4] < 3)) && (0 <= kernel_ds[i_4])) && (kernel_ds[i_4] < 3)) && (0 <= in_channel_locals[i_4])) && (in_channel_locals[i_4] < 64)))) {
        condval_1 = weight[(((((((int64_t)((int)threadIdx.x)) * (int64_t)1728) + (((int64_t)in_channel_locals[((int64_t)i_4)]) * (int64_t)27)) + (((int64_t)kernel_ds[((int64_t)i_4)]) * (int64_t)9)) + (((int64_t)kernel_rows[((int64_t)i_4)]) * (int64_t)3)) + ((int64_t)kernel_cols[((int64_t)i_4)]))];
      } else if (((feat_offsets[i_4] < 1728) && (((int)threadIdx.x) < 64))) {
        condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      ((half_t*)weight_tile)[(((((((((int)threadIdx.x) >> 6) * 1024) + (i_4 * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((i_4 & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_4 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_4 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))] = (((feat_offsets[i_4] < 1728) && (((int)threadIdx.x) < 64)) ? condval_1 : half_t(0x0p+0f/*0.000000e+00*/));
    }
    __syncthreads();
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)input_tile)[0])));
      tl::initialize_wgmma_descriptor<1, 128, 64>(desc_b, (&(((half_t*)weight_tile)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a + ((i_5 * 2048) >> 4)), uint64_t(desc_b + 0), ((uint32_t*)(acc + (i_5 * 64))), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 128);
    }
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 128; ++i_6) {
    if ((i_6 & 31) < 16) {
      if (0 <= (((((((batch_ids[(i_6 >> 5)] * 209715200) + (((i_6 & 31) >> 1) * 26214400)) + ((((int)threadIdx.x) & 3) * 6553600)) + ((i_6 & 1) * 3276800)) + (out_ds[(i_6 >> 5)] * 102400)) + (out_rows[(i_6 >> 5)] * 320)) + out_cols[(i_6 >> 5)])) {
        if ((((((((((((i_6 & 31) >> 1) * 26214400) + ((((int)threadIdx.x) & 3) * 6553600)) + ((i_6 & 1) * 3276800)) + (out_ds[(i_6 >> 5)] * 102400)) + (out_rows[(i_6 >> 5)] * 320)) + out_cols[(i_6 >> 5)]) / 209715200) + ((((((((((i_6 & 31) >> 1) * 26214400) + ((((int)threadIdx.x) & 3) * 6553600)) + ((i_6 & 1) * 3276800)) + (out_ds[(i_6 >> 5)] * 102400)) + (out_rows[(i_6 >> 5)] * 320)) + out_cols[(i_6 >> 5)]) % 209715200) >> 31)) + batch_ids[(i_6 >> 5)]) < 1) {
          output_flat[(((((((((int64_t)batch_ids[(((int64_t)i_6) >> (int64_t)5)]) * (int64_t)209715200) + (((((int64_t)i_6) & (int64_t)31) >> (int64_t)1) * (int64_t)26214400)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)6553600)) + ((((int64_t)i_6) & (int64_t)1) * (int64_t)3276800)) + (((int64_t)out_ds[(((int64_t)i_6) >> (int64_t)5)]) * (int64_t)102400)) + (((int64_t)out_rows[(((int64_t)i_6) >> (int64_t)5)]) * (int64_t)320)) + ((int64_t)out_cols[(((int64_t)i_6) >> (int64_t)5)]))] = ((half_t)acc[(((((i_6 >> 6) * 64) + (((i_6 & 31) >> 1) * 4)) + (((i_6 & 63) >> 5) * 2)) + (i_6 & 1))]);
        }
      }
    }
  }
}

