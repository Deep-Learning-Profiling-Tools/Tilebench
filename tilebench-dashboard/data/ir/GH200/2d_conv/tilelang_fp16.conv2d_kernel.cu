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

extern "C" __global__ void conv2d_kernel_kernel(const half_t* __restrict__ input, half_t* __restrict__ output_flat, const half_t* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(128, 1) conv2d_kernel_kernel(const half_t* __restrict__ input, half_t* __restrict__ output_flat, const half_t* __restrict__ weight) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* weight_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* input_tile = ((void*)((char*)buf_dyn_shmem + 4096));
  float acc[64];
  int bhw_offsets[2];
  int batch_ids[2];
  int out_rows[2];
  int out_cols[2];
  int feat_offsets[16];
  int in_channel_locals[16];
  int kernel_rows[16];
  int kernel_cols[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    bhw_offsets[i_1] = ((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (i_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2));
    batch_ids[i_1] = ((bhw_offsets[i_1] / 102400) + ((bhw_offsets[i_1] % 102400) >> 31));
    int hw_id = ((102400 & ((bhw_offsets[i_1] % 102400) >> 31)) + (bhw_offsets[i_1] % 102400));
    out_rows[i_1] = ((hw_id / 320) + ((hw_id % 320) >> 31));
    out_cols[i_1] = ((320 & ((hw_id % 320) >> 31)) + (hw_id % 320));
  }
  for (int feat_block = 0; feat_block < 72; ++feat_block) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      feat_offsets[i_2] = ((feat_block * 16) + i_2);
      in_channel_locals[i_2] = ((feat_offsets[i_2] / 9) + ((feat_offsets[i_2] % 9) >> 31));
      int kernel_rem = ((9 & ((feat_offsets[i_2] % 9) >> 31)) + (feat_offsets[i_2] % 9));
      kernel_rows[i_2] = ((kernel_rem / 3) + ((kernel_rem % 3) >> 31));
      kernel_cols[i_2] = ((3 & ((kernel_rem % 3) >> 31)) + (kernel_rem % 3));
    }
    __syncthreads();
    if ((((int)threadIdx.x) % 4) == 0) {
      #pragma unroll
      for (int i_3 = 0; i_3 < 8; ++i_3) {
        for (int vec_s = 0; vec_s < 4; ++vec_s) {
          int in_row = ((out_rows[(i_3 >> 2)] + kernel_rows[(((i_3 & 3) * 4) + vec_s)]) - 1);
          int in_col = ((out_cols[(i_3 >> 2)] + kernel_cols[(((i_3 & 3) * 4) + vec_s)]) - 1);
          int in_channel = in_channel_locals[(((i_3 & 3) * 4) + vec_s)];
          half_t condval;
          if ((((((((bhw_offsets[(i_3 >> 2)] < 102400) && (feat_offsets[(((i_3 & 3) * 4) + vec_s)] < 1152)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320))) && (((((0 <= in_channel) && (in_channel < 128)) && (0 <= batch_ids[(i_3 >> 2)])) && (batch_ids[(i_3 >> 2)] < 1)))) {
            condval = input[((((((int64_t)batch_ids[(((int64_t)i_3) >> (int64_t)2)]) * (int64_t)13107200) + (((int64_t)in_channel) * (int64_t)102400)) + (((int64_t)in_row) * (int64_t)320)) + ((int64_t)in_col))];
          } else if (((((((bhw_offsets[(i_3 >> 2)] < 102400) && (feat_offsets[(((i_3 & 3) * 4) + vec_s)] < 1152)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320))) {
            condval = half_t(0x0p+0f/*0.000000e+00*/);
          }
          ((half_t*)input_tile)[(((((((((int)threadIdx.x) >> 5) * 256) + ((i_3 >> 2) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_3 & 3) >> 1)) & 1) * 8)) + ((i_3 & 1) * 4)) + vec_s)] = (((((((bhw_offsets[(i_3 >> 2)] < 102400) && (feat_offsets[(((i_3 & 3) * 4) + vec_s)] < 1152)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320)) ? condval : half_t(0x0p+0f/*0.000000e+00*/));
        }
      }
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      half_t condval_1;
      if (((feat_offsets[i_4] < 1152)) && (((((((0 <= kernel_cols[i_4]) && (kernel_cols[i_4] < 3)) && (0 <= kernel_rows[i_4])) && (kernel_rows[i_4] < 3)) && (0 <= in_channel_locals[i_4])) && (in_channel_locals[i_4] < 128)))) {
        condval_1 = weight[((((((int64_t)((int)threadIdx.x)) * (int64_t)1152) + (((int64_t)in_channel_locals[((int64_t)i_4)]) * (int64_t)9)) + (((int64_t)kernel_rows[((int64_t)i_4)]) * (int64_t)3)) + ((int64_t)kernel_cols[((int64_t)i_4)]))];
      } else if ((feat_offsets[i_4] < 1152)) {
        condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      ((half_t*)weight_tile)[(((((((((int)threadIdx.x) >> 6) * 1024) + (i_4 * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((i_4 & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_4 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_4 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))] = ((feat_offsets[i_4] < 1152) ? condval_1 : half_t(0x0p+0f/*0.000000e+00*/));
    }
    __syncthreads();
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<3, 1, 16>(desc_a, (&(((half_t*)input_tile)[0])));
      tl::initialize_wgmma_descriptor<1, 128, 64>(desc_b, (&(((half_t*)weight_tile)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a + 0), uint64_t(desc_b + 0), ((uint32_t*)(acc + 0)), 1);
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 64);
    }
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 64; ++i_5) {
    int output_idx = ((((((batch_ids[(i_5 >> 5)] * 13107200) + (((i_5 & 31) >> 1) * 819200)) + ((((int)threadIdx.x) & 3) * 204800)) + ((i_5 & 1) * 102400)) + (out_rows[(i_5 >> 5)] * 320)) + out_cols[(i_5 >> 5)]);
    if (0 <= output_idx) {
      if (output_idx < 13107200) {
        output_flat[output_idx] = ((half_t)acc[(((((i_5 & 31) >> 1) * 4) + ((i_5 >> 5) * 2)) + (i_5 & 1))]);
      }
    }
  }
}

