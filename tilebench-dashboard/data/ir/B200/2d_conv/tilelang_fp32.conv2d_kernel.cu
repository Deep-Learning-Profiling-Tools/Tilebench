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

extern "C" __global__ void conv2d_kernel_kernel(const float* __restrict__ input, float* __restrict__ output_flat, const float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(64, 1) conv2d_kernel_kernel(const float* __restrict__ input, float* __restrict__ output_flat, const float* __restrict__ weight) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* weight_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* input_tile = ((void*)((char*)buf_dyn_shmem + 8192));
  float acc[128];
  int bhw_offsets[8];
  int batch_ids[8];
  int out_rows[8];
  int out_cols[8];
  int feat_offsets[4];
  int in_channel_locals[4];
  int kernel_rows[4];
  int kernel_cols[4];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    bhw_offsets[i_1] = (((((int)blockIdx.x) * 64) + (i_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2));
    batch_ids[i_1] = ((bhw_offsets[i_1] / 102400) + ((bhw_offsets[i_1] % 102400) >> 31));
    int hw_id = ((102400 & ((bhw_offsets[i_1] % 102400) >> 31)) + (bhw_offsets[i_1] % 102400));
    out_rows[i_1] = ((hw_id / 320) + ((hw_id % 320) >> 31));
    out_cols[i_1] = ((320 & ((hw_id % 320) >> 31)) + (hw_id % 320));
  }
  for (int feat_block = 0; feat_block < 72; ++feat_block) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      feat_offsets[i_2] = (((feat_block * 16) + ((((int)threadIdx.x) & 3) * 4)) + i_2);
      in_channel_locals[i_2] = ((feat_offsets[i_2] / 9) + ((feat_offsets[i_2] % 9) >> 31));
      int kernel_rem = ((9 & ((feat_offsets[i_2] % 9) >> 31)) + (feat_offsets[i_2] % 9));
      kernel_rows[i_2] = ((kernel_rem / 3) + ((kernel_rem % 3) >> 31));
      kernel_cols[i_2] = ((3 & ((kernel_rem % 3) >> 31)) + (kernel_rem % 3));
    }
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      for (int vec_s = 0; vec_s < 4; ++vec_s) {
        int in_row = ((out_rows[((i_3 * 2) + (((int)threadIdx.x) >> 5))] + kernel_rows[vec_s]) - 1);
        int in_col = ((out_cols[((i_3 * 2) + (((int)threadIdx.x) >> 5))] + kernel_cols[vec_s]) - 1);
        int in_channel = in_channel_locals[vec_s];
        float condval;
        if ((((((((bhw_offsets[((i_3 * 2) + (((int)threadIdx.x) >> 5))] < 102400) && (feat_offsets[vec_s] < 1152)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320))) && (((((0 <= in_channel) && (in_channel < 128)) && (0 <= batch_ids[((i_3 * 2) + (((int)threadIdx.x) >> 5))])) && (batch_ids[((i_3 * 2) + (((int)threadIdx.x) >> 5))] < 1)))) {
          condval = input[((((((int64_t)batch_ids[((((int64_t)i_3) * (int64_t)2) + (((int64_t)((int)threadIdx.x)) >> (int64_t)5))]) * (int64_t)13107200) + (((int64_t)in_channel) * (int64_t)102400)) + (((int64_t)in_row) * (int64_t)320)) + ((int64_t)in_col))];
        } else if (((((((bhw_offsets[((i_3 * 2) + (((int)threadIdx.x) >> 5))] < 102400) && (feat_offsets[vec_s] < 1152)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320))) {
          condval = 0x0p+0f/*0.000000e+00*/;
        }
        ((float*)input_tile)[(((((i_3 * 256) + ((((int)threadIdx.x) >> 2) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4)) + vec_s)] = (((((((bhw_offsets[((i_3 * 2) + (((int)threadIdx.x) >> 5))] < 102400) && (feat_offsets[vec_s] < 1152)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320)) ? condval : 0x0p+0f/*0.000000e+00*/);
      }
    }
    if ((((int)threadIdx.x) >> 2) == 0) {
      #pragma unroll
      for (int i_4 = 0; i_4 < 512; ++i_4) {
        float condval_1;
        if (((feat_offsets[(i_4 >> 7)] < 1152)) && (((((((0 <= kernel_cols[(i_4 >> 7)]) && (kernel_cols[(i_4 >> 7)] < 3)) && (0 <= kernel_rows[(i_4 >> 7)])) && (kernel_rows[(i_4 >> 7)] < 3)) && (0 <= in_channel_locals[(i_4 >> 7)])) && (in_channel_locals[(i_4 >> 7)] < 128)))) {
          condval_1 = weight[(((((((int64_t)i_4) & (int64_t)127) * (int64_t)1152) + (((int64_t)in_channel_locals[(((int64_t)i_4) >> (int64_t)7)]) * (int64_t)9)) + (((int64_t)kernel_rows[(((int64_t)i_4) >> (int64_t)7)]) * (int64_t)3)) + ((int64_t)kernel_cols[(((int64_t)i_4) >> (int64_t)7)]))];
        } else if ((feat_offsets[(i_4 >> 7)] < 1152)) {
          condval_1 = 0x0p+0f/*0.000000e+00*/;
        }
        ((float*)weight_tile)[(((((((((i_4 & 127) >> 5) * 512) + ((((int)threadIdx.x) & 3) * 128)) + ((i_4 >> 7) * 32)) + (((((i_4 & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + ((((i_4 >> 8) + ((i_4 & 15) >> 3)) & 1) * 8)) + (((((i_4 & 255) >> 7) + ((i_4 & 7) >> 2)) & 1) * 4)) + (i_4 & 3))] = ((feat_offsets[(i_4 >> 7)] < 1152) ? condval_1 : 0x0p+0f/*0.000000e+00*/);
      }
    }
    __syncthreads();
    {
      float A_local[16];
      float B_local[16];
      for (int ki = 0; ki < 2; ++ki) {
        for (int i_5 = 0; i_5 < 4; ++i_5) {
          tl::ptx_ldmatrix_x4((&(((float*)input_tile)[((((i_5 * 256) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4))])), (&(A_local[(i_5 * 4)])));
        }
        for (int i_6 = 0; i_6 < 4; ++i_6) {
          for (int j = 0; j < 4; ++j) {
            B_local[((i_6 * 4) + j)] = ((float*)weight_tile)[((((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_6 >> 1) * 512)) + (ki * 256)) + ((j & 1) * 128)) + ((((int)threadIdx.x) & 3) * 32)) + ((((i_6 & 1) + (j & 1)) & 1) * 16)) + ((((j >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2))];
          }
        }
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + ((i_7 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_7 * 4)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 4)));
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + (((i_7 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_7 * 4)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 4) + 2)));
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 128; ++i_8) {
    int output_idx = (((((((batch_ids[(i_8 >> 4)] * 13107200) + ((((int)threadIdx.x) >> 5) * 6553600)) + (((i_8 & 15) >> 1) * 819200)) + ((((int)threadIdx.x) & 3) * 204800)) + ((i_8 & 1) * 102400)) + (out_rows[(i_8 >> 4)] * 320)) + out_cols[(i_8 >> 4)]);
    if (0 <= output_idx) {
      if (output_idx < 13107200) {
        output_flat[output_idx] = acc[(((((i_8 >> 5) * 32) + (((i_8 & 15) >> 1) * 4)) + (((i_8 & 31) >> 4) * 2)) + (i_8 & 1))];
      }
    }
  }
}

