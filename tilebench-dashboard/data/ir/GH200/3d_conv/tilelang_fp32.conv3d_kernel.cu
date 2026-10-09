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

extern "C" __global__ void conv3d_kernel_kernel(const float* __restrict__ input, float* __restrict__ output_flat, const float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(64, 1) conv3d_kernel_kernel(const float* __restrict__ input, float* __restrict__ output_flat, const float* __restrict__ weight) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* input_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* weight_tile = ((void*)((char*)buf_dyn_shmem + 8192));
  float acc[128];
  int bdhw_offsets[16];
  int batch_ids[16];
  int out_ds[16];
  int out_rows[16];
  int out_cols[16];
  int feat_offsets[4];
  int in_channel_locals[4];
  int kernel_ds[4];
  int kernel_rows[4];
  int kernel_cols[4];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    bdhw_offsets[i_1] = (((((int)blockIdx.x) * 128) + (i_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2));
    batch_ids[i_1] = ((bdhw_offsets[i_1] / 3276800) + ((bdhw_offsets[i_1] % 3276800) >> 31));
    int dhw_id = ((3276800 & ((bdhw_offsets[i_1] % 3276800) >> 31)) + (bdhw_offsets[i_1] % 3276800));
    out_ds[i_1] = ((dhw_id / 102400) + ((dhw_id % 102400) >> 31));
    out_rows[i_1] = (((102400 & ((dhw_id % 102400) >> 31)) + (dhw_id % 102400)) / 320);
    out_cols[i_1] = ((320 & ((dhw_id % 320) >> 31)) + (dhw_id % 320));
  }
  for (int feat_block = 0; feat_block < 108; ++feat_block) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      feat_offsets[i_2] = (((feat_block * 16) + ((((int)threadIdx.x) & 3) * 4)) + i_2);
      in_channel_locals[i_2] = ((feat_offsets[i_2] / 27) + ((feat_offsets[i_2] % 27) >> 31));
      int kernel_rem = ((27 & ((feat_offsets[i_2] % 27) >> 31)) + (feat_offsets[i_2] % 27));
      kernel_ds[i_2] = ((kernel_rem / 9) + ((kernel_rem % 9) >> 31));
      kernel_rows[i_2] = (((9 & ((kernel_rem % 9) >> 31)) + (kernel_rem % 9)) / 3);
      kernel_cols[i_2] = ((3 & ((kernel_rem % 3) >> 31)) + (kernel_rem % 3));
    }
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      for (int vec_s = 0; vec_s < 4; ++vec_s) {
        int in_d = ((out_ds[((i_3 * 2) + (((int)threadIdx.x) >> 5))] + kernel_ds[vec_s]) - 1);
        int in_row = ((out_rows[((i_3 * 2) + (((int)threadIdx.x) >> 5))] + kernel_rows[vec_s]) - 1);
        int in_col = ((out_cols[((i_3 * 2) + (((int)threadIdx.x) >> 5))] + kernel_cols[vec_s]) - 1);
        int in_channel = in_channel_locals[vec_s];
        float condval;
        if ((((((((((bdhw_offsets[((i_3 * 2) + (((int)threadIdx.x) >> 5))] < 3276800) && (feat_offsets[vec_s] < 1728)) && (0 <= in_d)) && (in_d < 32)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320))) && (((((0 <= in_channel) && (in_channel < 64)) && (0 <= batch_ids[((i_3 * 2) + (((int)threadIdx.x) >> 5))])) && (batch_ids[((i_3 * 2) + (((int)threadIdx.x) >> 5))] < 1)))) {
          condval = input[(((((((int64_t)batch_ids[((((int64_t)i_3) * (int64_t)2) + (((int64_t)((int)threadIdx.x)) >> (int64_t)5))]) * (int64_t)209715200) + (((int64_t)in_channel) * (int64_t)3276800)) + (((int64_t)in_d) * (int64_t)102400)) + (((int64_t)in_row) * (int64_t)320)) + ((int64_t)in_col))];
        } else if (((((((((bdhw_offsets[((i_3 * 2) + (((int)threadIdx.x) >> 5))] < 3276800) && (feat_offsets[vec_s] < 1728)) && (0 <= in_d)) && (in_d < 32)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320))) {
          condval = 0x0p+0f/*0.000000e+00*/;
        }
        ((float*)input_tile)[(((((i_3 * 256) + ((((int)threadIdx.x) >> 2) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4)) + vec_s)] = (((((((((bdhw_offsets[((i_3 * 2) + (((int)threadIdx.x) >> 5))] < 3276800) && (feat_offsets[vec_s] < 1728)) && (0 <= in_d)) && (in_d < 32)) && (0 <= in_row)) && (in_row < 320)) && (0 <= in_col)) && (in_col < 320)) ? condval : 0x0p+0f/*0.000000e+00*/);
      }
    }
    if ((((int)threadIdx.x) >> 2) == 0) {
      #pragma unroll
      for (int i_4 = 0; i_4 < 256; ++i_4) {
        float condval_1;
        if (((feat_offsets[(i_4 >> 6)] < 1728)) && (((((((((0 <= kernel_cols[(i_4 >> 6)]) && (kernel_cols[(i_4 >> 6)] < 3)) && (0 <= kernel_rows[(i_4 >> 6)])) && (kernel_rows[(i_4 >> 6)] < 3)) && (0 <= kernel_ds[(i_4 >> 6)])) && (kernel_ds[(i_4 >> 6)] < 3)) && (0 <= in_channel_locals[(i_4 >> 6)])) && (in_channel_locals[(i_4 >> 6)] < 64)))) {
          condval_1 = weight[((((((((int64_t)i_4) & (int64_t)63) * (int64_t)1728) + (((int64_t)in_channel_locals[(((int64_t)i_4) >> (int64_t)6)]) * (int64_t)27)) + (((int64_t)kernel_ds[(((int64_t)i_4) >> (int64_t)6)]) * (int64_t)9)) + (((int64_t)kernel_rows[(((int64_t)i_4) >> (int64_t)6)]) * (int64_t)3)) + ((int64_t)kernel_cols[(((int64_t)i_4) >> (int64_t)6)]))];
        } else if ((feat_offsets[(i_4 >> 6)] < 1728)) {
          condval_1 = 0x0p+0f/*0.000000e+00*/;
        }
        ((float*)weight_tile)[(((((((((i_4 & 63) >> 5) * 512) + ((((int)threadIdx.x) & 3) * 128)) + ((i_4 >> 6) * 32)) + (((((i_4 & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + ((((i_4 >> 7) + ((i_4 & 15) >> 3)) & 1) * 8)) + (((((i_4 & 127) >> 6) + ((i_4 & 7) >> 2)) & 1) * 4)) + (i_4 & 3))] = ((feat_offsets[(i_4 >> 6)] < 1728) ? condval_1 : 0x0p+0f/*0.000000e+00*/);
      }
    }
    __syncthreads();
    {
      float A_local[32];
      float B_local[8];
      for (int ki = 0; ki < 2; ++ki) {
        for (int i_5 = 0; i_5 < 8; ++i_5) {
          tl::ptx_ldmatrix_x4((&(((float*)input_tile)[((((i_5 * 256) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4))])), (&(A_local[(i_5 * 4)])));
        }
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          for (int j = 0; j < 4; ++j) {
            B_local[((i_6 * 4) + j)] = ((float*)weight_tile)[(((((((((((int)threadIdx.x) >> 5) * 512) + (ki * 256)) + ((j & 1) * 128)) + ((((int)threadIdx.x) & 3) * 32)) + (((i_6 + (j & 1)) & 1) * 16)) + ((((j >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2))];
          }
        }
        for (int i_7 = 0; i_7 < 8; ++i_7) {
          for (int j_1 = 0; j_1 < 2; ++j_1) {
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + ((i_7 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_7 * 4)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 4)));
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + (((i_7 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_7 * 4)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 4) + 2)));
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 128; ++i_8) {
    int output_idx = ((((((((batch_ids[(i_8 >> 3)] * 209715200) + ((((int)threadIdx.x) >> 5) * 104857600)) + (((i_8 & 7) >> 1) * 26214400)) + ((((int)threadIdx.x) & 3) * 6553600)) + ((i_8 & 1) * 3276800)) + (out_ds[(i_8 >> 3)] * 102400)) + (out_rows[(i_8 >> 3)] * 320)) + out_cols[(i_8 >> 3)]);
    if (0 <= output_idx) {
      if (output_idx < 209715200) {
        output_flat[output_idx] = acc[(((((i_8 >> 4) * 16) + (((i_8 & 7) >> 1) * 4)) + (((i_8 & 15) >> 3) * 2)) + (i_8 & 1))];
      }
    }
  }
}

