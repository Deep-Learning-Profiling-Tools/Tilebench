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

extern "C" __global__ void conv1d_kernel_kernel(const float* __restrict__ input, float* __restrict__ output_flat, const float* __restrict__ weight);
extern "C" __global__ void __launch_bounds__(64, 1) conv1d_kernel_kernel(const float* __restrict__ input, float* __restrict__ output_flat, const float* __restrict__ weight) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* input_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* weight_tile = ((void*)((char*)buf_dyn_shmem + 4096));
  float acc[64];
  int bl_offsets[8];
  int batch_ids[8];
  int out_ls[8];
  int feat_offsets[4];
  int in_channel_locals[4];
  int kernel_ls[4];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    bl_offsets[i_1] = (((((int)blockIdx.x) * 64) + (i_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2));
    batch_ids[i_1] = ((bl_offsets[i_1] / 2621440) + ((bl_offsets[i_1] % 2621440) >> 31));
    out_ls[i_1] = ((2621440 & ((bl_offsets[i_1] % 2621440) >> 31)) + (bl_offsets[i_1] % 2621440));
  }
  for (int feat_block = 0; feat_block < 24; ++feat_block) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      feat_offsets[i_2] = (((feat_block * 16) + ((((int)threadIdx.x) & 3) * 4)) + i_2);
      in_channel_locals[i_2] = ((feat_offsets[i_2] / 3) + ((feat_offsets[i_2] % 3) >> 31));
      kernel_ls[i_2] = ((3 & ((feat_offsets[i_2] % 3) >> 31)) + (feat_offsets[i_2] % 3));
    }
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      for (int vec_s = 0; vec_s < 4; ++vec_s) {
        int in_l = ((out_ls[((i_3 * 2) + (((int)threadIdx.x) >> 5))] + kernel_ls[vec_s]) - 1);
        int in_channel = in_channel_locals[vec_s];
        float condval;
        if ((((((bl_offsets[((i_3 * 2) + (((int)threadIdx.x) >> 5))] < 2621440) && (feat_offsets[vec_s] < 384)) && (0 <= in_l)) && (in_l < 2621440))) && (((((0 <= in_channel) && (in_channel < 128)) && (0 <= batch_ids[((i_3 * 2) + (((int)threadIdx.x) >> 5))])) && (batch_ids[((i_3 * 2) + (((int)threadIdx.x) >> 5))] < 1)))) {
          condval = input[(((((int64_t)batch_ids[((((int64_t)i_3) * (int64_t)2) + (((int64_t)((int)threadIdx.x)) >> (int64_t)5))]) * (int64_t)335544320) + (((int64_t)in_channel) * (int64_t)2621440)) + ((int64_t)in_l))];
        } else if (((((bl_offsets[((i_3 * 2) + (((int)threadIdx.x) >> 5))] < 2621440) && (feat_offsets[vec_s] < 384)) && (0 <= in_l)) && (in_l < 2621440))) {
          condval = 0x0p+0f/*0.000000e+00*/;
        }
        ((float*)input_tile)[(((((i_3 * 256) + ((((int)threadIdx.x) >> 2) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4)) + vec_s)] = (((((bl_offsets[((i_3 * 2) + (((int)threadIdx.x) >> 5))] < 2621440) && (feat_offsets[vec_s] < 384)) && (0 <= in_l)) && (in_l < 2621440)) ? condval : 0x0p+0f/*0.000000e+00*/);
      }
    }
    if ((((int)threadIdx.x) >> 2) == 0) {
      #pragma unroll
      for (int i_4 = 0; i_4 < 256; ++i_4) {
        float condval_1;
        if (((feat_offsets[(i_4 >> 6)] < 384)) && (((((0 <= kernel_ls[(i_4 >> 6)]) && (kernel_ls[(i_4 >> 6)] < 3)) && (0 <= in_channel_locals[(i_4 >> 6)])) && (in_channel_locals[(i_4 >> 6)] < 128)))) {
          condval_1 = weight[((((((int64_t)((int)blockIdx.y)) * (int64_t)24576) + ((((int64_t)i_4) & (int64_t)63) * (int64_t)384)) + (((int64_t)in_channel_locals[(((int64_t)i_4) >> (int64_t)6)]) * (int64_t)3)) + ((int64_t)kernel_ls[(((int64_t)i_4) >> (int64_t)6)]))];
        } else if ((feat_offsets[(i_4 >> 6)] < 384)) {
          condval_1 = 0x0p+0f/*0.000000e+00*/;
        }
        ((float*)weight_tile)[(((((((((i_4 & 63) >> 5) * 512) + ((((int)threadIdx.x) & 3) * 128)) + ((i_4 >> 6) * 32)) + (((((i_4 & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + ((((i_4 >> 7) + ((i_4 & 15) >> 3)) & 1) * 8)) + (((((i_4 & 127) >> 6) + ((i_4 & 7) >> 2)) & 1) * 4)) + (i_4 & 3))] = ((feat_offsets[(i_4 >> 6)] < 384) ? condval_1 : 0x0p+0f/*0.000000e+00*/);
      }
    }
    {
      float A_local[16];
      float B_local[8];
      __syncthreads();
      for (int ki = 0; ki < 2; ++ki) {
        for (int i_5 = 0; i_5 < 4; ++i_5) {
          tl::ptx_ldmatrix_x4((&(((float*)input_tile)[((((i_5 * 256) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4))])), (&(A_local[(i_5 * 4)])));
        }
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          for (int j = 0; j < 4; ++j) {
            B_local[((i_6 * 4) + j)] = ((float*)weight_tile)[(((((((((((int)threadIdx.x) >> 5) * 512) + (ki * 256)) + ((j & 1) * 128)) + ((((int)threadIdx.x) & 3) * 32)) + (((i_6 + (j & 1)) & 1) * 16)) + ((((j >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2))];
          }
        }
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          for (int j_1 = 0; j_1 < 2; ++j_1) {
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + ((i_7 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_7 * 4)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 4)));
            tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(acc + (((i_7 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_7 * 4)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 4) + 2)));
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 64; ++i_8) {
    int output_idx = (((((((batch_ids[(i_8 >> 3)] * 335544320) + (((int)blockIdx.y) * 167772160)) + ((((int)threadIdx.x) >> 5) * 83886080)) + (((i_8 & 7) >> 1) * 20971520)) + ((((int)threadIdx.x) & 3) * 5242880)) + ((i_8 & 1) * 2621440)) + out_ls[(i_8 >> 3)]);
    if (0 <= output_idx) {
      if (output_idx < 335544320) {
        output_flat[output_idx] = acc[(((((i_8 >> 4) * 16) + (((i_8 & 7) >> 1) * 4)) + (((i_8 & 15) >> 3) * 2)) + (i_8 & 1))];
      }
    }
  }
}

