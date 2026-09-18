#!/bin/bash
#SBATCH --partition=contrib-B200
#SBATCH --qos=kzhou6
#SBATCH --gres=gpu:B200.180gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=tilebench_b2
#SBATCH --output=/projects/kzhou6/bcui2/research/tilebench/Tilebench/outputs/batch2-%j.out
#SBATCH --error=/projects/kzhou6/bcui2/research/tilebench/Tilebench/outputs/batch2-%j.err

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
module load git 2>/dev/null || true
unset PROMPT_COMMAND

eval "$(/projects/kzhou6/bcui2/env_software/miniconda3/bin/conda shell.bash hook)"
conda activate tilebench_env

cd /projects/kzhou6/bcui2/research/tilebench/Tilebench

# Batch 2 operators (22): mix of slow / medium / fast
OPS=(
  # slow (7): matmul / conv / attention / norm
  batched_matmul flash_decode linear_self_attention 3d_conv 1d_conv rmsnorm batch_normalization
  # medium (8): incl. 2 timeout (kl_divergence, histogramming)
  kl_divergence histogramming fused_activation top_k_selection jacobi_stencil_2d weight_dequant dequantize_rowwise quantize_global
  # fast (7)
  leaky_relu matrix_copy reverse_array interleave 2d_max_pooling moe_topk_gating radix_sort
)

bash tilebench/profiling/run_batch.sh batch2 "${OPS[@]}"
