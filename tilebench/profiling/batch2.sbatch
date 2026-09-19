#!/bin/bash
# Site-specific options (partition, QOS, GPU type) are given on the command line,
# from inside the repository and with the tilebench environment already active
# (sbatch exports the submitting environment to the job):
#   GPU=B200 sbatch --partition=<partition> --qos=<qos> --gres=gpu:<type>:1 \
#       tilebench/profiling/batch2.sbatch
# Slurm runs a copy of this file from its spool directory, so the repository is
# located from the submit directory, not from the script path.
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=tilebench_b2
#SBATCH --output=batch2-%j.out
#SBATCH --error=batch2-%j.err

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
module load git 2>/dev/null || true
unset PROMPT_COMMAND

REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)" || exit 1

# Batch 2 operators (22): mix of slow / medium / fast
OPS=(
  # slow (7): matmul / conv / attention / norm
  batched_matmul flash_decode linear_self_attention 3d_conv 1d_conv rmsnorm batch_normalization
  # medium (8): incl. 2 timeout (kl_divergence, histogramming)
  kl_divergence histogramming fused_activation top_k_selection jacobi_stencil_2d weight_dequant dequantize_rowwise quantize_global
  # fast (7)
  leaky_relu matrix_copy reverse_array interleave 2d_max_pooling moe_topk_gating radix_sort
)

exec bash "$REPO/tilebench/profiling/run_batch.sh" batch2 "${OPS[@]}"
