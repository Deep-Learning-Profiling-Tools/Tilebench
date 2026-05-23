#!/bin/bash
#SBATCH --partition=contrib-B200
#SBATCH --qos=kzhou6
#SBATCH --gres=gpu:B200.180gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=tb_desc
#SBATCH --output=/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/desc_rerun_%j.out
#SBATCH --error=/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/desc_rerun_%j.err
#
# Submit with:  sbatch --export=ALL,OPS="argmax flash_attention ..." tilebench_run/descriptor_rerun.sbatch

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

eval "$(/projects/kzhou6/bcui2/env_software/miniconda3/bin/conda shell.bash hook)"
conda activate tilebench_env
module load git 2>/dev/null || true
unset PROMPT_COMMAND

cd /projects/kzhou6/bcui2/research/tilebench/Tilebench

# OPS env var carries the list of ops to run, e.g.
#   sbatch --export=ALL,OPS="argmax dropout flash_attention" descriptor_rerun.sbatch
if [ -z "${OPS:-}" ]; then
    echo "ERROR: sbatch must be submitted with --export=ALL,OPS=\"op1 op2 ...\"" >&2
    exit 1
fi

bash tilebench_run/descriptor_rerun.sh
