#!/bin/bash
#SBATCH --partition=contrib-B200
#SBATCH --qos=kzhou6
#SBATCH --gres=gpu:B200.180gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=tb_llm_claude
#SBATCH --output=/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/llm_sweep_claude-%j.out
#SBATCH --error=/projects/kzhou6/bcui2/research/tilebench/Tilebench/tilebench_run/llm_sweep_claude-%j.err

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

eval "$(/projects/kzhou6/bcui2/env_software/miniconda3/bin/conda shell.bash hook)"
conda activate tilebench_env
module load git 2>/dev/null || true
unset PROMPT_COMMAND

cd /projects/kzhou6/bcui2/research/tilebench/Tilebench

# CLAUDE_API_KEY is inherited from the submitting shell (sbatch default
# --export=ALL). Verify it's present so we fail fast rather than burn a
# job slot just to crash 3s after spawn.
if [ -z "${CLAUDE_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: CLAUDE_API_KEY / ANTHROPIC_API_KEY not exported into sbatch env" >&2
    exit 1
fi

bash tilebench_run/llm_sweep.sh claude-opus-4-7 high 10 0.80
