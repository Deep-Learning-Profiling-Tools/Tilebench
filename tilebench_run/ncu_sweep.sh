#!/usr/bin/env bash
# One-shot NCU sweep — run from a B200 node.
#
# Profiles every (op, dtype, backend) triple in tilebench_run/ncu_catalogue.json
# (~45 ops × 3 dtypes × 2 backends ≈ 270 NCU reports), then post-processes them
# into per-op markdown writeups.
#
# Outputs:
#   tilebench_run/ncu/<op>/<backend>_<dtype>.ncu-rep      raw NCU reports
#   tilebench_run/ncu/sweep_log.json                       per-pair status
#   tilebench_run/ncu/sweep_failures.md                    failures, if any
#   results/ncu_comparison/<op>.md                         per-op markdown
#
# Resumable: if a .ncu-rep already exists or sweep_log.json marks the pair as
# ok, ncu_driver.py skips it. Just re-run to fill in failures.
#
# Usage (on a B200 node):
#   bash tilebench_run/ncu_sweep.sh
#
# Background + log:
#   nohup bash tilebench_run/ncu_sweep.sh > /tmp/ncu_sweep.log 2>&1 &
#   tail -f /tmp/ncu_sweep.log

set -euo pipefail

# --- env ---
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Activate conda env. If `conda activate` isn't already wired into this shell
# (typical for non-interactive bash), source the hook first. Adjust CONDA_BASE
# below if your conda lives elsewhere.
if ! command -v conda >/dev/null 2>&1; then
    CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate tilebench_env

module load git 2>/dev/null || true
unset PROMPT_COMMAND

# --- locate repo root (one level up from this script) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "[ncu_sweep] start $(date -Iseconds)"
echo "[ncu_sweep] repo: $REPO_ROOT"
echo "[ncu_sweep] gpu:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

# 1) NCU on every (op, dtype, backend). Auto-skips already-done pairs.
PYTHONPATH=. python -u tilebench_run/ncu_driver.py

# 2) Per-op markdown writeups under results/ncu_comparison/.
PYTHONPATH=. python -u tilebench_run/ncu_writeup.py

echo "============================================================"
echo "[ncu_sweep] DONE $(date -Iseconds)"
echo "  raw reports:     tilebench_run/ncu/<op>/<backend>_<dtype>.ncu-rep"
echo "  per-op writeup:  results/ncu_comparison/<op>.md"
echo "  sweep log:       tilebench_run/ncu/sweep_log.json"
echo "  failures (if any): tilebench_run/ncu/sweep_failures.md"
echo "============================================================"
