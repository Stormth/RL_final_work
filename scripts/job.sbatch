#!/bin/bash
# Slurm 提交模板：把命令原样传给脚本，例如
#   sbatch --gres=gpu:1 scripts/job.sbatch python -m rl_algo.eval.baseline --data data/processed/test.jsonl --out results/base
# 需要按集群改的只有 --partition / --account / conda 路径。
#SBATCH --job-name=rl-algo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# conda（按实际路径改）
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate rl-algo 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results outputs

echo "===== job ${SLURM_JOB_ID:-local} on $(hostname) ====="
nvidia-smi || true
echo "===== command ====="
printf '%q ' "$@"; echo

"$@"
