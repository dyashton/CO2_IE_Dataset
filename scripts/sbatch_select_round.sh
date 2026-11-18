#!/bin/bash
#SBATCH --job-name=co2_al_select
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_select_%j.out
#SBATCH --error=logs/slurm_select_%j.err

# AL propose+select for one round.
#   sbatch --export=ALL,ROUND_ID=round_001,RL_PROPOSALS=/path/to/proposals.csv \
#     scripts/sbatch_select_round.sh

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCHDIR:-/lustre/isaac24/scratch/$USER}/DFT_Calculations/uv-cache}"
mkdir -p "$UV_CACHE_DIR"

ROUND_ID="${ROUND_ID:?set ROUND_ID e.g. round_001}"
RL_PROPOSALS="${RL_PROPOSALS:?set RL_PROPOSALS to MGW proposals.csv}"

echo "[select] host=$(hostname) round=$ROUND_ID rl=$RL_PROPOSALS"
if [[ ! -f "$RL_PROPOSALS" ]]; then
  echo "[select] missing RL proposals: $RL_PROPOSALS" >&2
  exit 1
fi

bash scripts/run_discovery_round.sh "$ROUND_ID" --rl-proposals "$RL_PROPOSALS"
echo "[select] done -> artifacts/discovery/${ROUND_ID}/selected.csv"
