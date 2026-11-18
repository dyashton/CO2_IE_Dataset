#!/bin/bash
#SBATCH --job-name=co2_ingest_rt
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_ingest_retrain_%j.out
#SBATCH --error=logs/slurm_ingest_retrain_%j.err

# After DFT finishes:
#   sbatch --dependency=afterok:<DFT_JOBID> scripts/sbatch_ingest_retrain.sh
#
# Optional exports:
#   ENERGIES  (default: $SCRATCHDIR/co2_discovery/round_000/energies.csv)
#   ROUND_ID  (default: round_000)

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCHDIR:-/lustre/isaac24/scratch/$USER}/DFT_Calculations/uv-cache}"
mkdir -p "$UV_CACHE_DIR"

ROUND_ID="${ROUND_ID:-round_000}"
ENERGIES="${ENERGIES:-${SCRATCHDIR:-/lustre/isaac24/scratch/$USER}/co2_discovery/${ROUND_ID}/energies.csv}"
SUMMARY="${SUMMARY:-$ROOT/artifacts/discovery/${ROUND_ID}/summary.md}"

echo "[ingest_retrain] host=$(hostname) energies=$ENERGIES round=$ROUND_ID"
if [[ ! -f "$ENERGIES" ]]; then
  echo "[ingest_retrain] missing energies.csv: $ENERGIES" >&2
  exit 1
fi

uv run python scripts/ingest_discovery_round.py \
  --energies "$ENERGIES" \
  --round-id "$ROUND_ID" \
  --summary "$SUMMARY"

uv run python scripts/retrain_binding_after_round.py --sync-mgw

echo "[ingest_retrain] done"
