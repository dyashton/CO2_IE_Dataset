#!/bin/bash
#SBATCH --job-name=co2_disc_dft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_discovery_dft_%j.out
#SBATCH --error=logs/slurm_discovery_dft_%j.err

# DFT a selected.csv batch on scratch via run_parallel.py.
# Usage:
#   sbatch --export=ALL,SELECTED=artifacts/discovery/round_000/selected.csv,\
#     RUN_DIR=$SCRATCH/co2_discovery/round_000 \
#     scripts/sbatch_discovery_dft.sh

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

SELECTED="${SELECTED:?set SELECTED to selected.csv}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to scratch run directory}"
JOBS="${JOBS:-6}"
THREADS="${THREADS:-5}"

mkdir -p "$RUN_DIR"
echo "[sbatch] host=$(hostname) selected=$SELECTED run_dir=$RUN_DIR"

uv run python scripts/run_parallel.py \
  --csv "$SELECTED" \
  --n 0 \
  --jobs "$JOBS" \
  --threads "$THREADS" \
  --quick \
  --to dft \
  --run-dir "$RUN_DIR"

echo "[sbatch] done. ingest with:"
echo "  uv run python scripts/ingest_discovery_round.py --energies $RUN_DIR/energies.csv"
