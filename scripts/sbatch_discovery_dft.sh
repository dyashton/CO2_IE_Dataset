#!/bin/bash
#SBATCH --job-name=co2_disc_dft
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_discovery_dft_%j.out
#SBATCH --error=logs/slurm_discovery_dft_%j.err

# DFT a selected.csv batch on scratch via run_parallel.py.
# Usage:
#   sbatch --export=ALL,SELECTED=$PWD/artifacts/discovery/round_001/selected.csv,\
#     RUN_DIR=$SCRATCHDIR/co2_discovery/round_001 \
#     scripts/sbatch_discovery_dft.sh
# Longer wall: sbatch --partition=long --qos=long --time=2-00:00:00 ...

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCHDIR:-/lustre/isaac24/scratch/$USER}/DFT_Calculations/uv-cache}"
mkdir -p "$UV_CACHE_DIR"

# Parent monorepo tools (CREST/xTB); CO2_IE_Dataset env.sh only sees local tools/.
TOOLS_ROOT="${CO2_TOOLS_ROOT:-$(cd "$ROOT/.." && pwd)/tools}"
if [[ -x "$TOOLS_ROOT/crest/crest" ]]; then
  export PATH="$TOOLS_ROOT/crest:$PATH"
  export CREST="$TOOLS_ROOT/crest/crest"
fi
if [[ -x "$TOOLS_ROOT/xtb/bin/xtb" ]]; then
  export PATH="$TOOLS_ROOT/xtb/bin:$PATH"
  export XTB="$TOOLS_ROOT/xtb/bin/xtb"
  export XTBPATH="${XTBPATH:-$TOOLS_ROOT/xtb/share/xtb}"
fi
if [[ -f "$ROOT/../scripts/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/../scripts/env.sh" || true
fi

SELECTED="${SELECTED:?set SELECTED to selected.csv}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to scratch run directory}"
JOBS="${JOBS:-6}"
THREADS="${THREADS:-5}"

mkdir -p "$RUN_DIR"
echo "[sbatch] host=$(hostname) selected=$SELECTED run_dir=$RUN_DIR crest=$(command -v crest || echo MISSING)"

if ! command -v crest >/dev/null 2>&1; then
  echo "[sbatch] crest not on PATH" >&2
  exit 1
fi

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
