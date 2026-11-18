#!/usr/bin/env bash
# Slurm-free discovery round: RL generate → AL select → DFT → ingest/retrain.
#
# Usage (from CO2_IE_Dataset, with sibling Molecule_Generation_Workflow):
#   bash scripts/run_local_round.sh round_003
#   JOBS=3 THREADS=4 bash scripts/run_local_round.sh round_004
#
# Laptop defaults (32 GB): JOBS=3 THREADS=4
# Cluster-style: JOBS=6 THREADS=5
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ROUND="${1:?round id e.g. round_003}"
MGW="${MGW:-$ROOT/../Molecule_Generation_Workflow}"
MGW="$(cd "$MGW" && pwd)"
SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCHDIR:-${HOME}/co2_discovery_runs}}"
RUN_DIR="${RUN_DIR:-$SCRATCH_ROOT/$ROUND}"
JOBS="${JOBS:-3}"
THREADS="${THREADS:-4}"
PROPOSALS="${MGW_PROPOSALS_OUT:-$MGW/artifacts/discovery/${ROUND}_proposals.csv}"

if [[ ! -f "$MGW/scripts/run_co2_binding_generate.sh" ]]; then
  echo "[local] MGW generate script missing under $MGW" >&2
  exit 1
fi
if [[ ! -f artifacts/binding_predictor/pack.joblib ]]; then
  echo "[local] missing artifacts/binding_predictor/pack.joblib — run scripts/fetch_binding_pack.sh" >&2
  exit 1
fi

mkdir -p logs artifacts/discovery/"$ROUND" "$RUN_DIR" "$MGW/artifacts/discovery"

export MGW_ROOT="$MGW"
export MGW_PROPOSALS_OUT="$PROPOSALS"
export MGW_OBJECTIVE="${MGW_OBJECTIVE:-co2_binding_surrogate}"
export MGW_BINDING_PACK="${MGW_BINDING_PACK:-$MGW/artifacts/binding_predictor/pack.joblib}"
export MGW_BINDING_HEAD="${MGW_BINDING_HEAD:-ie}"
export MGW_BINDING_STRONG_BONUS="${MGW_BINDING_STRONG_BONUS:-0.5}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_ROOT/uv-cache}"
mkdir -p "$UV_CACHE_DIR"

# Prefer CO2_IE pack if MGW pack is missing (fresh laptop clone).
if [[ ! -f "$MGW_BINDING_PACK" && -f artifacts/binding_predictor/pack.joblib ]]; then
  echo "[local] syncing binding pack CO2_IE → MGW"
  bash "$MGW/scripts/fetch_binding_pack.sh" "$ROOT/artifacts/binding_predictor"
fi

echo "[local] === $ROUND ==="
echo "[local] MGW=$MGW jobs=$JOBS threads=$THREADS run_dir=$RUN_DIR"

echo "[local] 1/4 RL generate → $PROPOSALS"
(
  cd "$MGW"
  bash scripts/run_co2_binding_generate.sh
)

echo "[local] 2/4 AL propose + UCB select"
bash scripts/run_discovery_round.sh "$ROUND" --rl-proposals "$PROPOSALS"
SELECTED="$ROOT/artifacts/discovery/$ROUND/selected.csv"
if [[ ! -f "$SELECTED" ]]; then
  echo "[local] missing selected.csv: $SELECTED" >&2
  exit 1
fi

echo "[local] 3/4 DFT via run_parallel.py"
# Parent monorepo tools if present (Isaac); local installs usually already on PATH.
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
if [[ -f "$ROOT/scripts/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/env.sh" || true
fi
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export PSI_NUM_THREADS="$THREADS"
export PSI4_NUM_THREADS="$THREADS"

uv run python scripts/run_parallel.py \
  --csv "$SELECTED" \
  --n 0 \
  --jobs "$JOBS" \
  --threads "$THREADS" \
  --quick \
  --to dft \
  --run-dir "$RUN_DIR"

ENERGIES="$RUN_DIR/energies.csv"
if [[ ! -f "$ENERGIES" ]]; then
  echo "[local] missing energies.csv: $ENERGIES" >&2
  exit 1
fi

echo "[local] 4/4 ingest + retrain + sync MGW pack"
SUMMARY="$ROOT/artifacts/discovery/$ROUND/summary.md"
uv run python scripts/ingest_discovery_round.py \
  --energies "$ENERGIES" \
  --round-id "$ROUND" \
  --summary "$SUMMARY"
uv run python scripts/retrain_binding_after_round.py --sync-mgw

echo "[local] done $ROUND"
echo "[local] selected=$SELECTED"
echo "[local] energies=$ENERGIES"
echo "[local] summary=$SUMMARY"
