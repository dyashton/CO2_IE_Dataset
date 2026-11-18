#!/usr/bin/env bash
# Orchestrator helper: propose → gate/select → (optional) submit DFT.
# Usage:
#   bash scripts/run_discovery_round.sh round_000 \
#     [--rl-proposals /path/to/proposals.csv] [--submit]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ROUND="${1:?round id e.g. round_000}"
shift || true
RL_PROPOSALS=""
SUBMIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rl-proposals) RL_PROPOSALS="$2"; shift 2 ;;
    --submit) SUBMIT=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

DIR="artifacts/discovery/$ROUND"
mkdir -p "$DIR" logs

# Prefer growing discovery labels so later rounds exclude DFT hits.
SEED_CSV="data/compiled_gm_be.csv"
if [[ -f data/compiled_gm_be_discovery.csv ]]; then
  SEED_CSV="data/compiled_gm_be_discovery.csv"
fi

PROPOSE_ARGS=(--out "$DIR/proposals.csv" --seed-csv "$SEED_CSV")
if [[ -n "$RL_PROPOSALS" ]]; then
  PROPOSE_ARGS+=(--rl-proposals "$RL_PROPOSALS")
fi

uv run python scripts/propose_broad_hosts.py "${PROPOSE_ARGS[@]}"

uv run python scripts/gate_and_select_dft_batch.py \
  --proposals "$DIR/proposals.csv" \
  --out-gated "$DIR/gated.csv" \
  --out-selected "$DIR/selected.csv" \
  --pack artifacts/binding_predictor/pack.joblib

echo "[round] selected -> $DIR/selected.csv"
if [[ "$SUBMIT" -eq 1 ]]; then
  RUN_DIR="${SCRATCH:-$ROOT/runs}/co2_discovery/$ROUND"
  mkdir -p "$RUN_DIR"
  cat > "$DIR/submit.json" <<EOF
{"selected": "$DIR/selected.csv", "run_dir": "$RUN_DIR", "jobs": 6, "threads": 5}
EOF
  sbatch --export=ALL,SELECTED="$ROOT/$DIR/selected.csv",RUN_DIR="$RUN_DIR" \
    scripts/sbatch_discovery_dft.sh
  echo "[round] submitted DFT to $RUN_DIR"
fi
