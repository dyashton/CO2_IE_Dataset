#!/usr/bin/env bash
# Chain multi-round RL → AL select → DFT → ingest/retrain via Slurm dependencies.
#
# Usage (from CO2_IE_Dataset):
#   bash scripts/run_discovery_campaign.sh --from 1 --to 5
#   bash scripts/run_discovery_campaign.sh --from 2 --to 5          # resume
#   bash scripts/run_discovery_campaign.sh --from 1 --to 3 --dry-run
#
# Skips a round when energies.csv + summary.md already exist (unless --force).
# Round ids are round_001, round_002, ... (zero-padded).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FROM=1
TO=1
FORCE=0
DRY=0
MGW="${MGW:-$ROOT/../Molecule_Generation_Workflow}"
SCRATCH_ROOT="${SCRATCHDIR:-/lustre/isaac24/scratch/$USER}/co2_discovery"
DFT_PARTITION="${DFT_PARTITION:-campus}"
DFT_QOS="${DFT_QOS:-campus}"
DFT_TIME="${DFT_TIME:-24:00:00}"
ACCOUNT="${SLURM_ACCOUNT:-ACF-UTK0011}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --mgw) MGW="$2"; shift 2 ;;
    --scratch) SCRATCH_ROOT="$2"; shift 2 ;;
    --dft-partition) DFT_PARTITION="$2"; shift 2 ;;
    --dft-qos) DFT_QOS="$2"; shift 2 ;;
    --dft-time) DFT_TIME="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if (( FROM > TO )); then
  echo "[campaign] --from ($FROM) > --to ($TO)" >&2
  exit 1
fi
MGW="$(cd "$MGW" && pwd)"
if [[ ! -f "$MGW/scripts/hpc/sbatch_rl_generate.sh" ]]; then
  echo "[campaign] MGW generate sbatch missing under $MGW" >&2
  exit 1
fi
if [[ ! -f artifacts/binding_predictor/pack.joblib ]]; then
  echo "[campaign] missing artifacts/binding_predictor/pack.joblib" >&2
  exit 1
fi

mkdir -p logs "$MGW/logs" "$MGW/artifacts/discovery"
if [[ "$DRY" -eq 0 ]]; then
  mkdir -p "$SCRATCH_ROOT"
fi

# Avoid $(submit): command substitution subshells break the dry-run counter.
_LAST_JOB=""
submit() {
  # submit <description> <sbatch args...>  → sets _LAST_JOB
  local desc="$1"
  shift
  if [[ "$DRY" -eq 1 ]]; then
    _DRY_ID=$((_DRY_ID + 1))
    echo "[dry-run] $desc" >&2
    echo "  sbatch $*" >&2
    _LAST_JOB="dry${_DRY_ID}"
    return 0
  fi
  _LAST_JOB="$(sbatch --parsable "$@")"
  echo "[campaign] $desc -> job $_LAST_JOB" >&2
}

PREV_JOB=""
_DRY_ID=0

for ((n = FROM; n <= TO; n++)); do
  ROUND="$(printf 'round_%03d' "$n")"
  ROUND_DIR="$ROOT/artifacts/discovery/$ROUND"
  RUN_DIR="$SCRATCH_ROOT/$ROUND"
  PROPOSALS="$MGW/artifacts/discovery/${ROUND}_proposals.csv"
  SUMMARY="$ROUND_DIR/summary.md"
  ENERGIES="$RUN_DIR/energies.csv"

  mkdir -p "$ROUND_DIR"
  if [[ "$DRY" -eq 0 ]]; then
    mkdir -p "$RUN_DIR"
  fi

  if [[ "$FORCE" -eq 0 && -f "$ENERGIES" && -f "$SUMMARY" ]]; then
    echo "[campaign] skip $ROUND (energies + summary exist; pass --force to redo)"
    continue
  fi

  echo "[campaign] === $ROUND ==="

  dep=()
  if [[ -n "$PREV_JOB" ]]; then
    dep=(--dependency="afterok:${PREV_JOB}")
  fi

  # 1) RL generate (GPU) — per-round proposals path
  submit "$ROUND rl-generate" \
    "${dep[@]}" \
    --account="$ACCOUNT" \
    --chdir="$MGW" \
    --export="ALL,MGW_ROOT=$MGW,MGW_PROPOSALS_OUT=$PROPOSALS,MGW_BINDING_HEAD=ie,MGW_BINDING_STRONG_BONUS=${MGW_BINDING_STRONG_BONUS:-0.5}" \
    "$MGW/scripts/hpc/sbatch_rl_generate.sh"
  GEN_JOB="$_LAST_JOB"

  # 2) AL select (CPU)
  submit "$ROUND al-select" \
    --dependency="afterok:${GEN_JOB}" \
    --account="$ACCOUNT" \
    --chdir="$ROOT" \
    --export="ALL,ROUND_ID=$ROUND,RL_PROPOSALS=$PROPOSALS" \
    "$ROOT/scripts/sbatch_select_round.sh"
  SEL_JOB="$_LAST_JOB"

  # 3) DFT (CPU heavy)
  submit "$ROUND dft" \
    --dependency="afterok:${SEL_JOB}" \
    --account="$ACCOUNT" \
    --partition="$DFT_PARTITION" \
    --qos="$DFT_QOS" \
    --time="$DFT_TIME" \
    --chdir="$ROOT" \
    --export="ALL,SELECTED=$ROUND_DIR/selected.csv,RUN_DIR=$RUN_DIR" \
    "$ROOT/scripts/sbatch_discovery_dft.sh"
  DFT_JOB="$_LAST_JOB"

  # 4) Ingest + retrain + sync MGW pack
  submit "$ROUND ingest-retrain" \
    --dependency="afterok:${DFT_JOB}" \
    --account="$ACCOUNT" \
    --chdir="$ROOT" \
    --export="ALL,ROUND_ID=$ROUND,ENERGIES=$ENERGIES,SUMMARY=$SUMMARY" \
    "$ROOT/scripts/sbatch_ingest_retrain.sh"
  ING_JOB="$_LAST_JOB"

  PREV_JOB="$ING_JOB"
  echo "[campaign] $ROUND chain: gen=$GEN_JOB select=$SEL_JOB dft=$DFT_JOB ingest=$ING_JOB"
done

echo "[campaign] submitted rounds ${FROM}..${TO} (last job=${PREV_JOB:-none})"
echo "[campaign] monitor: squeue -u \$USER"
echo "[campaign] for longer DFT wall: DFT_PARTITION=long DFT_QOS=long DFT_TIME=2-00:00:00 $0 --from $FROM --to $TO"
