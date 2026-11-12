#!/usr/bin/env bash
# Install binding_predictor joblibs into artifacts/binding_predictor/.
# Usage:
#   bash scripts/fetch_binding_pack.sh
#   bash scripts/fetch_binding_pack.sh /path/to/binding_predictor.tar.gz
#   bash scripts/fetch_binding_pack.sh /path/to/binding_predictor/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${BINDING_PACK_DIR:-$ROOT/artifacts/binding_predictor}"
# Prefer sibling MGW tarball path used on ISAAC after RL setup.
MGW_TAR="$ROOT/../Molecule_Generation_Workflow/artifacts/binding_predictor.tar.gz"
MGW_DIR="$ROOT/../Molecule_Generation_Workflow/artifacts/binding_predictor"
NEEDED=(pack.joblib ie.joblib be.joblib penalty.joblib strong.joblib)

mkdir -p "$DEST"

copy_dir() {
  local src="$1"
  echo "[fetch] copying from $src -> $DEST"
  for f in "${NEEDED[@]}"; do
    if [[ ! -f "$src/$f" ]]; then
      echo "[fetch] missing required $src/$f" >&2
      exit 1
    fi
    cp -f "$src/$f" "$DEST/$f"
  done
  for f in metrics.json oof.csv ablations.json begm.joblib; do
    [[ -f "$src/$f" ]] && cp -f "$src/$f" "$DEST/$f" || true
  done
}

extract_archive() {
  local archive="$1"
  local tmp
  tmp="$(mktemp -d)"
  echo "[fetch] extracting $archive"
  tar -xzf "$archive" -C "$tmp"
  if [[ -f "$tmp/pack.joblib" ]]; then
    copy_dir "$tmp"
  elif [[ -f "$tmp/binding_predictor/pack.joblib" ]]; then
    copy_dir "$tmp/binding_predictor"
  else
    found="$(find "$tmp" -name pack.joblib | head -1 || true)"
    if [[ -z "$found" ]]; then
      echo "[fetch] pack.joblib not found in archive" >&2
      rm -rf "$tmp"
      exit 1
    fi
    copy_dir "$(dirname "$found")"
  fi
  rm -rf "$tmp"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,8p' "$0"
  exit 0
fi

if [[ $# -ge 1 ]]; then
  SRC="$1"
  if [[ -d "$SRC" ]]; then
    copy_dir "$SRC"
  elif [[ -f "$SRC" ]]; then
    extract_archive "$SRC"
  else
    echo "[fetch] not a file or directory: $SRC" >&2
    exit 1
  fi
elif [[ -f "$DEST/pack.joblib" ]]; then
  echo "[fetch] pack already present at $DEST"
elif [[ -f "$ROOT/artifacts/binding_predictor.tar.gz" ]]; then
  extract_archive "$ROOT/artifacts/binding_predictor.tar.gz"
elif [[ -f "$MGW_TAR" ]]; then
  extract_archive "$MGW_TAR"
elif [[ -d "$MGW_DIR" && -f "$MGW_DIR/pack.joblib" ]]; then
  copy_dir "$MGW_DIR"
else
  echo "[fetch] no pack found. Build one with:" >&2
  echo "  bash scripts/pack_binding_artifacts.sh" >&2
  echo "Then: bash scripts/fetch_binding_pack.sh /path/to/binding_predictor.tar.gz" >&2
  exit 1
fi

ls -lh "$DEST"
