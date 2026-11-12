#!/usr/bin/env bash
# Build a portable tarball of binding_predictor joblibs for HPC fetch.
# Core pack only (~230MB uncompressed / ~70MB gz) — skips begm_direct.
# Usage:
#   bash scripts/pack_binding_artifacts.sh
#   bash scripts/pack_binding_artifacts.sh /path/to/out.tar.gz
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${BINDING_PACK_SRC:-$ROOT/artifacts/binding_predictor}"
OUT="${1:-$ROOT/artifacts/binding_predictor.tar.gz}"

if [[ ! -d "$SRC" ]]; then
  echo "missing source dir: $SRC" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
tmp="$(mktemp -d)"
mkdir -p "$tmp/binding_predictor"
REQUIRED=(pack.joblib ie.joblib be.joblib penalty.joblib strong.joblib)
OPTIONAL=(metrics.json oof.csv ablations.json begm.joblib)
for f in "${REQUIRED[@]}"; do
  if [[ ! -f "$SRC/$f" ]]; then
    echo "missing required $SRC/$f" >&2
    rm -rf "$tmp"
    exit 1
  fi
  cp -f "$SRC/$f" "$tmp/binding_predictor/$f"
done
for f in "${OPTIONAL[@]}"; do
  [[ -f "$SRC/$f" ]] && cp -f "$SRC/$f" "$tmp/binding_predictor/$f" || true
done
tar -czf "$OUT" -C "$tmp" binding_predictor
rm -rf "$tmp"
ls -lh "$OUT"
echo "Copy to HPC scratch / GitHub Releases, then:"
echo "  bash scripts/fetch_binding_pack.sh $OUT"
