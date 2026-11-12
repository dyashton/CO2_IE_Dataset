#!/usr/bin/env bash
# Tar only the discovery-hpc manifest (source + small CSVs). Joblibs stay separate.
# Usage:
#   bash scripts/pack_hpc_bundle.sh
#   bash scripts/pack_hpc_bundle.sh /path/to/co2_ie_discovery_hpc.tar.gz
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/scripts/hpc_manifest.txt"
OUT="${1:-$ROOT/artifacts/co2_ie_discovery_hpc.tar.gz}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "missing $MANIFEST" >&2
  exit 1
fi

mapfile -t PATHS < <(grep -vE '^\s*(#|$)' "$MANIFEST")
missing=0
for p in "${PATHS[@]}"; do
  if [[ ! -e "$ROOT/$p" ]]; then
    echo "missing: $p" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
tmp="$(mktemp -d)"
stage="$tmp/CO2_IE_Dataset"
mkdir -p "$stage"
for p in "${PATHS[@]}"; do
  mkdir -p "$stage/$(dirname "$p")"
  cp -a "$ROOT/$p" "$stage/$p"
done
tar -czf "$OUT" -C "$tmp" CO2_IE_Dataset
rm -rf "$tmp"
ls -lh "$OUT"
echo "On HPC:"
echo "  tar -xzf $(basename "$OUT") && cd CO2_IE_Dataset && git checkout discovery-hpc  # or use this tarball alone"
echo "Prefer: git clone + git checkout discovery-hpc (versioned)."
echo "Then: bash scripts/fetch_binding_pack.sh /path/to/binding_predictor.tar.gz"
