#!/usr/bin/env bash
# Install CREST Linux binary into tools/crest/crest (not conda/uv). Needs xtb on PATH.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/tools/crest"
BIN="$DEST/crest"
URL="https://github.com/crest-lab/crest/releases/download/latest/crest-gnu-12-ubuntu-latest.tar.xz"
ARCHIVE="/tmp/crest-gnu-12-ubuntu-latest.tar.xz"
EXTRACT="/tmp/crest_extract_$$"

if [[ -x "$BIN" && -f "$BIN" ]]; then
  echo "crest already at $BIN"
  "$BIN" --version || true
  exit 0
fi

echo "Downloading $URL"
curl -fsSL -L -o "$ARCHIVE" "$URL"
rm -rf "$DEST" "$EXTRACT"
mkdir -p "$DEST" "$EXTRACT"
tar -xJf "$ARCHIVE" -C "$EXTRACT"
FOUND="$(find "$EXTRACT" -type f -name crest | head -1)"
if [[ -z "$FOUND" ]]; then
  echo "crest binary not found in archive" >&2
  exit 1
fi
cp "$FOUND" "$BIN"
chmod +x "$BIN"
rm -rf "$EXTRACT"
echo "Installed:"
"$BIN" --version || true
echo
echo "Add to PATH:"
echo "  export PATH=\"$DEST:\$PATH\""
echo "CREST calls xtb; also:"
echo "  export PATH=\"$ROOT/tools/xtb/bin:\$PATH\""
echo "  export XTBPATH=\"$ROOT/tools/xtb/share/xtb\""
