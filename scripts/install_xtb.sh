#!/usr/bin/env bash
# Install official xtb 6.7.1 Linux binary into tools/xtb/ (not conda/uv).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/tools/xtb"
URL="https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz"
ARCHIVE="/tmp/xtb-6.7.1-linux-x86_64.tar.xz"

mkdir -p "$ROOT/tools"
if [[ -x "$DEST/bin/xtb" ]]; then
  echo "xtb already at $DEST/bin/xtb"
  "$DEST/bin/xtb" --version
  exit 0
fi

echo "Downloading $URL"
curl -fsSL -o "$ARCHIVE" "$URL"
rm -rf "$DEST"
mkdir -p "$ROOT/tools"
tar -xJf "$ARCHIVE" -C "$ROOT/tools"
if [[ -d "$ROOT/tools/xtb-dist" ]]; then
  mv "$ROOT/tools/xtb-dist" "$DEST"
fi
chmod +x "$DEST/bin/xtb"
echo "Installed:"
"$DEST/bin/xtb" --version
echo
echo "Add to PATH:"
echo "  export PATH=\"$DEST/bin:\$PATH\""
echo "  export XTBPATH=\"$DEST/share/xtb\""
