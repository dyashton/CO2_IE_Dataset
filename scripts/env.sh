#!/usr/bin/env bash
# Portable local environment defaults. Cluster modules can override these.
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  SCRIPT_PATH="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  SCRIPT_PATH="${(%):-%N}"
else
  SCRIPT_PATH="$0"
fi
ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"

if [[ -x "$ROOT/tools/crest/crest" ]]; then
  export CREST="$ROOT/tools/crest/crest"
  export PATH="$(dirname "$CREST"):$PATH"
fi
if [[ -x "$ROOT/tools/xtb/bin/xtb" ]]; then
  export XTB="$ROOT/tools/xtb/bin/xtb"
  export XTBPATH="${XTBPATH:-$ROOT/tools/xtb/share/xtb}"
  export PATH="$(dirname "$XTB"):$PATH"
fi

export PSI4="${PSI4:-$(command -v psi4 || true)}"
export PSI4_EXE="${PSI4_EXE:-${PSI4:-}}"
export PSI4_NUM_THREADS="${PSI4_NUM_THREADS:-1}"
export PSI_NUM_THREADS="${PSI_NUM_THREADS:-$PSI4_NUM_THREADS}"
