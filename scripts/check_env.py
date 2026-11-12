#!/usr/bin/env python3
"""Check executables required by the trajectory workflow."""

from __future__ import annotations

import os
import shutil
import sys


def main() -> None:
    names = ("crest", "xtb", "psi4", "mctc-gcp", "dftd4")
    missing = []
    for name in names:
        configured = os.environ.get(name.upper().replace("-", "_"))
        path = configured if configured and os.path.isfile(configured) else shutil.which(name)
        status = path or "MISSING"
        print(f"{name:10s} {status}")
        if path is None:
            missing.append(name)
    if missing:
        print(f"missing: {', '.join(missing)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
