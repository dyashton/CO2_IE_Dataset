from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from co2ie import PSI4_METHOD
from co2ie.conformers import Xyz

ENERGY_RE = [
    re.compile(r"@DF-\S+\s+Final Energy:\s+([-+]?\d+\.\d+)", re.I),
    re.compile(r"Final Energy:\s+([-+]?\d+\.\d+)", re.I),
    re.compile(r"total energy\s*=\s*([-+]?\d+\.\d+)", re.I),
]


def write_input(xyz: Xyz, path: Path, *, charge: int = 0, multiplicity: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    geom = "\n".join(
        f"  {s:2s} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}"
        for s, p in zip(xyz.symbols, xyz.coords)
    )
    text = f"""memory 2 GB

molecule {{
{charge} {multiplicity}
{geom}
}}

set {{
  scf_type df
}}

energy('{PSI4_METHOD}')
"""
    path.write_text(text)


def parse_energy(text: str) -> float | None:
    for rx in ENERGY_RE:
        m = rx.search(text)
        if m:
            return float(m.group(1))
    return None


def psi4_bin() -> str | None:
    return shutil.which("psi4")


def run_psi4(inp: Path, timeout: int = 3600) -> float | None:
    exe = psi4_bin()
    if exe is None:
        return None
    inp = inp.resolve()
    out = inp.with_suffix(".out")
    if out.is_file():
        out.unlink()
    # cwd is inp.parent; pass basenames so relative --run-dir paths work
    proc = subprocess.run(
        [exe, inp.name, out.name],
        cwd=inp.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    blob = ""
    if out.is_file():
        # reject stale/failed runs that left an empty or ancient out
        if out.stat().st_mtime + 1 < inp.stat().st_mtime:
            return None
        blob += out.read_text()
    blob += "\n" + (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode not in (0, None) and "Final Energy" not in blob:
        return None
    return parse_energy(blob)
