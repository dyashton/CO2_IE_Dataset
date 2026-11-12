from __future__ import annotations

import os
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


def write_optimization_input(
    xyz: Xyz,
    path: Path,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    memory: str = "2 GB",
    geom_maxiter: int = 100,
) -> None:
    """Write an unconstrained r²SCAN-3c Psi4 geometry optimization input."""
    path.parent.mkdir(parents=True, exist_ok=True)
    geom = "\n".join(
        f"  {s:2s} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}"
        for s, p in zip(xyz.symbols, xyz.coords)
    )
    path.write_text(
        f"""memory {memory}

molecule {{
{charge} {multiplicity}
{geom}
}}

set {{
  scf_type df
  geom_maxiter {geom_maxiter}
  g_convergence gau_tight
}}

optimize('{PSI4_METHOD}')
"""
    )


def run_psi4_optimization(
    inp: Path, *, timeout: int = 3600, threads: int | None = None
) -> Path:
    """Run Psi4 optimization and return its output path."""
    exe = psi4_bin()
    if exe is None:
        raise FileNotFoundError("psi4 not found on PATH")
    inp = inp.resolve()
    out = inp.with_suffix(".out")
    env = os.environ.copy()
    conda_lib = Path(exe).resolve().parent.parent / "lib"
    if conda_lib.is_dir():
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(conda_lib), env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)
    thread_count = str(threads) if threads and threads > 0 else (
        env.get("PSI_NUM_THREADS") or env.get("PSI4_NUM_THREADS")
    )
    command = [str(exe), "-i", inp.name, "-o", out.name]
    if thread_count and thread_count.isdigit():
        command.extend(["-n", thread_count])
    proc = subprocess.run(
        command,
        cwd=inp.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError(
            f"Psi4 optimization failed for {inp.name}: "
            f"{detail[-1] if detail else f'exit code {proc.returncode}'}"
        )
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(f"Psi4 produced no output for {inp.name}")
    return out


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
    threads = os.environ.get("PSI_NUM_THREADS") or os.environ.get("PSI4_NUM_THREADS")
    env = os.environ.copy()
    conda_lib = Path(exe).resolve().parent.parent / "lib"
    if conda_lib.is_dir():
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(conda_lib), env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)
    proc = subprocess.run(
        [
            str(exe),
            "-i",
            inp.name,
            "-o",
            out.name,
            *(["-n", threads] if threads and threads.isdigit() else []),
        ],
        cwd=inp.parent,
        env=env,
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
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError(
            f"Psi4 failed for {inp.name}: "
            f"{detail[-1] if detail else f'exit code {proc.returncode}'}"
        )
    energy = parse_energy(blob)
    if energy is None:
        raise RuntimeError(f"Psi4 produced no parseable energy for {inp.name}")
    return energy
