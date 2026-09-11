from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from co2ie import ROOT
from co2ie.conformers import read_xyz, write_xyz

ENERGY_RE = [
    re.compile(r"TOTAL ENERGY\s+([-+]?\d+\.\d+)\s*Eh", re.I),
    re.compile(r"\|\s*TOTAL ENERGY\s+([-+]?\d+\.\d+)\s+Eh", re.I),
]


@dataclass
class XtbOpt:
    xyz_path: Path
    energy: float | None
    ok: bool


def xtb_bin(root: Path | None = None) -> Path | None:
    root = root or ROOT
    local = root / "tools" / "xtb" / "bin" / "xtb"
    if local.is_file() and os.access(local, os.X_OK):
        return local
    found = shutil.which("xtb")
    return Path(found) if found else None


def xtb_env(root: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    root = root or ROOT
    share = root / "tools" / "xtb" / "share" / "xtb"
    if share.is_dir():
        env["XTBPATH"] = str(share)
        env["PATH"] = str(root / "tools" / "xtb" / "bin") + os.pathsep + env.get("PATH", "")
    return env


def _parse_energy(text: str) -> float | None:
    for rx in ENERGY_RE:
        m = rx.search(text)
        if m:
            return float(m.group(1))
    return None


def opt_xyz(
    xyz_in: Path,
    xyz_out: Path,
    *,
    timeout: int = 180,
    fix_atoms: list[int] | None = None,
    constrain_co2: bool = False,
    anchor_atom: int | None = None,
    anchor_distance: float | None = None,
) -> XtbOpt:
    """GFN2-xTB geometry opt.

    constrain_co2: freeze CO2 internals (last 3 atoms = C,O,O) via distance/angle.
    anchor_atom/distance: soft tether of CO2 carbon to a host atom (1-based in file).
    fix_atoms: legacy soft $fix (prefer constrain_co2).
    """
    bin_ = xtb_bin()
    if bin_ is None:
        raise FileNotFoundError("xtb not found; run bash scripts/install_xtb.sh")
    with tempfile.TemporaryDirectory(prefix="xtb_") as td:
        work = Path(td)
        src = work / "start.xyz"
        shutil.copy(xyz_in, src)
        cmd = [str(bin_), str(src), "--gfn", "2", "--opt", "--norestart"]
        lines: list[str] = []
        if constrain_co2:
            xyz = read_xyz(xyz_in)
            n = xyz.n_atoms
            i_c, i_o1, i_o2 = n - 3, n - 2, n - 1
            lines.append("$constrain")
            lines.append("force constant=2.0")
            lines.append(f"distance: {i_c + 1}, {i_o1 + 1}, 1.16")
            lines.append(f"distance: {i_c + 1}, {i_o2 + 1}, 1.16")
            lines.append(f"angle: {i_o1 + 1}, {i_c + 1}, {i_o2 + 1}, 180.0")
            if anchor_atom is not None and anchor_distance is not None and anchor_atom >= 0:
                lines.append(
                    f"distance: {anchor_atom + 1}, {i_c + 1}, {anchor_distance:.4f}"
                )
            lines.append("$end")
        elif fix_atoms:
            ones = ",".join(str(i + 1) for i in fix_atoms)
            lines.append("$fix")
            lines.append(f"atoms: {ones}")
            lines.append("$end")
        if lines:
            (work / "xcontrol").write_text("\n".join(lines) + "\n")
            cmd.extend(["--input", "xcontrol"])
        proc = subprocess.run(
            cmd,
            cwd=work,
            env=xtb_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        opt = work / "xtbopt.xyz"
        if not opt.is_file():
            return XtbOpt(xyz_path=xyz_out, energy=None, ok=False)
        xyz = read_xyz(opt)
        xyz.comment = f"gfn2-xtb opt from {xyz_in.name}"
        write_xyz(xyz, xyz_out)
        e = _parse_energy(log)
        if e is None:
            e = _parse_energy(opt.read_text())
        if e is not None:
            xyz_out.with_suffix(".energy").write_text(f"{e:.12f}\n")
        return XtbOpt(xyz_path=xyz_out, energy=e, ok=True)


def sp_xyz(xyz_in: Path, *, timeout: int = 120) -> float | None:
    """Single-point GFN2-xTB energy (Hartree)."""
    bin_ = xtb_bin()
    if bin_ is None:
        raise FileNotFoundError("xtb not found; run bash scripts/install_xtb.sh")
    with tempfile.TemporaryDirectory(prefix="xtb_sp_") as td:
        work = Path(td)
        src = work / "mol.xyz"
        shutil.copy(xyz_in, src)
        proc = subprocess.run(
            [str(bin_), str(src), "--gfn", "2", "--norestart"],
            cwd=work,
            env=xtb_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return _parse_energy(log)



def opt_conformers(mol_dir: Path) -> list[XtbOpt]:
    results: list[XtbOpt] = []
    for p in sorted(mol_dir.glob("conf_[0-9][0-9].xyz")):
        results.append(opt_xyz(p, mol_dir / f"{p.stem}.xtb.xyz"))
    return results
