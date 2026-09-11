from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from co2ie import ROOT
from co2ie.conformers import Xyz, split_multi_xyz, write_xyz
from co2ie.xtb import xtb_env

ENERGY_IN_COMMENT = re.compile(r"[-+]?\d+\.\d+")


@dataclass
class CrestConf:
    path: Path
    energy: float | None
    index: int


def crest_bin(root: Path | None = None) -> Path | None:
    root = root or ROOT
    local = root / "tools" / "crest" / "crest"
    if local.is_file() and os.access(local, os.X_OK):
        return local
    found = shutil.which("crest")
    return Path(found) if found else None


def _parse_comment_energy(comment: str) -> float | None:
    nums = [float(x) for x in ENERGY_IN_COMMENT.findall(comment)]
    if not nums:
        return None
    hartree = [x for x in nums if x < -1.0]
    if hartree:
        return min(hartree)
    return nums[0]


def parse_crest_energies(path: Path) -> list[float | None]:
    """Parse crest.energies if present. Values may be relative kcal/mol."""
    if not path.is_file():
        return []
    out: list[float | None] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            out.append(float(parts[-1]))
        except ValueError:
            continue
    return out


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _find_conformers_xyz(workdir: Path) -> Path | None:
    # Preferred final ensemble; CREGEN may leave it empty for NCI jobs.
    for name in ("crest_conformers.xyz",):
        p = workdir / name
        if _nonempty(p):
            return p
    # Intermediate ensembles still usable for ranking/poses.
    for p in sorted(workdir.glob("crest_rotamers_*.xyz"), reverse=True):
        if _nonempty(p):
            return p
    p = workdir / "crest_best.xyz"
    if _nonempty(p):
        return p
    return None


def _sane_frame(xyz: Xyz) -> bool:
    if xyz.n_atoms < 2:
        return False
    return float(np.std(xyz.coords)) > 0.05


def extract_conformers(multi_xyz: Path, dest_dir: Path) -> list[CrestConf]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    frames = split_multi_xyz(multi_xyz)
    file_e = parse_crest_energies(multi_xyz.parent / "crest.energies")
    confs: list[CrestConf] = []
    for i, xyz in enumerate(frames, start=1):
        if not _sane_frame(xyz):
            continue
        p = dest_dir / f"conf_{len(confs) + 1:03d}.xyz"
        e = _parse_comment_energy(xyz.comment)
        if e is None and i - 1 < len(file_e):
            e = file_e[i - 1]
        xyz.comment = xyz.comment or f"crest conf {i}"
        write_xyz(xyz, p)
        if e is not None:
            p.with_suffix(".energy").write_text(f"{e:.12f}\n")
        confs.append(CrestConf(path=p, energy=e, index=len(confs) + 1))
    return confs


def _load_ensemble(output_dir: Path) -> list[CrestConf]:
    multi = _find_conformers_xyz(output_dir)
    if multi is None:
        return []
    if multi.name != "crest_conformers.xyz":
        shutil.copy(multi, output_dir / "crest_conformers.xyz")
        multi = output_dir / "crest_conformers.xyz"
    return extract_conformers(multi, output_dir)


def run_crest(
    xyz_file: Path,
    output_dir: Path,
    extra_args: list[str],
    *,
    timeout: int = 5400,
) -> list[CrestConf]:
    """Run CREST in output_dir. Reuse existing ensemble if present; on timeout, salvage."""
    exe = crest_bin()
    if exe is None:
        raise FileNotFoundError("crest not found; run bash scripts/install_crest.sh")
    output_dir.mkdir(parents=True, exist_ok=True)
    src = output_dir / "input.xyz"
    if Path(xyz_file).resolve() != src.resolve():
        shutil.copy(xyz_file, src)
    for old in output_dir.glob("conf_*.xyz"):
        old.unlink(missing_ok=True)
    # (2) don't re-run if a prior attempt already left an ensemble
    existing = _load_ensemble(output_dir)
    if existing:
        print(f"    reuse existing CREST ensemble ({len(existing)} confs)", flush=True)
        return existing
    env = xtb_env()
    env["PATH"] = str(exe.parent) + os.pathsep + env.get("PATH", "")
    # cwd is output_dir; pass basename so relative --run-dir paths still work
    cmd = [str(exe), "input.xyz", *extra_args]
    timed_out = False
    stdout = stderr = ""
    try:
        proc = subprocess.run(
            cmd,
            cwd=output_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout, stderr = proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stderr += f"\n[co2ie] CREST timed out after {timeout}s; trying to salvage ensemble.\n"
        print(f"    CREST timeout after {timeout}s — salvaging if possible", flush=True)
    (output_dir / "crest_stdout.txt").write_text(stdout + "\n" + stderr)
    confs = _load_ensemble(output_dir)
    if not confs and timed_out:
        return []
    return confs


def select_best_poses(confs: list[CrestConf], n: int) -> list[CrestConf]:
    ranked = sorted(confs, key=lambda c: (c.energy is None, c.energy if c.energy is not None else 0.0))
    return ranked[:n]
