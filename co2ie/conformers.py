from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from co2ie import N_CONF


@dataclass
class Xyz:
    symbols: list[str]
    coords: np.ndarray  # (n, 3)
    comment: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


def write_xyz(xyz: Xyz, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{xyz.n_atoms}", xyz.comment]
    for s, p in zip(xyz.symbols, xyz.coords):
        lines.append(f"{s:2s} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}")
    path.write_text("\n".join(lines) + "\n")


def split_multi_xyz(path: Path) -> list[Xyz]:
    """Split a concatenated XYZ file into frames."""
    raw = path.read_text().strip().splitlines()
    out: list[Xyz] = []
    i = 0
    while i < len(raw):
        if not raw[i].strip():
            i += 1
            continue
        n = int(raw[i].split()[0])
        comment = raw[i + 1] if i + 1 < len(raw) else ""
        symbols: list[str] = []
        coords: list[list[float]] = []
        for line in raw[i + 2 : i + 2 + n]:
            parts = line.split()
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        out.append(Xyz(symbols=symbols, coords=np.asarray(coords, dtype=float), comment=comment))
        i += 2 + n
    return out


def read_xyz(path: Path) -> Xyz:
    lines = path.read_text().strip().splitlines()
    n = int(lines[0])
    comment = lines[1] if len(lines) > 1 else ""
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return Xyz(symbols=symbols, coords=np.asarray(coords, dtype=float), comment=comment)


def mol_to_xyz(mol: Chem.Mol, conf_id: int = -1, comment: str = "") -> Xyz:
    conf = mol.GetConformer(conf_id)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=float,
    )
    return Xyz(symbols=symbols, coords=coords, comment=comment)


def embed_conformers(smiles: str, n_conf: int = N_CONF, seed: int = 7) -> list[Xyz]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.pruneRmsThresh = 0.5
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params)
    if not ids:
        rc = AllChem.EmbedMolecule(mol, params)
        if rc != 0:
            return []
        ids = [0]
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
    except Exception:
        try:
            AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=200)
        except Exception:
            pass
    return [mol_to_xyz(mol, int(i), comment=f"{smiles} conf {i}") for i in ids]


def write_host_conformers(mol_dir: Path, smiles: str, n_conf: int = N_CONF) -> list[Path]:
    mol_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, xyz in enumerate(embed_conformers(smiles, n_conf=n_conf)):
        p = mol_dir / f"conf_{i:02d}.xyz"
        write_xyz(xyz, p)
        paths.append(p)
    return paths
