from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import yaml
from rdkit import Chem

from co2ie import ROOT


@dataclass(frozen=True)
class Candidate:
    id: str
    smiles: str
    source: str
    n_heavy: int


def _truthy(v: object) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes"}


def load_sources(path: Path | None = None) -> dict:
    p = path or (ROOT / "data" / "sources.yaml")
    return yaml.safe_load(p.read_text())


def _resolve(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (root / p).resolve()


def _smiles_col(fieldnames: list[str] | None) -> str:
    if not fieldnames:
        raise ValueError("empty csv")
    for name in fieldnames:
        if name.lower() == "smiles":
            return name
    raise ValueError(f"no SMILES column in {fieldnames}")


def _iter_smiles(csv_path: Path, require_gate_ok: bool) -> list[str]:
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        col = _smiles_col(r.fieldnames)
        gate_col = None
        if r.fieldnames:
            for name in r.fieldnames:
                if name.lower() == "gate_ok":
                    gate_col = name
                    break
        out: list[str] = []
        for row in r:
            if require_gate_ok:
                if gate_col is None or not _truthy(row.get(gate_col, "")):
                    continue
            smi = (row.get(col) or "").strip()
            if smi:
                out.append(smi)
        return out


def _smallest(smiles_list: list[str], n: int, seen: set[str], source: str) -> list[Candidate]:
    scored: list[tuple[int, str]] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            continue
        scored.append((mol.GetNumHeavyAtoms(), can))
    scored.sort()
    picked: list[Candidate] = []
    for n_ha, can in scored:
        if can in seen:
            continue
        seen.add(can)
        cid = f"{source}_{len(picked):02d}"
        picked.append(Candidate(id=cid, smiles=can, source=source, n_heavy=n_ha))
        if len(picked) >= n:
            break
    return picked


def sample_small(sources_path: Path | None = None, root: Path | None = None) -> list[Candidate]:
    """5 smallest gated GuacaMol + 5 smallest AxelRolov. mine/ is unused in v1."""
    root = root or ROOT
    cfg = load_sources(sources_path)
    seen: set[str] = set()
    out: list[Candidate] = []
    for key in ("guacamol", "axelrolov"):
        block = cfg[key]
        path = _resolve(root, block["path"])
        n = int(block.get("n", 5))
        require = bool(block.get("require_gate_ok", False))
        smiles = _iter_smiles(path, require)
        out.extend(_smallest(smiles, n, seen, source=key))
    return out


def sample_from_csv(csv_path: Path, n: int = 10, source: str = "csv") -> list[Candidate]:
    """First `n` valid SMILES from a CSV (order preserved)."""
    smiles = _iter_smiles(csv_path, require_gate_ok=False)
    out: list[Candidate] = []
    seen: set[str] = set()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"[sample] skip unparseable: {smi[:80]}")
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            continue
        seen.add(can)
        out.append(
            Candidate(
                id=f"{source}_{len(out):02d}",
                smiles=can,
                source=source,
                n_heavy=mol.GetNumHeavyAtoms(),
            )
        )
        if len(out) >= n:
            break
    return out


def write_catalog(cands: list[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "smiles", "source", "n_heavy"])
        w.writeheader()
        for c in cands:
            w.writerow({"id": c.id, "smiles": c.smiles, "source": c.source, "n_heavy": c.n_heavy})
