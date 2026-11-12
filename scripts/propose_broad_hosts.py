#!/usr/bin/env python3
"""Build a broad unlabeled host pool: libraries + RL proposals + elite mutants."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]

# Simple side-chain / heteroatom swap SMARTS mutations (broad, cheap).
_MUTATION_SMARTS = (
    ("[cH]", "[c](N)"),
    ("[cH]", "[c](O)"),
    ("[CH2]", "[CH](N)"),
    ("[NH2]", "[NH]C"),
    ("[OH]", "[NH2]"),
    ("O", "N"),
)


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_smiles_column(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        key = next(
            (c for c in reader.fieldnames if c.upper() in {"SMILES", "SMI"}),
            reader.fieldnames[0],
        )
        out = []
        for row in reader:
            can = canonicalize((row.get(key) or "").strip())
            if can:
                out.append(can)
        return out


def mutate(smiles: str, rng: random.Random, n: int = 4) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    out: list[str] = []
    for _ in range(n * 3):
        if len(out) >= n:
            break
        smarts_from, smarts_to = rng.choice(_MUTATION_SMARTS)
        patt = Chem.MolFromSmarts(smarts_from)
        repl = Chem.MolFromSmiles(smarts_to)
        if patt is None or repl is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue
        idx = rng.choice(matches)
        try:
            new = AllChem.ReplaceSubstructs(mol, patt, repl, replaceAll=False)[0]
            Chem.SanitizeMol(new)
            can = Chem.MolToSmiles(new, canonical=True)
            if can and can != smiles:
                out.append(can)
        except Exception:
            continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-csv", type=Path, default=ROOT / "data" / "compiled_gm_be.csv")
    parser.add_argument("--rl-proposals", type=Path, default=None)
    parser.add_argument(
        "--library",
        type=Path,
        action="append",
        default=[],
        help="Extra library CSV (repeatable). Defaults from configs/discovery.yaml libs.",
    )
    parser.add_argument("--elite-top-n", type=int, default=20)
    parser.add_argument("--elite-mutants", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    labeled = set(load_smiles_column(args.seed_csv))
    rows: list[dict[str, str]] = []
    seen: set[str] = set(labeled)

    def add(smiles: str, source: str) -> None:
        if smiles in seen:
            return
        seen.add(smiles)
        rows.append({"SMILES": smiles, "source": source})

    libs = args.library or [
        ROOT / "data" / "azamacro_host_library.csv",
        ROOT / "data" / "amine_ie_designed.csv",
        ROOT / "data" / "robust_200_candidates.csv",
        ROOT / "data" / "peptide_co2_binders_150.csv",
        ROOT / "data" / "diverse_small_molecules_300.csv",
    ]
    for path in libs:
        for smi in load_smiles_column(path):
            add(smi, f"library:{path.name}")

    if args.rl_proposals and args.rl_proposals.is_file():
        for smi in load_smiles_column(args.rl_proposals):
            add(smi, "rl_proposal")

    # Elite mutants from strongest BE_GM in seed.
    elites: list[tuple[float, str]] = []
    with args.seed_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            smi = canonicalize((row.get("SMILES") or "").strip())
            if not smi:
                continue
            try:
                begm = float(row.get("BE_GM") or row.get("be_gm") or "nan")
            except ValueError:
                continue
            elites.append((begm, smi))
    elites.sort(key=lambda x: x[0])
    for begm, smi in elites[: args.elite_top_n]:
        for mut in mutate(smi, rng, n=max(1, args.elite_mutants // max(args.elite_top_n, 1))):
            add(mut, f"elite_mutant:{begm:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["SMILES", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[propose] {len(rows)} unlabeled hosts -> {args.out} (excluded {len(labeled)} labeled)")


if __name__ == "__main__":
    main()
