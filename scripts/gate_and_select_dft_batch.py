#!/usr/bin/env python3
"""Cheap gates + Bayesian UCB acquisition for DFT batch selection.

Acquisition maximizes UCB on -IE (lower IE = stronger interaction):
  score = -mu_ie + kappa * sigma
with scaffold diversity caps and an exploration quota.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
ORGANIC = frozenset({"C", "H", "N", "O", "F", "S"})


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def scaffold(smiles: str) -> str:
    try:
        value = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
        return value or f"acyclic:{smiles}"
    except Exception:
        return f"bad:{smiles}"


def gate(
    smiles: str,
    *,
    elements: frozenset[str],
    heavy_min: int,
    heavy_max: int,
    charge_min: int,
    charge_max: int,
) -> tuple[bool, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "parse"
    if any(a.GetSymbol() not in elements for a in mol.GetAtoms()):
        return False, "element"
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < heavy_min or n_heavy > heavy_max:
        return False, "size"
    charge = Chem.GetFormalCharge(mol)
    if charge < charge_min or charge > charge_max:
        return False, "charge"
    return True, "ok"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def score_with_pack(smiles_list: list[str], pack_path: Path) -> dict[str, dict[str, float]]:
    """Return SMILES -> {ie, uncertainty, strong_prob, reward}."""
    import os
    import sys

    mgw = ROOT.parent / "Molecule_Generation_Workflow"
    if mgw.is_dir() and str(mgw) not in sys.path:
        sys.path.insert(0, str(mgw))
    vendor = mgw / "vendor"
    if vendor.is_dir() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    os.environ["MGW_BINDING_PACK"] = str(pack_path)
    os.environ["MGW_BINDING_HEAD"] = "ie"
    from objectives.co2_binding_surrogate import BindingSurrogateScorer

    scorer = BindingSurrogateScorer(pack_path=pack_path)
    out: dict[str, dict[str, float]] = {}
    for smi in smiles_list:
        pred = scorer.predict_one(smi)
        if pred is None:
            continue
        reward = -pred["binding"] + 0.5 * pred["strong_prob"]
        out[smi] = {
            "ie": pred["ie"],
            "uncertainty": pred["uncertainty"],
            "strong_prob": pred["strong_prob"],
            "reward": reward,
            "ood": pred["ood"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--out-gated", type=Path, required=True)
    parser.add_argument("--out-selected", type=Path, required=True)
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--explore-fraction", type=float, default=0.25)
    parser.add_argument("--max-per-scaffold", type=int, default=2)
    parser.add_argument("--kappa", type=float, default=1.5)
    parser.add_argument("--pack", type=Path, default=ROOT / "artifacts" / "binding_predictor" / "pack.joblib")
    parser.add_argument(
        "--score-missing",
        action="store_true",
        help="Run binding pack on rows lacking ie/uncertainty (slow; 3D features).",
    )
    parser.add_argument("--heavy-min", type=int, default=4)
    parser.add_argument("--heavy-max", type=int, default=80)
    parser.add_argument("--charge-min", type=int, default=-2)
    parser.add_argument("--charge-max", type=int, default=2)
    args = parser.parse_args()

    raw = load_rows(args.proposals)
    gated: list[dict[str, str]] = []
    reject = defaultdict(int)
    for row in raw:
        smi = canonicalize((row.get("SMILES") or "").strip())
        if not smi:
            reject["parse"] += 1
            continue
        ok, reason = gate(
            smi,
            elements=ORGANIC,
            heavy_min=args.heavy_min,
            heavy_max=args.heavy_max,
            charge_min=args.charge_min,
            charge_max=args.charge_max,
        )
        if not ok:
            reject[reason] += 1
            continue
        gated.append({**row, "SMILES": smi})

    def _has_score(row: dict[str, str]) -> bool:
        for key in ("ie", "binding", "uncertainty"):
            raw_v = (row.get(key) or "").strip()
            if not raw_v:
                continue
            try:
                if math.isfinite(float(raw_v)):
                    return True
            except ValueError:
                continue
        return False

    # Prefer existing proposal columns (RL export). Optionally score the rest.
    pack_scores: dict[str, dict[str, float]] = {}
    need = [r["SMILES"] for r in gated if not _has_score(r)]
    if args.score_missing and need and args.pack.is_file():
        print(f"[select] scoring {len(need)}/{len(gated)} molecules with pack (slow)")
        try:
            pack_scores = score_with_pack(need, args.pack)
        except Exception as exc:  # pragma: no cover
            print(f"[select] pack scoring failed ({exc}); using proposal columns")
    elif need:
        print(
            f"[select] {len(need)} rows lack scores → high-uncertainty explore "
            "(pass --score-missing to run the binding pack)"
        )

    scored: list[dict[str, object]] = []
    for row in gated:
        smi = row["SMILES"]
        ps = pack_scores.get(smi)
        if ps is not None:
            mu = float(ps["ie"])
            sigma = float(ps["uncertainty"])
            reward = float(ps["reward"])
            ood = int(ps["ood"])
            strong = float(ps["strong_prob"])
        else:
            try:
                mu = float(row.get("ie") or row.get("binding") or "nan")
            except ValueError:
                mu = float("nan")
            try:
                sigma = float(row.get("uncertainty") or "nan")
            except ValueError:
                sigma = float("nan")
            try:
                reward = float(row.get("reward") or "nan")
            except ValueError:
                reward = float("nan")
            ood = int(float(row.get("ood_gated") or 0))
            try:
                strong = float(row.get("strong_prob") or "nan")
            except ValueError:
                strong = float("nan")
        if not math.isfinite(mu):
            # Unscored → pure explore (high sigma).
            mu = 0.0
            sigma = 5.0 if not math.isfinite(sigma) else max(sigma, 5.0)
        if not math.isfinite(sigma):
            sigma = 1.0
        ucb = -mu + args.kappa * sigma
        scored.append(
            {
                "SMILES": smi,
                "source": row.get("source", ""),
                "ie": mu,
                "uncertainty": sigma,
                "ucb": ucb,
                "reward": reward if math.isfinite(reward) else ucb,
                "strong_prob": strong if math.isfinite(strong) else "",
                "ood_gated": ood,
                "scaffold": scaffold(smi),
            }
        )

    args.out_gated.parent.mkdir(parents=True, exist_ok=True)
    with args.out_gated.open("w", newline="") as f:
        fields = [
            "SMILES",
            "source",
            "ie",
            "uncertainty",
            "ucb",
            "reward",
            "strong_prob",
            "ood_gated",
            "scaffold",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in scored:
            writer.writerow({k: row[k] for k in fields})

    n_explore = max(1, int(round(args.n * args.explore_fraction)))
    n_exploit = max(0, args.n - n_explore)

    by_ucb = sorted(scored, key=lambda r: float(r["ucb"]), reverse=True)
    by_sigma = sorted(scored, key=lambda r: float(r["uncertainty"]), reverse=True)

    selected: list[dict[str, object]] = []
    scaffold_counts: dict[str, int] = defaultdict(int)
    chosen: set[str] = set()

    def take(pool: list[dict[str, object]], k: int, bucket: str) -> None:
        added = 0
        for row in pool:
            if added >= k or len(selected) >= args.n:
                break
            smi = str(row["SMILES"])
            scaf = str(row["scaffold"])
            if smi in chosen:
                continue
            if scaffold_counts[scaf] >= args.max_per_scaffold:
                continue
            chosen.add(smi)
            scaffold_counts[scaf] += 1
            item = dict(row)
            item["_bucket"] = bucket
            selected.append(item)
            added += 1

    take(by_ucb, n_exploit, "exploit")
    take(by_sigma, n_explore, "explore")
    # Top up if diversity caps left slots empty.
    take(by_ucb, args.n, "fill")

    final = selected[: args.n]
    with args.out_selected.open("w", newline="") as f:
        fields = [
            "SMILES",
            "source",
            "ie",
            "uncertainty",
            "ucb",
            "reward",
            "strong_prob",
            "ood_gated",
            "scaffold",
            "select_bucket",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in final:
            writer.writerow(
                {
                    "SMILES": row["SMILES"],
                    "source": row.get("source", ""),
                    "ie": f"{float(row['ie']):.6f}",
                    "uncertainty": f"{float(row['uncertainty']):.6f}",
                    "ucb": f"{float(row['ucb']):.6f}",
                    "reward": (
                        f"{float(row['reward']):.6f}"
                        if isinstance(row["reward"], (int, float))
                        and math.isfinite(float(row["reward"]))
                        else row["reward"]
                    ),
                    "strong_prob": row["strong_prob"],
                    "ood_gated": row["ood_gated"],
                    "scaffold": row["scaffold"],
                    "select_bucket": row.get("_bucket", ""),
                }
            )

    print(
        f"[select] gated={len(gated)} reject={dict(reject)} "
        f"selected={len(final)} -> {args.out_selected}"
    )


if __name__ == "__main__":
    main()
