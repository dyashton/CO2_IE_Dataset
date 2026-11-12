#!/usr/bin/env python3
"""Ingest DFT energies from a discovery round into the growing compiled CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "SMILES",
    "BE_GM",
    "BE",
    "IE",
    "E_host_penalty_kcal",
    "source",
    "run",
    "mol_id",
    "pose_id",
    "status",
]


def _float(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        raw = (row.get(key) or "").strip()
        if not raw:
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return None


def load_energies(path: Path) -> list[dict[str, str]]:
    """Accept parallel energies.csv or catalog-style DFT outputs."""
    rows_out: list[dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = (row.get("SMILES") or row.get("smiles") or "").strip()
            if not smiles:
                continue
            ie = _float(row, "IE", "ie", "ie_kcal_mol", "psi4_ie")
            be = _float(row, "BE", "be")
            begm = _float(row, "BE_GM", "be_gm")
            penalty = _float(row, "E_host_penalty_kcal", "e_host_penalty_kcal")
            if ie is None and be is None and begm is None:
                continue
            if be is None and ie is not None:
                be = ie
            if penalty is None:
                penalty = 0.0
            if begm is None and be is not None:
                begm = be + penalty
            status = (row.get("status") or "ok").strip() or "ok"
            if status not in {"ok", "OK", "success"}:
                continue
            rows_out.append(
                {
                    "SMILES": smiles,
                    "BE_GM": f"{begm:.6f}",
                    "BE": f"{be:.6f}",
                    "IE": f"{ie if ie is not None else be:.6f}",
                    "E_host_penalty_kcal": f"{penalty:.6f}",
                    "source": (row.get("source") or "discovery").strip(),
                    "run": (row.get("run") or path.parent.name).strip(),
                    "mol_id": (row.get("mol_id") or "").strip(),
                    "pose_id": (row.get("pose_id") or "").strip(),
                    "status": "ok",
                }
            )
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energies", type=Path, required=True)
    parser.add_argument(
        "--compiled",
        type=Path,
        default=ROOT / "data" / "compiled_gm_be_discovery.csv",
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=ROOT / "data" / "compiled_gm_be.csv",
        help="Seed snapshot used for best-BE_GM comparison",
    )
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--round-id", default="")
    args = parser.parse_args()

    new_rows = load_energies(args.energies)
    if not new_rows:
        raise SystemExit(f"no usable energy rows in {args.energies}")

    existing: list[dict[str, str]] = []
    if args.compiled.is_file():
        with args.compiled.open(newline="") as f:
            existing = list(csv.DictReader(f))
    elif args.seed.is_file():
        with args.seed.open(newline="") as f:
            existing = list(csv.DictReader(f))

    seen = {(r.get("SMILES") or "").strip() for r in existing}
    added = 0
    for row in new_rows:
        if row["SMILES"] in seen:
            continue
        existing.append(row)
        seen.add(row["SMILES"])
        added += 1

    args.compiled.parent.mkdir(parents=True, exist_ok=True)
    with args.compiled.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in existing:
            writer.writerow({k: row.get(k, "") for k in FIELDS})

    seed_best = None
    if args.seed.is_file():
        with args.seed.open(newline="") as f:
            for row in csv.DictReader(f):
                val = _float(row, "BE_GM", "be_gm")
                if val is None:
                    continue
                seed_best = val if seed_best is None else min(seed_best, val)

    new_best = min(float(r["BE_GM"]) for r in new_rows)
    overall_best = min(float(r["BE_GM"]) for r in existing if (r.get("BE_GM") or "").strip())

    summary_path = args.summary or args.energies.parent / "summary.md"
    lines = [
        f"# Discovery ingest {args.round_id or args.energies.parent.name}",
        "",
        f"- energies file: `{args.energies}`",
        f"- new labeled rows added: **{added}** (parsed {len(new_rows)})",
        f"- compiled size: **{len(existing)}** → `{args.compiled}`",
        f"- batch best BE_GM: **{new_best:.3f}**",
        f"- overall best BE_GM: **{overall_best:.3f}**",
    ]
    if seed_best is not None:
        lines.append(f"- seed best BE_GM: **{seed_best:.3f}**")
        lines.append(
            f"- beat seed best: **{'yes' if overall_best < seed_best - 1e-6 else 'no'}**"
        )
    lines.append("")
    lines.append("## Top new (this batch)")
    for row in sorted(new_rows, key=lambda r: float(r["BE_GM"]))[:10]:
        lines.append(f"- {float(row['BE_GM']):7.3f}  `{row['SMILES']}`")
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
