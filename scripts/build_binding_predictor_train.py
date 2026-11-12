#!/usr/bin/env python3
"""Freeze the CO2 binding-predictor training table from compiled_gm_be."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from co2ie.surrogate import canonicalize_smiles

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GM = ROOT / "data" / "compiled_gm_be.csv"
DEFAULT_IE_EXTRA = ROOT / "data" / "compiled_4.csv"
DEFAULT_OUT = ROOT / "data" / "binding_predictor_train.csv"
DEFAULT_CARD = ROOT / "docs" / "BINDING_PREDICTOR_DATA.md"


def _load_gm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"SMILES", "IE", "BE", "BE_GM", "E_host_penalty_kcal", "source"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    rows: list[dict] = []
    for row in df.to_dict("records"):
        can = canonicalize_smiles(row.get("SMILES"))
        try:
            ie = float(row["IE"])
            be = float(row["BE"])
            be_gm = float(row["BE_GM"])
            penalty = float(row["E_host_penalty_kcal"])
        except (TypeError, ValueError):
            continue
        if can is None or not all(np.isfinite(v) for v in (ie, be, be_gm, penalty)):
            continue
        rows.append(
            {
                "SMILES": can,
                "IE": ie,
                "BE": be,
                "BE_GM": be_gm,
                "E_host_penalty_kcal": penalty,
                "source": str(row["source"]),
                "label_set": "gm_be",
            }
        )
    if not rows:
        raise ValueError(f"{path} has no valid gm_be rows")
    clean = pd.DataFrame(rows)
    # One row per canonical SMILES; mean numeric labels, keep first source.
    return (
        clean.groupby("SMILES", as_index=False)
        .agg(
            IE=("IE", "mean"),
            BE=("BE", "mean"),
            BE_GM=("BE_GM", "mean"),
            E_host_penalty_kcal=("E_host_penalty_kcal", "mean"),
            source=("source", "first"),
            label_set=("label_set", "first"),
        )
        .sort_values("SMILES")
        .reset_index(drop=True)
    )


def _ie_extra_union(gm: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Optional IE-only rows from compiled_4 not already in gm_be."""
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "SMILES" not in df.columns or "IE" not in df.columns:
        return pd.DataFrame()
    known = set(gm["SMILES"])
    rows: list[dict] = []
    for row in df.to_dict("records"):
        can = canonicalize_smiles(row.get("SMILES"))
        if can is None or can in known:
            continue
        try:
            ie = float(row["IE"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ie):
            continue
        source = str(row.get("source", "compiled_4"))
        rows.append(
            {
                "SMILES": can,
                "IE": ie,
                "BE": np.nan,
                "BE_GM": np.nan,
                "E_host_penalty_kcal": np.nan,
                "source": source,
                "label_set": "ie_only",
            }
        )
    if not rows:
        return pd.DataFrame()
    extra = pd.DataFrame(rows)
    return (
        extra.groupby("SMILES", as_index=False)
        .agg(
            IE=("IE", "mean"),
            BE=("BE", "first"),
            BE_GM=("BE_GM", "first"),
            E_host_penalty_kcal=("E_host_penalty_kcal", "first"),
            source=("source", "first"),
            label_set=("label_set", "first"),
        )
        .sort_values("SMILES")
        .reset_index(drop=True)
    )


def build(
    *,
    gm_be: Path = DEFAULT_GM,
    ie_extra: Path = DEFAULT_IE_EXTRA,
    out: Path = DEFAULT_OUT,
    card: Path = DEFAULT_CARD,
    no_ie_extra: bool = False,
) -> dict:
    gm = _load_gm(gm_be)
    extra = pd.DataFrame() if no_ie_extra else _ie_extra_union(gm, ie_extra)
    out_df = pd.concat([gm, extra], ignore_index=True) if len(extra) else gm
    out_df["penalty_clamped"] = np.maximum(0.0, out_df["E_host_penalty_kcal"].to_numpy())
    out_df["strong_neg6"] = (out_df["BE_GM"] < -6.0).astype("float")
    out_df["strong_neg7"] = (out_df["BE_GM"] < -7.0).astype("float")

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    pen = gm["E_host_penalty_kcal"].to_numpy(dtype=float)
    try:
        rel = out.relative_to(ROOT)
    except ValueError:
        rel = out
    card_text = f"""# Binding predictor training table

Source: `{rel}`

## Contents

| Set | n | Labels |
|-----|--:|--------|
| gm_be (`compiled_gm_be`) | {len(gm)} | IE, BE, BE_GM, E_host_penalty_kcal |
| ie_only (`compiled_4` novel SMILES) | {len(extra)} | IE only |
| **total rows** | **{len(out_df)}** | |

## Label physics

- `IE`: host–CO2 interaction energy (kcal/mol, more negative = stronger interaction)
- `BE`: binding energy before host reorganization
- `BE_GM = BE + E_host_penalty_kcal`
- `penalty_clamped = max(0, E_host_penalty_kcal)` — negative raw penalties are method noise (GFN2-xTB vs r²SCAN-3c conformer rank reversal), not physical binding bonuses

## Penalty diagnostics (gm_be only)

- frac negative raw penalty: {(pen < 0).mean():.3f}
- min / median / max: {pen.min():.3f} / {np.median(pen):.3f} / {pen.max():.3f}
- BE_GM < -6 prevalence: {(gm["BE_GM"] < -6).mean():.3f}
- BE_GM < -7 prevalence: {(gm["BE_GM"] < -7).mean():.3f}

## CV

Bemis–Murcko scaffold GroupKFold (5-fold), same protocol as `scripts/train_surrogate.py`.

## Excluded

- `peptide_co2_binders_150.csv` (class-only, no energies)
- `amine_2400` (narrow diagnostic IE snapshot; not main train pool)
- Carbamate / chemisorption ΔH literature labels (not in this dataset)
"""
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(card_text)
    print(f"[save] {out} rows={len(out_df)} gm_be={len(gm)} ie_only={len(extra)}")
    print(f"[save] {card}")
    summary = {
        "out": str(out),
        "n_total": int(len(out_df)),
        "n_gm_be": int(len(gm)),
        "n_ie_only": int(len(extra)),
        "penalty_frac_negative": float((pen < 0).mean()),
    }
    print(json.dumps(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gm-be", type=Path, default=DEFAULT_GM)
    parser.add_argument("--ie-extra", type=Path, default=DEFAULT_IE_EXTRA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--no-ie-extra", action="store_true")
    args = parser.parse_args()
    build(
        gm_be=args.gm_be,
        ie_extra=args.ie_extra,
        out=args.out,
        card=args.card,
        no_ie_extra=args.no_ie_extra,
    )


if __name__ == "__main__":
    main()
