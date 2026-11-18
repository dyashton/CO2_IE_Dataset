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


def _clip_gm_be(
    gm: pd.DataFrame,
    *,
    mode: str = "p1p99",
    lo: float | None = None,
    hi: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Drop gm_be rows with extreme BE_GM (train-only; compiled archives untouched).

    mode:
      - p1p99: keep BE_GM within empirical [1st, 99th] percentiles (default)
      - none: keep all finite rows
      - fixed: keep BE_GM within [lo, hi] (both required)
    """
    mode = str(mode).strip().lower()
    n_before = int(len(gm))
    if n_before == 0:
        return gm, {"mode": mode, "n_before": 0, "n_after": 0, "n_dropped": 0}

    if mode in {"none", "off", "disable", "disabled"}:
        return gm, {
            "mode": "none",
            "n_before": n_before,
            "n_after": n_before,
            "n_dropped": 0,
        }

    be_gm = gm["BE_GM"].to_numpy(dtype=float)
    if mode in {"p1p99", "percentile", "quantiles"}:
        lo_v = float(np.nanpercentile(be_gm, 1))
        hi_v = float(np.nanpercentile(be_gm, 99))
        mode_out = "p1p99"
    elif mode == "fixed":
        if lo is None or hi is None:
            raise ValueError("fixed BE_GM clip requires --be-gm-lo and --be-gm-hi")
        lo_v, hi_v = float(lo), float(hi)
        mode_out = "fixed"
    else:
        raise ValueError(f"unknown be_gm_clip mode: {mode!r}")

    if not (np.isfinite(lo_v) and np.isfinite(hi_v) and lo_v <= hi_v):
        raise ValueError(f"invalid BE_GM clip bounds: lo={lo_v} hi={hi_v}")

    keep = (gm["BE_GM"] >= lo_v) & (gm["BE_GM"] <= hi_v)
    clipped = gm.loc[keep].reset_index(drop=True)
    info = {
        "mode": mode_out,
        "lo": lo_v,
        "hi": hi_v,
        "n_before": n_before,
        "n_after": int(len(clipped)),
        "n_dropped": int(n_before - len(clipped)),
    }
    print(
        f"[clip] BE_GM {mode_out} keep [{lo_v:.3f}, {hi_v:.3f}] "
        f"gm_be {n_before}→{len(clipped)} (dropped {info['n_dropped']})"
    )
    return clipped, info


def build(
    *,
    gm_be: Path = DEFAULT_GM,
    ie_extra: Path = DEFAULT_IE_EXTRA,
    out: Path = DEFAULT_OUT,
    card: Path = DEFAULT_CARD,
    no_ie_extra: bool = False,
    be_gm_clip: str = "p1p99",
    be_gm_lo: float | None = None,
    be_gm_hi: float | None = None,
) -> dict:
    gm_raw = _load_gm(gm_be)
    gm, clip_info = _clip_gm_be(
        gm_raw, mode=be_gm_clip, lo=be_gm_lo, hi=be_gm_hi
    )
    if len(gm) == 0:
        raise ValueError("BE_GM clip removed all gm_be rows")
    extra = pd.DataFrame() if no_ie_extra else _ie_extra_union(gm, ie_extra)
    out_df = pd.concat([gm, extra], ignore_index=True) if len(extra) else gm
    out_df["penalty_clamped"] = np.maximum(0.0, out_df["E_host_penalty_kcal"].to_numpy())
    # Strong flags only meaningful on gm_be rows (ie_only has NaN BE_GM).
    out_df["strong_neg6"] = (out_df["BE_GM"] < -6.0).astype("float")
    out_df["strong_neg7"] = (out_df["BE_GM"] < -7.0).astype("float")

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    pen = gm["E_host_penalty_kcal"].to_numpy(dtype=float)
    try:
        rel = out.relative_to(ROOT)
    except ValueError:
        rel = out
    clip_line = "none (all gm_be kept)"
    if clip_info.get("mode") == "p1p99":
        clip_line = (
            f"drop gm_be outside BE_GM [p1, p99] = "
            f"[{clip_info['lo']:.3f}, {clip_info['hi']:.3f}] "
            f"(dropped {clip_info['n_dropped']})"
        )
    elif clip_info.get("mode") == "fixed":
        clip_line = (
            f"drop gm_be outside BE_GM [{clip_info['lo']:.3f}, {clip_info['hi']:.3f}] "
            f"(dropped {clip_info['n_dropped']})"
        )
    card_text = f"""# Binding predictor training table

Source: `{rel}`

## Contents

| Set | n | Labels |
|-----|--:|--------|
| gm_be (after BE_GM clip) | {len(gm)} | IE, BE, BE_GM, E_host_penalty_kcal |
| ie_only (`compiled_4` novel SMILES) | {len(extra)} | IE only |
| **total rows** | **{len(out_df)}** | |

## Label physics

- `IE`: host–CO2 interaction energy (kcal/mol, more negative = stronger interaction)
- `BE`: binding energy before host reorganization
- `BE_GM = BE + E_host_penalty_kcal`
- `penalty_clamped = max(0, E_host_penalty_kcal)` — negative raw penalties are method noise (GFN2-xTB vs r²SCAN-3c conformer rank reversal), not physical binding bonuses

## Train-only BE_GM clip

- {clip_line}
- Compiled discovery/seed CSVs are **not** modified (provenance kept).

## Penalty diagnostics (gm_be only, post-clip)

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
        "be_gm_clip": clip_info,
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
    parser.add_argument(
        "--be-gm-clip",
        default="p1p99",
        choices=("p1p99", "none", "fixed"),
        help="Train-only BE_GM filter (default p1p99). Compiled CSVs unchanged.",
    )
    parser.add_argument("--be-gm-lo", type=float, default=None)
    parser.add_argument("--be-gm-hi", type=float, default=None)
    args = parser.parse_args()
    build(
        gm_be=args.gm_be,
        ie_extra=args.ie_extra,
        out=args.out,
        card=args.card,
        no_ie_extra=args.no_ie_extra,
        be_gm_clip=args.be_gm_clip,
        be_gm_lo=args.be_gm_lo,
        be_gm_hi=args.be_gm_hi,
    )


if __name__ == "__main__":
    main()
