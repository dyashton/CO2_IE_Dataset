#!/usr/bin/env python3
"""Train / score linear Bradley–Terry CO2 preference model (molecule-grouped CV)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from co2ie import ROOT
from co2ie.preference import (
    MARGIN_KCAL,
    N_FOLDS,
    N_PARTNERS,
    RANDOM_SEED,
    build_compiled_csv,
    cross_validate,
    load_compiled,
    load_model,
    make_pairs,
    save_model,
    smiles_to_fp,
    spearman_score_vs_ei,
    train_full,
)


def _xgb_cv(mols, n_folds: int = N_FOLDS) -> dict:
    """Molecule-grouped XGBoost IE regression; Spearman of (-pred) vs (-EI) ≡ pred vs EI."""
    fps = np.stack([smiles_to_fp(m.smiles) for m in mols])  # type: ignore[arg-type]
    eis = np.array([m.ei for m in mols], dtype=np.float64)
    groups = np.array([m.mol_id for m in mols])
    gkf = GroupKFold(n_splits=min(n_folds, len(mols)))
    folds = []
    for fold, (tr, te) in enumerate(gkf.split(fps, eis, groups), start=1):
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.5,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=4,
        )
        model.fit(fps[tr], eis[tr])
        pred = model.predict(fps[te])
        from scipy import stats

        rho, _ = stats.spearmanr(pred, eis[te])
        # pair acc from predicted EI (same-source, margin) on test mols only
        sources = [m.source for m in mols]
        rng = np.random.default_rng(RANDOM_SEED + fold)
        te_pairs, te_y = make_pairs(
            te.tolist(), eis, sources, margin=MARGIN_KCAL, n_partners=N_PARTNERS, rng=rng
        )
        if te_pairs:
            te_list = te.tolist()
            pos = {idx: k for k, idx in enumerate(te_list)}
            correct = 0
            for (i, j), y in zip(te_pairs, te_y):
                pred_y = 1 if pred[pos[i]] < pred[pos[j]] else 0
                correct += int(pred_y == y)
            pair_acc = correct / len(te_pairs)
        else:
            pair_acc = float("nan")
        folds.append(
            {
                "fold": fold,
                "n_test_mols": len(te),
                "test_spearman": float(rho),
                "test_pair_acc": float(pair_acc),
            }
        )
    return {"folds": folds, "n_mols": len(mols)}


def cmd_train(args: argparse.Namespace) -> None:
    compiled = Path(args.compiled)
    if args.rebuild or not compiled.is_file():
        compiled = build_compiled_csv(compiled)
        print(f"[compiled] wrote {compiled}")

    mols = load_compiled(compiled)
    print(f"[data] {len(mols)} molecules")
    by = {}
    for m in mols:
        by[m.source] = by.get(m.source, 0) + 1
    print(f"[data] by source: {by}")

    cv = cross_validate(mols, n_folds=args.folds, margin=args.margin, n_partners=args.partners)
    print("\n[BTL linear] molecule-grouped CV")
    for f in cv["folds"]:
        print(
            f"  fold {f['fold']}: train_mols={f['n_train_mols']} test_mols={f['n_test_mols']} "
            f"train_pairs={f['n_train_pairs']} test_pairs={f['n_test_pairs']} "
            f"test_pair_acc={f['test_pair_acc']:.3f} test_spearman={f['test_spearman']:.3f}"
        )
    accs = [f["test_pair_acc"] for f in cv["folds"] if f["test_pair_acc"] == f["test_pair_acc"]]
    rhos = [f["test_spearman"] for f in cv["folds"] if f["test_spearman"] == f["test_spearman"]]
    print(f"  mean test_pair_acc={float(np.mean(accs)):.3f}  mean test_spearman={float(np.mean(rhos)):.3f}")

    if not args.skip_xgb:
        xgb = _xgb_cv(mols, n_folds=args.folds)
        print("\n[XGB IE regression] molecule-grouped CV (same GroupKFold spirit)")
        for f in xgb["folds"]:
            print(
                f"  fold {f['fold']}: test_mols={f['n_test_mols']} "
                f"test_pair_acc={f['test_pair_acc']:.3f} test_spearman={f['test_spearman']:.3f}"
            )
        xacc = [f["test_pair_acc"] for f in xgb["folds"] if f["test_pair_acc"] == f["test_pair_acc"]]
        xrho = [f["test_spearman"] for f in xgb["folds"] if f["test_spearman"] == f["test_spearman"]]
        print(f"  mean test_pair_acc={float(np.mean(xacc)):.3f}  mean test_spearman={float(np.mean(xrho)):.3f}")

    model = train_full(mols, margin=args.margin, n_partners=args.partners)
    out = Path(args.out)
    save_model(model, out)
    print(f"\n[save] {out}")

    metrics_path = out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"btl_cv": cv, "xgb_cv": None if args.skip_xgb else xgb}, indent=2))
    print(f"[save] {metrics_path}")


def cmd_score(args: argparse.Namespace) -> None:
    model = load_model(Path(args.model))
    for smi in args.smiles:
        s = model.score_smiles(smi)
        print(f"{smi}\t{s if s is not None else 'INVALID'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Rebuild compiled (optional), CV, fit, save model")
    t.add_argument("--compiled", type=Path, default=ROOT / "data" / "compiled.csv")
    t.add_argument("--rebuild", action="store_true", help="Rewrite compiled.csv from test + test_1")
    t.add_argument("--out", type=Path, default=ROOT / "artifacts" / "preference_linear.joblib")
    t.add_argument("--folds", type=int, default=N_FOLDS)
    t.add_argument("--margin", type=float, default=MARGIN_KCAL)
    t.add_argument("--partners", type=int, default=N_PARTNERS)
    t.add_argument("--skip-xgb", action="store_true")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("score", help="Score SMILES with a saved model (higher = stronger binder)")
    s.add_argument("--model", type=Path, default=ROOT / "artifacts" / "preference_linear.joblib")
    s.add_argument("smiles", nargs="+")
    s.set_defaults(func=cmd_score)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
