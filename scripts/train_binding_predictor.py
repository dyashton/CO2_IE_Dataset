#!/usr/bin/env python3
"""Train the hybrid CO2 binding predictor (IE/BE + clamped penalty → BE_GM)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, XGBRegressor

from co2ie.surrogate import (
    FEATURE_SET_3D,
    _BROAD_DESCRIPTOR_NAMES,
    feature_names,
    featurize_smiles,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "binding_predictor_train.csv"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "binding_predictor"

REG_MODELS = ("xgboost", "extra_trees", "random_forest")
GATE = {
    "ie_spearman": 0.75,
    "ie_rmse": 1.05,
    "be_spearman": 0.70,
    "be_rmse": 1.10,
    "begm_compose_spearman": 0.60,
    "strong_pr_auc": 0.35,
    "mw_partial_gap": 0.05,
}


def _scaffold_group(smiles: str) -> str:
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    return scaffold or f"acyclic:{smiles}"


def _pair_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    ok = total = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] == y_true[j]:
                continue
            ok += int((y_true[i] < y_true[j]) == (y_pred[i] < y_pred[j]))
            total += 1
    return float(ok / total) if total else float("nan")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "spearman": float("nan"),
            "pair_accuracy": float("nan"),
        }
    return {
        "n": int(yt.size),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)) if yt.size > 1 else float("nan"),
        "spearman": float(spearmanr(yt, yp).statistic) if yt.size > 2 else float("nan"),
        "pair_accuracy": _pair_accuracy(yt, yp),
    }


def _partial_spearman(y: np.ndarray, pred: np.ndarray, mw: np.ndarray) -> float:
    """Spearman of residuals after linear MW regression (anti-hack check)."""
    mask = np.isfinite(y) & np.isfinite(pred) & np.isfinite(mw)
    if mask.sum() < 5:
        return float("nan")
    y_m, p_m, w = y[mask], pred[mask], mw[mask].reshape(-1, 1)
    y_res = y_m - Ridge(alpha=1.0).fit(w, y_m).predict(w)
    p_res = p_m - Ridge(alpha=1.0).fit(w, p_m).predict(w)
    return float(spearmanr(y_res, p_res).statistic)


def _reg_model(seed: int, kind: str, *, objective: str = "reg:squarederror"):
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=400,
            max_features=0.7,
            min_samples_leaf=2,
            n_jobs=4,
            random_state=seed,
        )
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=400,
            max_features=0.5,
            min_samples_leaf=2,
            n_jobs=4,
            random_state=seed,
        )
    if kind != "xgboost":
        raise ValueError(f"unknown model: {kind}")
    return XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.035,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.45,
        reg_lambda=3.0,
        objective=objective,
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
    )


def _clf_model(seed: int, kind: str):
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_features=0.7,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=4,
            random_state=seed,
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_features=0.5,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=4,
            random_state=seed,
        )
    return XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.5,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
        scale_pos_weight=1.0,
    )


def _probe_model(seed: int) -> XGBRegressor:
    model = _reg_model(seed, "xgboost")
    model.set_params(n_estimators=250, max_depth=3, learning_rate=0.05)
    return model


def _stable_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    cumulative: float,
    max_features: int,
) -> np.ndarray:
    n_desc = len(_BROAD_DESCRIPTOR_NAMES)
    dmean = X_train[:, :n_desc].mean(axis=0)
    dscale = X_train[:, :n_desc].std(axis=0)
    dscale[dscale < 1e-8] = 1.0
    probe_X = X_train.copy()
    probe_X[:, :n_desc] = (probe_X[:, :n_desc] - dmean) / dscale
    probe = _probe_model(seed)
    y_scale = float(y_train.std()) or 1.0
    probe.fit(probe_X, (y_train - y_train.mean()) / y_scale)
    importance = np.asarray(probe.feature_importances_, dtype=np.float64)
    order = np.argsort(-importance)
    positive = order[importance[order] > 0]
    if positive.size == 0:
        return order[: min(max_features, X_train.shape[1])]
    mass = np.cumsum(importance[positive])
    keep = int(np.searchsorted(mass, cumulative * mass[-1]) + 1)
    keep = min(max(keep, 32), max_features)
    return positive[:keep]


def _transform_xy(
    X_train: np.ndarray,
    X_test: np.ndarray,
    selected: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_desc = len(_BROAD_DESCRIPTOR_NAMES)
    dmean = X_train[:, :n_desc].mean(axis=0)
    dscale = X_train[:, :n_desc].std(axis=0)
    dscale[dscale < 1e-8] = 1.0
    xtr = X_train.copy()
    xte = X_test.copy()
    xtr[:, :n_desc] = (xtr[:, :n_desc] - dmean) / dscale
    xte[:, :n_desc] = (xte[:, :n_desc] - dmean) / dscale
    if selected is not None:
        xtr = xtr[:, selected]
        xte = xte[:, selected]
    return xtr, xte, dmean, dscale


def _fit_predict_reg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    selected: np.ndarray | None,
    kind: str,
    objective: str = "reg:squarederror",
    mw_train: np.ndarray | None = None,
    mw_test: np.ndarray | None = None,
    residualize_mw: bool = False,
    target_transform: str = "identity",
) -> tuple[np.ndarray, dict]:
    xtr, xte, dmean, dscale = _transform_xy(X_train, X_test, selected)
    y = y_train.astype(np.float64).copy()
    mw_model = None
    if residualize_mw:
        assert mw_train is not None and mw_test is not None
        mw_model = Ridge(alpha=1.0).fit(mw_train.reshape(-1, 1), y)
        y = y - mw_model.predict(mw_train.reshape(-1, 1))
    if target_transform == "log1p":
        y = np.log1p(np.maximum(y, 0.0))
    ymean = float(y.mean())
    yscale = float(y.std()) or 1.0
    model = _reg_model(seed, kind, objective=objective)
    model.fit(xtr, (y - ymean) / yscale)
    pred = model.predict(xte) * yscale + ymean
    if target_transform == "log1p":
        pred = np.expm1(pred)
        pred = np.maximum(pred, 0.0)
    if mw_model is not None:
        pred = pred + mw_model.predict(mw_test.reshape(-1, 1))
    return pred, {
        "model": model,
        "descriptor_mean": dmean,
        "descriptor_scale": dscale,
        "target_mean": ymean,
        "target_scale": yscale,
        "mw_model": mw_model,
        "residualize_mw": bool(residualize_mw),
        "target_transform": target_transform,
        "objective": objective,
        "model_kind": kind,
    }


def _topk_recall(y_true: np.ndarray, score: np.ndarray, *, frac: float = 0.1, thr: float = -7.0) -> float:
    mask = np.isfinite(y_true) & np.isfinite(score)
    yt, sc = y_true[mask], score[mask]
    if yt.size == 0:
        return float("nan")
    k = max(1, int(round(frac * yt.size)))
    top = np.argsort(sc)[:k]  # more negative = better binder
    hits = yt[top] < thr
    positives = yt < thr
    if not positives.any():
        return float("nan")
    return float(hits.sum() / positives.sum())


def _feature_dominance(names: list[str], importances: np.ndarray) -> dict:
    proxies = ("NOCount", "TPSA", "MolWt", "NumHeteroatoms", "NumHAcceptors")
    total = float(np.sum(np.maximum(importances, 0.0))) or 1.0
    share = 0.0
    for name, imp in zip(names, importances):
        if any(p in name for p in proxies):
            share += max(float(imp), 0.0)
    return {
        "proxy_importance_share": float(share / total),
        "flag_gt_40pct": bool(share / total > 0.40),
    }


def _morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _load_featurized(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    needed = {"SMILES", "IE", "BE", "BE_GM", "E_host_penalty_kcal", "source", "label_set"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}; run build_binding_predictor_train.py")
    features: list[np.ndarray] = []
    mws: list[float] = []
    kept: list[int] = []
    for i, row in enumerate(df.to_dict("records")):
        smi = str(row["SMILES"])
        feat = featurize_smiles(smi, FEATURE_SET_3D)
        mol = Chem.MolFromSmiles(smi)
        if feat is None or mol is None or not np.all(np.isfinite(feat)):
            continue
        features.append(feat.astype(np.float32))
        mws.append(float(Descriptors.MolWt(mol)))
        kept.append(i)
    clean = df.iloc[kept].reset_index(drop=True)
    return clean, np.stack(features), np.asarray(mws, dtype=np.float64)


def _oof_regression(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    mw: np.ndarray,
    *,
    seed: int,
    cumulative: float,
    max_features: int,
    mask: np.ndarray | None = None,
    residualize_mw: bool = False,
    objective: str = "reg:squarederror",
    target_transform: str = "identity",
    model_names: tuple[str, ...] = REG_MODELS,
) -> tuple[dict[str, np.ndarray], dict, np.ndarray, str]:
    """Scaffold OOF for one target; returns oof_by_model, fold metrics, stable idx, best kind."""
    if mask is None:
        mask = np.isfinite(y)
    idx = np.flatnonzero(mask)
    oof_by_model = {k: np.full(len(y), np.nan) for k in model_names}
    folds: dict[str, list] = {k: [] for k in model_names}
    fold_selected: list[np.ndarray] = []
    X_m, y_m, g_m, mw_m = X[idx], y[idx], groups[idx], mw[idx]
    n_splits = min(5, len(np.unique(g_m)))
    splitter = GroupKFold(n_splits=n_splits)
    local_oof = {k: np.full(len(y_m), np.nan) for k in model_names}
    for fold_i, (tr, te) in enumerate(splitter.split(X_m, y_m, groups=g_m), start=1):
        selected = _stable_selection(
            X_m[tr],
            y_m[tr],
            seed=seed,
            cumulative=cumulative,
            max_features=max_features,
        )
        fold_selected.append(selected)
        for kind in model_names:
            pred, _ = _fit_predict_reg(
                X_m[tr],
                y_m[tr],
                X_m[te],
                seed=seed,
                selected=selected,
                kind=kind,
                objective=objective if kind == "xgboost" else "reg:squarederror",
                mw_train=mw_m[tr],
                mw_test=mw_m[te],
                residualize_mw=residualize_mw,
                target_transform=target_transform,
            )
            local_oof[kind][te] = pred
            folds[kind].append({"fold": fold_i, **_metrics(y_m[te], pred)})
    for kind in model_names:
        oof_by_model[kind][idx] = local_oof[kind]
    model_metrics = {k: _metrics(y, oof_by_model[k]) for k in model_names}
    best = max(
        model_names,
        key=lambda k: (model_metrics[k]["spearman"], -model_metrics[k]["mae"]),
    )
    votes = np.zeros(X.shape[1], dtype=np.int32)
    for selected in fold_selected:
        votes[selected] += 1
    threshold = (len(fold_selected) + 1) // 2
    stable = np.flatnonzero(votes >= threshold)
    if len(stable) < 32:
        stable = np.argsort(-votes)[: min(max_features, X.shape[1])]
    return oof_by_model, {"folds": folds, "model_metrics": model_metrics}, stable, best


def _fit_final_reg(
    X: np.ndarray,
    y: np.ndarray,
    mw: np.ndarray,
    *,
    stable: np.ndarray,
    kind: str,
    seed: int,
    residualize_mw: bool,
    objective: str,
    target_transform: str,
    target_name: str,
    mask: np.ndarray | None = None,
) -> dict:
    if mask is None:
        mask = np.isfinite(y)
    X_m, y_m, mw_m = X[mask], y[mask], mw[mask]
    ensemble = []
    fitted = None
    for ens_seed in range(seed, seed + 5):
        _, cur = _fit_predict_reg(
            X_m,
            y_m,
            X_m,
            seed=ens_seed,
            selected=stable,
            kind=kind,
            objective=objective if kind == "xgboost" else "reg:squarederror",
            mw_train=mw_m,
            mw_test=mw_m,
            residualize_mw=residualize_mw,
            target_transform=target_transform,
        )
        if fitted is None:
            fitted = cur
        ensemble.append(cur["model"])
    assert fitted is not None
    all_names = feature_names(FEATURE_SET_3D)
    mean_imp = np.mean([m.feature_importances_ for m in ensemble], axis=0)
    ranked = sorted(
        (
            {
                "feature": all_names[int(i)],
                "importance": float(imp),
            }
            for i, imp in zip(stable, mean_imp)
        ),
        key=lambda d: d["importance"],
        reverse=True,
    )
    return {
        **fitted,
        "models": ensemble,
        "model": ensemble[0],
        "ensemble_size": len(ensemble),
        "feature_set": FEATURE_SET_3D,
        "all_feature_names": all_names,
        "feature_indices": stable.tolist(),
        "feature_names": [all_names[i] for i in stable],
        "descriptor_names": list(_BROAD_DESCRIPTOR_NAMES),
        "selected_features": ranked,
        "target": target_name,
        "cv_strategy": "bemis_murcko_scaffold_grouped",
        "feature_dominance": _feature_dominance(
            [all_names[i] for i in stable], np.asarray(mean_imp)
        ),
    }


def _oof_classifier(
    X: np.ndarray,
    y_bin: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    cumulative: float,
    max_features: int,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict, np.ndarray, str]:
    idx = np.flatnonzero(mask & np.isfinite(y_bin))
    X_m, y_m, g_m = X[idx], y_bin[idx].astype(int), groups[idx]
    kinds = REG_MODELS
    local = {k: np.full(len(y_m), np.nan) for k in kinds}
    folds: dict[str, list] = {k: [] for k in kinds}
    fold_selected: list[np.ndarray] = []
    n_splits = min(5, len(np.unique(g_m)))
    # Use continuous proxy for feature selection via y as soft labels
    y_soft = y_m.astype(np.float64)
    for fold_i, (tr, te) in enumerate(
        GroupKFold(n_splits=n_splits).split(X_m, y_m, groups=g_m), start=1
    ):
        selected = _stable_selection(
            X_m[tr],
            y_soft[tr],
            seed=seed,
            cumulative=cumulative,
            max_features=max_features,
        )
        fold_selected.append(selected)
        xtr, xte, _, _ = _transform_xy(X_m[tr], X_m[te], selected)
        pos = max(int((y_m[tr] == 1).sum()), 1)
        neg = max(int((y_m[tr] == 0).sum()), 1)
        for kind in kinds:
            model = _clf_model(seed, kind)
            if kind == "xgboost":
                model.set_params(scale_pos_weight=neg / pos)
            model.fit(xtr, y_m[tr])
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(xte)[:, 1]
            else:
                prob = model.predict(xte)
            local[kind][te] = prob
            ap = (
                float(average_precision_score(y_m[te], prob))
                if len(np.unique(y_m[te])) > 1
                else float("nan")
            )
            folds[kind].append({"fold": fold_i, "n": int(len(te)), "pr_auc": ap})
    oof = np.full(len(y_bin), np.nan)
    metrics = {}
    for kind in kinds:
        full = np.full(len(y_bin), np.nan)
        full[idx] = local[kind]
        metrics[kind] = {
            "pr_auc": float(average_precision_score(y_bin[idx], local[kind]))
            if len(np.unique(y_m)) > 1
            else float("nan"),
            "folds": folds[kind],
        }
    best = max(kinds, key=lambda k: (metrics[k]["pr_auc"],))
    oof[idx] = local[best]
    votes = np.zeros(X.shape[1], dtype=np.int32)
    for selected in fold_selected:
        votes[selected] += 1
    threshold = (len(fold_selected) + 1) // 2
    stable = np.flatnonzero(votes >= threshold)
    if len(stable) < 32:
        stable = np.argsort(-votes)[: min(max_features, X.shape[1])]
    return oof, metrics, stable, best


def _fit_final_clf(
    X: np.ndarray,
    y_bin: np.ndarray,
    *,
    stable: np.ndarray,
    kind: str,
    seed: int,
    mask: np.ndarray,
) -> dict:
    X_m = X[mask]
    y_m = y_bin[mask].astype(int)
    xtr, _, dmean, dscale = _transform_xy(X_m, X_m, stable)
    pos = max(int((y_m == 1).sum()), 1)
    neg = max(int((y_m == 0).sum()), 1)
    base = _clf_model(seed, kind)
    if kind == "xgboost":
        base.set_params(scale_pos_weight=neg / pos)
        base.fit(xtr, y_m)
        model = base
        calibrator = None
    else:
        # Isotonic calibration on internal CV of the train set.
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        model.fit(xtr, y_m)
        calibrator = "isotonic"
    return {
        "model": model,
        "models": [model],
        "descriptor_mean": dmean,
        "descriptor_scale": dscale,
        "feature_set": FEATURE_SET_3D,
        "all_feature_names": feature_names(FEATURE_SET_3D),
        "feature_indices": stable.tolist(),
        "feature_names": [feature_names(FEATURE_SET_3D)[i] for i in stable],
        "descriptor_names": list(_BROAD_DESCRIPTOR_NAMES),
        "target": "strong_BE_GM_lt_m6",
        "model_kind": kind,
        "calibrator": calibrator,
        "target_mean": 0.0,
        "target_scale": 1.0,
    }


def _predict_payload(payload: dict, X: np.ndarray, mw: np.ndarray | None = None) -> np.ndarray:
    n_desc = len(payload["descriptor_names"])
    Xt = np.asarray(X, dtype=np.float32).copy()
    Xt[:, :n_desc] = (Xt[:, :n_desc] - payload["descriptor_mean"]) / payload["descriptor_scale"]
    Xt = Xt[:, np.asarray(payload["feature_indices"], dtype=int)]
    models = list(payload.get("models", [payload["model"]]))
    preds = np.asarray([m.predict(Xt) for m in models], dtype=np.float64)
    # classifiers
    if hasattr(models[0], "predict_proba"):
        probs = np.asarray([m.predict_proba(Xt)[:, 1] for m in models], dtype=np.float64)
        return probs.mean(axis=0)
    pred = preds.mean(axis=0) * float(payload["target_scale"]) + float(payload["target_mean"])
    if payload.get("target_transform") == "log1p":
        pred = np.expm1(pred)
        pred = np.maximum(pred, 0.0)
    mw_model = payload.get("mw_model")
    if mw_model is not None and mw is not None:
        pred = pred + mw_model.predict(mw.reshape(-1, 1))
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--importance-cumulative", type=float, default=0.95)
    parser.add_argument("--max-features", type=int, default=512)
    parser.add_argument(
        "--penalty-objective",
        default="reg:pseudohubererror",
        choices=("reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"),
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Assume --data already exists",
    )
    args = parser.parse_args()

    if not args.skip_build or not args.data.is_file():
        import importlib.util

        build_path = ROOT / "scripts" / "build_binding_predictor_train.py"
        spec = importlib.util.spec_from_file_location(
            "build_binding_predictor_train", build_path
        )
        assert spec is not None and spec.loader is not None
        build_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_mod)
        build_mod.build(out=args.data)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df, X, mw = _load_featurized(args.data)
    groups = np.asarray([_scaffold_group(s) for s in df["SMILES"]])
    gm_mask = (df["label_set"] == "gm_be").to_numpy()
    ie = df["IE"].to_numpy(dtype=float)
    be = df["BE"].to_numpy(dtype=float)
    be_gm = df["BE_GM"].to_numpy(dtype=float)
    penalty_raw = df["E_host_penalty_kcal"].to_numpy(dtype=float)
    penalty_clamped = np.maximum(0.0, penalty_raw)
    strong = (be_gm < -6.0).astype(float)
    strong[~gm_mask] = np.nan

    print(
        f"[data] rows={len(df)} gm_be={int(gm_mask.sum())} "
        f"features={X.shape[1]} scaffolds={len(np.unique(groups))}"
    )

    ablations: dict = {}
    oof_table = df[["SMILES", "IE", "BE", "BE_GM", "E_host_penalty_kcal", "source", "label_set"]].copy()
    oof_table["mw"] = mw

    # --- IE (gm_be only for ship/gates; ie_only expansion hurts scaffold OOF) ---
    print("[train] IE (gm_be)")
    ie_oof, ie_meta, ie_stable, ie_best = _oof_regression(
        X, ie, groups, mw, seed=args.seed,
        cumulative=args.importance_cumulative, max_features=args.max_features,
        mask=gm_mask, residualize_mw=False,
    )
    ie_oof_mw, ie_meta_mw, _, _ = _oof_regression(
        X, ie, groups, mw, seed=args.seed,
        cumulative=args.importance_cumulative, max_features=args.max_features,
        mask=gm_mask, residualize_mw=True,
    )
    print("[train] IE (all rows incl ie_only) ablation")
    ie_all_oof, ie_all_meta, _, ie_all_best = _oof_regression(
        X, ie, groups, mw, seed=args.seed,
        cumulative=args.importance_cumulative, max_features=args.max_features,
        residualize_mw=False,
    )
    oof_table["pred_ie"] = ie_oof[ie_best]
    oof_table["pred_ie_mwres"] = ie_oof_mw[
        max(ie_oof_mw, key=lambda k: ie_meta_mw["model_metrics"][k]["spearman"])
    ]
    oof_table["pred_ie_allrows"] = ie_all_oof[ie_all_best]
    ie_payload = _fit_final_reg(
        X, ie, mw, stable=ie_stable, kind=ie_best, seed=args.seed,
        residualize_mw=False, objective="reg:squarederror",
        target_transform="identity", target_name="IE_kcal_mol", mask=gm_mask,
    )
    joblib.dump(ie_payload, out_dir / "ie.joblib")

    # --- BE (gm_be only) ---
    print("[train] BE")
    be_oof, be_meta, be_stable, be_best = _oof_regression(
        X, be, groups, mw, seed=args.seed,
        cumulative=args.importance_cumulative, max_features=args.max_features,
        mask=gm_mask, residualize_mw=False,
    )
    be_oof_mw, be_meta_mw, _, _ = _oof_regression(
        X, be, groups, mw, seed=args.seed,
        cumulative=args.importance_cumulative, max_features=args.max_features,
        mask=gm_mask, residualize_mw=True,
    )
    oof_table["pred_be"] = be_oof[be_best]
    oof_table["pred_be_mwres"] = be_oof_mw[
        max(be_oof_mw, key=lambda k: be_meta_mw["model_metrics"][k]["spearman"])
    ]
    be_payload = _fit_final_reg(
        X, be, mw, stable=be_stable, kind=be_best, seed=args.seed,
        residualize_mw=False, objective="reg:squarederror",
        target_transform="identity", target_name="BE_kcal_mol", mask=gm_mask,
    )
    joblib.dump(be_payload, out_dir / "be.joblib")

    # --- Penalty variants ---
    # Ship path: clamped + log1p (won first-pass ablation vs Huber/MSE/raw).
    print("[train] penalty clamped log1p (ship)")
    pen_oof, pen_meta, pen_stable, pen_best = _oof_regression(
        X, penalty_clamped, groups, mw, seed=args.seed,
        cumulative=args.importance_cumulative, max_features=args.max_features,
        mask=gm_mask, residualize_mw=False,
        objective="reg:squarederror", target_transform="log1p",
    )
    print("[train] penalty clamped + huber ablation")
    pen_huber_oof, pen_huber_meta, _, pen_huber_best = _oof_regression(
        X, penalty_clamped, groups, mw, seed=args.seed,
        cumulative=args.importance_cumulative, max_features=args.max_features,
        mask=gm_mask, residualize_mw=False,
        objective=args.penalty_objective, target_transform="identity",
    )
    print("[train] penalty raw MSE ablation")
    pen_raw_oof, pen_raw_meta, _, pen_raw_best = _oof_regression(
        X, penalty_raw, groups, mw, seed=args.seed,
        cumulative=args.importance_cumulative, max_features=args.max_features,
        mask=gm_mask, residualize_mw=False,
        objective="reg:squarederror", target_transform="identity",
    )
    print("[train] penalty clamped MSE ablation")
    pen_mse_oof, pen_mse_meta, _, pen_mse_best = _oof_regression(
        X, penalty_clamped, groups, mw, seed=args.seed,
        cumulative=args.importance_cumulative, max_features=args.max_features,
        mask=gm_mask, residualize_mw=False,
        objective="reg:squarederror", target_transform="identity",
    )
    oof_table["pred_penalty"] = pen_oof[pen_best]
    oof_table["pred_penalty_huber"] = pen_huber_oof[pen_huber_best]
    oof_table["pred_penalty_raw"] = pen_raw_oof[pen_raw_best]
    oof_table["pred_penalty_mse"] = pen_mse_oof[pen_mse_best]
    pen_payload = _fit_final_reg(
        X, penalty_clamped, mw, stable=pen_stable, kind=pen_best, seed=args.seed,
        residualize_mw=False, objective="reg:squarederror",
        target_transform="log1p", target_name="E_host_penalty_clamped_log1p",
        mask=gm_mask,
    )
    pen_payload["force_nonnegative"] = True
    joblib.dump(pen_payload, out_dir / "penalty.joblib")

    # --- Direct BE_GM ablation ---
    print("[train] direct BE_GM")
    begm_oof, begm_meta, begm_stable, begm_best = _oof_regression(
        X, be_gm, groups, mw, seed=args.seed,
        cumulative=args.importance_cumulative, max_features=args.max_features,
        mask=gm_mask, residualize_mw=False,
    )
    oof_table["pred_begm_direct"] = begm_oof[begm_best]
    begm_payload = _fit_final_reg(
        X, be_gm, mw, stable=begm_stable, kind=begm_best, seed=args.seed,
        residualize_mw=False, objective="reg:squarederror",
        target_transform="identity", target_name="BE_GM_direct", mask=gm_mask,
    )
    joblib.dump(begm_payload, out_dir / "begm_direct.joblib")

    # Compose BE_GM variants from OOF
    oof_table["pred_begm_compose"] = oof_table["pred_be"] + oof_table["pred_penalty"]
    oof_table["pred_begm_compose_rawpen"] = oof_table["pred_be"] + oof_table["pred_penalty_raw"]
    oof_table["pred_begm_compose_huber"] = oof_table["pred_be"] + oof_table["pred_penalty_huber"]
    oof_table["pred_begm_compose_mse"] = oof_table["pred_be"] + oof_table["pred_penalty_mse"]

    # --- Strong binder classifier ---
    print("[train] strong-binder classifier BE_GM<-6")
    strong_oof, strong_meta, strong_stable, strong_best = _oof_classifier(
        X, strong, groups, seed=args.seed,
        cumulative=args.importance_cumulative, max_features=args.max_features,
        mask=gm_mask,
    )
    oof_table["pred_strong_prob"] = strong_oof
    strong_payload = _fit_final_clf(
        X, strong, stable=strong_stable, kind=strong_best, seed=args.seed, mask=gm_mask,
    )
    joblib.dump(strong_payload, out_dir / "strong.joblib")

    # Metrics + gates
    ie_m = _metrics(ie, oof_table["pred_ie"].to_numpy())
    ie_m["spearman_mw_partial"] = _partial_spearman(ie, oof_table["pred_ie"].to_numpy(), mw)
    ie_mw_m = _metrics(ie, oof_table["pred_ie_mwres"].to_numpy())
    be_m = _metrics(np.where(gm_mask, be, np.nan), oof_table["pred_be"].to_numpy())
    be_m["spearman_mw_partial"] = _partial_spearman(
        np.where(gm_mask, be, np.nan), oof_table["pred_be"].to_numpy(), mw
    )
    compose_m = _metrics(np.where(gm_mask, be_gm, np.nan), oof_table["pred_begm_compose"].to_numpy())
    direct_m = _metrics(np.where(gm_mask, be_gm, np.nan), oof_table["pred_begm_direct"].to_numpy())
    compose_raw_m = _metrics(
        np.where(gm_mask, be_gm, np.nan), oof_table["pred_begm_compose_rawpen"].to_numpy()
    )
    compose_huber_m = _metrics(
        np.where(gm_mask, be_gm, np.nan), oof_table["pred_begm_compose_huber"].to_numpy()
    )
    compose_mse_m = _metrics(
        np.where(gm_mask, be_gm, np.nan), oof_table["pred_begm_compose_mse"].to_numpy()
    )
    ie_all_m = _metrics(ie, oof_table["pred_ie_allrows"].to_numpy())
    strong_idx = gm_mask & np.isfinite(strong_oof)
    strong_pr = (
        float(average_precision_score(strong[strong_idx], strong_oof[strong_idx]))
        if strong_idx.any() and len(np.unique(strong[strong_idx])) > 1
        else float("nan")
    )
    prevalence = float(np.nanmean(strong[gm_mask]))

    ablations = {
        "direct_vs_compose": {"direct": direct_m, "compose_clamped_huber": compose_m},
        "penalty_label": {
            "clamped_log1p_compose": compose_m,
            "clamped_huber_compose": compose_huber_m,
            "raw_mse_compose": compose_raw_m,
            "clamped_mse_compose": compose_mse_m,
        },
        "ie_train_pool": {
            "gm_be_only": ie_m,
            "gm_be_plus_compiled4": ie_all_m,
        },
        "mw_residualized": {
            "ie_raw": ie_m,
            "ie_mwres": ie_mw_m,
            "be_raw": be_m,
            "be_mwres": _metrics(
                np.where(gm_mask, be, np.nan), oof_table["pred_be_mwres"].to_numpy()
            ),
        },
        "topk_recall_begm_lt7_at_10pct": {
            "ie_only": _topk_recall(be_gm, oof_table["pred_ie"].to_numpy()),
            "begm_compose": _topk_recall(be_gm, oof_table["pred_begm_compose"].to_numpy()),
            "begm_direct": _topk_recall(be_gm, oof_table["pred_begm_direct"].to_numpy()),
            "strong_prob": _topk_recall(
                be_gm, -oof_table["pred_strong_prob"].to_numpy()
            ),  # higher prob → negate for argsort-as-more-neg
        },
    }
    # Fix strong topk: want high prob first
    sc = oof_table["pred_strong_prob"].to_numpy()
    mask = gm_mask & np.isfinite(be_gm) & np.isfinite(sc)
    if mask.any():
        yt, scores = be_gm[mask], sc[mask]
        k = max(1, int(round(0.1 * yt.size)))
        top = np.argsort(-scores)[:k]
        pos = yt < -7.0
        ablations["topk_recall_begm_lt7_at_10pct"]["strong_prob"] = (
            float((yt[top] < -7.0).sum() / pos.sum()) if pos.any() else float("nan")
        )

    gates = {
        "ie_spearman": {
            "value": ie_m["spearman"],
            "threshold": GATE["ie_spearman"],
            "pass": bool(ie_m["spearman"] >= GATE["ie_spearman"]),
        },
        "ie_rmse": {
            "value": ie_m["rmse"],
            "threshold": GATE["ie_rmse"],
            "pass": bool(ie_m["rmse"] <= GATE["ie_rmse"]),
        },
        "be_spearman": {
            "value": be_m["spearman"],
            "threshold": GATE["be_spearman"],
            "pass": bool(be_m["spearman"] >= GATE["be_spearman"]),
        },
        "be_rmse": {
            "value": be_m["rmse"],
            "threshold": GATE["be_rmse"],
            "pass": bool(be_m["rmse"] <= GATE["be_rmse"]),
        },
        "begm_compose_spearman": {
            "value": compose_m["spearman"],
            "threshold": GATE["begm_compose_spearman"],
            "pass": bool(compose_m["spearman"] >= GATE["begm_compose_spearman"]),
        },
        "strong_pr_auc": {
            "value": strong_pr,
            "threshold": GATE["strong_pr_auc"],
            "pass": bool(strong_pr >= GATE["strong_pr_auc"]),
        },
        "ie_mw_partial_gap": {
            "value": abs(ie_m["spearman"] - ie_m["spearman_mw_partial"])
            if np.isfinite(ie_m["spearman_mw_partial"])
            else float("nan"),
            "threshold": GATE["mw_partial_gap"],
            "pass": bool(
                np.isfinite(ie_m["spearman_mw_partial"])
                and abs(ie_m["spearman"] - ie_m["spearman_mw_partial"])
                <= GATE["mw_partial_gap"]
            ),
        },
    }
    ship_begm = bool(gates["begm_compose_spearman"]["pass"])
    reward_head = "begm_compose" if ship_begm else "ie"

    # OOD train bank (Morgan)
    fps = []
    bank_smiles = []
    for smi in df.loc[gm_mask, "SMILES"]:
        fp = _morgan_fp(smi)
        if fp is not None:
            fps.append(fp)
            bank_smiles.append(smi)
    # Mahalanobis on selected IE descriptor features (standardized)
    n_desc = len(_BROAD_DESCRIPTOR_NAMES)
    desc = X[gm_mask, :n_desc]
    dmean = desc.mean(axis=0)
    dstd = desc.std(axis=0)
    dstd[dstd < 1e-8] = 1.0
    Z = (desc - dmean) / dstd
    # Low-dim PCA-ish: use top variance columns to keep covariance stable
    var = Z.var(axis=0)
    keep = np.argsort(-var)[:32]
    Zk = Z[:, keep]
    cov = np.cov(Zk, rowvar=False)
    cov = cov + np.eye(cov.shape[0]) * 1e-3
    cov_inv = np.linalg.pinv(cov)
    # CV cutoff: 95th percentile of train Mahalanobis
    diff = Zk - Zk.mean(axis=0)
    mahal = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    mahal_cutoff = float(np.percentile(mahal, 95))

    # OOF residual std for uncertainty scale
    ie_resid = ie - oof_table["pred_ie"].to_numpy()
    ie_resid = ie_resid[np.isfinite(ie_resid)]
    be_resid = be[gm_mask] - oof_table.loc[gm_mask, "pred_be"].to_numpy()
    be_resid = be_resid[np.isfinite(be_resid)]
    pen_resid = penalty_clamped[gm_mask] - oof_table.loc[gm_mask, "pred_penalty"].to_numpy()
    pen_resid = pen_resid[np.isfinite(pen_resid)]

    pack = {
        "ie_path": str(out_dir / "ie.joblib"),
        "be_path": str(out_dir / "be.joblib"),
        "penalty_path": str(out_dir / "penalty.joblib"),
        "strong_path": str(out_dir / "strong.joblib"),
        "begm_direct_path": str(out_dir / "begm_direct.joblib"),
        "reward_head": reward_head,
        "ship_begm_compose": ship_begm,
        "ood": {
            "tanimoto_floor": 0.35,
            "mahal_cutoff": mahal_cutoff,
            "mahal_feature_indices": keep.tolist(),
            "mahal_mean": Zk.mean(axis=0).tolist(),
            "mahal_cov_inv": cov_inv.tolist(),
            "descriptor_mean": dmean.tolist(),
            "descriptor_scale": dstd.tolist(),
            "bank_smiles": bank_smiles,
        },
        "uncertainty": {
            "ie_oof_rmse": float(np.sqrt(np.mean(ie_resid**2))) if ie_resid.size else 1.0,
            "be_oof_rmse": float(np.sqrt(np.mean(be_resid**2))) if be_resid.size else 1.0,
            "penalty_oof_rmse": float(np.sqrt(np.mean(pen_resid**2))) if pen_resid.size else 1.0,
        },
        "gates": gates,
        "feature_set": FEATURE_SET_3D,
    }
    # Store fingerprints as bit strings for portability
    pack["ood"]["bank_fp_bitstrings"] = [fp.ToBitString() for fp in fps]
    joblib.dump(pack, out_dir / "pack.joblib")

    # Also dump a composed begm stub payload referencing be+penalty for MGW
    joblib.dump(
        {
            "mode": "compose",
            "be_path": str(out_dir / "be.joblib"),
            "penalty_path": str(out_dir / "penalty.joblib"),
            "target": "BE_GM_compose",
        },
        out_dir / "begm.joblib",
    )

    metrics = {
        "data_path": str(args.data),
        "n_rows": int(len(df)),
        "n_gm_be": int(gm_mask.sum()),
        "cv_strategy": "bemis_murcko_scaffold_grouped",
        "selected_models": {
            "ie": ie_best,
            "be": be_best,
            "penalty": pen_best,
            "begm_direct": begm_best,
            "strong": strong_best,
        },
        "ie": ie_m,
        "ie_mwres": ie_mw_m,
        "be": be_m,
        "penalty_clamped": _metrics(
            np.where(gm_mask, penalty_clamped, np.nan), oof_table["pred_penalty"].to_numpy()
        ),
        "begm_compose": compose_m,
        "begm_direct": direct_m,
        "strong": {
            "pr_auc": strong_pr,
            "prevalence": prevalence,
            "enrichment": float(strong_pr / prevalence) if prevalence > 0 else float("nan"),
            "model_metrics": strong_meta,
        },
        "feature_dominance": {
            "ie": ie_payload.get("feature_dominance"),
            "be": be_payload.get("feature_dominance"),
            "penalty": pen_payload.get("feature_dominance"),
        },
        "gates": gates,
        "reward_head": reward_head,
        "all_gates_pass": all(g["pass"] for g in gates.values()),
        "ablations": ablations,
        "penalty_model_meta": pen_meta["model_metrics"],
        "ie_model_meta": ie_meta["model_metrics"],
        "be_model_meta": be_meta["model_metrics"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "ablations.json").write_text(json.dumps(ablations, indent=2) + "\n")
    oof_table.to_csv(out_dir / "oof.csv", index=False)

    print(json.dumps({"gates": gates, "reward_head": reward_head}, indent=2))
    print(f"[save] {out_dir}")


if __name__ == "__main__":
    main()
