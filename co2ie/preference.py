"""Bradley–Terry pairwise preference scorer for CO2 binding.

Molecules are the grouping unit: CV folds never put the same molecule in
train and test, and pairs are built only within a fold's molecule set so
pair-level labels cannot leak across folds.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from co2ie import ROOT

MARGIN_KCAL = 0.5
N_PARTNERS = 50
FP_BITS = 2048
FP_RADIUS = 2
N_FOLDS = 5
RANDOM_SEED = 7


@dataclass(frozen=True)
class MolRow:
    smiles: str
    ei: float
    source: str
    mol_id: str


@dataclass
class PreferenceModel:
    """Linear BTL: s(x) = w·x (bias cancels in pairwise Δs)."""

    coef: np.ndarray  # shape (n_features,)
    n_bits: int = FP_BITS
    radius: int = FP_RADIUS

    def score_fp(self, fp: np.ndarray) -> float:
        return float(fp @ self.coef)

    def score_smiles(self, smiles: str) -> float | None:
        fp = smiles_to_fp(smiles, n_bits=self.n_bits, radius=self.radius)
        if fp is None:
            return None
        return self.score_fp(fp)


def smiles_to_fp(
    smiles: str, *, n_bits: int = FP_BITS, radius: int = FP_RADIUS
) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float64)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def load_compiled(path: Path) -> list[MolRow]:
    rows: list[MolRow] = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            smi = (row.get("SMILES") or row.get("smiles") or "").strip()
            ei_s = (row.get("EI") or row.get("EInt") or "").strip()
            src = (row.get("source") or "unknown").strip()
            if not smi or not ei_s:
                continue
            try:
                ei = float(ei_s)
            except ValueError:
                continue
            if smiles_to_fp(smi) is None:
                continue
            rows.append(MolRow(smiles=smi, ei=ei, source=src, mol_id=f"{src}_{i}"))
    return rows


def build_compiled_csv(out: Path | None = None) -> Path:
    """Merge test.csv + test_1.csv with source tags."""
    out = out or (ROOT / "data" / "compiled.csv")
    rows: list[tuple[str, float, str]] = []

    test = ROOT / "data" / "test.csv"
    with test.open(newline="") as f:
        for row in csv.DictReader(f):
            smi = (row.get("SMILES") or "").strip()
            e = (row.get("EInt") or "").strip()
            if not smi or not e:
                continue
            try:
                rows.append((smi, float(e), "test"))
            except ValueError:
                continue

    test1 = ROOT / "data" / "test_1.csv"
    with test1.open(newline="") as f:
        for row in csv.DictReader(f):
            smi = (row.get("SMILES") or "").strip()
            e = (row.get("Binding_Energy (kcal/mol)") or "").strip()
            if not smi or not e:
                continue
            try:
                rows.append((smi, float(e), "test_1"))
            except ValueError:
                continue

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["SMILES", "EI", "source"])
        w.writeheader()
        for smi, ei, src in rows:
            w.writerow({"SMILES": smi, "EI": ei, "source": src})
    return out


def _better(ei_a: float, ei_b: float) -> bool:
    """More negative EI = stronger binder."""
    return ei_a < ei_b


def make_pairs(
    indices: list[int],
    eis: np.ndarray,
    sources: list[str],
    *,
    margin: float = MARGIN_KCAL,
    n_partners: int = N_PARTNERS,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Build (i,j) pairs with y=1 if molecule i binds better than j.

    Only same-source pairs with |EI_i - EI_j| >= margin.
    Each molecule gets up to n_partners random eligible partners.
    Caller must pass indices from a single CV split only (no cross-fold pairs).
    """
    by_source: dict[str, list[int]] = {}
    for i in indices:
        by_source.setdefault(sources[i], []).append(i)

    pairs: list[tuple[int, int]] = []
    labels: list[int] = []
    for members in by_source.values():
        for i in members:
            eligible = [
                j
                for j in members
                if j != i and abs(float(eis[i] - eis[j])) >= margin
            ]
            if not eligible:
                continue
            rng.shuffle(eligible)
            for j in eligible[:n_partners]:
                y = 1 if _better(float(eis[i]), float(eis[j])) else 0
                pairs.append((i, j))
                labels.append(y)
    return pairs, np.asarray(labels, dtype=np.int32)


def _pair_matrix(
    pairs: list[tuple[int, int]], fps: np.ndarray
) -> np.ndarray:
    """X_pair[k] = fp_i - fp_j for pair (i,j)."""
    if not pairs:
        return np.zeros((0, fps.shape[1]), dtype=np.float64)
    return np.stack([fps[i] - fps[j] for i, j in pairs], axis=0)


def fit_btl(
    fps: np.ndarray,
    pairs: list[tuple[int, int]],
    y: np.ndarray,
    *,
    C: float = 1.0,
) -> PreferenceModel:
    X = _pair_matrix(pairs, fps)
    if len(X) < 10:
        raise RuntimeError(f"too few pairs to fit BTL ({len(X)})")
    clf = LogisticRegression(
        C=C,
        fit_intercept=False,  # bias cancels in Δs = w·(x_a-x_b)
        solver="liblinear",
        max_iter=500,
        random_state=RANDOM_SEED,
    )
    clf.fit(X, y)
    return PreferenceModel(coef=clf.coef_.ravel().astype(np.float64))


def pair_accuracy(
    model: PreferenceModel,
    fps: np.ndarray,
    pairs: list[tuple[int, int]],
    y: np.ndarray,
) -> float:
    if not pairs:
        return float("nan")
    X = _pair_matrix(pairs, fps)
    scores = X @ model.coef
    pred = (scores > 0).astype(np.int32)
    return float(np.mean(pred == y))


def spearman_score_vs_ei(model: PreferenceModel, fps: np.ndarray, eis: np.ndarray) -> float:
    s = fps @ model.coef
    # higher s should mean better binder → more negative EI → negative correlation expected
    # report Spearman between s and -EI (higher s ↔ stronger binding)
    rho, _ = stats.spearmanr(s, -eis)
    return float(rho)


def cross_validate(
    mols: list[MolRow],
    *,
    n_folds: int = N_FOLDS,
    margin: float = MARGIN_KCAL,
    n_partners: int = N_PARTNERS,
) -> dict:
    """Molecule-grouped CV: GroupKFold on mol_id; pairs only within each split."""
    n = len(mols)
    fps = np.stack([smiles_to_fp(m.smiles) for m in mols])  # type: ignore[arg-type]
    eis = np.array([m.ei for m in mols], dtype=np.float64)
    sources = [m.source for m in mols]
    groups = np.array([m.mol_id for m in mols])

    gkf = GroupKFold(n_splits=min(n_folds, n))
    rng = np.random.default_rng(RANDOM_SEED)

    fold_metrics: list[dict] = []
    for fold, (tr, te) in enumerate(gkf.split(fps, eis, groups), start=1):
        tr_idx = tr.tolist()
        te_idx = te.tolist()
        # leakage guard: no shared mol_ids
        assert set(groups[tr_idx]).isdisjoint(set(groups[te_idx]))

        tr_pairs, tr_y = make_pairs(
            tr_idx, eis, sources, margin=margin, n_partners=n_partners, rng=rng
        )
        te_pairs, te_y = make_pairs(
            te_idx, eis, sources, margin=margin, n_partners=n_partners, rng=rng
        )
        model = fit_btl(fps, tr_pairs, tr_y)
        fold_metrics.append(
            {
                "fold": fold,
                "n_train_mols": len(tr_idx),
                "n_test_mols": len(te_idx),
                "n_train_pairs": len(tr_pairs),
                "n_test_pairs": len(te_pairs),
                "train_pair_acc": pair_accuracy(model, fps, tr_pairs, tr_y),
                "test_pair_acc": pair_accuracy(model, fps, te_pairs, te_y),
                "test_spearman": spearman_score_vs_ei(model, fps[te_idx], eis[te_idx]),
            }
        )
    return {"folds": fold_metrics, "n_mols": n}


def train_full(
    mols: list[MolRow],
    *,
    margin: float = MARGIN_KCAL,
    n_partners: int = N_PARTNERS,
) -> PreferenceModel:
    fps = np.stack([smiles_to_fp(m.smiles) for m in mols])  # type: ignore[arg-type]
    eis = np.array([m.ei for m in mols], dtype=np.float64)
    sources = [m.source for m in mols]
    idx = list(range(len(mols)))
    rng = np.random.default_rng(RANDOM_SEED)
    pairs, y = make_pairs(idx, eis, sources, margin=margin, n_partners=n_partners, rng=rng)
    return fit_btl(fps, pairs, y)


def save_model(model: PreferenceModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"coef": model.coef, "n_bits": model.n_bits, "radius": model.radius},
        path,
    )


def load_model(path: Path) -> PreferenceModel:
    d = joblib.load(path)
    return PreferenceModel(coef=np.asarray(d["coef"]), n_bits=d["n_bits"], radius=d["radius"])
