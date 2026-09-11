#!/usr/bin/env python3
"""Self-check: molecule-grouped splits never leak; synthetic BTL is separable."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from co2ie.preference import (
    MolRow,
    PreferenceModel,
    cross_validate,
    fit_btl,
    make_pairs,
)


def test_group_disjoint() -> None:
    mols = [
        MolRow(smiles="CCO", ei=-1.0, source="a", mol_id="a_0"),
        MolRow(smiles="CCC", ei=-2.0, source="a", mol_id="a_1"),
        MolRow(smiles="CCCC", ei=-3.0, source="a", mol_id="a_2"),
        MolRow(smiles="CCCO", ei=-1.5, source="a", mol_id="a_3"),
        MolRow(smiles="CCCCO", ei=-2.5, source="a", mol_id="a_4"),
        MolRow(smiles="CCCCCO", ei=-0.5, source="a", mol_id="a_5"),
    ]
    groups = np.array([m.mol_id for m in mols])
    X = np.arange(len(mols)).reshape(-1, 1)
    y = np.zeros(len(mols))
    gkf = GroupKFold(n_splits=3)
    for tr, te in gkf.split(X, y, groups):
        assert set(groups[tr]).isdisjoint(set(groups[te])), "molecule leaked across fold"


def test_pairs_same_source_only() -> None:
    eis = np.array([-1.0, -3.0, -2.0, -5.0])
    sources = ["a", "a", "b", "b"]
    rng = np.random.default_rng(0)
    pairs, _ = make_pairs([0, 1, 2, 3], eis, sources, margin=0.5, n_partners=10, rng=rng)
    for i, j in pairs:
        assert sources[i] == sources[j]


def test_synthetic_btl() -> None:
    # separable: bit 0 on => better (more neg EI)
    rng = np.random.default_rng(0)
    n, d = 40, 16
    fps = rng.integers(0, 2, size=(n, d)).astype(np.float64)
    eis = -fps[:, 0] * 3.0 + rng.normal(0, 0.05, size=n)
    sources = ["syn"] * n
    idx = list(range(n))
    pairs, y = make_pairs(idx, eis, sources, margin=0.2, n_partners=20, rng=rng)
    model = fit_btl(fps, pairs, y, C=10.0)
    X = np.stack([fps[i] - fps[j] for i, j in pairs])
    pred = (X @ model.coef > 0).astype(int)
    acc = float(np.mean(pred == y))
    assert acc > 0.9, f"synthetic pair acc {acc}"


if __name__ == "__main__":
    test_group_disjoint()
    test_pairs_same_source_only()
    test_synthetic_btl()
    print("ok: leakage guards + synthetic BTL")
