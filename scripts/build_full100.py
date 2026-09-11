#!/usr/bin/env python3
"""Build data/full100.csv: 100 unlabeled SMILES from MGW + PropMolFlow.

Always force-includes PropMolFlow/results/top25_smiles.csv (top binder candidates),
then fills remaining slots with diverse MGW / PropMolFlow / AxelRolov picks.

Excludes anything with a stored IE (test.csv EInt, PropMolFlow DFT benches,
prior CO2_IE_Dataset energies). Ranking signals (reward/heur/HAE) are allowed.
"""
from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
KOSTAS = ROOT.parent
OUT = ROOT / "data" / "full100.csv"
N_HA_MAX = 40

# block sizes for the *remaining* slots after forced top candidates
N_MGW = 30
N_PMF_RANKED = 20
N_PMF_LARGE = 15
N_AXEL = 10
N_TOTAL = 100

MGW_TOP_BAND = 500  # reward pool before diversity
PMF_TOP_BAND = 400
AXEL_TOP_BAND = 200

TOP25_PATH = KOSTAS / "PropMolFlow" / "results" / "top25_smiles.csv"


@dataclass
class Cand:
    smiles: str
    source: str
    n_heavy: int
    rank_score: float
    note: str
    fp: object | None = None


def _canon(smi: str) -> tuple[str, int] | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol), mol.GetNumHeavyAtoms()


def _fp(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # ponytail: MorganGenerator API varies by rdkit; BitVect is enough for Tanimoto
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _truthy(v: object) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes"}


def _read_smiles_col(fieldnames: list[str] | None) -> str:
    if not fieldnames:
        raise ValueError("empty csv")
    for name in fieldnames:
        if name and name.lower() == "smiles":
            return name
    raise ValueError(f"no SMILES column in {fieldnames}")


def _load_exclusion() -> set[str]:
    excluded: set[str] = set()

    def add_smi(raw: str) -> None:
        raw = (raw or "").strip()
        if not raw:
            return
        got = _canon(raw)
        if got:
            excluded.add(got[0])

    # test.csv EInt labels
    p = ROOT / "data" / "test.csv"
    if p.is_file():
        with p.open(newline="") as f:
            r = csv.DictReader(f)
            col = _read_smiles_col(r.fieldnames)
            for row in r:
                add_smi(row.get(col, ""))

    # PropMolFlow DFT benches
    for rel, scol in (
        ("PropMolFlow/results/dft_benchmark_full.csv", "smiles"),
        ("PropMolFlow/results/ie_push/seeds.csv", "smiles"),
    ):
        p = KOSTAS / rel
        if not p.is_file():
            continue
        with p.open(newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                add_smi(row.get(scol, "") or row.get("input_smiles", ""))

    # prior CO2_IE_Dataset runs with any IE or catalog smiles
    runs = ROOT / "runs"
    if runs.is_dir():
        for run_dir in runs.iterdir():
            if not run_dir.is_dir():
                continue
            cat = run_dir / "catalog.csv"
            if cat.is_file():
                with cat.open(newline="") as f:
                    r = csv.DictReader(f)
                    col = _read_smiles_col(r.fieldnames)
                    for row in r:
                        add_smi(row.get(col, ""))
            en = run_dir / "energies.csv"
            if en.is_file():
                with en.open(newline="") as f:
                    r = csv.DictReader(f)
                    if not r.fieldnames:
                        continue
                    scol = _read_smiles_col(r.fieldnames)
                    ie_col = next(
                        (c for c in r.fieldnames if c and c.lower() == "ie_kcal_mol"),
                        None,
                    )
                    for row in r:
                        ie = (row.get(ie_col) or "").strip() if ie_col else ""
                        if ie:
                            add_smi(row.get(scol, ""))
    return excluded


def _maxmin_pick(pool: list[Cand], n: int, already: list[Cand]) -> list[Cand]:
    """Greedy max-min Tanimoto diversity; seed with highest |rank_score| then farthest."""
    if n <= 0 or not pool:
        return []
    remaining = list(pool)
    for c in remaining:
        if c.fp is None:
            c.fp = _fp(c.smiles)
    remaining = [c for c in remaining if c.fp is not None]
    if not remaining:
        return []

    picked: list[Cand] = []
    refs = [c for c in already if c.fp is not None]

    # first: best score in pool (more positive reward / more negative HAE handled by caller sort)
    remaining.sort(key=lambda c: c.rank_score, reverse=True)
    first = remaining.pop(0)
    picked.append(first)
    refs.append(first)

    while len(picked) < n and remaining:
        best_i = -1
        best_d = -1.0
        for i, c in enumerate(remaining):
            sims = [DataStructs.TanimotoSimilarity(c.fp, r.fp) for r in refs]
            d = 1.0 - max(sims) if sims else 1.0
            if d > best_d:
                best_d = d
                best_i = i
        picked.append(remaining.pop(best_i))
        refs.append(picked[-1])
    return picked


def _ingest(
    path: Path,
    *,
    source: str,
    score_cols: list[str],
    note: str,
    require_gate: bool = False,
    higher_better: bool = True,
    excluded: set[str],
    seen: set[str],
) -> list[Cand]:
    if not path.is_file():
        print(f"[warn] missing {path}")
        return []
    out: list[Cand] = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        scol = _read_smiles_col(r.fieldnames)
        gate = next((c for c in (r.fieldnames or []) if c.lower() == "gate_ok"), None)
        for row in r:
            if require_gate and (gate is None or not _truthy(row.get(gate, ""))):
                continue
            raw = (row.get(scol) or "").strip()
            got = _canon(raw)
            if got is None:
                continue
            can, n_ha = got
            if n_ha > N_HA_MAX or can in excluded or can in seen:
                continue
            score = None
            for col in score_cols:
                if col in row and (row[col] or "").strip():
                    try:
                        score = float(row[col])
                        break
                    except ValueError:
                        continue
            if score is None:
                score = 0.0
            # normalize so higher is always "better" for sorting
            rank = score if higher_better else -score
            out.append(
                Cand(smiles=can, source=source, n_heavy=n_ha, rank_score=rank, note=note)
            )
    return out


def _force_top25(excluded: set[str], seen: set[str]) -> list[Cand]:
    """Always include PropMolFlow results/top25_smiles.csv (unlabeled top binder candidates)."""
    out: list[Cand] = []
    if not TOP25_PATH.is_file():
        print(f"[warn] missing forced top list {TOP25_PATH}")
        return out
    with TOP25_PATH.open(newline="") as f:
        r = csv.DictReader(f)
        scol = _read_smiles_col(r.fieldnames)
        for i, row in enumerate(r):
            raw = (row.get(scol) or "").strip()
            got = _canon(raw)
            if got is None:
                print(f"[top25] skip unparseable row {i}: {raw[:60]}")
                continue
            can, n_ha = got
            if n_ha > N_HA_MAX:
                print(f"[top25] skip n_ha={n_ha}>max: {can}")
                continue
            if can in excluded:
                print(f"[top25] skip labeled: {can}")
                continue
            if can in seen:
                continue
            seen.add(can)
            out.append(
                Cand(
                    smiles=can,
                    source="propmolflow_top25",
                    n_heavy=n_ha,
                    rank_score=float(25 - i),  # list order as priority
                    note="forced_top25_smiles",
                    fp=_fp(can),
                )
            )
    return out


def build() -> list[Cand]:
    excluded = _load_exclusion()
    print(f"[exclude] {len(excluded)} labeled / prior SMILES")
    seen: set[str] = set()
    picked: list[Cand] = []
    skipped_excl = 0

    def take(cands: list[Cand], n: int, band: int) -> list[Cand]:
        nonlocal skipped_excl
        if n <= 0:
            return []
        uniq: dict[str, Cand] = {}
        for c in cands:
            if c.smiles in excluded:
                skipped_excl += 1
                continue
            if c.smiles in seen:
                continue
            prev = uniq.get(c.smiles)
            if prev is None or c.rank_score > prev.rank_score:
                uniq[c.smiles] = c
        pool = sorted(uniq.values(), key=lambda c: c.rank_score, reverse=True)[:band]
        sel = _maxmin_pick(pool, n, picked)
        for c in sel:
            seen.add(c.smiles)
        return sel

    # --- 0: forced PropMolFlow top binder candidates ---
    block0 = _force_top25(excluded, seen)
    picked.extend(block0)
    print(f"[0] forced top25_smiles: {len(block0)}")
    remain = N_TOTAL - len(picked)
    # scale diversity blocks to fill remaining slots (default mix 30/20/15/10 = 75)
    base = N_MGW + N_PMF_RANKED + N_PMF_LARGE + N_AXEL
    n_mgw = max(0, round(remain * N_MGW / base))
    n_pmf = max(0, round(remain * N_PMF_RANKED / base))
    n_large = max(0, round(remain * N_PMF_LARGE / base))
    n_axel = max(0, remain - n_mgw - n_pmf - n_large)
    print(f"[quota] remain={remain} → MGW {n_mgw}, PMF_ranked {n_pmf}, large {n_large}, axel {n_axel}")

    # --- A: MGW ---
    mgw_paths = [
        (
            KOSTAS
            / "Molecule_Generation_Workflow/publication/co2/data/scored_components_nconf2.csv",
            True,
            ["reward"],
            "mgw_scored",
        ),
        (
            KOSTAS / "Molecule_Generation_Workflow/publication/co2/data/top_hits.csv",
            False,
            ["reward"],
            "mgw_top_hits",
        ),
        (
            KOSTAS
            / "Molecule_Generation_Workflow/inference_outputs/co2_best_ep2050/scored_smiles.csv",
            False,
            ["reward"],
            "mgw_ep2050",
        ),
    ]
    mgw_pool: list[Cand] = []
    for path, gate, cols, note in mgw_paths:
        mgw_pool.extend(
            _ingest(
                path,
                source="guacamol",
                score_cols=cols,
                note=note,
                require_gate=gate,
                higher_better=True,
                excluded=excluded,
                seen=seen,
            )
        )
    block_a = take(mgw_pool, n_mgw, MGW_TOP_BAND)
    for c in block_a:
        c.source = "guacamol"
    picked.extend(block_a)
    print(f"[A] MGW picked {len(block_a)}/{n_mgw}")

    # --- B: PropMolFlow ranked / dft candidates ---
    pmf_ranked: list[Cand] = []
    top25_dft = KOSTAS / "PropMolFlow/sampling_input/co2_large/top25_dft_candidates.csv"
    pmf_ranked.extend(
        _ingest(
            top25_dft,
            source="propmolflow_ranked",
            score_cols=["predicted_rank_score"],
            note="top25_dft_candidates",
            require_gate=False,
            higher_better=True,
            excluded=excluded,
            seen=seen,
        )
    )
    samp = KOSTAS / "PropMolFlow/sampling_input"
    gated_extra: list[Cand] = []
    if samp.is_dir():
        for ranked in sorted(samp.glob("co2_large_rerank*/ranked.csv")):
            pmf_ranked.extend(
                _ingest(
                    ranked,
                    source="propmolflow_ranked",
                    score_cols=["co2_reward", "predicted_rank_score"],
                    note=ranked.parent.name,
                    require_gate=False,
                    higher_better=True,
                    excluded=excluded,
                    seen=seen,
                )
            )
            gated_extra.extend(
                _ingest(
                    ranked,
                    source="propmolflow_ranked",
                    score_cols=["co2_reward", "predicted_rank_score"],
                    note=ranked.parent.name + "_gated",
                    require_gate=True,
                    higher_better=True,
                    excluded=excluded,
                    seen=seen,
                )
            )
    pmf_pool = gated_extra + pmf_ranked
    block_b = take(pmf_pool, n_pmf, PMF_TOP_BAND)
    picked.extend(block_b)
    print(f"[B] PropMolFlow ranked picked {len(block_b)}/{n_pmf}")

    # --- C: co2_large stratified + diversity ---
    large = KOSTAS / "PropMolFlow/data/co2_large_raw/co2_large.csv"
    large_pool = _ingest(
        large,
        source="propmolflow_large",
        score_cols=["co2_heur"],
        note="co2_large",
        require_gate=False,
        higher_better=False,
        excluded=excluded,
        seen=seen,
    )
    bins = [(1, 12), (13, 20), (21, 28), (29, 40)]
    stratified: list[Cand] = []
    for lo, hi in bins:
        band = [c for c in large_pool if lo <= c.n_heavy <= hi]
        band.sort(key=lambda c: c.rank_score, reverse=True)
        stratified.extend(band[:80])
    block_c = take(stratified if stratified else large_pool, n_large, len(stratified) or 400)
    picked.extend(block_c)
    print(f"[C] PropMolFlow large picked {len(block_c)}/{n_large}")

    # --- D: AxelRolov ---
    axel = (
        KOSTAS
        / "Molecule_Generation_Workflow/publication/co2/data/axelrolov_scored_nconf2.csv"
    )
    axel_pool = _ingest(
        axel,
        source="axelrolov",
        score_cols=["HAE"],
        note="axelrolov_hae",
        require_gate=False,
        higher_better=False,
        excluded=excluded,
        seen=seen,
    )
    block_d = take(axel_pool, n_axel, AXEL_TOP_BAND)
    picked.extend(block_d)
    print(f"[D] AxelRolov picked {len(block_d)}/{n_axel}")

    if len(picked) < N_TOTAL:
        need = N_TOTAL - len(picked)
        print(f"[backfill] need {need} more")
        extra_pool = [
            c for c in mgw_pool + large_pool if c.smiles not in seen and c.smiles not in excluded
        ]
        extra = take(extra_pool, need, max(need * 20, 200))
        for c in extra:
            c.note = c.note + "|backfill"
        picked.extend(extra)

    print(f"[done] total {len(picked)}")
    return picked[:N_TOTAL]


def _motif_counts(smiles_list: list[str]) -> dict[str, int]:
    pats = {
        "COOH": Chem.MolFromSmarts("C(=O)[OH]"),
        "imidazole": Chem.MolFromSmarts("c1ncc[nH]1"),
        "ether": Chem.MolFromSmarts("[OD2]([#6])[#6]"),
        "amine": Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"),
        "thioether": Chem.MolFromSmarts("[SD2]"),
    }
    counts = {k: 0 for k in pats}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for k, q in pats.items():
            if q is not None and mol.HasSubstructMatch(q):
                counts[k] += 1
    return counts


def main() -> None:
    picked = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["id", "smiles", "source", "n_heavy", "rank_score", "note"]
        )
        w.writeheader()
        for i, c in enumerate(picked):
            w.writerow(
                {
                    "id": f"full_{i:02d}",
                    "smiles": c.smiles,
                    "source": c.source,
                    "n_heavy": c.n_heavy,
                    "rank_score": f"{c.rank_score:.6g}",
                    "note": c.note,
                }
            )
    print(f"[write] {OUT}  ({len(picked)} rows)")

    by_src: dict[str, int] = {}
    for c in picked:
        by_src[c.source] = by_src.get(c.source, 0) + 1
    n_ha = [c.n_heavy for c in picked]
    print("[summary] by source:", dict(sorted(by_src.items())))
    print(
        f"[summary] n_heavy min/med/max = {min(n_ha)}/{statistics.median(n_ha):.0f}/{max(n_ha)}"
    )
    print("[summary] motifs:", _motif_counts([c.smiles for c in picked]))


if __name__ == "__main__":
    main()
