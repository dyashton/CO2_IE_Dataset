# CO2 Binding Predictor

Hybrid physisorption surrogate trained on `compiled_gm_be` DFT labels.

## Label physics

| Symbol | Meaning |
|--------|---------|
| `IE` | Host–CO₂ interaction energy (kcal/mol; more negative = stronger) |
| `BE` | Binding energy before host reorganization |
| `E_host_penalty_kcal` | Host strain / reorganization |
| `BE_GM` | `BE + E_host_penalty_kcal` (geometry-minimized binding) |

This is **non-covalent / physisorption** IE–BE chemistry from the existing pose→DFT pipeline. It is **not** amine carbamate chemisorption ΔH (those labels are not in this dataset).

## Why decompose BE_GM?

Direct 2D `BE_GM` regression is weak (scaffold Spearman ~0.44). Oracle ablation shows `BE_OOF + true_penalty` reaches ~0.79. MACE host-strain calibration failed (Spearman ≈ 0 vs DFT penalty). This pack instead learns `penalty` from DFT labels with:

- **Clamp:** `penalty_clamped = max(0, E_host_penalty)` so negative method noise cannot become fake binding bonuses (~37% of raw penalties are slightly negative from GFN2-xTB vs r²SCAN-3c rank reversal).
- **Ship transform:** regress `log1p(penalty_clamped)` and invert at predict-time (won vs Huber/MSE/raw on BE_GM compose ranking). Huber/MSE kept as ablations.

Compose: `BE_GM_hat = BE_hat + penalty_hat`.

A calibrated **strong-binder classifier** (`BE_GM < -6`) counters regression shrinkage of the elite tail.

## Features

Descriptor-first: `broad_rdkit_3d_v1` (RDKit descriptors + Morgan/MACCS/etc. + ETKDG shape: asphericity, PMI, …). ChemBERTa fusion is **out of scope for the default ship path** (ablation-only if revisited).

Anti-hack: MW-partialled Spearman and optional MW-residualized heads are reported in `metrics.json`.

## Train

```bash
cd CO2_IE_Dataset
uv run python scripts/build_binding_predictor_train.py
uv run python scripts/train_binding_predictor.py
```

Artifacts: `artifacts/binding_predictor/{ie,be,penalty,begm,begm_direct,strong,pack}.joblib`, `metrics.json`, `oof.csv`, `ablations.json`.

Data card: `docs/BINDING_PREDICTOR_DATA.md`.

## Gates (scaffold OOF) and latest result

| Head | Gate | Latest (`artifacts/binding_predictor/metrics.json`) |
|------|------|------|
| IE (gm_be) | Spearman ≥ 0.75, RMSE ≤ 1.05 | ρ≈0.73, RMSE≈1.07 (near miss; ship anyway as best head) |
| BE | Spearman ≥ 0.70, RMSE ≤ 1.10 | ρ≈0.698, RMSE≈1.09 |
| BE_GM compose | Spearman ≥ 0.60 | ρ≈0.48 (beats direct 0.44; still below gate) |
| Strong `BE_GM < -6` | PR-AUC ≥ 0.35 | **pass** ≈0.38 (enrichment ≈2.9×) |
| Anti-hack | IE MW-partial gap ≤ 0.05 | gap≈0.14 (size still leaks into IE) |

`pack.joblib` sets `reward_head=ie` when compose misses the BE_GM gate. IE alone recovers more `BE_GM < -7` in top-10% than compose/direct/strong in the current ablation.

## MGW / RL hook

```bash
export MGW_OBJECTIVE=co2_binding_surrogate
export MGW_BINDING_PACK=/path/to/CO2_IE_Dataset/artifacts/binding_predictor/pack.joblib
# optional:
export MGW_BINDING_HEAD=begm_compose   # or ie
export MGW_BINDING_STRONG_BONUS=0.5
export MGW_BINDING_MAX_UNCERTAINTY=    # default 2× OOF RMSE
```

Example yaml: `Molecule_Generation_Workflow/train_selfies_co2_binding.yaml`.

**OOD gate:** max Morgan Tanimoto to train bank < 0.35 **or** descriptor Mahalanobis above the train 95th percentile → gated off (trees go quiet OOD; ensemble disagreement alone is insufficient).

Reward: `-binding_hat + strong_bonus * P(strong) − inorganic/charged/MW penalties`.
