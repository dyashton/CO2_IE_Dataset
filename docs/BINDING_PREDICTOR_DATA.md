# Binding predictor training table

Source: `data/binding_predictor_train.csv`

## Contents

| Set | n | Labels |
|-----|--:|--------|
| gm_be (after BE_GM clip) | 1167 | IE, BE, BE_GM, E_host_penalty_kcal |
| ie_only (`compiled_4` novel SMILES) | 457 | IE only |
| **total rows** | **1624** | |

## Label physics

- `IE`: host–CO2 interaction energy (kcal/mol, more negative = stronger interaction)
- `BE`: binding energy before host reorganization
- `BE_GM = BE + E_host_penalty_kcal`
- `penalty_clamped = max(0, E_host_penalty_kcal)` — negative raw penalties are method noise (GFN2-xTB vs r²SCAN-3c conformer rank reversal), not physical binding bonuses

## Train-only BE_GM clip

- drop gm_be outside BE_GM [p1, p99] = [-7.929, 1.863] (dropped 24)
- Compiled discovery/seed CSVs are **not** modified (provenance kept).

## Penalty diagnostics (gm_be only, post-clip)

- frac negative raw penalty: 0.000
- min / median / max: 0.000 / 0.001 / 8.002
- BE_GM < -6 prevalence: 0.093
- BE_GM < -7 prevalence: 0.025

## CV

Bemis–Murcko scaffold GroupKFold (5-fold), same protocol as `scripts/train_surrogate.py`.

## Excluded

- `peptide_co2_binders_150.csv` (class-only, no energies)
- `amine_2400` (narrow diagnostic IE snapshot; not main train pool)
- Carbamate / chemisorption ΔH literature labels (not in this dataset)
