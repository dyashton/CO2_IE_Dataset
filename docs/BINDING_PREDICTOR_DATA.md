# Binding predictor training table

Source: `data/binding_predictor_train.csv`

## Contents

| Set | n | Labels |
|-----|--:|--------|
| gm_be (`compiled_gm_be`) | 1047 | IE, BE, BE_GM, E_host_penalty_kcal |
| ie_only (`compiled_4` novel SMILES) | 434 | IE only |
| **total rows** | **1481** | |

## Label physics

- `IE`: host–CO2 interaction energy (kcal/mol, more negative = stronger interaction)
- `BE`: binding energy before host reorganization
- `BE_GM = BE + E_host_penalty_kcal`
- `penalty_clamped = max(0, E_host_penalty_kcal)` — negative raw penalties are method noise (GFN2-xTB vs r²SCAN-3c conformer rank reversal), not physical binding bonuses

## Penalty diagnostics (gm_be only)

- frac negative raw penalty: 0.000
- min / median / max: 0.000 / 0.004 / 16.677
- BE_GM < -6 prevalence: 0.113
- BE_GM < -7 prevalence: 0.039

## CV

Bemis–Murcko scaffold GroupKFold (5-fold), same protocol as `scripts/train_surrogate.py`.

## Excluded

- `peptide_co2_binders_150.csv` (class-only, no energies)
- `amine_2400` (narrow diagnostic IE snapshot; not main train pool)
- Carbamate / chemisorption ΔH literature labels (not in this dataset)
