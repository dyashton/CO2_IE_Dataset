# CO2 interaction-energy dataset

SMILES → RDKit ETKDG → **CREST host** → **site-seeded CO₂** (N/O outward normals) → **xTB opt with CO₂ atoms frozen** → geometry filter → rank by xTB IE → Psi4 **r²SCAN-3c** interaction energy.

CREST is used **only for the host**. Free NCI complex search is disabled (it was letting CO₂ covalently fuse). Psi4 `ie_kcal_mol` is the label.

This reports **interaction energy**, not binding affinity.

## Setup

```bash
cd CO2_IE_Dataset
uv sync
bash scripts/install_xtb.sh
bash scripts/install_crest.sh
```

Psi4 is a **CLI**, not a uv dep. Install separately (e.g. conda-forge `psi4` + `dftd4-python` + `gcp-correction`) and put `psi4` on `PATH`.

## Stages

```bash
uv run python scripts/build_small.py --to sample
uv run python scripts/build_small.py --to xyz
uv run python scripts/build_small.py --to crest_host
uv run python scripts/build_small.py --to crest_complex   # site seeds + constrained-CO2 xTB
uv run python scripts/build_small.py --to dft
```

Resume from a stage (e.g. reuse hosts, only rebuild poses):

```bash
uv run python scripts/build_small.py --run-dir runs/test_n5 --csv data/test.csv --n 5 \
  --from crest_complex --to crest_complex
uv run python scripts/build_small.py --run-dir runs/test_n5 --csv data/test.csv --n 5 \
  --from dft --to dft
```

From a CSV (first N SMILES):

```bash
uv run python scripts/build_small.py --csv data/test.csv --n 5 --quick --to dft
```

`--quick` only affects CREST **host** search.

## Pose QC

```bash
uv run python scripts/check_poses.py runs/test_n5
```

Valid pose: both C–O ∈ [1.10, 1.25] Å, O–C–O ≥ 160°, min host–CO₂ distance ≥ 2.2 Å.

## Checks

```bash
uv run python tests/check_conformers.py
uv run python tests/check_site_poses.py       # needs xtb
uv run python tests/check_crest_ethanol.py    # needs crest; host search only
```
