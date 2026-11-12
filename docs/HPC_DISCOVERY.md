# HPC: CO₂ discovery (AL + DFT + retrain)

Sibling of MGW `co2-binding-hpc`. Clone **this** branch next to RL; do **not** rsync the 13 GB laptop tree.

## Clone (preferred)

```bash
cd ${SCRATCH:-$HOME}/co2_discovery
git clone git@github.com:dyashton/CO2_IE_Dataset.git
cd CO2_IE_Dataset
git checkout discovery-hpc
uv sync
# optional: uv sync --extra ml   # ChemBERTa / MACE stacks only
```

Layout on cluster:

```text
$SCRATCH/co2_discovery/
  Molecule_Generation_Workflow/   # co2-binding-hpc
  CO2_IE_Dataset/                 # discovery-hpc (this repo)
  binding_predictor.tar.gz        # shared pack (~70MB) — not in git
```

## Binding pack (not in git)

Joblibs are gitignored. On a machine that already has them:

```bash
bash scripts/pack_binding_artifacts.sh
# → artifacts/binding_predictor.tar.gz
scp artifacts/binding_predictor.tar.gz cluster:$SCRATCH/co2_discovery/
```

On HPC:

```bash
bash scripts/fetch_binding_pack.sh $SCRATCH/co2_discovery/binding_predictor.tar.gz
# or, if MGW already unpacked the same tarball:
bash scripts/fetch_binding_pack.sh
```

## Chemistry stack (DFT)

Psi4 / xTB / CREST are **external** CLIs (not uv deps):

```bash
bash scripts/install_xtb.sh    # if not already on PATH
bash scripts/install_crest.sh
# install Psi4 via conda-forge (psi4 + dftd4-python + gcp-correction)
uv run python scripts/check_env.py
```

## Round loop (all on HPC)

```bash
# 1) RL generate (in sibling MGW) → proposals.csv
# 2) Propose + UCB select
bash scripts/run_discovery_round.sh round_000 \
  --rl-proposals ../Molecule_Generation_Workflow/artifacts/discovery/proposals.csv

# 3) DFT batch
sbatch --export=ALL,SELECTED=$PWD/artifacts/discovery/round_000/selected.csv,\
RUN_DIR=$SCRATCH/co2_discovery/round_000 \
  scripts/sbatch_discovery_dft.sh

# 4) Ingest + retrain + sync MGW pack
uv run python scripts/ingest_discovery_round.py \
  --energies $SCRATCH/co2_discovery/round_000/energies.csv \
  --round-id round_000 \
  --summary artifacts/discovery/round_000/summary.md
uv run python scripts/retrain_binding_after_round.py --sync-mgw
```

## Smoke (before a 48-mol array)

```bash
bash scripts/fetch_binding_pack.sh
uv run python scripts/propose_broad_hosts.py \
  --out artifacts/discovery/smoke/proposals.csv --elite-mutants 20
uv run python scripts/gate_and_select_dft_batch.py \
  --proposals artifacts/discovery/smoke/proposals.csv \
  --out-gated artifacts/discovery/smoke/gated.csv \
  --out-selected artifacts/discovery/smoke/selected.csv \
  --n 3 --pack artifacts/binding_predictor/pack.joblib
uv run python scripts/build_small.py \
  --csv artifacts/discovery/smoke/selected.csv \
  --n 3 --quick --to dft \
  --run-dir $SCRATCH/co2_discovery/smoke3
```

## Offline bundle (no git)

If the cluster cannot reach GitHub:

```bash
# on laptop (repo root):
bash scripts/pack_hpc_bundle.sh
bash scripts/pack_binding_artifacts.sh
scp artifacts/co2_ie_discovery_hpc.tar.gz artifacts/binding_predictor.tar.gz cluster:...
```

Manifest of included paths: [`scripts/hpc_manifest.txt`](../scripts/hpc_manifest.txt).

## Reward honesty

Acquisition and RL use pack head **`ie`**. DFT records **`BE_GM`**. See [`DISCOVERY_LOOP.md`](DISCOVERY_LOOP.md).
