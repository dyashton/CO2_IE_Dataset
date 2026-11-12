# CO2 interaction-energy dataset

SMILES → RDKit ETKDG → **CREST host** → **site-seeded CO₂** (N/O outward normals) → **xTB opt with CO₂ atoms frozen** → geometry filter → rank by xTB IE → Psi4 **r²SCAN-3c** interaction energy.

CREST is used **only for the host**. Free NCI complex search is disabled (it was letting CO₂ covalently fuse). Psi4 `ie_kcal_mol` is the label.

This reports **interaction energy**, not binding affinity.

## Closed-loop discovery (RL + Bayesian AL)

RL generator lives in sibling `Molecule_Generation_Workflow` (`co2-binding-hpc` branch). This repo owns acquisition, DFT, ingest, and surrogate retrain:

```bash
# After RL writes proposals.csv on HPC:
bash scripts/run_discovery_round.sh round_000 --rl-proposals /path/to/proposals.csv
sbatch --export=ALL,SELECTED=$PWD/artifacts/discovery/round_000/selected.csv,\
RUN_DIR=$SCRATCH/co2_discovery/round_000 scripts/sbatch_discovery_dft.sh
uv run python scripts/ingest_discovery_round.py \
  --energies $SCRATCH/co2_discovery/round_000/energies.csv --round-id round_000
uv run python scripts/retrain_binding_after_round.py --sync-mgw
```

Full runbook: [`docs/DISCOVERY_LOOP.md`](docs/DISCOVERY_LOOP.md).  
HPC clone (lean branch, no 13 GB rsync): [`docs/HPC_DISCOVERY.md`](docs/HPC_DISCOVERY.md) — use branch **`discovery-hpc`**.

## Setup

```bash
cd CO2_IE_Dataset
git checkout discovery-hpc   # on HPC
uv sync                      # discovery core; add --extra ml for MACE/ChemBERTa
bash scripts/fetch_binding_pack.sh /path/to/binding_predictor.tar.gz
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

## Surrogate labels

The generator trains on the existing descriptor surrogate. Physics labels are an offline oracle for elites, not the inner-loop reward. xTB energies are not used.

```bash
uv run python scripts/label_physics_ie.py --csv candidates.csv --elites 20
uv run python scripts/label_physics_ie.py --smiles CN CCN --poses 100
```

`artifacts/physics_labels.csv` has `SMILES`, `IE`, and `source=physics_mace_omol` so `scripts/train_surrogate.py` can append it later. `reward` is `-IE`.

Run independent batches in parallel on a local machine (six workers × five
threads is suitable for a 32-core / 32-GB laptop):

```bash
uv run python scripts/run_parallel.py \
  --csv data/compiled_pre_full100.csv \
  --n 392 \
  --jobs 6 \
  --threads 5 \
  --quick \
  --to dft \
  --run-dir runs/compiled_pre_full100_parallel
```

Each batch has an isolated run directory. The runner resumes completed
batches, writes `catalog.csv`, `energies.csv`, and
`parallel_manifest.json` at the top level, and applies the same bounded
thread settings to CREST, xTB, and Psi4.

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

## Preference agent (LLM A vs B)

Library + CLI to compare two hosts for CO₂ binding preference (descriptors, sites, few-shots, A/B self-consistency). See [docs/preference_agent.md](docs/preference_agent.md) for the improvement / RL loop.

```bash
uv run python scripts/preference_agent_cli.py init-shots
uv run python scripts/preference_agent_cli.py compare 'CCO' 'c1ccncc1'
uv run python scripts/preference_agent_cli.py eval --pairs 50 --refresh-shots
```

## Chem LM preference / IE scorer

Frozen **ChemBERTa** embeddings + Ridge regression on `compiled.csv` EIs (molecule-grouped CV). Predicts interaction energy; pairwise accuracy from predicted IE order. Not a chat judge.

```bash
uv run python scripts/train_chem_lm.py train
uv run python scripts/train_chem_lm.py score --model artifacts/chem_lm_ridge.joblib 'CCO'
```

Outputs: `artifacts/chem_lm_ridge.joblib`, `artifacts/chem_lm_ridge.metrics.json`, embedding cache under `artifacts/chem_lm_cache/`.

## Generic dimer trajectory dataset

The trajectory workflow accepts the DES370K-compatible dimer CSV produced in
`Intermolecular_Pose_Proposal/data/processed/des370k_5000_dimers.csv` (or a
JSONL file) with `dimer_id`, `smiles_a`, and `smiles_b`. It embeds both
fragments with explicit hydrogens, records their graphs, creates deterministic
rigid placements, runs CREST for each placement, and optimizes each selected
complex with unconstrained Psi4 r²SCAN-3c. A missing `charge` field is
computed as `charge_a + charge_b`.

Source the local executable defaults and validate them:

```bash
source scripts/env.sh
uv run python scripts/check_env.py
```

Run a small validation without starting quantum calculations:

```bash
uv run python scripts/trajectory_dataset.py \
  --input ../Intermolecular_Pose_Proposal/data/processed/des370k_5000_dimers.csv \
  --start 4000 --stop 5000 --validate-only
```

The local shard for the final 1,000 dimers is resumable:

```bash
uv run python scripts/trajectory_dataset.py \
  --input ../Intermolecular_Pose_Proposal/data/processed/des370k_5000_dimers.csv \
  --start 4000 --stop 5000 \
  --run-dir runs/des370k_trajectory_local_1000 \
  --n-seeds 4 --n-crest 1 --threads 1
```

Each trajectory is stored under `trajectories/<trajectory_id>/` as
`trajectory.xyz`, `trajectory.jsonl`, `trajectory.npz`, `initial.xyz`,
`crest_seed.xyz`, `opt.in`, `opt.out`, and `opt.xyz`. Successful rows are
checkpointed in `dimers.csv`; failures are retained in `failed.csv`.
Existing completed trajectories are reused unless `--force` is supplied.
Independent HPC/local outputs can be combined with:

```bash
uv run python scripts/merge_trajectory_shards.py \
  runs/des370k_trajectory_hpc_4000 \
  runs/des370k_trajectory_local_1000 \
  --output-dir runs/des370k_trajectory_merged
```


Pairwise preference only: given hosts A and B, the model picks which binds CO₂ more strongly. Labels from true `EInt` (more negative wins). Prompt includes a physisorption rubric, RDKit descriptors, explicit polar-site inventory, and held-out few-shot pairs. Reports pair accuracy. Not used in the DFT pipeline or RL loop.

```bash
# OPENAI_API_KEY from CO2_IE_Dataset/.env (or export)
uv run python scripts/llm_judge_ei.py --mock --pairs 25           # no API
uv run python scripts/llm_judge_ei.py --pairs 50 --model gpt-4.1
uv run python scripts/llm_judge_ei.py --all --pairs 100 --model gpt-4.1
```

Outputs: `data/llm_judge_pairs.csv` (cached/resumable by model+prompt tag), `artifacts/llm_judge_metrics.json`.
