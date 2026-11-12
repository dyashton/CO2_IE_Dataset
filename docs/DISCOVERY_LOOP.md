# Closed-loop CO₂ discovery (RL + Bayesian AL + DFT)

Reviewer framing: **closed-loop molecular discovery with reinforcement learning and Bayesian active learning.**

| Role | Tooling |
|------|---------|
| RL generator | Sibling [`Molecule_Generation_Workflow`](../../Molecule_Generation_Workflow/docs/HPC_RL.md) → `proposals.csv` |
| Bayesian acquisition | [`scripts/gate_and_select_dft_batch.py`](../scripts/gate_and_select_dft_batch.py) (UCB on −IE + uncertainty, scaffold caps) |
| Expensive oracle | CREST → constrained-CO₂ xTB → Psi4 via [`run_parallel.py`](../scripts/run_parallel.py) |
| Surrogate retrain | [`scripts/retrain_binding_after_round.py`](../scripts/retrain_binding_after_round.py) |

Config defaults: [`configs/discovery.yaml`](../configs/discovery.yaml).

## Orchestrator round

```bash
cd CO2_IE_Dataset

# 1) Optional: pull RL proposals from MGW on HPC
#    scp cluster:…/artifacts/discovery/proposals.csv /tmp/rl_proposals.csv

bash scripts/run_discovery_round.sh round_000 \
  --rl-proposals /tmp/rl_proposals.csv

# 2) DFT on HPC
sbatch --export=ALL,SELECTED=$PWD/artifacts/discovery/round_000/selected.csv,\
RUN_DIR=$SCRATCH/co2_discovery/round_000 \
  scripts/sbatch_discovery_dft.sh

# 3) Ingest
uv run python scripts/ingest_discovery_round.py \
  --energies $SCRATCH/co2_discovery/round_000/energies.csv \
  --round-id round_000 \
  --summary artifacts/discovery/round_000/summary.md

# 4) Retrain surrogate + sync MGW pack
uv run python scripts/retrain_binding_after_round.py --sync-mgw
```

## Smoke test (new machine / before a 48-mol array)

```bash
# 3 molecules through the existing pipeline
uv run python scripts/build_small.py \
  --csv artifacts/discovery/round_000/selected.csv \
  --n 3 --quick --to dft \
  --run-dir runs/discovery_smoke3
```

## Portable layout (all-on-HPC)

Preferred cluster layout — see [`HPC_DISCOVERY.md`](HPC_DISCOVERY.md):

- Clone `Molecule_Generation_Workflow` @ `co2-binding-hpc` and this repo @ `discovery-hpc` under `$SCRATCH/co2_discovery/`
- Shared: binding pack tarball (`scripts/fetch_binding_pack.sh`), round dirs under `$SCRATCH/co2_discovery/`
- Do **not** rsync laptop `artifacts/` / `runs/` / `.venv` (~13 GB)

## Reward honesty

RL and acquisition currently optimize **IE** (pack `reward_head=ie`). DFT still records **BE_GM**. Do not claim BE_GM-optimized RL until compose Spearman clears the 0.6 gate in `artifacts/binding_predictor/metrics.json`.
