#!/usr/bin/env python3
"""Rebuild binding_predictor_train.csv from discovery labels and retrain the pack.

Refreshes RL reward artifacts after each DFT round (active-learning retrain step).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiled",
        type=Path,
        default=ROOT / "data" / "compiled_gm_be_discovery.csv",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=ROOT / "data" / "binding_predictor_train.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "binding_predictor",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--sync-mgw",
        action="store_true",
        help="Also copy pack into sibling Molecule_Generation_Workflow/artifacts",
    )
    args = parser.parse_args()

    if not args.compiled.is_file():
        raise SystemExit(f"missing compiled discovery csv: {args.compiled}")

    # Prefer the existing builder if present; otherwise copy BE_GM rows as IE targets.
    builder = ROOT / "scripts" / "build_binding_predictor_train.py"
    if builder.is_file():
        cmd = [
            "uv",
            "run",
            "python",
            str(builder),
            "--gm-be",
            str(args.compiled),
            "--out",
            str(args.train_csv),
        ]
        try:
            subprocess.run(cmd, check=True, cwd=ROOT)
        except subprocess.CalledProcessError:
            _write_minimal_train(args.compiled, args.train_csv)
    else:
        _write_minimal_train(args.compiled, args.train_csv)

    if args.skip_train:
        print(f"[retrain] wrote train csv only: {args.train_csv}")
        return

    train = ROOT / "scripts" / "train_binding_predictor.py"
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(train),
            "--data",
            str(args.train_csv),
            "--out-dir",
            str(args.out_dir),
        ],
        check=True,
        cwd=ROOT,
    )
    print(f"[retrain] pack refreshed under {args.out_dir}")

    if args.sync_mgw:
        mgw_fetch = (
            ROOT.parent
            / "Molecule_Generation_Workflow"
            / "scripts"
            / "fetch_binding_pack.sh"
        )
        if mgw_fetch.is_file():
            subprocess.run(["bash", str(mgw_fetch), str(args.out_dir)], check=True)
            print("[retrain] synced pack to Molecule_Generation_Workflow")


def _write_minimal_train(compiled: Path, out: Path) -> None:
    """Map discovery rows into a train table the binding trainer can read."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with compiled.open(newline="") as f_in, out.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = [
            "SMILES",
            "IE",
            "BE",
            "BE_GM",
            "E_host_penalty_kcal",
            "source",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"[retrain] wrote minimal train csv {out}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
