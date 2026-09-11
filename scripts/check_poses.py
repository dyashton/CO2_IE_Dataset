#!/usr/bin/env python3
"""Diagnose CO2 pose geometries under a run directory."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from co2ie.co2_pose import co2_geometry_ok, co2_geometry_stats
from co2ie.conformers import read_xyz


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path, help="e.g. runs/test_n5")
    args = p.parse_args()
    eng = args.run_dir / "energies.csv"
    rows = list(csv.DictReader(eng.open())) if eng.is_file() else []
    ie_of = {(r["mol_id"], r["pose_id"]): r.get("ie_kcal_mol", "") for r in rows}
    mols = args.run_dir / "mols"
    print(f"{'mol':10s} {'pose':10s} {'ok':>3} {'r1':>6} {'r2':>6} {'ang':>7} {'dC':>6} {'dO':>6} {'ie':>10}")
    for mol_dir in sorted(mols.glob("*")):
        if not mol_dir.is_dir():
            continue
        for pose in sorted((mol_dir / "complexes").glob("pose_[0-9][0-9][0-9].xyz")):
            xyz = read_xyz(pose)
            s = co2_geometry_stats(xyz)
            ok = co2_geometry_ok(xyz)
            ie = ie_of.get((mol_dir.name, pose.stem), "")
            print(
                f"{mol_dir.name:10s} {pose.stem:10s} {str(ok):>3} "
                f"{s['r_co1']:6.3f} {s['r_co2']:6.3f} {s['angle']:7.1f} "
                f"{s['min_c_host']:6.3f} {s['min_o_host']:6.3f} {ie:>10}"
            )


if __name__ == "__main__":
    main()
