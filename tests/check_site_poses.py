#!/usr/bin/env python3
"""Ethanol: site seeds + constrained-CO2 xTB opt must yield ≥1 geometry-valid pose."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from co2ie.co2_pose import (
    co2_geometry_ok,
    generate_co2,
    pose_sites,
    site_complex,
)
from co2ie.conformers import embed_conformers, read_xyz, write_xyz
from co2ie.xtb import opt_xyz, xtb_bin

if xtb_bin() is None:
    print("skip: xtb not found")
    sys.exit(0)

xyzs = embed_conformers("CCO", n_conf=1)
assert xyzs, "ethanol ETKDG failed"
host = xyzs[0]
sites = pose_sites(host, max_sites=8)
assert sites, "expected ≥1 site"
with tempfile.TemporaryDirectory(prefix="site_etoh_") as td:
    root = Path(td)
    write_xyz(generate_co2(), root / "co2.xyz")
    ok_n = 0
    for s in sites:
        seed, meta = site_complex(host, s)
        seed_path = root / f"seed_{s.tag}.xyz"
        write_xyz(seed, seed_path)
        out = root / f"opt_{s.tag}.xyz"
        opt = opt_xyz(
            seed_path,
            out,
            constrain_co2=True,
            anchor_atom=s.binding_atom if s.binding_atom >= 0 else None,
            anchor_distance=float(meta["co2_distance"]),
        )
        if opt.ok and co2_geometry_ok(read_xyz(out)):
            ok_n += 1
    assert ok_n >= 1, f"expected ≥1 valid constrained-CO2 pose, got {ok_n}"
    print("ok", len(sites), "sites,", ok_n, "valid after constrained-CO2 xTB opt")
