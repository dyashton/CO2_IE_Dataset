#!/usr/bin/env python3
"""Ethanol + CO2: CREST host then --nci must yield ≥2 complex frames. Skip if crest missing."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from co2ie.co2_pose import dummy_complex, generate_co2
from co2ie.conformers import embed_conformers, read_xyz, write_xyz
from co2ie.crest import crest_bin, run_crest

if crest_bin() is None:
    print("skip: crest not found (bash scripts/install_crest.sh)")
    sys.exit(0)

xyzs = embed_conformers("CCO", n_conf=4)
assert xyzs, "ethanol ETKDG failed"
with tempfile.TemporaryDirectory(prefix="crest_etoh_") as td:
    root = Path(td)
    host_xyz = root / "host.xyz"
    write_xyz(xyzs[0], host_xyz)
    host_confs = run_crest(host_xyz, root / "crest_host", ["-gfn2", "-quick"], timeout=1800)
    assert host_confs, "CREST host produced no structures"
    host = read_xyz(host_confs[0].path)
    seed, _ = dummy_complex(host)
    seed_path = root / "seed.xyz"
    write_xyz(seed, seed_path)
    write_xyz(generate_co2(), root / "co2.xyz")
    nci = run_crest(seed_path, root / "nci", ["-nci", "-gfn2", "-noreftopo", "-quick"], timeout=1800)
    assert len(nci) >= 2, f"expected ≥2 NCI poses, got {len(nci)}"
    print("ok", len(host_confs), "host confs,", len(nci), "NCI poses")
