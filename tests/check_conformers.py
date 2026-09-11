#!/usr/bin/env python3
"""Ethanol SMILES → ≥1 xyz whose atom count matches RDKit AddHs."""
from __future__ import annotations

from rdkit import Chem

from co2ie.conformers import embed_conformers

smi = "CCO"
mol = Chem.AddHs(Chem.MolFromSmiles(smi))
assert mol is not None
n = mol.GetNumAtoms()
xyzs = embed_conformers(smi, n_conf=4)
assert len(xyzs) >= 1, f"expected ≥1 conformer, got {len(xyzs)}"
assert xyzs[0].n_atoms == n, f"atom count {xyzs[0].n_atoms} != {n}"
print("ok", len(xyzs), "conformers,", n, "atoms")
