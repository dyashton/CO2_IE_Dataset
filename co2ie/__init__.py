from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARTREE_TO_KCAL = 627.509
PSI4_METHOD = "r2scan-3c"
N_CONF = 4
N_HOST = 3
N_POSES = 5
MAX_SITES_PER_HOST = 8
XTB_IE_MAX = 20.0
XTB_IE_MIN = -40.0
DUMMY_CO2_DISTANCE = 3.0
DUMMY_POSE_TYPE = "dummy_centroid"

ENERGY_CSV_COLUMNS = [
    "mol_id",
    "pose_id",
    "host_conf_id",
    "smiles",
    "host_energy_xtb",
    "complex_energy_xtb",
    "psi4_method",
    "psi4_host_energy",
    "psi4_co2_energy",
    "psi4_complex_energy",
    "ie_kcal_mol",
]
