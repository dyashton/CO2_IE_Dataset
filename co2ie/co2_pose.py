"""CO2 geometry + site-seeded placements (CREST host only; no free NCI complex search)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from co2ie import DUMMY_CO2_DISTANCE, DUMMY_POSE_TYPE
from co2ie.complexes import combine_host_co2
from co2ie.conformers import Xyz

CO2_CO = 1.16
# Valid intact CO2 after constrained opt
CO2_CO_MIN, CO2_CO_MAX = 1.10, 1.22
CO2_ANGLE_MIN = 165.0
MIN_HOST_CO2_DIST = 2.20
SEED_CLEARANCE = 2.55


@dataclass(frozen=True)
class PoseSite:
    tag: str
    carbon: np.ndarray
    axis: np.ndarray
    binding_atom: int
    distance: float


def _principal_axis(coords: np.ndarray) -> np.ndarray:
    c = coords - coords.mean(axis=0)
    if c.shape[0] < 2:
        return np.array([0.0, 0.0, 1.0])
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    axis = vt[0]
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return axis / n


def co2_linear(carbon: np.ndarray, axis: np.ndarray) -> Xyz:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = np.cross(a, tmp)
    perp = perp / (np.linalg.norm(perp) + 1e-12)
    r = CO2_CO
    coords = np.vstack([carbon, carbon + r * perp, carbon - r * perp])
    return Xyz(symbols=["C", "O", "O"], coords=coords, comment="isolated CO2")


def generate_co2() -> Xyz:
    """Linear O=C=O, C–O = 1.16 Å."""
    return co2_linear(np.zeros(3), np.array([0.0, 0.0, 1.0]))


def isolated_co2() -> Xyz:
    return generate_co2()


def dummy_complex(host: Xyz, distance: float = DUMMY_CO2_DISTANCE) -> tuple[Xyz, dict]:
    """Legacy centroid seed (kept for tests). Prefer site_complexes()."""
    centroid = host.coords.mean(axis=0)
    axis = _principal_axis(host.coords)
    carbon = centroid + distance * axis
    co2 = co2_linear(carbon, axis)
    xyz = combine_host_co2(host, co2, comment=f"{DUMMY_POSE_TYPE} d={distance}")
    meta = {
        "co2_pose_type": DUMMY_POSE_TYPE,
        "co2_distance": distance,
        "binding_atom": "",
    }
    return xyz, meta


def _neighbors(host: Xyz, i: int, r_cut: float = 1.35) -> list[int]:
    d = np.linalg.norm(host.coords - host.coords[i], axis=1)
    return [j for j, dj in enumerate(d) if j != i and dj < r_cut]


def _outward(host: Xyz, i: int) -> np.ndarray:
    nxyz = [host.coords[j] for j in _neighbors(host, i) if host.symbols[j] != "H"]
    if not nxyz:
        nxyz = [host.coords[j] for j in _neighbors(host, i)]
    if not nxyz:
        return np.array([0.0, 0.0, 1.0])
    mean = np.mean(nxyz, axis=0)
    v = host.coords[i] - mean
    nrm = np.linalg.norm(v)
    if nrm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return v / nrm


def _has_h(host: Xyz, i: int) -> bool:
    return any(host.symbols[j] == "H" for j in _neighbors(host, i))


def pose_sites(host: Xyz, max_sites: int = 12) -> list[PoseSite]:
    """PropMolFlow-style sites from host xyz: arom/aliphatic N, OH, ether/carbonyl O."""
    sites: list[PoseSite] = []
    for i, sym in enumerate(host.symbols):
        if sym == "N":
            axis = _outward(host, i)
            # ponytail: treat all N alike from xyz (no aromaticity); NH vs N by H neighbors
            dist = 3.0
            tag = f"NH{i}" if _has_h(host, i) else f"N{i}"
            sites.append(
                PoseSite(
                    tag=tag,
                    carbon=host.coords[i] + dist * axis,
                    axis=axis,
                    binding_atom=i,
                    distance=dist,
                )
            )
        elif sym == "O":
            axis = _outward(host, i)
            dist = 2.9
            tag = f"OH{i}" if _has_h(host, i) else f"Oc{i}"
            sites.append(
                PoseSite(
                    tag=tag,
                    carbon=host.coords[i] + dist * axis,
                    axis=axis,
                    binding_atom=i,
                    distance=dist,
                )
            )
    if not sites:
        axis = _principal_axis(host.coords)
        c = host.coords.mean(axis=0) + 3.2 * axis
        sites.append(
            PoseSite(tag="centroid", carbon=c, axis=axis, binding_atom=-1, distance=3.2)
        )
    uniq: list[PoseSite] = []
    seen: set[tuple[float, ...]] = set()
    for s in sites:
        key = tuple(np.round(s.carbon, 2))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
        if len(uniq) >= max_sites:
            break
    return uniq


def site_complex(host: Xyz, site: PoseSite) -> tuple[Xyz, dict]:
    """Place CO2 at site; push out along axis until host clearance is met."""
    carbon = np.array(site.carbon, dtype=float)
    axis = site.axis / (np.linalg.norm(site.axis) + 1e-12)
    dist = float(site.distance)
    for _ in range(20):
        co2 = co2_linear(carbon, axis)
        xyz = combine_host_co2(host, co2, comment=f"site {site.tag} d={dist:.2f}")
        stats = co2_geometry_stats(xyz)
        if min(stats["min_c_host"], stats["min_o_host"]) >= SEED_CLEARANCE:
            break
        carbon = carbon + 0.25 * axis
        dist += 0.25
    else:
        co2 = co2_linear(carbon, axis)
        xyz = combine_host_co2(host, co2, comment=f"site {site.tag} d={dist:.2f}")
    meta = {
        "co2_pose_type": f"site_{site.tag}",
        "co2_distance": dist,
        "binding_atom": "" if site.binding_atom < 0 else str(site.binding_atom),
    }
    return xyz, meta


def co2_indices(complex_xyz: Xyz) -> tuple[int, int, int]:
    """Last three atoms must be C,O,O."""
    n = complex_xyz.n_atoms
    if n < 3:
        raise ValueError("complex too small for CO2")
    return n - 3, n - 2, n - 1


def co2_geometry_stats(complex_xyz: Xyz) -> dict:
    i_c, i_o1, i_o2 = co2_indices(complex_xyz)
    c, o1, o2 = complex_xyz.coords[i_c], complex_xyz.coords[i_o1], complex_xyz.coords[i_o2]
    r1 = float(np.linalg.norm(o1 - c))
    r2 = float(np.linalg.norm(o2 - c))
    v1, v2 = o1 - c, o2 - c
    cosang = float(np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12))
    ang = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
    host = complex_xyz.coords[:i_c]
    d_c = float(np.linalg.norm(host - c, axis=1).min()) if len(host) else 99.0
    d_o = float(
        min(
            np.linalg.norm(host - o1, axis=1).min() if len(host) else 99.0,
            np.linalg.norm(host - o2, axis=1).min() if len(host) else 99.0,
        )
    )
    return {
        "r_co1": r1,
        "r_co2": r2,
        "angle": ang,
        "min_c_host": d_c,
        "min_o_host": d_o,
    }


def co2_geometry_ok(complex_xyz: Xyz) -> bool:
    """Intact linear CO2, not covalently fused to host."""
    i_c, i_o1, i_o2 = co2_indices(complex_xyz)
    if [complex_xyz.symbols[i] for i in (i_c, i_o1, i_o2)] != ["C", "O", "O"]:
        return False
    s = co2_geometry_stats(complex_xyz)
    if not (CO2_CO_MIN <= s["r_co1"] <= CO2_CO_MAX):
        return False
    if not (CO2_CO_MIN <= s["r_co2"] <= CO2_CO_MAX):
        return False
    if s["angle"] < CO2_ANGLE_MIN:
        return False
    if min(s["min_c_host"], s["min_o_host"]) < MIN_HOST_CO2_DIST:
        return False
    return True
