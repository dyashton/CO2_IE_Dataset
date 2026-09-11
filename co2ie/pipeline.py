from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from co2ie import (
    ENERGY_CSV_COLUMNS,
    HARTREE_TO_KCAL,
    MAX_SITES_PER_HOST,
    N_HOST,
    N_POSES,
    PSI4_METHOD,
    ROOT,
    XTB_IE_MAX,
    XTB_IE_MIN,
)
from co2ie.catalog import Candidate, sample_from_csv, sample_small, write_catalog
from co2ie.co2_pose import (
    co2_geometry_ok,
    generate_co2,
    pose_sites,
    site_complex,
)
from co2ie.conformers import read_xyz, write_host_conformers, write_xyz
from co2ie.crest import CrestConf, crest_bin, run_crest, select_best_poses
from co2ie.psi4_jobs import psi4_bin, run_psi4, write_input
from co2ie.xtb import opt_xyz, sp_xyz, xtb_bin

STAGES = ("sample", "xyz", "crest_host", "crest_complex", "dft")


def _fmt(e: float | None) -> str:
    return "" if e is None else f"{e:.10f}"


def _read_energy(path: Path) -> float | None:
    ep = path.with_suffix(".energy")
    if not ep.is_file():
        return None
    try:
        return float(ep.read_text().strip())
    except ValueError:
        return None


def _write_energies(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ENERGY_CSV_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in ENERGY_CSV_COLUMNS})


def _cands(*, ethanol: bool = False, csv_path: Path | None = None, n: int = 10) -> list[Candidate]:
    if ethanol:
        return [Candidate(id="ethanol", smiles="CCO", source="mine", n_heavy=3)]
    if csv_path is not None:
        return sample_from_csv(csv_path, n=n, source=csv_path.stem)
    return sample_small()


def _stage_xyz(run_dir: Path, cands: list[Candidate]) -> None:
    print(f"[xyz] ETKDG for {len(cands)} molecules → {run_dir / 'mols'}")
    mols_dir = run_dir / "mols"
    for i, c in enumerate(cands, start=1):
        mol_dir = mols_dir / c.id
        mol_dir.mkdir(parents=True, exist_ok=True)
        (mol_dir / "smiles.txt").write_text(c.smiles + "\n")
        paths = write_host_conformers(mol_dir / "rdkit", c.smiles)
        print(f"  [{i}/{len(cands)}] {c.id}  {c.smiles}  n_conf={len(paths)}")
    print("[xyz] done")


def _crest_args(*, nci: bool, quick: bool, mquick: bool = False) -> list[str]:
    args = ["-nci", "-gfn2", "-noreftopo"] if nci else ["-gfn2"]
    if mquick:
        args.append("-mquick")
    elif quick:
        args.append("-quick")
    return args


def _stage_crest_host(run_dir: Path, cands: list[Candidate], *, quick: bool = False) -> None:
    if crest_bin() is None:
        raise FileNotFoundError("crest not found; run bash scripts/install_crest.sh")
    if xtb_bin() is None:
        raise FileNotFoundError("xtb not found; run bash scripts/install_xtb.sh")
    mode = "quick" if quick else "full"
    print(f"[crest_host] CREST -gfn2 ({mode}) for {len(cands)} molecules; keep top {N_HOST}")
    for i, c in enumerate(cands, start=1):
        mol_dir = run_dir / "mols" / c.id
        host_dir = mol_dir / "host"
        existing = sorted(host_dir.glob("crest_conf_*.xyz")) if host_dir.is_dir() else []
        if existing:
            print(f"  [{i}/{len(cands)}] {c.id}  skip (already have {len(existing)} hosts)")
            continue
        rdkit = sorted((mol_dir / "rdkit").glob("conf_*.xyz"))
        if not rdkit:
            print(f"  [{i}/{len(cands)}] {c.id}  skip (no rdkit xyz)")
            continue
        # Large hosts: -mquick; hard cap 90 min then salvage/RDKit (don't block the run)
        n_atoms = int(rdkit[0].read_text().splitlines()[0].split()[0])
        use_mquick = quick and n_atoms >= 40
        timeout = 5400  # 90 min
        tag = "mquick" if use_mquick else ("quick" if quick else "full")
        print(
            f"  [{i}/{len(cands)}] {c.id}  CREST host ({tag}, {n_atoms} atoms, timeout={timeout}s) …",
            flush=True,
        )
        work = mol_dir / "crest_host"
        confs = run_crest(
            rdkit[0],
            work,
            _crest_args(nci=False, quick=quick, mquick=use_mquick),
            timeout=timeout,
        )
        if not confs:
            print(f"  [{i}/{len(cands)}] {c.id}  CREST empty/timeout → RDKit fallback")
            fallback = work / "conf_001.xyz"
            work.mkdir(parents=True, exist_ok=True)
            shutil.copy(rdkit[0], fallback)
            confs = [CrestConf(path=fallback, energy=None, index=1)]
        top = select_best_poses(confs, N_HOST)
        if host_dir.exists():
            shutil.rmtree(host_dir)
        host_dir.mkdir(parents=True)
        for j, conf in enumerate(top, start=1):
            dest = host_dir / f"crest_conf_{j:03d}.xyz"
            opt = opt_xyz(conf.path, dest)
            if not opt.ok:
                shutil.copy(conf.path, dest)
        print(f"  [{i}/{len(cands)}] {c.id}  {len(confs)} crest → {len(top)} hosts (xTB-opt)")
    print("[crest_host] done")


def _stage_crest_complex(run_dir: Path, cands: list[Candidate], *, quick: bool = False) -> None:
    """Site-seeded CO2 poses; freeze CO2 atoms in xTB opt. CREST NCI is not used."""
    del quick  # host CREST quick only; pose stage ignores it
    if xtb_bin() is None:
        raise FileNotFoundError("xtb not found; run bash scripts/install_xtb.sh")
    print(
        f"[poses] site seeds + xTB opt (CO2 frozen); "
        f"≤{MAX_SITES_PER_HOST} sites/host → top {N_POSES} by xtb IE"
    )
    co2 = generate_co2()
    co2_path = run_dir / "co2.xyz"
    write_xyz(co2, co2_path)
    co2_opt = run_dir / "co2.xtb.xyz"
    opt_co2 = opt_xyz(co2_path, co2_opt)
    e_co2 = opt_co2.energy if opt_co2.ok else sp_xyz(co2_path)
    if e_co2 is None:
        raise RuntimeError("xTB CO2 energy failed")
    (run_dir / "co2.xtb.energy").write_text(f"{e_co2:.12f}\n")
    print(f"  CO2 xtb E = {e_co2:.8f} Eh")

    for i, c in enumerate(cands, start=1):
        mol_dir = run_dir / "mols" / c.id
        hosts = sorted((mol_dir / "host").glob("crest_conf_*.xyz"))
        pose_dir = mol_dir / "complexes"
        if pose_dir.exists():
            for old in list(pose_dir.glob("pose_*")) + list(pose_dir.glob("seed_*")) + list(
                pose_dir.glob("opt_*")
            ):
                old.unlink()
        pose_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{i}/{len(cands)}] {c.id}  {len(hosts)} hosts → site seeds …", flush=True)
        candidates: list[dict] = []
        for hp in hosts:
            host_id = hp.stem
            host = read_xyz(hp)
            e_host = _read_energy(hp)
            if e_host is None:
                e_host = sp_xyz(hp)
            if e_host is None:
                print(f"    {host_id}: skip (no host xtb energy)")
                continue
            sites = pose_sites(host, max_sites=MAX_SITES_PER_HOST)
            print(f"    {host_id}: {len(sites)} sites", flush=True)
            for s in sites:
                seed, meta = site_complex(host, s)
                seed_path = pose_dir / f"seed_{host_id}_{s.tag}.xyz"
                write_xyz(seed, seed_path)
                out_tmp = pose_dir / f"opt_{host_id}_{s.tag}.xyz"
                anchor = s.binding_atom if s.binding_atom >= 0 else None
                opt = opt_xyz(
                    seed_path,
                    out_tmp,
                    constrain_co2=True,
                    anchor_atom=anchor,
                    anchor_distance=float(meta["co2_distance"]),
                )
                if not opt.ok or opt.energy is None:
                    continue
                geom = read_xyz(out_tmp)
                if not co2_geometry_ok(geom):
                    continue
                ie = (opt.energy - e_host - e_co2) * HARTREE_TO_KCAL
                if ie > XTB_IE_MAX or ie < XTB_IE_MIN:
                    continue
                candidates.append(
                    {
                        "ie": ie,
                        "e_complex": opt.energy,
                        "e_host": e_host,
                        "path": out_tmp,
                        "host_id": host_id,
                        "tag": s.tag,
                        "meta": meta,
                    }
                )
        candidates.sort(key=lambda d: d["ie"])
        top = candidates[:N_POSES]
        lines = []
        for j, d in enumerate(top, start=1):
            dest = pose_dir / f"pose_{j:03d}.xyz"
            shutil.copy(d["path"], dest)
            dest.with_suffix(".energy").write_text(f"{d['e_complex']:.12f}\n")
            dest.with_suffix(".host").write_text(d["host_id"] + "\n")
            dest.with_suffix(".site").write_text(
                f"{d['tag']}\t{d['meta']['binding_atom']}\t{d['ie']:.4f}\n"
            )
            lines.append(
                f"pose_{j:03d} {d['host_id']} {d['tag']} xtb_ie={d['ie']:.3f} {d['path']}\n"
            )
        (pose_dir / "selected.txt").write_text("".join(lines))
        if top:
            print(
                f"  [{i}/{len(cands)}] {c.id}  "
                f"{len(candidates)} valid → selected {len(top)} "
                f"(best xtb IE {top[0]['ie']:.2f} kcal/mol)"
            )
        else:
            print(f"  [{i}/{len(cands)}] {c.id}  {len(candidates)} valid → selected 0")
    print("[poses] done")

def _stage_dft(run_dir: Path, cands: list[Candidate]) -> None:
    co2_path = run_dir / "co2.xyz"
    if not co2_path.is_file():
        write_xyz(generate_co2(), co2_path)
    co2 = read_xyz(co2_path)
    have_psi4 = psi4_bin() is not None
    if have_psi4:
        print(f"[dft] Psi4 {PSI4_METHOD} single points")
    else:
        print("[dft] psi4 not on PATH — writing inputs only (energies blank)")
    psi4_root = run_dir / "psi4"
    co2_inp = psi4_root / "co2.in"
    write_input(co2, co2_inp)
    if have_psi4:
        print("  CO2 monomer …", flush=True)
    co2_e = run_psi4(co2_inp) if have_psi4 else None
    host_cache: dict[tuple[str, str], float | None] = {}
    rows: list[dict] = []
    out = run_dir / "energies.csv"

    def _flush() -> None:
        _write_energies(out, rows)
        print(f"  [dft] checkpoint {len(rows)} rows → {out}", flush=True)

    for i, c in enumerate(cands, start=1):
        mol_dir = run_dir / "mols" / c.id
        psi4_dir = mol_dir / "psi4"
        poses = sorted((mol_dir / "complexes").glob("pose_[0-9][0-9][0-9].xyz"))
        print(f"  [{i}/{len(cands)}] {c.id}  {len(poses)} poses", flush=True)
        for pose_path in poses:
            pose_id = pose_path.stem
            hid_file = pose_path.with_suffix(".host")
            host_conf_id = hid_file.read_text().strip() if hid_file.is_file() else ""
            host_path = mol_dir / "host" / f"{host_conf_id}.xyz"
            row = {
                "mol_id": c.id,
                "pose_id": pose_id,
                "host_conf_id": host_conf_id,
                "smiles": c.smiles,
                "host_energy_xtb": _fmt(_read_energy(host_path) if host_path.is_file() else None),
                "complex_energy_xtb": _fmt(_read_energy(pose_path)),
                "psi4_method": PSI4_METHOD,
                "psi4_host_energy": "",
                "psi4_co2_energy": _fmt(co2_e),
                "psi4_complex_energy": "",
                "ie_kcal_mol": "",
            }
            if host_path.is_file():
                host_xyz = read_xyz(host_path)
                host_inp = psi4_dir / f"host_{host_conf_id}.in"
                write_input(host_xyz, host_inp)
                key = (c.id, host_conf_id)
                if key not in host_cache:
                    if have_psi4:
                        print(f"    host {host_conf_id} …", flush=True)
                    host_cache[key] = run_psi4(host_inp) if have_psi4 else None
                host_e = host_cache[key]
                row["psi4_host_energy"] = _fmt(host_e)
            else:
                host_e = None
            cx = read_xyz(pose_path)
            cx_inp = psi4_dir / f"{pose_id}.in"
            write_input(cx, cx_inp)
            if have_psi4:
                print(f"    {pose_id} …", flush=True)
            cx_e = run_psi4(cx_inp) if have_psi4 else None
            row["psi4_complex_energy"] = _fmt(cx_e)
            if host_e is not None and co2_e is not None and cx_e is not None:
                row["ie_kcal_mol"] = f"{(cx_e - host_e - co2_e) * HARTREE_TO_KCAL:.6f}"
            rows.append(row)
            _flush()
    print(f"[dft] finished {len(rows)} rows → {out}")


def run(
    run_dir: Path,
    stage: str,
    *,
    ethanol: bool = False,
    csv_path: Path | None = None,
    n: int = 10,
    quick: bool = False,
    start: str = "sample",
) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage}; want {STAGES}")
    if start not in STAGES:
        raise ValueError(f"unknown start {start}; want {STAGES}")
    stop = STAGES.index(stage)
    begin = STAGES.index(start)
    if begin > stop:
        raise ValueError(f"--from {start} is after --to {stage}")
    cands = _cands(ethanol=ethanol, csv_path=csv_path, n=n)
    if ethanol:
        label = "ethanol"
        use_quick = True
    elif csv_path is not None:
        label = f"{csv_path.stem}_n{len(cands)}"
        use_quick = quick
    else:
        label = "small10"
        use_quick = quick
    print(f"[run] {label}  stages {start}→{stage}  run_dir={run_dir}")
    if begin <= 0:
        write_catalog(cands, run_dir / "catalog.csv")
        print(f"[sample] {len(cands)} molecules → {run_dir / 'catalog.csv'}")
        for c in cands:
            print(f"  {c.id}  n_heavy={c.n_heavy}  {c.smiles}")
    else:
        # reuse existing catalog if present
        cat = run_dir / "catalog.csv"
        if cat.is_file():
            print(f"[sample] reuse {cat}")
        else:
            write_catalog(cands, cat)
            print(f"[sample] wrote {cat}")
    if stop == 0:
        return
    if begin <= 1:
        _stage_xyz(run_dir, cands)
    if stop == 1:
        return
    if begin <= 2:
        _stage_crest_host(run_dir, cands, quick=use_quick)
    if stop == 2:
        return
    if begin <= 3:
        _stage_crest_complex(run_dir, cands, quick=use_quick)
    if stop == 3:
        return
    if begin <= 4:
        _stage_dft(run_dir, cands)
    print("[run] finished")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="CO2 interaction-energy dataset (CREST poses, Psi4 labels).")
    p.add_argument("--to", choices=STAGES, default="dft")
    p.add_argument("--from", dest="start", choices=STAGES, default="sample", help="First stage to run (skip earlier).")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--ethanol", action="store_true", help="Ethanol + CO2 smoke run instead of the 10-mol slice.")
    p.add_argument("--csv", type=Path, default=None, help="SMILES CSV (e.g. data/test.csv); uses first --n rows.")
    p.add_argument("--n", type=int, default=10, help="How many molecules from --csv (default 10).")
    p.add_argument("--quick", action="store_true", help="Pass CREST -quick for host search only.")
    args = p.parse_args(argv)
    csv_path = args.csv.resolve() if args.csv else None
    if args.ethanol:
        default_dir = ROOT / "runs" / "ethanol"
    elif csv_path is not None:
        default_dir = ROOT / "runs" / f"{csv_path.stem}_n{args.n}"
    else:
        default_dir = ROOT / "runs" / "small10"
    run_dir = args.run_dir or default_dir
    run(
        run_dir,
        args.to,
        ethanol=args.ethanol,
        csv_path=csv_path,
        n=args.n,
        quick=args.quick,
        start=args.start,
    )


if __name__ == "__main__":
    main()
