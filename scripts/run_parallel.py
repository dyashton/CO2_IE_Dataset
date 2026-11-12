#!/usr/bin/env python3
"""Run independent CO2 workflow batches in parallel on one machine."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "build_small.py"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_rows(path: Path, n: int) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "SMILES" not in {
            name.upper() for name in reader.fieldnames
        }:
            raise ValueError(f"{path} must contain a SMILES column")
        rows = list(reader)
    if n > 0:
        rows = rows[:n]
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return list(reader.fieldnames), rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _chunks(rows: list[dict[str, str]], count: int) -> list[list[dict[str, str]]]:
    return [rows[i::count] for i in range(count) if rows[i::count]]


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(path)


def _worker(
    *,
    batch: dict[str, Any],
    csv_path: Path,
    run_dir: Path,
    to_stage: str,
    from_stage: str,
    quick: bool,
    threads: int,
) -> dict[str, Any]:
    batch_dir = run_dir / batch["batch_id"]
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_csv = batch_dir / csv_path.name
    log_path = batch_dir / "pipeline.log"
    command = [
        sys.executable,
        str(PIPELINE),
        "--run-dir",
        str(batch_dir / "run"),
        "--csv",
        str(batch_csv),
        "--n",
        str(batch["row_count"]),
        "--from",
        from_stage,
        "--to",
        to_stage,
    ]
    if quick:
        command.append("--quick")
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = str(threads)
    # Psi4 honors PSI_NUM_THREADS; the alias keeps other compiled components bounded.
    env["PSI_NUM_THREADS"] = str(threads)
    env["PSI4_NUM_THREADS"] = str(threads)
    env["PYTHONUNBUFFERED"] = "1"
    started = _now()
    with log_path.open("w") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "batch_id": batch["batch_id"],
        "status": "completed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "started": started,
        "finished": _now(),
        "log": str(log_path),
        "command": command,
    }


def _merge_csvs(
    *,
    run_dir: Path,
    name: str,
    output_name: str,
    qualify_column: str,
) -> int:
    paths = sorted(run_dir.glob(f"batch_*/run/{name}"))
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in paths:
        batch_id = path.parents[1].name
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            if not fieldnames:
                fieldnames = ["batch_id", *reader.fieldnames]
            for row in reader:
                row = dict(row)
                row["batch_id"] = batch_id
                if row.get(qualify_column):
                    row[qualify_column] = f"{batch_id}__{row[qualify_column]}"
                rows.append(row)
    if fieldnames:
        _write_csv(run_dir / output_name, fieldnames, rows)
    return len(rows)


def _validate_energies(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {"rows": 0, "with_ie": 0, "bad_ie": 0, "bad_recomputed_ie": 0}
    rows = with_ie = bad_ie = bad_recomputed_ie = 0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows += 1
            ie = (row.get("ie_kcal_mol") or "").strip()
            if not ie:
                continue
            try:
                float(ie)
            except ValueError:
                bad_ie += 1
                continue
            with_ie += 1
            try:
                host = float(row["psi4_host_energy"])
                co2 = float(row["psi4_co2_energy"])
                complex_ = float(row["psi4_complex_energy"])
                expected = (complex_ - host - co2) * 627.509
                if abs(expected - float(ie)) > 1e-4:
                    bad_recomputed_ie += 1
            except (KeyError, ValueError):
                bad_recomputed_ie += 1
    return {
        "rows": rows,
        "with_ie": with_ie,
        "bad_ie": bad_ie,
        "bad_recomputed_ie": bad_recomputed_ie,
    }


def _duplicate_count(path: Path, columns: tuple[str, ...]) -> int:
    if not path.is_file():
        return 0
    values: list[str] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            value = "\x1f".join((row.get(column) or "").strip() for column in columns)
            if value:
                values.append(value)
    return len(values) - len(set(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--n", type=int, default=0, help="Rows to use; 0 means all.")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--from", dest="from_stage", default="sample")
    parser.add_argument("--to", dest="to_stage", default="dft")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Rerun completed batches instead of reusing their manifest status.",
    )
    args = parser.parse_args()
    if args.jobs < 1 or args.threads < 1:
        parser.error("--jobs and --threads must be positive")
    csv_path = args.csv.resolve()
    run_dir = args.run_dir.resolve()
    fieldnames, rows = _read_rows(csv_path, args.n)
    batches = _chunks(rows, args.jobs)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "parallel_manifest.json"
    manifest: dict[str, Any] = {
        "input_csv": str(csv_path),
        "input_rows": len(rows),
        "jobs": args.jobs,
        "threads": args.threads,
        "quick": args.quick,
        "from_stage": args.from_stage,
        "to_stage": args.to_stage,
        "started": _now(),
        "batches": {},
    }
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text())
            if (
                previous.get("input_csv") == str(csv_path)
                and previous.get("input_rows") == len(rows)
            ):
                manifest = previous
        except json.JSONDecodeError:
            pass
    reusable_config = (
        manifest.get("quick") == args.quick
        and manifest.get("from_stage") == args.from_stage
        and manifest.get("to_stage") == args.to_stage
    )
    manifest.update(
        {
            "input_csv": str(csv_path),
            "input_rows": len(rows),
            "jobs": args.jobs,
            "threads": args.threads,
            "quick": args.quick,
            "from_stage": args.from_stage,
            "to_stage": args.to_stage,
        }
    )

    batch_specs: list[dict[str, Any]] = []
    for index, batch_rows in enumerate(batches, start=1):
        batch_id = f"batch_{index:03d}"
        batch_dir = run_dir / batch_id
        batch_csv = batch_dir / csv_path.name
        _write_csv(batch_csv, fieldnames, batch_rows)
        spec = {"batch_id": batch_id, "row_count": len(batch_rows)}
        batch_specs.append(spec)
        manifest.setdefault("batches", {}).setdefault(batch_id, spec)
    _save_manifest(manifest_path, manifest)

    pending = []
    for spec in batch_specs:
        state = manifest["batches"].get(spec["batch_id"], {})
        if (
            not args.no_skip
            and reusable_config
            and state.get("status") == "completed"
            and state.get("returncode") == 0
        ):
            print(f"[skip] {spec['batch_id']} already completed")
        else:
            pending.append(spec)
    print(
        f"[parallel] {len(rows)} rows, {len(batch_specs)} batches, "
        f"{len(pending)} pending, jobs={args.jobs}, threads={args.threads}"
    )
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                _worker,
                batch=spec,
                csv_path=csv_path,
                run_dir=run_dir,
                to_stage=args.to_stage,
                from_stage=args.from_stage,
                quick=args.quick,
                threads=args.threads,
            ): spec
            for spec in pending
        }
        for future in as_completed(futures):
            result = future.result()
            manifest["batches"][result["batch_id"]] = result
            _save_manifest(manifest_path, manifest)
            print(
                f"[{result['status']}] {result['batch_id']} "
                f"returncode={result['returncode']}",
                flush=True,
            )

    catalog_rows = _merge_csvs(
        run_dir=run_dir,
        name="catalog.csv",
        output_name="catalog.csv",
        qualify_column="id",
    )
    energy_rows = _merge_csvs(
        run_dir=run_dir,
        name="energies.csv",
        output_name="energies.csv",
        qualify_column="mol_id",
    )
    energy_stats = _validate_energies(run_dir / "energies.csv")
    catalog_duplicates = _duplicate_count(run_dir / "catalog.csv", ("id",))
    energy_duplicates = _duplicate_count(
        run_dir / "energies.csv", ("mol_id", "pose_id")
    )
    failed = [
        batch_id
        for batch_id, state in manifest["batches"].items()
        if state.get("status") != "completed"
    ]
    manifest["finished"] = _now()
    manifest["failed_batches"] = failed
    manifest["merged"] = {
        "catalog_rows": catalog_rows,
        "energy_rows": energy_rows,
        "input_rows": len(rows),
        "catalog_duplicate_ids": catalog_duplicates,
        "energy_duplicate_pose_ids": energy_duplicates,
        **energy_stats,
    }
    _save_manifest(manifest_path, manifest)
    print(f"[merge] catalog rows={catalog_rows} → {run_dir / 'catalog.csv'}")
    print(f"[merge] energy rows={energy_rows} → {run_dir / 'energies.csv'}")
    verification = {
        **energy_stats,
        "catalog_duplicate_ids": catalog_duplicates,
        "energy_duplicate_pose_ids": energy_duplicates,
    }
    print(f"[verify] {json.dumps(verification, sort_keys=True)}")
    if (
        failed
        or energy_stats["bad_ie"]
        or energy_stats["bad_recomputed_ie"]
        or catalog_duplicates
        or energy_duplicates
    ):
        print(f"[failed] batches={failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
