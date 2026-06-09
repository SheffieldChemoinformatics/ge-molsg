#!/usr/bin/env python
"""End-to-end workflow: SMILES file -> surfaces -> WKS descriptors / BoF vectors.

Reads a ``.smi`` or ``.csv`` file of SMILES (optionally with an ID column and/or
an activity-label column, or split across separate active/decoy ``.smi`` files),
generates a molecular surface for each, computes per-vertex WKS descriptors, and
optionally encodes them into per-molecule Bag-of-Features vectors against a
codebook.

Codebook handling
-----------------
* ``--codebook PATH`` existing  -> loaded and used.
* ``--codebook PATH`` missing   -> prompts to confirm generating one and saving
  to that path (or exit).
* no ``--codebook``             -> a codebook is built from a random sample of
  ``--codebook-sample`` molecules and saved to ``<output>/codebook.npy``.

Outputs (all but BoF are temporary by default)
----------------------------------------------
* BoF vectors      -> always written (the primary deliverable), unless --no-bof.
* WKS descriptors  -> written only with --keep-wks (or --wks-only).
* surface .npy     -> written only with --keep-surfaces.
* min-energy PDBs  -> written only with --save-pdb DIR.

Front-end (SMILES -> surface) failures are logged and skipped; a summary is
printed at the end.

Examples
--------
    python scripts/smiles_to_descriptors.py ligands.smi --out run/
    python scripts/smiles_to_descriptors.py library.csv --out run/ \
        --has-header --smiles-col smiles --id-col name --label-col activity \
        --num-confs 20 --n-jobs -1 --keep-wks
    python scripts/smiles_to_descriptors.py actives.smi decoys.smi --out run/ \
        --labels-from-files --codebook existing_cb.npy
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# ge_molsg is imported lazily in main() so --help works without it installed.


# --------------------------------------------------------------------------- #
# Input record                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    smiles: str
    mol_id: Optional[str] = None
    label: Optional[str] = None


# --------------------------------------------------------------------------- #
# Input parsing                                                               #
# --------------------------------------------------------------------------- #
def _looks_like_smiles_header(first_field: str) -> bool:
    """Heuristic: a header cell is a word like 'smiles'/'smi', not a SMILES."""
    return first_field.strip().lower() in {"smiles", "smi", "structure", "canonical_smiles"}


def read_smi(path: Path, label: Optional[str] = None) -> List[Record]:
    """Read a whitespace-delimited .smi file: ``SMILES [ID]`` per line."""
    records: List[Record] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if lineno == 0 and _looks_like_smiles_header(parts[0]):
                continue  # skip a header row if present
            smiles = parts[0]
            mol_id = parts[1] if len(parts) > 1 else None
            records.append(Record(smiles=smiles, mol_id=mol_id, label=label))
    return records


def read_csv(
    path: Path,
    has_header: bool,
    smiles_col: str,
    id_col: Optional[str],
    label_col: Optional[str],
) -> List[Record]:
    """Read SMILES (and optional ID/label) from a CSV file.

    Column selectors may be names (when ``--has-header``) or 0-based indices.
    """
    records: List[Record] = []
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return records

    if has_header:
        header = [h.strip() for h in rows[0]]
        body = rows[1:]

        def col_idx(sel: str) -> int:
            if sel in header:
                return header.index(sel)
            return int(sel)  # allow numeric index even with a header
    else:
        body = rows

        def col_idx(sel: str) -> int:
            return int(sel)

    si = col_idx(smiles_col)
    ii = col_idx(id_col) if id_col is not None else None
    li = col_idx(label_col) if label_col is not None else None

    for row in body:
        if not row or si >= len(row):
            continue
        smiles = row[si].strip()
        if not smiles:
            continue
        mol_id = row[ii].strip() if ii is not None and ii < len(row) else None
        label = row[li].strip() if li is not None and li < len(row) else None
        records.append(Record(smiles=smiles, mol_id=mol_id or None, label=label or None))
    return records


def load_records(args) -> List[Record]:
    """Dispatch input parsing across the supported modes."""
    records: List[Record] = []
    if args.labels_from_files:
        # Each input file's stem (minus trailing 's') labels its molecules,
        # e.g. ligands.smi -> 'ligand', decoys.smi -> 'decoy'.
        for path in args.input:
            p = Path(path)
            label = p.stem[:-1] if p.stem.endswith("s") else p.stem
            records.extend(read_smi(p, label=label))
    else:
        for path in args.input:
            p = Path(path)
            if p.suffix.lower() == ".csv":
                records.extend(
                    read_csv(p, args.has_header, args.smiles_col,
                             args.id_col, args.label_col)
                )
            else:
                records.extend(read_smi(p))
    return records


# --------------------------------------------------------------------------- #
# Naming                                                                      #
# --------------------------------------------------------------------------- #
def assign_names(records: List[Record]) -> None:
    """Fill missing IDs with InChIKeys; de-duplicate all names in place."""
    from rdkit import Chem

    seen: dict[str, int] = {}
    for rec in records:
        name = rec.mol_id
        if not name:
            mol = Chem.MolFromSmiles(rec.smiles)
            name = Chem.MolToInchiKey(mol) if mol is not None else "INVALID"
        # de-duplicate
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        rec.mol_id = name


# --------------------------------------------------------------------------- #
# Front-end worker: SMILES -> surface .npy (one molecule)                     #
# --------------------------------------------------------------------------- #
@dataclass
class FrontEndConfig:
    surf_dir: str
    seed: int = 42
    num_confs: int = 1
    max_attempts: int = 50
    maxiters: int = 500
    save_pdb_dir: Optional[str] = None


def _front_end_worker(task):
    """Generate one surface .npy from a record. Returns (id, path_or_None, err)."""
    rec, cfg = task
    try:
        # Import inside the worker so the process pool pickles cleanly.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import smiles_to_surface_npy as s2s

        mol = s2s.embed_mol(
            rec.smiles, seed=cfg.seed, num_confs=cfg.num_confs,
            max_attempts=cfg.max_attempts, maxiters=cfg.maxiters,
        )
        if cfg.save_pdb_dir:
            from rdkit import Chem

            os.makedirs(cfg.save_pdb_dir, exist_ok=True)
            Chem.MolToPDBFile(mol, os.path.join(cfg.save_pdb_dir, f"{rec.mol_id}.pdb"))

        verts, faces = s2s.gen_mesh(mol)
        conf = mol.GetConformer()
        atom_pos = np.asarray(conf.GetPositions(), dtype=np.float64)
        q = s2s.mmff94_partial_charges(mol)
        esp = s2s.coulombic_esp_on_mesh(atom_pos, verts[:, :3], q)

        out_path = os.path.join(cfg.surf_dir, f"{rec.mol_id}.npy")
        s2s.save_surface_npy(out_path, verts[:, :3], faces, esp)
        return (rec.mol_id, out_path, None)
    except Exception as exc:  # noqa: BLE001 - we want to log any failure
        return (rec.mol_id, None, f"{type(exc).__name__}: {exc}")


def generate_surfaces(records, cfg: FrontEndConfig, n_jobs: int, log) -> List[tuple]:
    """Run the front-end over all records, serial or pooled. Returns successes.

    Each success is ``(record, surface_path)``; failures are logged and skipped.
    """
    tasks = [(rec, cfg) for rec in records]
    results: List[tuple] = []
    n_fail = 0

    def handle(mol_id, path, err, rec):
        nonlocal n_fail
        if err is None:
            results.append((rec, path))
        else:
            n_fail += 1
            log.warning("FAILED %s: %s", mol_id, err)

    if n_jobs == 1:
        for rec, task in zip(records, tasks):
            mol_id, path, err = _front_end_worker(task)
            handle(mol_id, path, err, rec)
    else:
        workers = os.cpu_count() if n_jobs < 0 else n_jobs
        rec_by_task = {id(t): r for t, r in zip(tasks, records)}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_front_end_worker, t): t for t in tasks}
            for fut in as_completed(futs):
                mol_id, path, err = fut.result()
                handle(mol_id, path, err, rec_by_task[id(futs[fut])])

    log.info("Surface generation: %d ok, %d failed", len(results), n_fail)
    print(f"Surfaces: {len(results)} generated, {n_fail} failed")
    return results


# --------------------------------------------------------------------------- #
# Codebook                                                                     #
# --------------------------------------------------------------------------- #
def resolve_codebook(args, descriptors, log):
    """Load an existing codebook or build one from a random descriptor sample.

    Returns the codebook array, or None if BoF is disabled / aborted.
    """
    import ge_molsg as gm

    cb_path = Path(args.codebook) if args.codebook else None

    if cb_path and cb_path.exists():
        log.info("Loading codebook from %s", cb_path)
        return np.load(cb_path)

    if cb_path and not cb_path.exists():
        resp = input(
            f"Codebook '{cb_path}' does not exist. "
            f"Generate one and save it there? [y/N] "
        ).strip().lower()
        if resp != "y":
            print("Aborting at user request.")
            sys.exit(0)
        save_to = cb_path
    else:
        save_to = Path(args.out) / "codebook.npy"

    # Build from a random sample of molecules.
    rng = np.random.default_rng(args.seed)
    n_sample = min(args.codebook_sample, len(descriptors))
    idx = rng.choice(len(descriptors), size=n_sample, replace=False)
    sample = [descriptors[i] for i in idx]

    if args.subsample_per_mol:
        sample = gm.subsample_descriptors(sample, args.subsample_per_mol)
    pool = gm.sample_descriptor_pool(sample)
    codebook = gm.build_codebook(pool, n_codewords=args.n_codewords,
                                 random_state=args.seed)

    save_to.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_to, codebook)
    log.info("Built codebook %s from %d molecules -> %s",
             codebook.shape, n_sample, save_to)
    print(f"Codebook {codebook.shape} built from {n_sample} molecules -> {save_to}")
    return codebook


# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("smiles_to_descriptors")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(out_dir / "workflow.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    return log


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="+", help="Input .smi/.csv file(s).")
    p.add_argument("--out", "-o", default="ge_molsg_run", help="Output directory.")

    # CSV parsing
    p.add_argument("--has-header", action="store_true", help="CSV has a header row.")
    p.add_argument("--smiles-col", default="smiles",
                   help="SMILES column name or index (default: smiles / 0).")
    p.add_argument("--id-col", default=None, help="ID column name or index (optional).")
    p.add_argument("--label-col", default=None,
                   help="Activity-label column name or index (optional).")
    p.add_argument("--labels-from-files", action="store_true",
                   help="Treat each input .smi file as one class (by filename).")

    # Conformer / front-end
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-confs", type=int, default=1,
                   help="Conformers per molecule; >1 keeps lowest-energy (default 1).")
    p.add_argument("--max-attempts", type=int, default=50)
    p.add_argument("--maxiters", type=int, default=500)
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Front-end worker processes; -1 = all CPUs (default 1).")

    # Descriptor / ge_molsg
    p.add_argument("--n-neighbors", type=int, default=100)
    p.add_argument("--n-components", type=int, default=100)
    p.add_argument("--evals", type=int, default=50)
    p.add_argument("--elec-weight", type=float, default=0.3)

    # Codebook / BoF
    p.add_argument("--codebook", default=None,
                   help="Codebook .npy path; if missing, prompts to build there.")
    p.add_argument("--codebook-sample", type=int, default=100,
                   help="Molecules sampled to build a codebook (default 100).")
    p.add_argument("--n-codewords", type=int, default=1000)
    p.add_argument("--subsample-per-mol", type=int, default=None,
                   help="Optional FPS subsample per molecule for the codebook pool.")
    p.add_argument("--bof-knn", type=int, default=3)
    p.add_argument("--no-bof", action="store_true", help="Skip BoF encoding.")
    p.add_argument("--wks-only", action="store_true",
                   help="Compute and keep WKS descriptors; skip BoF.")

    # Retention (all temporary by default except BoF)
    p.add_argument("--keep-surfaces", action="store_true",
                   help="Keep generated surface .npy files.")
    p.add_argument("--keep-wks", action="store_true",
                   help="Keep per-molecule WKS descriptor .npy files.")
    p.add_argument("--save-pdb", default=None,
                   help="Directory to save min-energy conformer PDBs (optional).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out)
    log = setup_logging(out_dir)

    import ge_molsg as gm

    # 1. Parse input ------------------------------------------------------- #
    records = load_records(args)
    if not records:
        print("Error: no SMILES records parsed from input.", file=sys.stderr)
        sys.exit(1)
    assign_names(records)
    log.info("Parsed %d records", len(records))
    print(f"Parsed {len(records)} records")

    # 2. Front-end: SMILES -> surfaces (parallel, fault-tolerant) ---------- #
    surf_dir = (out_dir / "surfaces") if args.keep_surfaces \
        else Path(tempfile.mkdtemp(prefix="ge_molsg_surf_"))
    fe_cfg = FrontEndConfig(
        surf_dir=str(surf_dir), seed=args.seed, num_confs=args.num_confs,
        max_attempts=args.max_attempts, maxiters=args.maxiters,
        save_pdb_dir=args.save_pdb,
    )
    if args.keep_surfaces:
        surf_dir.mkdir(parents=True, exist_ok=True)
    successes = generate_surfaces(records, fe_cfg, args.n_jobs, log)
    if not successes:
        print("Error: no surfaces generated; see log.", file=sys.stderr)
        sys.exit(1)

    # 3. WKS descriptors (ge_molsg, main process) ------------------------- #
    surfaces = [gm.load_surface_npy(path, name=rec.mol_id) for rec, path in successes]
    config = gm.GEMolSGConfig(
        n_neighbors=args.n_neighbors, n_components=args.n_components,
        evals=args.evals, elec_weight=args.elec_weight,
    )
    descriptors = gm.compute_wks_batch(surfaces, config, n_jobs=1, progress=True)

    if args.keep_wks or args.wks_only:
        wks_dir = out_dir / "wks"
        wks_dir.mkdir(parents=True, exist_ok=True)
        for (rec, _), desc in zip(successes, descriptors):
            np.save(wks_dir / f"{rec.mol_id}_wks.npy", desc)
        print(f"WKS descriptors -> {wks_dir}")

    # 4. BoF encoding ------------------------------------------------------ #
    if args.no_bof or args.wks_only:
        _summary(successes, records, out_dir, log)
        return

    codebook = resolve_codebook(args, descriptors, log)
    vectors = np.vstack(
        [gm.knn_histogram(d, codebook, knn=args.bof_knn) for d in descriptors]
    )
    np.save(out_dir / "bof_vectors.npy", vectors)

    index = [{"id": rec.mol_id, "label": rec.label} for rec, _ in successes]
    (out_dir / "bof_index.json").write_text(json.dumps(index, indent=2))
    print(f"BoF vectors {vectors.shape} -> {out_dir / 'bof_vectors.npy'}")
    print(f"Index -> {out_dir / 'bof_index.json'}")

    _summary(successes, records, out_dir, log)


def _summary(successes, records, out_dir, log) -> None:
    n_ok, n_total = len(successes), len(records)
    msg = f"Done: {n_ok}/{n_total} molecules processed ({n_total - n_ok} failed)."
    log.info(msg)
    print(msg)
    print(f"Log: {out_dir / 'workflow.log'}")


if __name__ == "__main__":
    main()
