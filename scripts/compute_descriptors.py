#!/usr/bin/env python
"""Compute per-vertex WKS descriptors for a directory of surface files.

Loads every ``.npy`` surface in an input directory, computes its WKS descriptor
(optionally in parallel), and saves the descriptors to an output directory.

Example
-------
    python scripts/compute_descriptors.py surfaces/ descriptors/ \
        --n-neighbours 100 --n-components 100 --evals 50 --n-jobs -1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ge_molsg import GEMolSGConfig, compute_wks_batch, load_surface_npy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory of .npy surfaces.")
    parser.add_argument("output_dir", type=Path, help="Directory for descriptors.")
    parser.add_argument("--n-neighbors", type=int, default=100)
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--evals", type=int, default=50)
    parser.add_argument("--elec-weight", type=float, default=0.3)
    parser.add_argument("--n-jobs", type=int, default=1, help="-1 uses all CPUs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_dir.glob("*.npy"))
    if not paths:
        raise SystemExit(f"No .npy files found in {args.input_dir}")

    surfaces = [load_surface_npy(str(p), name=p.stem) for p in paths]
    config = GEMolSGConfig(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        evals=args.evals,
        elec_weight=args.elec_weight,
    )

    descriptors = compute_wks_batch(
        surfaces, config, n_jobs=args.n_jobs, progress=True
    )

    for path, desc in zip(paths, descriptors):
        out = args.output_dir / f"{path.stem}_wks.npy"
        np.save(out, desc)

    print(f"Wrote {len(descriptors)} descriptors to {args.output_dir}")


if __name__ == "__main__":
    main()
