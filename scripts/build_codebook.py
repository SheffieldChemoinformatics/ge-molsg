#!/usr/bin/env python
"""Build a Bag-of-Features codebook from a sample of surfaces.

Computes WKS descriptors for a set of sample surfaces, pools them (optionally
farthest-point-subsampled per molecule), fits a MiniBatchKMeans codebook, and
saves it as a ``.npy`` array.

Example
-------
    python scripts/build_codebook.py surfaces/ codebook.npy \
        --n-codewords 1000 --sample-per-mol 200 --n-jobs -1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ge_molsg import (
    GEMolSGConfig,
    build_codebook,
    compute_wks_batch,
    load_surface_npy,
    sample_descriptor_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory of .npy surfaces.")
    parser.add_argument("output", type=Path, help="Output codebook .npy path.")
    parser.add_argument("--n-codewords", type=int, default=1000)
    parser.add_argument("--sample-per-mol", type=int, default=None)
    parser.add_argument("--n-neighbors", type=int, default=100)
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--evals", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1, help="-1 uses all CPUs.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = sorted(args.input_dir.glob("*.npy"))
    if not paths:
        raise SystemExit(f"No .npy files found in {args.input_dir}")

    surfaces = [load_surface_npy(str(p), name=p.stem) for p in paths]
    config = GEMolSGConfig(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        evals=args.evals,
    )

    descriptors = compute_wks_batch(
        surfaces, config, n_jobs=args.n_jobs, progress=True
    )
    pool = sample_descriptor_pool(descriptors, n_per_mol=args.sample_per_mol)
    codebook = build_codebook(
        pool, n_codewords=args.n_codewords, random_state=args.random_state
    )

    np.save(args.output, codebook)
    print(f"Saved codebook {codebook.shape} to {args.output}")


if __name__ == "__main__":
    main()
