#!/usr/bin/env python
"""Encode surfaces into Bag-of-Features vectors against a codebook.

Loads a precomputed codebook, computes WKS descriptors for each input surface,
aggregates them into per-molecule BoF vectors, and writes a single stacked
array plus an index of surface names.

Example
-------
    python scripts/encode_surfaces.py surfaces/ codebook.npy vectors.npy \
        --bof-knn 3 --n-jobs -1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ge_molsg import (
    GEMolSGConfig,
    compute_wks_batch,
    knn_histogram,
    load_surface_npy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory of .npy surfaces.")
    parser.add_argument("codebook", type=Path, help="Codebook .npy path.")
    parser.add_argument("output", type=Path, help="Output vectors .npy path.")
    parser.add_argument("--bof-knn", type=int, default=3)
    parser.add_argument("--n-neighbors", type=int, default=100)
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--evals", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1, help="-1 uses all CPUs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = sorted(args.input_dir.glob("*.npy"))
    if not paths:
        raise SystemExit(f"No .npy files found in {args.input_dir}")

    codebook = np.load(args.codebook)
    surfaces = [load_surface_npy(str(p), name=p.stem) for p in paths]
    config = GEMolSGConfig(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        evals=args.evals,
    )

    descriptors = compute_wks_batch(
        surfaces, config, n_jobs=args.n_jobs, progress=True
    )
    vectors = np.vstack(
        [knn_histogram(d, codebook, knn=args.bof_knn) for d in descriptors]
    )

    np.save(args.output, vectors)
    index_path = args.output.with_suffix(".index.json")
    index_path.write_text(json.dumps([p.stem for p in paths], indent=2))
    print(f"Saved vectors {vectors.shape} to {args.output}")
    print(f"Saved name index to {index_path}")


if __name__ == "__main__":
    main()
