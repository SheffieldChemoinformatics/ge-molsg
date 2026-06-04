"""Command line interface for the ge_molsg package."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ge_molsg import (
    GEMolSGConfig,
    build_codebook,
    compute_wks_batch,
    knn_histogram,
    load_surface_npy,
    sample_descriptor_pool,
    subsample_descriptors,
)


def _load_surfaces(input_dir: Path):
    paths = sorted(input_dir.glob("*.npy"))
    if not paths:
        raise SystemExit(f"No .npy files found in {input_dir}")
    return paths, [load_surface_npy(str(p), name=p.stem) for p in paths]


def _config(args: argparse.Namespace) -> GEMolSGConfig:
    return GEMolSGConfig(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        evals=args.evals,
    )


def _cmd_descriptors(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths, surfaces = _load_surfaces(args.input_dir)
    descriptors = compute_wks_batch(
        surfaces, _config(args), n_jobs=args.n_jobs, progress=True
    )
    for path, desc in zip(paths, descriptors):
        np.save(args.output_dir / f"{path.stem}_wks.npy", desc)
    print(f"Wrote {len(descriptors)} descriptors to {args.output_dir}")


def _cmd_codebook(args: argparse.Namespace) -> None:
    _, surfaces = _load_surfaces(args.input_dir)
    descriptors = compute_wks_batch(
        surfaces, _config(args), n_jobs=args.n_jobs, progress=True
    )
    if args.sample_per_mol is not None:
        descriptors = subsample_descriptors(descriptors, args.sample_per_mol)
    pool = sample_descriptor_pool(descriptors)
    codebook = build_codebook(pool, n_codewords=args.n_codewords)
    np.save(args.output, codebook)
    print(f"Saved codebook {codebook.shape} to {args.output}")


def _cmd_encode(args: argparse.Namespace) -> None:
    _, surfaces = _load_surfaces(args.input_dir)
    codebook = np.load(args.codebook)
    descriptors = compute_wks_batch(
        surfaces, _config(args), n_jobs=args.n_jobs, progress=True
    )
    vectors = np.vstack(
        [knn_histogram(d, codebook, knn=args.bof_knn) for d in descriptors]
    )
    np.save(args.output, vectors)
    print(f"Saved vectors {vectors.shape} to {args.output}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-neighbors", type=int, default=100)
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--evals", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1, help="-1 uses all CPUs.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ge_molsg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_desc = sub.add_parser("descriptors", help="Compute per-vertex WKS descriptors.")
    p_desc.add_argument("input_dir", type=Path)
    p_desc.add_argument("output_dir", type=Path)
    _add_common(p_desc)
    p_desc.set_defaults(func=_cmd_descriptors)

    p_cb = sub.add_parser("codebook", help="Build a Bag-of-Features codebook.")
    p_cb.add_argument("input_dir", type=Path)
    p_cb.add_argument("output", type=Path)
    p_cb.add_argument("--n-codewords", type=int, default=1000)
    p_cb.add_argument("--sample-per-mol", type=int, default=None,
                      help="Vertices kept per molecule. Default: retain all.")
    _add_common(p_cb)
    p_cb.set_defaults(func=_cmd_codebook)

    p_enc = sub.add_parser("encode", help="Encode surfaces into BoF vectors.")
    p_enc.add_argument("input_dir", type=Path)
    p_enc.add_argument("codebook", type=Path)
    p_enc.add_argument("output", type=Path)
    p_enc.add_argument("--bof-knn", type=int, default=3)
    _add_common(p_enc)
    p_enc.set_defaults(func=_cmd_encode)

    return parser


def main(argv=None) -> None:
    """Entry point for the ``ge_molsg`` command."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
