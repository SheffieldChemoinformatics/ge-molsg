#!/usr/bin/env python
"""Reproduce paper retrieval results for a target from a Zenodo dataset.

Downloads a target's ESP-Npy surface dataset from Zenodo (a ``.tar.gz`` of
``ligand_<ID>`` / ``decoy_<ID>`` ``.npy`` surfaces), computes WKS descriptors,
encodes them to Bag-of-Features vectors against a provided codebook, then runs
chi-squared-kernel similarity retrieval using the paper's query set.

Queries are the active IDs listed in ``queries/<target>.csv`` under the column
``queries``; each is searched against the entire dataset. Per-query EF1% and
BEDROC are written out and their means reported.

Codebooks are loaded from ``example_codebooks/<target>.npy`` (or a path given
with ``--codebook``); if none is found one is built from the dataset.

The retrieval helpers follow the project's DUD-E evaluation utilities, adapted
to use the chi-squared kernel and to drive queries from the CSV.

Example
-------
    python scripts/evaluate_target.py --target ampc --out eval_ampc/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import chi2_kernel
from rdkit.ML.Scoring import Scoring


# --------------------------------------------------------------------------- #
# Download                                                                     #
# --------------------------------------------------------------------------- #
def fetch_dataset(target: str, dest: Path) -> Path:
    """Download and extract the target's ESP-Npy tarball from Zenodo.

    Uses a placeholder Zenodo URL; replace ``ZENODO_URL`` with the real record
    file URL. Returns the directory containing the extracted ``.npy`` surfaces.
    """
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / f"{target}_ESP_Npy.tar.gz"

    # ---- PLACEHOLDER: replace with the real Zenodo record file URL ----
    ZENODO_URL = f"https://zenodo.org/record/PLACEHOLDER/files/{target}_ESP_Npy.tar.gz"
    # -------------------------------------------------------------------

    if not tar_path.exists():
        print(f"Downloading {ZENODO_URL}")
        subprocess.run(["curl", "-L", "-o", str(tar_path), ZENODO_URL], check=True)

    extract_dir = dest / target
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(extract_dir)

    npys = list(extract_dir.rglob("*.npy"))
    if not npys:
        raise FileNotFoundError(f"No .npy surfaces found under {extract_dir}")
    return npys[0].parent


# --------------------------------------------------------------------------- #
# Retrieval helpers (adapted from dudez_eval_utils.py; chi-squared kernel)     #
# --------------------------------------------------------------------------- #
def _flatten(list_of_arrays: list) -> np.ndarray:
    """Concatenate a list of arrays into a single flat array."""
    return np.asarray([x for xs in list_of_arrays for x in xs]).flatten()


def results_gen(
    ranked_labels_raw: list,
    similarity: np.ndarray,
    fns: list,
    substring_fns: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rank by similarity, remove duplicate conformations, return clean arrays."""
    rankings = np.argsort(similarity)[::-1]
    ranked_labels = np.asarray(ranked_labels_raw)[rankings]
    ranked_fns = np.asarray(fns)[rankings]

    ind_del, fns_del, sim_del = [], [], []
    for n in substring_fns:
        matches = np.flatnonzero(np.char.find(ranked_fns, n) != -1)
        if matches.shape[0] > 1:
            extra = np.sort(matches)[1:].flatten()
            ind_del.append(extra)
            fns_del.append(ranked_fns[extra])
            sim_del.append([i for i in range(len(fns)) if fns[i] in list(ranked_fns[extra])])

    sim_del = _flatten(sim_del)
    ind_del = _flatten(ind_del)
    fns_del = _flatten(fns_del)

    if len(ind_del):
        ranked_labels = np.delete(ranked_labels, ind_del)
        similarity = np.delete(similarity, sim_del)

    clean_fns = np.asarray([f for f in ranked_fns if f not in fns_del])
    clean_labels = np.asarray(
        sorted([1 if f[0] == "l" else 0 for f in clean_fns], reverse=True)
    )
    return ranked_labels, similarity, clean_labels, clean_fns


def rank_metrics(ref_name: str, similarity: np.ndarray, results: np.ndarray) -> dict:
    """EF1% and BEDROC for one ranked query result (no AUC)."""
    similarity[::-1].sort()
    results_sim = np.hstack((similarity.reshape(-1, 1), results.reshape(-1, 1)))
    ef = Scoring.CalcEnrichment(results_sim, 1, [0.01])
    bedroc = Scoring.CalcBEDROC(results_sim, 1, 20)
    return {"RefMol": ref_name[:-4], "EF1%": ef, "BEDROC": bedroc}


def chi2_sim_fn(descs: list, i: int) -> np.ndarray:
    """Chi-squared-kernel similarity of molecule i against all others."""
    ref = np.asarray(descs[i]).reshape(1, -1)
    others = np.asarray([descs[z] for z in range(len(descs)) if z != i])
    return np.array([float(chi2_kernel(ref, q.reshape(1, -1)).flat[0]) for q in others])


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def load_surfaces(surf_dir: Path) -> tuple[list, list]:
    """Load all .npy surfaces; return (surfaces, filenames). Labels via prefix."""
    import ge_molsg as gm

    fns = sorted([f for f in os.listdir(surf_dir) if f.endswith(".npy")], reverse=True)
    surfaces = [gm.load_surface_npy(str(surf_dir / f), name=f[:-4]) for f in fns]
    return surfaces, fns


def load_query_ids(queries_csv: Path) -> list[str]:
    """Read query IDs from the 'queries' column of the target CSV."""
    df = pd.read_csv(queries_csv)
    if "queries" not in df.columns:
        raise ValueError(f"{queries_csv} has no 'queries' column")
    return [str(q).strip() for q in df["queries"].dropna()]


# --------------------------------------------------------------------------- #
# Codebook                                                                     #
# --------------------------------------------------------------------------- #
def resolve_codebook(args: argparse.Namespace, descriptors: list) -> np.ndarray:
    """Load the provided codebook, or build one from the dataset if absent."""
    import ge_molsg as gm

    cb_path = Path(args.codebook) if args.codebook else Path(args.codebook_dir) / f"{args.target}.npy"

    if cb_path.exists():
        print(f"Loading codebook: {cb_path}")
        return np.load(cb_path)

    print(f"Codebook {cb_path} not found; building from dataset.")
    rng = np.random.default_rng(args.seed)
    n = min(args.codebook_sample, len(descriptors))
    sample = [descriptors[i] for i in rng.choice(len(descriptors), n, replace=False)]
    pool = gm.sample_descriptor_pool(sample)
    codebook = gm.build_codebook(pool, n_codewords=args.n_codewords, random_state=args.seed)
    cb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cb_path, codebook)
    print(f"Built codebook {codebook.shape} -> {cb_path}")
    return codebook


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #
def evaluate(
    target: str,
    vectors: list,
    fns: list,
    query_ids: list[str],
    out_dir: Path,
) -> None:
    """Run chi-squared retrieval for each query ID; report mean EF1%/BEDROC."""
    result_dir = out_dir / "results"
    ranked_dir = out_dir / "ranked_lists"
    result_dir.mkdir(parents=True, exist_ok=True)
    ranked_dir.mkdir(parents=True, exist_ok=True)

    fns_arr = np.asarray(fns)
    labels = np.asarray([1 if f[0] == "l" else 0 for f in fns])
    stems = [f[:-4] for f in fns]  # filename without .npy

    # Map each query ID to its dataset index.
    query_inds = []
    for qid in query_ids:
        hits = [j for j, s in enumerate(stems) if s == qid or s.endswith(qid)]
        if hits:
            query_inds.append(hits[0])
        else:
            print(f"  WARNING: query '{qid}' not found in dataset; skipping")
    print(f"Queries resolved: {len(query_inds)}/{len(query_ids)}")

    ef1_all, bedroc_all = [], []
    for i in query_inds:
        other_idx = [z for z in range(len(fns_arr)) if z != i]
        sim = chi2_sim_fn(vectors, i)
        fns_rest = fns_arr[other_idx].tolist()
        substring_fns = [fns_arr[z].split(".")[0][:-2] for z in other_idx]
        raw_labels = [int(labels[z]) for z in other_idx]

        ranked_labels, similarity, clean_labels, ranked_fns = results_gen(
            raw_labels, sim, fns_rest, substring_fns
        )
        assert not np.isnan(similarity).any(), "NaN in similarity scores"

        metrics = rank_metrics(fns_arr[i], similarity, ranked_labels)

        ref_stem = fns_arr[i].split(".")[0]
        pd.DataFrame({"Mol Names": ranked_fns, "Similarity": similarity}).to_csv(
            ranked_dir / ref_stem
        )
        pd.DataFrame([metrics]).to_csv(result_dir / ref_stem)

        ef1_all.append(metrics["EF1%"][0])
        bedroc_all.append(metrics["BEDROC"])
        print(f"  {fns_arr[i]}  EF1%={metrics['EF1%'][0]:.4f}  BEDROC={metrics['BEDROC']:.4f}")

    print(f"\nTarget: {target}")
    print(f"Mean EF1%:   {np.mean(ef1_all):.4f}")
    print(f"Mean BEDROC: {np.mean(bedroc_all):.4f}")
    pd.DataFrame(
        {"target": [target], "mean_EF1%": [np.mean(ef1_all)], "mean_BEDROC": [np.mean(bedroc_all)]}
    ).to_csv(out_dir / "summary.csv", index=False)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target", required=True, help="Target name (e.g. ampc, xiap).")
    p.add_argument("--out", "-o", default=None, help="Output directory (default: eval_<target>).")
    p.add_argument("--data-dir", default=None,
                   help="Use an existing surface directory instead of downloading.")
    p.add_argument("--queries-dir", default="queries",
                   help="Directory holding <target>.csv (default: queries/).")
    p.add_argument("--codebook", default=None, help="Explicit codebook .npy path.")
    p.add_argument("--codebook-dir", default="example_codebooks",
                   help="Folder of per-target codebooks (default: example_codebooks/).")
    p.add_argument("--codebook-sample", type=int, default=100)
    p.add_argument("--n-codewords", type=int, default=1000)
    # descriptor params
    p.add_argument("--n-neighbors", type=int, default=100)
    p.add_argument("--n-components", type=int, default=100)
    p.add_argument("--evals", type=int, default=50)
    p.add_argument("--elec-weight", type=float, default=0.3)
    p.add_argument("--bof-knn", type=int, default=3)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out) if args.out else Path(f"eval_{args.target}")
    out_dir.mkdir(parents=True, exist_ok=True)

    import ge_molsg as gm

    # 1. Data: download from Zenodo or use a local directory.
    surf_dir = Path(args.data_dir) if args.data_dir else fetch_dataset(args.target, out_dir / "data")

    # 2. Load surfaces.
    surfaces, fns = load_surfaces(surf_dir)
    print(
        f"Loaded {len(surfaces)} surfaces "
        f"({sum(f[0] == 'l' for f in fns)} ligands, "
        f"{sum(f[0] != 'l' for f in fns)} decoys)"
    )

    # 3. WKS descriptors.
    config = gm.GEMolSGConfig(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        evals=args.evals,
        elec_weight=args.elec_weight,
    )
    descriptors = gm.compute_wks_batch(surfaces, config, n_jobs=args.n_jobs, progress=True)

    # 4. Codebook + BoF vectors.
    codebook = resolve_codebook(args, descriptors)
    vectors = [gm.knn_histogram(d, codebook, knn=args.bof_knn) for d in descriptors]

    # 5. Queries from the paper's CSV.
    query_ids = load_query_ids(Path(args.queries_dir) / f"{args.target}.csv")

    # 6. Retrieval + report.
    evaluate(args.target, vectors, fns, query_ids, out_dir)


if __name__ == "__main__":
    main()