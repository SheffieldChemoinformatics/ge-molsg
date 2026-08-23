"""Rebuild per-n_neighbors GE-MolSG codebooks.

Downloads/extracts whichever DUDE-Z targets are referenced by the sample set
and not already cached locally (this covers all 43 targets, ~47.8 GB total -
expect this to take a long time). Per-surface WKS descriptors are cached to
disk as they're computed, so an interrupted run can resume without redoing
finished work.
"""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import ge_molsg as gm

ZENODO_BASE_URL = "https://zenodo.org/records/20547837/files"
DATA_ROOT = BASE_DIR / "dude_z_data"
CODEBOOK_DIR = BASE_DIR / "codebooks"
CODEBOOK_SET_CSV = CODEBOOK_DIR / "gemolsg_codebook_set.csv"
DESC_CACHE_DIR = CODEBOOK_DIR / "_desc_cache"

# Same recipe as the original ge_molsg_cb.npy (see codebooks/README.md naming:
# 100eigs_100evals_15var_*knn_normalized_lap_0.3ew_1000cw), varying n_neighbors.
N_COMPONENTS, EVALS, VAR, EW, LAP_NORM = 100, 100, 15, 0.3, "normalized"
N_CODEWORDS = 1000
RANDOM_STATE = 42
SWEEP_N_NEIGHBORS = [10, 50, 100]
N_JOBS = -1

OUT_DIR = BASE_DIR / "experiments_out" / "codebook_bootstrap"
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(OUT_DIR / "codebook_bootstrap.log")],
    force=True)
log = logging.getLogger("codebook_bootstrap")


def fetch_target(target, data_root):
    data_root = Path(data_root)
    for cand in (data_root / target, data_root / target.upper(),
                 data_root / "DUDE-Z" / target):
        if (cand / "ESP_Npy").is_dir():
            return cand

    local_tar = None
    for tar in (data_root / f"{target}.tar.gz", data_root / "DUDE-Z" / f"{target}.tar.gz"):
        if tar.exists():
            local_tar = tar
            break

    def _download(dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{ZENODO_BASE_URL.rstrip('/')}/{target}.tar.gz"
        log.info("Downloading %s", url)
        try:
            subprocess.run(["curl", "-fSL", "-o", str(dest), url], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            urllib.request.urlretrieve(url, dest)

    if local_tar is None:
        local_tar = data_root / f"{target}.tar.gz"
        _download(local_tar)

    for attempt in range(2):
        try:
            log.info("Extracting %s", local_tar)
            with tarfile.open(local_tar, "r:gz") as tf:
                tf.extractall(data_root)
            break
        except (tarfile.TarError, EOFError, OSError) as exc:
            if attempt == 0:
                log.warning("Corrupt/incomplete archive %s (%s); re-downloading", local_tar, exc)
                local_tar.unlink(missing_ok=True)
                _download(local_tar)
            else:
                raise

    for cand in (data_root / target, data_root / target.upper()):
        if (cand / "ESP_Npy").is_dir():
            return cand
    hits = list(data_root.rglob(f"{target}/ESP_Npy")) or list(data_root.rglob("ESP_Npy"))
    if hits:
        return hits[0].parent
    raise FileNotFoundError(f"Extracted '{target}' but found no ESP_Npy under {data_root}")


def load_sample_set(csv_path):
    """Return list of (target, relative_npy_path) from gemolsg_codebook_set.csv."""
    df = pd.read_csv(csv_path)
    col = "Mol Names" if "Mol Names" in df.columns else df.columns[-1]
    entries = []
    for rel in df[col]:
        target = rel.split("/", 1)[0]
        entries.append((target, rel))
    return entries


def _wks_descriptor(surf_path, n_neighbors):
    mol = np.load(str(surf_path), allow_pickle=True)
    coords, esp = mol[0], mol[2]
    feat = np.concatenate([coords, (esp * EW).reshape(-1, 1)], axis=1)
    W = gm.compute_affinity(
        feat, n_neighbors=n_neighbors, backend="ckdtree",
        adaptive_bw=True, square_distances=False,
    )
    L = gm.graph_laplacian(W, laplacian_type=LAP_NORM)
    eigensystem = gm.compute_eigenpairs(
        L, n_components=N_COMPONENTS, drop_first=True, eigensolver="arpack",
    )
    return np.nan_to_num(gm.wks(eigensystem, evals=EVALS, variance=VAR, l2=True))


def _cache_key(target, rel_path):
    return rel_path.replace("/", "__")[:-4]  # strip .npy, filesystem-safe


def _descriptor_for_entry(target, rel_path, n_neighbors, cache_dir):
    cache_path = cache_dir / f"{_cache_key(target, rel_path)}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    surf_path = DATA_ROOT / target / "ESP_Npy" / Path(rel_path).name
    desc = _wks_descriptor(surf_path, n_neighbors)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, desc)
    return desc


def build_codebook_for_k(n_neighbors, entries):
    out_path = CODEBOOK_DIR / f"gemolsg_cb_{n_neighbors}.npy"
    if out_path.exists():
        log.info("gemolsg_cb_%d.npy already exists; skipping", n_neighbors)
        return

    cache_dir = DESC_CACHE_DIR / str(n_neighbors)
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info("Computing %d per-surface WKS descriptors at n_neighbors=%d", len(entries), n_neighbors)
    descriptors = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_descriptor_for_entry)(target, rel_path, n_neighbors, cache_dir)
        for target, rel_path in entries
    )

    pool = np.vstack(descriptors)
    log.info("Pooled descriptor matrix for n_neighbors=%d: %s", n_neighbors, pool.shape)

    codebook = gm.build_codebook(pool, n_codewords=N_CODEWORDS, random_state=RANDOM_STATE)
    np.save(out_path, codebook)
    log.info("Saved %s (%s)", out_path, codebook.shape)


def main():
    entries = load_sample_set(CODEBOOK_SET_CSV)
    targets = sorted({t for t, _ in entries})
    log.info("Codebook sample set: %d surfaces across %d targets", len(entries), len(targets))

    for target in targets:
        try:
            fetch_target(target, DATA_ROOT)
        except Exception:
            log.exception("Failed to fetch/extract %s; surfaces from this target will fail below", target)

    for n_neighbors in SWEEP_N_NEIGHBORS:
        build_codebook_for_k(n_neighbors, entries)

    log.info("Codebook bootstrap complete.")


if __name__ == "__main__":
    main()
