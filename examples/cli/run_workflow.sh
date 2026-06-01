#!/usr/bin/env bash
#
# examples/cli/run_workflow.sh
# ----------------------------
# End-to-end GE-MolSG workflow using the `ge_molsg` command line interface.
#
# The CLI exposes three subcommands, each operating on a directory of `.npy`
# surface files (object arrays of [vertices, faces, charges]):
#
#   ge_molsg descriptors  INPUT_DIR OUTPUT_DIR       -> per-vertex WKS descriptors
#   ge_molsg codebook     INPUT_DIR CODEBOOK.npy     -> Bag-of-Features codebook
#   ge_molsg encode       INPUT_DIR CODEBOOK OUTPUT  -> per-molecule BoF vectors
#
# Equivalent standalone scripts live in ../../scripts/ and take the same
# arguments (e.g. `python scripts/compute_descriptors.py ...`).
#
# Usage:
#   bash examples/cli/run_workflow.sh /path/to/surfaces /path/to/work_dir
#
set -euo pipefail

SURFACES="${1:?Usage: run_workflow.sh SURFACE_DIR WORK_DIR}"
WORK="${2:?Usage: run_workflow.sh SURFACE_DIR WORK_DIR}"

mkdir -p "$WORK"

# Shared descriptor parameters.
NN=100          # kNN graph connectivity
NCOMP=100       # eigenpairs retained (after dropping the trivial mode)
EVALS=50        # WKS descriptor width
NJOBS=-1        # parallel workers (-1 = all CPUs)

echo "==> 1. Per-vertex WKS descriptors"
ge_molsg descriptors "$SURFACES" "$WORK/descriptors" \
    --n-neighbors "$NN" --n-components "$NCOMP" --evals "$EVALS" --n-jobs "$NJOBS"

echo "==> 2. Bag-of-Features codebook"
ge_molsg codebook "$SURFACES" "$WORK/codebook.npy" \
    --n-codewords 1000 --sample-per-mol 200 \
    --n-neighbors "$NN" --n-components "$NCOMP" --evals "$EVALS" --n-jobs "$NJOBS"

echo "==> 3. Per-molecule BoF vectors"
ge_molsg encode "$SURFACES" "$WORK/codebook.npy" "$WORK/vectors.npy" \
    --bof-knn 3 \
    --n-neighbors "$NN" --n-components "$NCOMP" --evals "$EVALS" --n-jobs "$NJOBS"

echo "==> Done. Outputs in $WORK"
