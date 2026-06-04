# Examples

Two ways to run the GE-MolSG workflow, both equivalent to the standalone
scripts in [`../scripts/`](../scripts).

## Command line (`cli/run_workflow.sh`)

A shell script driving the `ge_molsg` CLI end to end: descriptors → codebook →
encode. Requires the package to be installed (`pip install -e .`), which
registers the `ge_molsg` command.

```bash
bash examples/cli/run_workflow.sh /path/to/surfaces /path/to/work_dir
```

The individual subcommands and the matching `scripts/*.py` take identical
arguments, so either entry point works.

## Notebook (`workflow.ipynb`)

A single self-contained notebook that runs the full library workflow in
process on the bundled thalidomide R-enantiomer surface — loading a surface,
inspecting the graph / Laplacian / eigensystem intermediates, computing WKS
descriptors, building a codebook, and encoding to Bag-of-Features vectors. No
external data files or CLI invocation needed.

## From SMILES (`../scripts/smiles_to_descriptors.py`)

End-to-end from a `.smi`/`.csv` file of SMILES through to WKS descriptors and
Bag-of-Features vectors: 3-D conformer generation, surface meshing, Coulombic
ESP, then the ge_molsg descriptor and codebook stages. Requires RDKit and the
external Node.js mesh generator. Failures are logged and skipped with a summary.

```bash
python scripts/smiles_to_descriptors.py library.csv --out run/ \
    --has-header --smiles-col smiles --id-col name --label-col activity \
    --num-confs 20 --n-jobs -1
```

## Reproducing paper retrieval results (`../scripts/evaluate_target.py`)

Downloads a target's ESP-Npy surface dataset from Zenodo, encodes it to
Bag-of-Features vectors against the codebook in `example_codebooks/<target>.npy`,
and runs chi-squared-kernel retrieval using the query set in
`queries/<target>.csv` (active IDs under a `queries` column). Reports mean EF1%
and BEDROC. Requires RDKit (for the enrichment/BEDROC scoring).

```bash
python scripts/evaluate_target.py --target ampc --out eval_ampc/
```
