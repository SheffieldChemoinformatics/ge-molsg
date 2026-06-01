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
