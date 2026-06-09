# Query sets

The query molecules used to report retrieval results in the paper, one CSV per
target named `<target>.csv` (e.g. `ampc.csv`, `xiap.csv`). Each CSV lists the
active molecule IDs to use as queries under a column named `queries`.

`scripts/evaluate_target.py` reads `<target>.csv` and searches each query
against the entire dataset, reporting mean EF1% and BEDROC.

Example `ampc.csv`:

    queries
    L0
    L1
    L2
