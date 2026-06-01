# Test data

## thalidomide_R.npy

Reference molecular surface for the thalidomide R enantiomer, stored as a NumPy
object array with the standard layout `[vertices, faces, charges]`:

- `vertices` — `(5738, 3)` float64 surface coordinates
- `faces` — `(11472, 3)` int64 triangle indices
- `charges` — `(5738, 1)` float64 per-vertex electrostatic potential

Loaded in tests via `tests.store.thalidomide_r()`. Fast invariant tests use
the synthetic surfaces from `tests/store.py` instead, to keep the suite quick.
