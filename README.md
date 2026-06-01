# GE-MolSG

WKS surface descriptors with a graph-Laplacian spectral embedding.

GE-MolSG computes per-vertex [Wave Kernel Signature](http://imagine.enpc.fr/~aubrym/projects/wks/index.html)
(WKS) descriptors on molecular surfaces. The spectral embedding is built
directly with SciPy — an adaptive-bandwidth Euclidean affinity matrix, a
normalized graph Laplacian, and the bottom eigenpairs with the trivial mode
dropped — and aggregated into a per-molecule Bag-of-Features vector against a
learned codebook.

Every stage is an importable function, so you can use as much or as little of
the pipeline as you need. The `GEMolSG` class is an optional convenience
wrapper that composes these functions.

## Installation

```bash
pip install numpy scipy scikit-learn tqdm
```

The neighbor search uses SciPy's cKDTree by default (scikit-learn is also
required, for the codebook and descriptor normalization). `tqdm` is only needed for the
optional progress bar.

## Usage

### Individual stages: graph, eigenvalues, eigenvectors, descriptor

Each step returns a plain object you can inspect or reuse:

```python
import ge_molsg as gm

surface = gm.load_surface_npy("mol.npy")            # or gm.MolSurface(vertices, faces, esp)
points  = surface.augmented_points(elec_weight=0.3)  # (N, 4): [x, y, z, esp * weight]

# 1. Affinity matrix (sparse W)
W = gm.compute_affinity(points, n_neighbors=100)

# 2. Graph Laplacian (sparse L)
L = gm.graph_laplacian(W, laplacian_type="normalized")

# 3. Eigenvalues and eigenvectors (trivial mode already dropped)
eigenvalues, eigenvectors = gm.compute_eigenpairs(L, n_components=100)

# 4. WKS descriptor from the eigensystem
descriptor = gm.wks((eigenvalues, eigenvectors), evals=50)   # (N, 50)
```

Returns: `W` and `L` are `scipy.sparse` matrices; `eigenvalues` is `(k,)`,
`eigenvectors` is `(N, k)`, and `descriptor` is `(N, evals)`.

Shortcuts:

```python
L = gm.build_laplacian(points, n_neighbors=100)      # steps 1-2 in one call
K = gm.knn_distance_graph(points, n_neighbors=100)   # raw kNN distance graph
sigma = gm.adaptive_bandwidth(K, n_neighbors=100)    # per-point bandwidths
```

To keep the trivial eigenpair (for example, to inspect the near-zero mode),
pass `drop_first=False`:

```python
eigenvalues, eigenvectors = gm.compute_eigenpairs(L, n_components=100, drop_first=False)
```

The solver is configurable:

```python
eigenvalues, eigenvectors = gm.compute_eigenpairs(
    L, n_components=100, eigensolver="arpack", eigen_tol=0.0
)
```

### One descriptor, end to end

```python
cfg = gm.GEMolSGConfig(n_components=100, n_neighbors=100, evals=50)
descriptor = gm.compute_wks(surface, cfg)            # runs steps 1-4 internally
```

### Many descriptors, optionally in parallel

`compute_wks_batch` runs descriptor generation across surfaces using a process
pool, returning results in input order. `n_jobs=1` runs serially; `n_jobs=-1`
uses all available CPUs.

```python
surfaces    = [gm.load_surface_npy(p) for p in paths]
descriptors = gm.compute_wks_batch(surfaces, cfg, n_jobs=-1, progress=True)
# list of (N_i, evals) arrays, in input order
```

### Codebook and Bag-of-Features

The descriptor stages above are sufficient on their own. To aggregate
per-vertex descriptors into fixed-length per-molecule vectors, fit a codebook
and encode against it:

```python
pool     = gm.sample_descriptor_pool(descriptors, n_per_mol=200)
codebook = gm.build_codebook(pool, n_codewords=1000)
vector   = gm.knn_histogram(descriptor, codebook, knn=3)
```

### Convenience wrapper

`GEMolSG` holds the config and a fitted codebook so they are not passed
repeatedly. It composes the functions above and adds nothing load-bearing.

```python
model = gm.GEMolSG(cfg, n_codewords=1000, n_jobs=-1, progress=True)
model.fit_codebook(sample_surfaces)        # generates descriptors, then fits the codebook
vector = model.transform(surface)          # one surface  -> BoF vector
matrix = model.transform_many(surfaces)    # many surfaces -> BoF matrix
```

## Configuration

`GEMolSGConfig` collects the descriptor parameters:

| Field | Default | Description |
| --- | --- | --- |
| `n_components` | 100 | Eigenpairs retained (after dropping the trivial mode). |
| `n_neighbors` | 100 | kNN graph connectivity. |
| `elec_weight` | 0.3 | ESP scale in the augmented point cloud. |
| `adaptive_bw` | True | Per-point adaptive bandwidth; if False, use fixed `sigma`. |
| `sigma` | None | Fixed bandwidth (used only when `adaptive_bw` is False). |
| `laplacian_type` | "normalized" | `normalized`, `unnormalized`, or `random_walk`. |
| `backend` | "ckdtree" | Neighbor backend (`"ckdtree"` or `"sklearn"`) or a custom callable. |
| `eigensolver` | "arpack" | `arpack`, `lobpcg`, or `dense`. |
| `eigen_tol` | 0.0 | Solver tolerance; 0 uses machine precision. |
| `evals` | 50 | WKS descriptor width (energy evaluations). |
| `variance` | 7 | WKS Gaussian variance multiplier. |
| `l2_normalize` | True | L2-normalize each per-vertex descriptor. |

Additional lower-level fields are available on `GEMolSGConfig`:
`square_distances`, `symmetrize`, `backend_kwargs`, and `random_state`.

## Custom neighbor backend

Any callable mapping `(points, k) -> (indices, distances)` with self-matches
excluded can be used in place of the built-in backends (`"ckdtree"`, the
default, and `"sklearn"`):

```python
def my_backend(points, k):
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(points).query(points, k=k + 1)
    return idx[:, 1:], dist[:, 1:]

W = gm.compute_affinity(points, n_neighbors=100, backend=my_backend)
```

For process-based parallelism the callable must be picklable (a module-level
function, not a lambda or closure).
