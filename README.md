# GE-MolSG

![image](https://github.com/SheffieldChemoinformatics/ge-molsg/blob/main/assets/ge_mol_sg_workflow.png?raw=true)

Spectral geometry descriptors encoded via graph-Laplacian approximation and
[Wave Kernel Signature](http://imagine.enpc.fr/~aubrym/projects/wks/index.html) (WKS).

GE-MolSG computes per-vertex WKS descriptors on molecular surfaces. 
The spectral embedding is built with SciPy — an adaptive-bandwidth Euclidean affinity matrix, a
normalized graph Laplacian, and the bottom eigenpairs with the trivial mode
dropped — and aggregated into a per-molecule Bag-of-Features vector against a
learned codebook.

The generation of molecular surfaces can be done in a variety of ways. 
We provide an example for generating a surface from SMILES, and we include 
the files required to reproduce our experiments and benchmarks under 
the folder `experiments/`.

Every stage of the transformation between surface and codebook generation 
is implemented as a separate function, so you can use as much or as little of
the pipeline as you need. The `GEMolSG` class is a high-level
wrapper that composes these functions.

## Installation using `uv` using Python 3.12

```bash
# Clone this repo
git clone https://github.com/SheffieldChemoinformatics/ge-molsg.git
cd ge-molsg
```


```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

The neighbor search uses SciPy's cKDTree by default (scikit-learn is also
required, for the codebook and descriptor normalization). `tqdm` is only needed for the
optional progress bar.

## Usage

### Surface creation (prep stage)
`ge_molsg` requires a Numpy representation of the surface along with its vertices, faces, and ESP values.
This is a preparation stage that can be performed as the user prefers. `ge_molsg` focusses on yielding embeddings and codebooks from the input surfaces.
In our early experiments we used two public modules for ESP prediction and surface generation:

- https://github.com/AstexUK/ESP_DNN
- https://github.com/AstexUK/esp-surface-generator 

Here we provide an example of generating a surface from a SMILES string using MMFF94 charges and `esp-surface-generator`, which requires `node` (version `v17.9.1` at the time of writing). To install `esp-surface-generator` follow these steps:

```bash
bash scripts/install_esp_surface_generator.sh
```

```bash
# Ensure that you're environment is correctly configured to run Node JS
# We're using Node 17 here. `nvm use 17`
python scripts/smiles_to_surface_npy.py --smiles "CC(=O)Nc1ccc(O)cc1" --out mol.npy
>>> Processing SMILES: CC(=O)Nc1ccc(O)cc1
Saved surface to mol.npy  (3988 vertices, 7972 faces)
Verified with ge_molsg: 3988 vertices, ESP range [-56.79, 57.04]
```

```bash
python -c "import numpy as np; d = np.load('mol.npy', allow_pickle=True); print([a.shape for a in d])"
>>> [(3988, 3), (7972, 3), (3988,)]
```

### One descriptor, end to end

```python
import ge_molsg as gm

surface = gm.load_surface_npy("mol.npy")            # or gm.MolSurface(vertices, faces, esp)
cfg = gm.GEMolSGConfig(n_components=100, n_neighbors=100, evals=50)
descriptor = gm.compute_wks(surface, cfg)            # runs steps 1-4 internally
```

```python
descriptor.shape
>>> (3988, 50)

descriptor[0]
>>> array([0.18905491, 0.19299358, 0.19670782, 0.2000639 , 0.20290116,
        ...
       0.07995553, 0.08174792, 0.08367599, 0.08569381, 0.08776426])
```

### Many descriptors, optionally in parallel

`compute_wks_batch` runs descriptor generation across surfaces using a process
pool, returning results in input order. `n_jobs=1` runs serially; `n_jobs=-1`
uses all available CPUs.

```python
surfaces = [gm.load_surface_npy(p) for p in paths]
descriptors = gm.compute_wks_batch(surfaces, cfg, n_jobs=-1, progress=True)
# list of (N_i, evals) arrays, in input order
```

### Codebook and Bag-of-Features

The descriptor stages above are sufficient on their own. To aggregate
per-vertex descriptors into fixed-length per-molecule vectors, fit a codebook
and encode against it:

```python
pool = gm.sample_descriptor_pool(descriptors, n_per_mol=1500)
codebook = gm.build_codebook(pool, n_codewords=1024)
vector = gm.knn_histogram(descriptor, codebook, knn=3)
```

```python
codebook.shape
>>> (1024, 50)
vector.shape
>>> (1024,)
```

### Individual stages: graph, eigenvalues, eigenvectors, descriptor

Each step returns a plain object you can inspect or reuse:

```python
surface = gm.load_surface_npy("mol.npy")
points = surface.augmented_points(elec_weight=0.3)  # (N, 4): [x, y, z, esp * weight]

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

### Convenience wrapper

`GEMolSG` holds the config and a fitted codebook so they are not passed
repeatedly. It composes the functions above and adds nothing load-bearing.

```python
model = gm.GEMolSG(cfg, n_codewords=1000, n_jobs=-1, progress=True)
model.fit_codebook(sample_surfaces)        # generates descriptors, then fits the codebook
vector = model.transform(surface)          # one surface  -> BoF vector
matrix = model.transform_many(surfaces)    # many surfaces -> BoF matrix
```

## Experiments

Install the additional dependencies required to run the experiment notebooks:

```bash
uv sync --extra experiments
```

To reproduce our experiments, run the notebooks inside `/experiments`. 
Note that some dependecies like `roshambo` and `oddt` cannot be run in recent 
Python versions. Hence the reproducibility of some experiments requires the creation 
of lower environment versions. We also recommend running the experiments in a cluster 
due to long processign times of molecular surfaces.

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
