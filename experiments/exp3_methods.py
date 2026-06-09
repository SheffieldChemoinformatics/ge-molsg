"""Method machinery for Experiment 3 (GM-MolSG / GE-MolSG and baselines).

Bundles the patch-feature assembly and the two geo descriptor paths used by the
DUDE-Z comparison, plus the linear/concat similarity fusion. Patch covariance
extraction lives in ``patches.py`` (``compute_patches``) and VLAD encoding in
``vlad.py`` (``vlad_bof``); this module provides everything else so the
experiment notebook stays thin.

Two geo descriptor paths are provided and they are intentionally configured
differently:

* :func:`geo_wks_gm` — the geo block of the fused GM-MolSG descriptor.
* :func:`geo_wks_ge` — the standalone GE-MolSG descriptor.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import normalize as sk_normalize

import ge_molsg as gm

VALID_PATCH_FEATURES = ["esp", "logp", "sdc", "sdx", "sa",
                        "si", "mean_curv", "gauss_curv"]


# --------------------------------------------------------------------------- #
# Feature-list parsing                                                         #
# --------------------------------------------------------------------------- #
def parse_feature_list(spec, valid_set, block_name):
    tags = [t.strip() for t in spec.split(",") if t.strip()]
    for t in tags:
        if t not in valid_set:
            raise ValueError(f"Unknown {block_name} feature '{t}'. Valid: {valid_set}")
    return tags


# --------------------------------------------------------------------------- #
# Per-vertex surface feature channels                                          #
# --------------------------------------------------------------------------- #
def shape_index_curvatures(V, F):
    import igl
    _, _, PV1, PV2, _ = igl.principal_curvature(V, F)
    k1, k2 = PV1, PV2
    si = (2 / np.pi) * np.arctan((k1 + k2) / (k1 - k2 + 1e-8))
    mc = (k1 + k2) / 2
    gc = (k1 ** 2 + k2 ** 2) / 2
    return np.column_stack([si, mc, gc])


def create_logp_potential(mol, atom_pos, surf_coords, max_radius=6.0, sigma=2.0):
    from rdkit.Chem import Crippen
    atom_logp = np.array([c[0] for c in Crippen._GetAtomContribs(mol)], dtype=np.float64)
    coords = np.asarray(atom_pos, dtype=np.float64)
    surf = np.asarray(surf_coords, dtype=np.float64)
    neg_inv_2s2 = -1.0 / (2.0 * sigma * sigma)
    r2_max = max_radius * max_radius
    result = np.empty(len(surf), dtype=np.float64)
    for i0 in range(0, len(surf), 2048):
        i1 = min(i0 + 2048, len(surf))
        diff = surf[i0:i1, None, :] - coords[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        w = np.exp(d2 * neg_inv_2s2)
        w[d2 > r2_max] = 0.0
        result[i0:i1] = w @ atom_logp
    return result


def project_atom_properties_to_surface(atom_dict, atom_coords, surface_coords,
                                       max_radius=6.0, sigma=2.5):
    atom_coords = np.asarray(atom_coords, dtype=np.float64)
    surface_coords = np.asarray(surface_coords, dtype=np.float64)
    na = len(atom_dict)
    props = np.array([[atom_dict[i]["sdc"], atom_dict[i]["sdx"], atom_dict[i]["sa"]]
                      for i in range(na)], dtype=np.float64)
    neg_inv_2s2 = -1.0 / (2.0 * sigma * sigma)
    r2_max = max_radius * max_radius
    ns = len(surface_coords)
    out = np.empty((ns, 3), dtype=np.float64)
    for i0 in range(0, ns, 2048):
        i1 = min(i0 + 2048, ns)
        diff = surface_coords[i0:i1, None, :] - atom_coords[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        w = np.exp(d2 * neg_inv_2s2)
        w[d2 > r2_max] = 0.0
        out[i0:i1] = w @ props
    return out[:, 0], out[:, 1], out[:, 2]




def build_patch_feat(vertices, faces, charges_raw, rdmol, patch_fts,
                     elec_weight=0.3, 
                     logp_sigma=2.5, logp_radius=6.0,
                     jazzy_sigma=2.5, jazzy_radius=6.0):
    """Assemble the per-vertex patch feature matrix from the requested channels.

    Columns are ordered to match ``patch_fts`` (any subset of
    VALID_PATCH_FEATURES).
    """
    charges_raw = np.asarray(charges_raw).flatten()
    pool = {}
    need = set(patch_fts)

    charges_for_elec = charges_raw.copy()
    pool["esp"] = charges_for_elec.reshape(-1, 1) * 0.4

    conf = rdmol.GetConformer()
    atom_pos = conf.GetPositions()
    logp_raw = create_logp_potential(rdmol, atom_pos, vertices,
                                     max_radius=logp_radius, sigma=logp_sigma).flatten()
    logp_for_lipo = logp_raw.copy()
    pool["logp"] = logp_for_lipo.reshape(-1, 1)

    si_arr = shape_index_curvatures(vertices, faces)
    pool["si"] = si_arr[:, 0:1]
    pool["mean_curv"] = si_arr[:, 1:2]
    pool["gauss_curv"] = si_arr[:, 2:3]

    if need & {"sdc", "sdx", "sa"}:
        from jazzy.core import (
            kallisto_molecule_from_rdkit_molecule,
            calculate_polar_strength_map,
            get_covalent_atom_idxs,
            get_charges_from_kallisto_molecule,
        )
        anb = get_covalent_atom_idxs(rdmol)
        km = kallisto_molecule_from_rdkit_molecule(rdmol)
        kc = get_charges_from_kallisto_molecule(km, 0)
        am = calculate_polar_strength_map(rdmol, km, anb, kc)
        sf, xf, af = project_atom_properties_to_surface(
            am, atom_pos, vertices, max_radius=jazzy_radius, sigma=jazzy_sigma)
        pool["sdc"] = np.log1p(sf.reshape(-1, 1))
        pool["sdx"] = np.log1p(xf.reshape(-1, 1))
        pool["sa"] = np.log1p(af.reshape(-1, 1))


    for key in pool:
        assert not np.isnan(pool[key]).any() and not np.isinf(pool[key]).any(), \
            f"build_patch_feat: NaN/inf in channel '{key}'"

    missing = [t for t in patch_fts if t not in pool]
    if missing:
        raise ValueError(f"build_patch_feat: unsupported patch channel(s): {missing}")

    patch_feat = np.hstack([pool[t] for t in patch_fts])
    assert not np.isnan(patch_feat).any() and not np.isinf(patch_feat).any(), \
        "build_patch_feat: NaN/inf in assembled patch_feat"
    return patch_feat


# --------------------------------------------------------------------------- #
# Geo descriptor paths                                                         #
# --------------------------------------------------------------------------- #
def _geo_wks(graph_features, n_components, evals, variance, nn, laplacian_type,
             square_distances):
    W = gm.compute_affinity(
        graph_features, n_neighbors=nn, backend="ckdtree",
        adaptive_bw=True, square_distances=square_distances,
    )
    L = gm.graph_laplacian(W, laplacian_type=laplacian_type)
    eigensystem = gm.compute_eigenpairs(
        L, n_components=n_components, drop_first=True, eigensolver="arpack",
    )
    return gm.wks(eigensystem, evals=evals, variance=variance, l2=True)


def geo_wks_gm(graph_features, n_components, evals, variance, nn, laplacian_type):
    return _geo_wks(graph_features, n_components, evals, variance, nn,
                    laplacian_type, square_distances=True)


def geo_wks_ge(graph_features, n_components, evals, variance, nn, laplacian_type):
    return _geo_wks(graph_features, n_components, evals, variance, nn,
                    laplacian_type, square_distances=False)


# --------------------------------------------------------------------------- #
# Geo BoF histogram (hard k-NN, global-sigma softmax mask; L1)                 #
# --------------------------------------------------------------------------- #
def knn_histogram(descriptors, codebook, knn=3, norm="l1"):
    sigma = np.sqrt(2 * np.median(pdist(codebook, "euclidean")))
    sq_dist = cdist(codebook, descriptors, metric="sqeuclidean").T
    softmax = np.exp(-0.5 * sq_dist / (sigma ** 2 + 1e-12))
    nn_idx = np.argsort(sq_dist, axis=1)[:, :knn]
    mask = np.zeros_like(softmax)
    for row, ix in zip(mask, nn_idx):
        row[ix] = 1.0
    return sk_normalize(mask.sum(axis=0).reshape(1, -1), norm=norm)[0]


# --------------------------------------------------------------------------- #
# Similarity fusion                                                            #
# --------------------------------------------------------------------------- #
def cosine_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    Xn = X / n
    return Xn @ Xn.T


def linear_fusion(S_geo, S_vlad, alpha):
    return alpha * S_geo + (1.0 - alpha) * S_vlad
