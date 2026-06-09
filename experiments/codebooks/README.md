# Experiment codebooks

Pre-fitted codebooks for all experiments. Load with `np.load(..., allow_pickle=True)`.

## Experiment 1 — GE-MolSG vs MolSG
- `ge_molsg_cb.npy` — GE-MolSG geo codebook (1000 × 100; k-means centres over WKS descriptors).
- `molsg.npy`       — MolSG baseline codebook (500 × 100; as used by the MolSG results-gen script).

## Experiment 2 — GE-MolSG vs ElectroShape vs ESP-Sim
- `ge_molsg_cb.npy` — GE-MolSG geo codebook (same file as Experiment 1).

## Experiment 3 — GM-MolSG vs baselines
- `ge_molsg_cb.npy`  — GE-MolSG geo codebook (standalone GE-MolSG method).
- `gm_geo_cb.npy`    — GM-MolSG geo codebook (1000 × 100; geo block of the fused descriptor).
- `gm_patch_cb.npy`  — GM-MolSG surface-patch VLAD codebook (128 × 36).

## Experiment 4 — GE-MolSG vs MMFF94
- `ge_molsg_cb.npy` — GE-MolSG geo codebook (same file as Experiments 1–3).
- `mmff94_cb.npy`   — MMFF94 codebook (1000 × 100; seeded_mmff94_precomp_bot_paper,
                       100eigs_100evals_15var_100knn_normalized_lap_0.3ew_400ss_1000cw).

