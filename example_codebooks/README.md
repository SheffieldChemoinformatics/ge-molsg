# Example codebooks

Per-target Bag-of-Features codebooks used to reproduce the paper's retrieval
results. Each file is a NumPy array of shape `(n_codewords, descriptor_dim)`
named `<target>.npy` (e.g. `ampc.npy`, `xiap.npy`).

`scripts/evaluate_target.py` loads `<target>.npy` from this folder by default.
If a target's codebook is absent, the script builds one from the downloaded
dataset and saves it here.
