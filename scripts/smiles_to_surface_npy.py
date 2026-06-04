"""SMILES to ge_molsg-compatible surface .npy.

Pipeline: SMILES -> 3-D conformer(s) -> molecular surface mesh (via an external
Node.js generator) -> Coulombic ESP on the mesh vertices -> object-array .npy
of ``[vertices, faces, esp]``.

Conformer generation supports two modes:
  * ``num_confs == 1`` (default): a single ETKDG embedding, MMFF-optimized.
  * ``num_confs > 1``: embed N conformers, MMFF-optimise each, keep the
    lowest-energy one (better geometry, more compute).

Note: the mesh step shells out to ``esp-surface-generator/mesh_from_stdin.js``
and therefore requires Node.js and that script to be present.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

PQR_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(PQR_FOLDER, exist_ok=True)


def vdw_radii_list(mol):
    pt = Chem.GetPeriodicTable()
    return [pt.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()]


def mol_to_pqr_block(
    mol,
    charges=None,
    resname: str = "MOL",
    chain: str = "_",
    resseq: int = 1,
) -> str:
    """Emit whitespace-separated PQR-like text for the mesh generator.

    Format per atom::

        ATOM<serial> <name> <resname> <chain> <resseq> <x> <y> <z> <charge> <radius>

    Heavy atoms are named ``<Element><running_index_within_element>`` (C1, C2,
    O3, ...); hydrogens are all named ``Hh``.
    """
    n = mol.GetNumAtoms()
    if n == 0:
        raise ValueError("Molecule has no atoms")
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer/coordinates")

    if charges is None:
        charges = [0.0] * n
    elif len(charges) != n:
        raise ValueError("charges must be None or length == number of atoms")

    radii = vdw_radii_list(mol)
    if len(radii) != n:
        raise ValueError("vdW radii length mismatch")

    conf = mol.GetConformer()
    elem_counts: dict[str, int] = {}
    chain = (chain or "_")[:1]
    resname = (resname or "MOL")[:3]

    lines = []
    for i, atom in enumerate(mol.GetAtoms()):
        serial = i + 1
        elem = atom.GetSymbol()
        if atom.GetAtomicNum() == 1:
            name = "Hh"
        else:
            elem_counts[elem] = elem_counts.get(elem, 0) + 1
            name = f"{elem}{elem_counts[elem]}"

        p = conf.GetAtomPosition(i)
        chg = float(charges[i])
        rad = float(radii[i])
        line = (
            f"ATOM{serial:>7d}"
            f"{name:>5s} {resname:<3s} {chain} {resseq:>3d}"
            f"{p.x:>12.3f}{p.y:>8.3f}{p.z:>8.3f}"
            f"{chg:>8.4f}{rad:>8.4f}"
        )
        lines.append(line)

    return "\n".join(lines) + "\n"


def pqr_block_to_mesh(pqr_text: str) -> str:
    """Run the Node.js mesh generator on a PQR block, returning its stdout."""
    p = subprocess.run(
        ["node", "esp-surface-generator/mesh_from_stdin.js"],
        input=pqr_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return p.stdout


def tmesh_to_numpy(mesh_text: str):
    """Parse the mesh generator's text output into (vertices, faces).

    Vertex lines have 9 whitespace-separated fields; the first three are XYZ.
    The integer after the vertex block is the face count; faces follow as
    repeating ``"3 0"`` headers each succeeded by three vertex-index lines.
    """
    lines = mesh_text.splitlines()

    vrows = []
    face_index = None
    nf = None
    for i in range(len(lines)):
        parts = lines[i].split(" ")
        if len(parts) == 9:
            vrows.append([float(x) for x in parts])
        elif vrows and len(parts) != 9:
            nf = int(lines[i])
            face_index = i
            break

    if face_index is None or nf is None:
        raise ValueError("Could not locate face-count line after vertices")

    vertices = np.asarray(vrows, dtype=np.float64)

    faces = np.empty((nf, 3), dtype=np.int32)
    f = 0
    i = face_index
    while i < len(lines) and f < nf:
        if lines[i] == "3 0":
            if i + 3 >= len(lines):
                raise ValueError("Face header '3 0' without enough index lines")
            faces[f] = (int(lines[i + 1]), int(lines[i + 2]), int(lines[i + 3]))
            f += 1
            i += 4
        else:
            i += 1

    if f != nf:
        raise ValueError(f"Decoded {f} faces, expected {nf}")

    return vertices[:, :3], faces


def coulombic_esp_on_mesh(
    atom_coords,
    surface_coords,
    partial_charges,
    max_radius: float = 20.0,
    esp_scale: float = 332.0,
    eps: float = 1e-12,
    return_float64: bool = True,
):
    """Coulombic ESP at mesh vertices: ``esp_scale * sum_i q_i / |r - r_i|``.

    Atoms beyond ``max_radius`` (A) of a vertex are excluded; zero charges are
    dropped for speed.
    """
    a = np.asarray(atom_coords, dtype=np.float64)
    s = np.asarray(surface_coords, dtype=np.float64)
    q = np.asarray(partial_charges, dtype=np.float64)

    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("atom_coords must be shaped (N, 3)")
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError("surface_coords must be shaped (M, 3)")
    if q.ndim != 1 or q.shape[0] != a.shape[0]:
        raise ValueError("partial_charges must be shaped (N,) and match atom_coords")

    nonzero = q != 0.0
    a = a[nonzero]
    q = q[nonzero]

    r2_cut = float(max_radius) ** 2
    diff = s[:, None, :] - a[None, :, :]
    d2 = np.einsum("mni,mni->mn", diff, diff)
    within = d2 <= r2_cut
    inv_r = 1.0 / (np.sqrt(d2) + eps)
    esp = (within * (q[None, :] * inv_r)).sum(axis=1) * float(esp_scale)
    if not return_float64:
        esp = esp.astype(np.float32, copy=False)
    return esp


def mmff94_partial_charges(mol, variant: str = "MMFF94"):
    """Return MMFF partial charges in the molecule's current atom order."""
    if mol.GetNumConformers() == 0:
        raise ValueError("mol has no conformers; provide a 3-D conformer first.")
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    if props is None:
        raise ValueError(f"MMFF parameters unavailable (variant={variant}).")
    return [props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]


def embed_mol(
    smiles: str,
    seed: int = 42,
    num_confs: int = 1,
    max_attempts: int = 50,
    maxiters: int = 500,
):
    """Parse SMILES, add Hs, embed conformer(s), MMFF-optimize.

    With ``num_confs == 1`` a single conformer is embedded and optimized. With
    ``num_confs > 1`` that many conformers are embedded, each MMFF-optimised,
    and only the lowest-energy conformer is retained on the returned molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.maxAttempts = int(max_attempts)
    params.numThreads = 1  # single-threaded: parallelism lives one level up

    if num_confs <= 1:
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError("3-D embedding failed.")
        if AllChem.MMFFOptimizeMolecule(mol, maxIters=maxiters) == -1:
            raise RuntimeError("MMFF force-field setup failed.")
        return mol

    cids = list(AllChem.EmbedMultipleConfs(mol, int(num_confs), params))
    if not cids:
        raise RuntimeError("3-D embedding failed for all conformers.")

    energies = []
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        raise RuntimeError("MMFF force-field setup failed.")
    for cid in cids:
        AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=maxiters)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        energies.append(ff.CalcEnergy())

    best = cids[int(np.argmin(energies))]
    best_conf = Chem.Conformer(mol.GetConformer(best))
    mol.RemoveAllConformers()
    mol.AddConformer(best_conf, assignId=True)
    return mol


def gen_mesh(mol):
    """Generate the surface mesh for a molecule with a 3-D conformer."""
    pqr_block = mol_to_pqr_block(mol)
    mesh = pqr_block_to_mesh(pqr_block)
    return tmesh_to_numpy(mesh)


def save_surface_npy(path, vertices, faces, esp) -> None:
    """Save ``[vertices, faces, esp]`` as a ge_molsg object-array .npy."""
    arr = np.empty(3, dtype=object)
    arr[0] = vertices
    arr[1] = faces
    arr[2] = esp
    np.save(str(path), arr)


def smiles_to_surface_npy(
    smiles: str,
    out: str | Path,
    seed: int = 42,
    num_confs: int = 1,
    max_attempts: int = 50,
    maxiters: int = 500,
) -> None:
    """SMILES -> conformer -> mesh -> ESP -> surface .npy."""
    mol = embed_mol(
        smiles, seed=seed, num_confs=num_confs,
        max_attempts=max_attempts, maxiters=maxiters,
    )
    verts, faces = gen_mesh(mol)

    conf = mol.GetConformer()
    atom_pos = np.asarray(conf.GetPositions(), dtype=np.float64)
    q = mmff94_partial_charges(mol)
    esp = coulombic_esp_on_mesh(atom_pos, verts[:, :3], q)

    save_surface_npy(out, verts[:, :3], faces, esp)
    print(f"Saved surface to {out}  ({verts.shape[0]} vertices, {faces.shape[0]} faces)")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a SMILES string to a ge_molsg-compatible .npy surface.",
    )
    parser.add_argument("--smiles", "-s", help="SMILES string (or omit to read stdin).")
    parser.add_argument("--out", "-o", default="mol.npy", help="Output .npy path.")
    parser.add_argument("--seed", type=int, default=42, help="Embedding RNG seed.")
    parser.add_argument("--num-confs", type=int, default=1,
                        help="Conformers to embed; >1 keeps the lowest-energy one.")
    parser.add_argument("--max-attempts", type=int, default=50,
                        help="ETKDG embedding attempts per conformer.")
    parser.add_argument("--maxiters", type=int, default=500,
                        help="MMFF optimization iterations per conformer.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    smiles = args.smiles
    if smiles is None:
        if sys.stdin.isatty():
            print("Enter SMILES: ", end="", flush=True)
        smiles = sys.stdin.readline().strip()
    if not smiles:
        print("Error: no SMILES provided.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing SMILES: {smiles}")
    smiles_to_surface_npy(
        smiles, out=args.out, seed=args.seed, num_confs=args.num_confs,
        max_attempts=args.max_attempts, maxiters=args.maxiters,
    )

    try:
        import ge_molsg as gm

        surface = gm.load_surface_npy(args.out)
        print(f"Verified with ge_molsg: {surface.n_vertices} vertices, "
              f"ESP range [{surface.esp.min():.2f}, {surface.esp.max():.2f}]")
    except ImportError:
        print("(ge_molsg not importable - skipping round-trip check)")


if __name__ == "__main__":
    main()
