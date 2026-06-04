from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
import numpy as np
import os

PQR_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(PQR_FOLDER, exist_ok=True)


def vdw_radii_list(mol: Chem.Mol):
    pt = Chem.GetPeriodicTable()
    return [pt.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()]

def load_pqr_block(mol) -> str:
    """
    Load a pre-formatted PQR file from a predefined folder using the
    molecule's name property (_Name) as the filename.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule with the '_Name' property set.

    Returns
    -------
    str
        The PQR file contents as a text block.
    """
    name = mol.GetProp("ID")
    if not name:
        raise ValueError("Molecule has no '_Name' property set.")

    pqr_path = os.path.join(PQR_FOLDER, f"{name}.pdb.pqr")

    if not os.path.isfile(pqr_path):
        raise FileNotFoundError(f"PQR file not found: {pqr_path}")

    with open(pqr_path, "r") as f:
        pqr_block = f.read()

    return pqr_block

def mol_to_pqr_block(
    mol: Chem.Mol,
    charges=None,
    resname: str = "MOL",
    chain: str = "_",
    resseq: int = 1,
) -> str:
    """
    Emit PQR-like text matching the requested format exactly (whitespace-separated):

    ATOM<serial> <name> <resname> <chain> <resseq> <x> <y> <z> <charge> <radius>

    Requirements implemented:
    - Record name is always "ATOM"
    - Atom serials are 1..N
    - Heavy atoms are named <Element><running_index_within_element>: C1, C2, O3, N4, ...
      (running indices are per element, not global)
    - All hydrogens are named exactly "Hh" (enumerated via serial, as in your example)
    - resname defaults to "MOL", chain defaults to "_", resseq defaults to 1
    - x/y/z printed with 3 decimals
    - charge/radius printed with 4 decimals
    """

    n = mol.GetNumAtoms()
    if n == 0:
        raise ValueError("Molecule has no atoms")
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer/coordinates")

    if charges is None:
        charges = [0.0] * n
    elif len(charges) != n:
        raise ValueError("charges must be None or a list of length equal to the number of atoms")

    radii = vdw_radii_list(mol)
    if len(radii) != n:
        raise ValueError("vdW radii length mismatch")

    conf = mol.GetConformer()

    # Running element counters for heavy-atom naming: C1,C2,... O1,O2,... etc.
    elem_counts = {}

    chain = (chain or "_")[:1]
    resname = (resname or "MOL")[:3]

    lines = []
    for i, atom in enumerate(mol.GetAtoms()):
        serial = i + 1
        elem = atom.GetSymbol()          # 'C', 'O', 'N', 'Cl', etc.
        anum = atom.GetAtomicNum()

        if anum == 1:
            name = "Hh"
        else:
            elem_counts[elem] = elem_counts.get(elem, 0) + 1
            name = f"{elem}{elem_counts[elem]}"

        p = conf.GetAtomPosition(i)
        chg = float(charges[i])
        rad = float(radii[i])

        # Spacing/alignment chosen to match your example closely
        line = (
            f"ATOM{serial:>7d}"
            f"{name:>5s} {resname:<3s} {chain} {resseq:>3d}"
            f"{p.x:>12.3f}{p.y:>8.3f}{p.z:>8.3f}"
            f"{chg:>8.4f}{rad:>8.4f}"
        )
        lines.append(line)

    return "\n".join(lines) + "\n"


def pqr_block_to_mesh(pqr_text: str) -> str:
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
    """
    Very simple parser using your rules:

    - Split by lines, then split each line by whitespace.
    - If len(parts) == 8 => it's a vertex coordinate line; collect floats.
    - Once vertex lines stop, the next non-empty line is the face-count integer (nf).
    - Then parse faces as repeating blocks:
        line == "3 0"  -> next 3 lines are vertex indices (one int per line)
      Repeat until nf faces are collected.

    Returns:
      vertices: (nv, 8) float64
      faces:    (nf, 3) int32
    """

    lines = [ln for ln in mesh_text.splitlines()]

    # ---- collect vertex rows (len == 8) ----
    vrows = []
    i = 0
    for i in range(len(lines)):
        parts = lines[i].split(" ")

        if len(parts) == 9:
            vrows.append([float(x) for x in parts])
            i += 1
        elif (len(vrows) > 0) and len(parts) != 9:
            nf = int(lines[i])
            face_index = i
            break
        else:
            continue


    vertices = np.asarray(vrows, dtype=np.float64)
    
    i += 1

    # ---- parse faces: look for '3 0' then take next 3 lines as indices ----
    faces = np.empty((nf, 3), dtype=np.int32)
    f = 0

    for i in range(face_index, len(lines)):
        if lines[i] == "3 0":
            if i + 3 >= len(lines):
                raise ValueError("Face header '3 0' found but not enough lines for indices")
            a = int(lines[i + 1])
            b = int(lines[i + 2])
            c = int(lines[i + 3])
            faces[f] = (a, b, c)
            f += 1
            i += 4
        else:
            i += 1

    if f != nf:
        raise ValueError(f"Decoded {f} faces, expected {nf}")

    return vertices[:,:3], faces


def coulombic_esp_on_mesh(
    atom_coords,
    surface_coords,
    partial_charges,
    max_radius=20.0,
    esp_scale=332.0,
    eps=1e-12,
    return_float64=True
):
    """
    Conventional Coulombic electrostatic potential (ESP) at mesh vertices:

        ESP(r) = esp_scale * sum_i ( q_i / |r - r_i| )

    using a cutoff: include atoms with |r-r_i| <= max_radius (Å).

    Parameters
    ----------
    atom_coords : (N, 3) array-like
        Atom XYZ coordinates in Å.
    surface_coords : (M, 3) array-like
        Mesh vertex XYZ coordinates in Å.
    partial_charges : (N,) array-like
        Partial charges in units of e.
    max_radius : float
        Cutoff radius in Å.
    esp_scale : float
        Scale factor (default 332.0). Set to 1.0 for unscaled q/r.
    eps : float
        Small value to avoid division by zero for coincident points.
    return_float64 : bool
        If True returns float64, else float32.

    Returns
    -------
    esp : (M,) np.ndarray
        ESP value at each mesh vertex.
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

    # Drop zero charges (matches the JS intent; speeds up computation)
    nonzero = q != 0.0
    a = a[nonzero]
    q = q[nonzero]

    r2_cut = float(max_radius) ** 2

    diff = s[:, None, :] - a[None, :, :]          # (M, N, 3)
    d2 = np.einsum("mni,mni->mn", diff, diff)     # (M, N)
    within = d2 <= r2_cut

    # conventional: q / r
    inv_r = 1.0 / (np.sqrt(d2) + eps)
    esp = (within * (q[None, :] * inv_r)).sum(axis=1)

    esp *= float(esp_scale)
    if not return_float64:
        esp = esp.astype(np.float32, copy=False)
    return esp

def mmff94_partial_charges(mol: Chem.Mol, *, conf_id: int = -1, variant: str = "MMFF94"):
    """
    Get MMFF94/MMFF94s partial charges for a molecule that already has a 3D conformer.

    Returns
    -------
    charges : list[float]
        MMFF partial charges in the current atom order of `mol`.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("mol has no conformers; provide a 3D conformer first.")

    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    if props is None:
        raise ValueError(f"MMFF parameters not available for this molecule (variant={variant}).")
    charges = [props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]
    return charges


def gen_mesh(
    mol,
    gen_charges=False,
):
    """
    Generate (or load) a molecular surface mesh and augment vertex features.

    Behavior:
    - Uses the RDKit InChIKey as filename. Saves/loads a NumPy .npz with 'verts' and 'faces'.
    - If the cache file exists, loads verts/faces and skips mesh generation.
    - If not, builds the mesh from the molecule, saves it, then augments features.

    Parameters
    ----------
    mol : RDKit Mol with 3D conformer
    charges : bool
        If True, append ESP (scaled) as an extra vertex column.
    logp : array_like or None
        Per-atom lipophilicity weights (length = num atoms). If provided, append lipophilicity potential.
    pharma : bool
        If True, append pharmacophore features projected to the surface.
    cache_dir : str
        Directory to store/load cached meshes.

    Returns
    -------
    verts : (M, 3 + extras) np.ndarray
        Vertex coordinates (and appended features if requested).
    faces : (F, 3) np.ndarray
        Triangle indices.
    """

    # load_pqr_block is for when ESP-DNN has been used and a PQR file exists in a folder
    # working towards integrating ESP-DNN into this single workflow if Tensorflow 1.X behaves

    pqr_block = mol_to_pqr_block(mol) #load_pqr_block(mol)
    mesh = pqr_block_to_mesh(pqr_block)
    verts, faces = tmesh_to_numpy(mesh)

    # Charges -> ESP field
    if gen_charges:
        conf = mol.GetConformer()
        atom_pos = np.asarray(conf.GetPositions(), dtype=np.float64)

        q = mmff94_partial_charges(mol, optimize=False)
        esp = coulombic_esp_on_mesh(atom_pos, verts[:, :3], q).reshape(-1, 1)
        verts = np.concatenate([verts, esp], axis=1)


    return verts, faces


# ---------------------------------------------------------------------------
# SMILES → .npy interface
# ---------------------------------------------------------------------------

def embed_mol(smiles: str, seed: int = 42) -> Chem.Mol:
    """Parse SMILES, add Hs, embed a 3-D conformer, and MMFF-optimise."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        raise RuntimeError("3-D embedding failed — try a different seed or SMILES.")
    ff_result = AllChem.MMFFOptimizeMolecule(mol)
    if ff_result == -1:
        raise RuntimeError("MMFF force-field setup failed.")
    return mol


def save_surface_npy(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    esp: np.ndarray,
) -> None:
    """Save [vertices, faces, esp] as a ge_molsg-compatible object-array .npy file."""
    arr = np.empty(3, dtype=object)
    arr[0] = vertices
    arr[1] = faces
    arr[2] = esp
    np.save(str(path), arr)
    print(f"Saved surface to {path}  ({vertices.shape[0]} vertices, {faces.shape[0]} faces)")


def smiles_to_surface_npy(
    smiles: str,
    out: str | Path,
    seed: int = 42,
) -> None:
    """Full pipeline: SMILES → 3-D conformer → mesh (via node) → ESP → .npy."""
    mol = embed_mol(smiles, seed=seed)
    verts, faces = gen_mesh(mol, gen_charges=False)

    # Compute ESP separately so we store it as a flat (N,) array
    conf = mol.GetConformer()
    atom_pos = np.asarray(conf.GetPositions(), dtype=np.float64)
    q = mmff94_partial_charges(mol)
    esp = coulombic_esp_on_mesh(atom_pos, verts[:, :3], q)

    save_surface_npy(out, verts[:, :3], faces, esp)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a SMILES string to a ge_molsg-compatible .npy surface file.",
    )
    parser.add_argument(
        "--smiles", "-s",
        help="SMILES string (or omit to read from stdin).",
    )
    parser.add_argument(
        "--out", "-o",
        default="mol.npy",
        help="Output .npy file path (default: mol.npy).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the 3-D conformer embedder (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
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
    smiles_to_surface_npy(smiles, out=args.out, seed=args.seed)

    try:
        import ge_molsg as gm
        surface = gm.load_surface_npy(args.out)
        print(f"Verified with ge_molsg: {surface.n_vertices} vertices, "
              f"ESP range [{surface.esp.min():.2f}, {surface.esp.max():.2f}]")
    except ImportError:
        print("(ge_molsg not importable — skipping round-trip check)")


if __name__ == "__main__":
    main()
