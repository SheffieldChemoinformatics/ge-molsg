"""Shared query resolution and DUDE-Z retrieval ranking utilities."""

import os
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.ML.Scoring import Scoring


def load_valid_surface_fns(surface_dir, pdb_dir):
    """Return surfaces with an RDKit-sanitizable same-stem PDB, in DUDE-Z order."""
    surface_dir = Path(surface_dir)
    pdb_dir = Path(pdb_dir)
    return [
        filename for filename in sorted(os.listdir(surface_dir), reverse=True)
        if filename.endswith(".npy")
        and Chem.MolFromPDBFile(
            str(pdb_dir / f"{filename[:-4]}.pdb"),
            removeHs=False,
            sanitize=True,
        ) is not None
    ]


def resolve_query_index(stems, query_id):
    """Resolve a query only by its exact ID or exact ``ligand_<ID>`` stem."""
    expected_stems = (f"ligand_{query_id}", query_id)
    matches = [index for index, stem in enumerate(stems) if stem in expected_stems]
    if len(matches) > 1:
        raise ValueError(f"Query ID {query_id!r} matches multiple candidate ligands")
    return matches[0] if matches else None


def dudez_duplicate_handling(similarities, labels, stems):
    """Keep the top-ranked filename per legacy two-character-stripped substring."""
    rankings = np.argsort(similarities)[::-1]
    ranked_labels = np.asarray(labels)[rankings]
    ranked_stems = np.asarray(stems)[rankings]
    rank_indices_to_delete = []
    score_indices_to_delete = []
    duplicate_stems = []

    for base in (stem[:-2] for stem in stems):
        matches = np.flatnonzero(np.char.find(ranked_stems, base) != -1)
        if len(matches) > 1:
            extras = matches[1:]
            rank_indices_to_delete.extend(extras)
            duplicate_stems.extend(ranked_stems[extras])
            score_indices_to_delete.extend(
                index for index, stem in enumerate(stems)
                if stem in ranked_stems[extras]
            )

    ranked_labels = np.delete(ranked_labels, rank_indices_to_delete)
    remaining_similarities = np.delete(similarities, score_indices_to_delete)
    return ranked_labels, remaining_similarities, len(set(duplicate_stems))


def retrieval_metrics(similarities, labels, stems, bedroc_alpha=20):
    """Calculate EF1% and BEDROC after the ElectroShape reference deduplication."""
    ranked_labels, remaining_similarities, removed_count = dudez_duplicate_handling(
        similarities, labels, stems
    )
    remaining_similarities[::-1].sort()
    ranked_scores = np.column_stack((remaining_similarities, ranked_labels))
    ef1 = Scoring.CalcEnrichment(ranked_scores, 1, [0.01])[0]
    bedroc = Scoring.CalcBEDROC(ranked_scores, 1, bedroc_alpha)
    return float(ef1), float(bedroc), removed_count
