"""
Utility functions: PDB loading, coordinate extraction, and output serialization.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import PDB

from .stability import StabilityReport


# ---------------------------------------------------------------------------
# PDB I/O
# ---------------------------------------------------------------------------

_PARSER = PDB.PDBParser(QUIET=True)


def load_pdb(path: str | Path) -> PDB.Structure.Structure:
    """Load a PDB file and return a BioPython Structure object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDB not found: {path}")
    return _PARSER.get_structure(path.stem, str(path))


def extract_ca_data(
    structure: PDB.Structure.Structure,
    chain_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract per-residue C-alpha coordinates, pLDDT scores, and residue names
    from a BioPython Structure.

    ESMFold stores pLDDT in the B-factor column. The API may return values in
    the 0–1 range rather than the canonical 0–100 range; this function detects
    and normalises both cases.

    Parameters
    ----------
    structure : BioPython Structure (as returned by load_pdb)
    chain_id  : specific chain to extract; if None, all chains are concatenated

    Returns
    -------
    ca_coords     : (N, 3) float array of CA coordinates in Å
    plddt_scores  : (N,)   float array of pLDDT values in 0–100 scale
    residue_names : list of 3-letter residue names, length N
    """
    model = structure[0]

    chains: List[PDB.Chain.Chain] = (
        [model[chain_id]] if chain_id is not None else list(model.get_chains())
    )

    ca_coords:     List[np.ndarray] = []
    plddt_scores:  List[float]      = []
    residue_names: List[str]        = []

    for chain in chains:
        for residue in chain.get_residues():
            # Skip HETATM (waters, ligands) — residue id[0] == ' ' for standard AA
            if residue.get_id()[0] != " ":
                continue
            if "CA" not in residue:
                continue

            ca = residue["CA"]
            ca_coords.append(ca.get_coord())

            b = float(ca.get_bfactor())
            # Detect 0–1 normalised pLDDT and scale to 0–100
            plddt_scores.append(b * 100.0 if b <= 1.0 else b)
            residue_names.append(residue.get_resname())

    if not ca_coords:
        raise ValueError(
            f"No CA atoms found in structure '{structure.get_id()}'. "
            "Check that the PDB contains standard ATOM records."
        )

    return (
        np.array(ca_coords,    dtype=float),
        np.array(plddt_scores, dtype=float),
        residue_names,
    )


def load_structures_from_dir(
    directory: str | Path,
    pattern: str = "*.pdb",
) -> List[Tuple[str, np.ndarray, np.ndarray, List[str]]]:
    """
    Load all PDB files matching *pattern* inside *directory*.

    Returns
    -------
    List of (pdb_path, ca_coords, plddt, residue_names) tuples, sorted by
    filename. Files that cannot be parsed are skipped with a warning.
    """
    directory = Path(directory)
    pdb_files = sorted(directory.glob(pattern))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files matching '{pattern}' in {directory}")

    results: List[Tuple[str, np.ndarray, np.ndarray, List[str]]] = []
    for pdb_path in pdb_files:
        try:
            structure = load_pdb(pdb_path)
            ca_coords, plddt, res_names = extract_ca_data(structure)
            print(
                f"  [load] {pdb_path.name}: {len(ca_coords)} residues, "
                f"avg pLDDT = {plddt.mean():.1f}"
            )
            results.append((str(pdb_path), ca_coords, plddt, res_names))
        except Exception as exc:
            print(f"  [warn] Skipping {pdb_path.name}: {exc}")

    return results


def group_by_length(
    entries: List[Tuple[str, np.ndarray, np.ndarray, List[str]]],
    tolerance: int = 5,
) -> List[List[Tuple[str, np.ndarray, np.ndarray, List[str]]]]:
    """
    Partition loaded structures into groups of similar residue count.

    Structures whose lengths differ by ≤ *tolerance* residues are placed in
    the same group (longest-first within each group).  Groups with ≥ 2 members
    can be analysed as an ensemble; single-member groups fall back to pLDDT mode.

    Parameters
    ----------
    entries   : output of load_structures_from_dir
    tolerance : maximum allowed residue-count difference within a group

    Returns
    -------
    List of groups; each group is a list of the same tuple format as entries.
    """
    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda e: len(e[1]), reverse=True)
    groups: List[List] = []

    for entry in sorted_entries:
        n = len(entry[1])
        placed = False
        for group in groups:
            ref_n = len(group[0][1])
            if abs(n - ref_n) <= tolerance:
                group.append(entry)
                placed = True
                break
        if not placed:
            groups.append([entry])

    return groups


# ---------------------------------------------------------------------------
# Output serialization
# ---------------------------------------------------------------------------

def report_to_dataframe(report: StabilityReport) -> pd.DataFrame:
    """Convert a StabilityReport's per-residue data to a pandas DataFrame."""
    rows = [
        {
            "residue_index":   r.residue_index,
            "residue_name":    r.residue_name,
            "plddt":           r.plddt,
            "rmsf_angstrom":   r.rmsf,
            "stability_score": r.stability_score,
            "classification":  r.classification.value,
        }
        for r in report.residues
    ]
    return pd.DataFrame(rows)


def report_to_dict(report: StabilityReport) -> dict:
    """Serialise a StabilityReport to a JSON-compatible dictionary."""
    return {
        "pdb_paths":    report.pdb_paths,
        "n_structures": report.n_structures,
        "mean_rmsd_angstrom": report.mean_rmsd,
        "pairwise_rmsd_matrix": (
            report.pairwise_rmsd_matrix.tolist()
            if report.pairwise_rmsd_matrix is not None
            else None
        ),
        "summary": {
            "n_residues": len(report.residues),
            "mean_plddt":   round(report.mean_plddt(), 2),
            "mean_rmsf":    round(report.mean_rmsf(), 4),
            "fraction_stable": round(report.fraction_stable(), 4),
            "n_stable":   report.n_stable,
            "n_moderate": report.n_moderate,
            "n_flexible": report.n_flexible,
        },
        "residues": [
            {
                "residue_index":   r.residue_index,
                "residue_name":    r.residue_name,
                "plddt":           round(r.plddt, 2),
                "rmsf_angstrom":   round(r.rmsf, 4),
                "stability_score": round(r.stability_score, 4),
                "classification":  r.classification.value,
            }
            for r in report.residues
        ],
        # Visualization-ready arrays (parallel lists for easy plotting)
        "visualization_arrays": {
            "residue_indices":  [r.residue_index  for r in report.residues],
            "plddt_values":     [round(r.plddt, 2)           for r in report.residues],
            "rmsf_values":      [round(r.rmsf, 4)            for r in report.residues],
            "stability_scores": [round(r.stability_score, 4) for r in report.residues],
            "classifications":  [r.classification.value      for r in report.residues],
        },
    }


def _json_default(obj):
    """JSON encoder fallback for numpy scalar types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: dict, output_path: str | Path) -> None:
    """Write *data* as a pretty-printed JSON file, auto-creating parent dirs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(data, fh, indent=2, default=_json_default)
    print(f"  [save] JSON -> {output_path}")


def save_csv(df: pd.DataFrame, output_path: str | Path) -> None:
    """Write *df* as a CSV file, auto-creating parent dirs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"  [save] CSV  -> {output_path}")


# ---------------------------------------------------------------------------
# StructureMetrics serialization
# ---------------------------------------------------------------------------

def structure_metrics_to_dict(metrics) -> dict:
    """Serialise a StructureMetrics object to a JSON-compatible dict."""
    return {
        "pdb_path":    metrics.pdb_path,
        "n_residues":  metrics.n_residues,
        "global": {
            "radius_of_gyration_angstrom": metrics.radius_of_gyration,
            "structural_entropy":          metrics.structural_entropy,
            "total_hbonds":                metrics.total_hbonds,
            "mean_sasa_angstrom2":         metrics.mean_sasa,
            "mean_packing_density":        metrics.mean_packing_density,
            "secondary_structure_source":  metrics.ss_source,
            "fraction_helix":  metrics.fraction_helix,
            "fraction_sheet":  metrics.fraction_sheet,
            "fraction_coil":   metrics.fraction_coil,
            "n_helix":  metrics.n_helix,
            "n_sheet":  metrics.n_sheet,
            "n_coil":   metrics.n_coil,
        },
        "per_residue": [
            {
                "residue_index":       r.residue_index,
                "residue_name":        r.residue_name,
                "secondary_structure": r.secondary_structure,
                "ss_label":            {"H": "helix", "E": "sheet", "C": "coil"}.get(r.secondary_structure, "coil"),
                "sasa_angstrom2":      round(r.sasa, 2),
                "packing_density":     r.packing_density,
                "contact_count":       r.contact_count,
                "mean_ca_distance_A":  round(r.mean_ca_distance, 3),
                "hbond_count":         r.hbond_count,
                "local_entropy":       round(r.local_entropy, 4),
            }
            for r in metrics.per_residue
        ],
        "hydrogen_bonds": [
            {
                "donor_index":    hb.donor_index,
                "acceptor_index": hb.acceptor_index,
                "energy_kcal_mol": hb.energy,
                "distance_H_O_A":  hb.distance_NH_O,
            }
            for hb in metrics.hbonds
        ],
        "contact_map":     metrics.contact_map.tolist(),
        "distance_matrix": metrics.distance_matrix.tolist(),
    }


def structure_metrics_to_dataframe(metrics) -> pd.DataFrame:
    """Convert per-residue StructureMetrics to a pandas DataFrame."""
    rows = [
        {
            "residue_index":       r.residue_index,
            "residue_name":        r.residue_name,
            "secondary_structure": r.secondary_structure,
            "sasa_angstrom2":      r.sasa,
            "packing_density":     r.packing_density,
            "contact_count":       r.contact_count,
            "mean_ca_distance_A":  r.mean_ca_distance,
            "hbond_count":         r.hbond_count,
            "local_entropy":       r.local_entropy,
        }
        for r in metrics.per_residue
    ]
    return pd.DataFrame(rows)
