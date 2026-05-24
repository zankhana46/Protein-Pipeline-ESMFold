"""
Member 1 — high-level analyzer interface for pipeline integration.

This module exposes StabilityAnalyzer, the integration point consumed by
the Member 4 fusion engine and any downstream module that needs per-residue
stability information.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .stability import (
    ResidueStability,
    StabilityClass,
    StabilityReport,
    analyze_ensemble,
    analyze_single_structure,
)
from .utils import (
    extract_ca_data,
    group_by_length,
    load_pdb,
    load_structures_from_dir,
    report_to_dataframe,
    report_to_dict,
    save_csv,
    save_json,
    structure_metrics_to_dataframe,
    structure_metrics_to_dict,
)
from .structure_analysis import StructureMetrics, run_structure_analysis

# Re-export so downstream code only needs to import from this module
__all__ = [
    "StabilityAnalyzer",
    "StabilityReport",
    "StabilityClass",
    "ResidueStability",
    "StructureMetrics",
]


class StabilityAnalyzer:
    """
    Top-level interface for Member 1 stability analysis.

    Accepts one or more ESMFold PDB files, performs Kabsch-alignment-based
    RMSD / RMSF computation (or pLDDT fallback for single structures), and
    returns structured StabilityReport objects.

    Parameters
    ----------
    output_dir       : directory where JSON / CSV artefacts are written
    chain_id         : chain to extract (default: first chain in model)
    group_tolerance  : max residue-count difference for ensemble grouping
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs/stability",
        chain_id: Optional[str] = None,
        group_tolerance: int = 5,
    ) -> None:
        self.output_dir     = Path(output_dir)
        self.chain_id       = chain_id
        self.group_tolerance = group_tolerance

    # ------------------------------------------------------------------
    # Primary analysis methods
    # ------------------------------------------------------------------

    def analyze_file(self, pdb_path: str | Path) -> StabilityReport:
        """
        Analyse a single PDB file using pLDDT-based RMSF approximation.

        Parameters
        ----------
        pdb_path : path to an ESMFold PDB file

        Returns
        -------
        StabilityReport
        """
        pdb_path = Path(pdb_path)
        structure = load_pdb(pdb_path)
        ca_coords, plddt, res_names = extract_ca_data(structure, chain_id=self.chain_id)
        return analyze_single_structure(
            ca_coords=ca_coords,
            plddt=plddt,
            residue_names=res_names,
            pdb_path=str(pdb_path),
        )

    def analyze_directory(
        self,
        directory: str | Path,
        pattern: str = "*.pdb",
    ) -> List[StabilityReport]:
        """
        Load all PDB files from *directory*, group by length, and run the
        appropriate analysis mode (ensemble vs single-structure) per group.

        Parameters
        ----------
        directory : path containing PDB files
        pattern   : glob to select files within the directory

        Returns
        -------
        List[StabilityReport] — one report per length group
        """
        entries = load_structures_from_dir(directory, pattern=pattern)
        groups  = group_by_length(entries, tolerance=self.group_tolerance)

        reports: List[StabilityReport] = []
        for group in groups:
            if len(group) == 1:
                pdb_path, ca_coords, plddt, res_names = group[0]
                report = analyze_single_structure(
                    ca_coords=ca_coords,
                    plddt=plddt,
                    residue_names=res_names,
                    pdb_path=pdb_path,
                )
            else:
                report = analyze_ensemble(
                    ca_coord_list=[e[1] for e in group],
                    plddt_list=[e[2] for e in group],
                    residue_names_list=[e[3] for e in group],
                    pdb_paths=[e[0] for e in group],
                )
            reports.append(report)

        return reports

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def save_report(
        self,
        report: StabilityReport,
        label: str = "stability",
    ) -> Tuple[Path, Path]:
        """
        Write *report* as JSON and CSV to self.output_dir.

        Returns
        -------
        (json_path, csv_path)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{label}_stability.json"
        csv_path  = self.output_dir / f"{label}_stability.csv"
        save_json(report_to_dict(report), json_path)
        save_csv(report_to_dataframe(report), csv_path)
        return json_path, csv_path

    # ------------------------------------------------------------------
    # Pipeline-facing accessors (consumed by Member 4 fusion engine)
    # ------------------------------------------------------------------

    def get_stable_residues(self, report: StabilityReport) -> List[int]:
        """Return residue indices classified as STABLE."""
        return [r.residue_index for r in report.residues if r.classification == StabilityClass.STABLE]

    def get_flexible_residues(self, report: StabilityReport) -> List[int]:
        """Return residue indices classified as FLEXIBLE."""
        return [r.residue_index for r in report.residues if r.classification == StabilityClass.FLEXIBLE]

    def get_stability_scores_array(self, report: StabilityReport) -> np.ndarray:
        """Return per-residue stability scores as a (N,) float array ∈ [0, 1]."""
        return np.array([r.stability_score for r in report.residues], dtype=float)

    def get_rmsf_array(self, report: StabilityReport) -> np.ndarray:
        """Return per-residue RMSF values as a (N,) float array in Å."""
        return np.array([r.rmsf for r in report.residues], dtype=float)

    def to_dict(self, report: StabilityReport) -> Dict[str, Any]:
        """Serialise *report* to a plain Python dict (JSON-compatible)."""
        return report_to_dict(report)

    # ------------------------------------------------------------------
    # Full structural analysis (algorithms 2-10)
    # ------------------------------------------------------------------

    def analyze_structure(
        self,
        pdb_path: str | Path,
        contact_cutoff: float = 8.0,
        packing_radius: float = 10.0,
        hbond_energy_cutoff: float = -0.5,
    ) -> StructureMetrics:
        """
        Run the full structural geometry and biophysics suite on one PDB file.

        Computes:
          - Distance matrix (NxN pairwise CA-CA)
          - Contact map (8 A binary, sequence-separated)
          - Secondary structure (DSSP or dihedral fallback)
          - SASA per residue (Shrake-Rupley)
          - Radius of gyration
          - Packing density per residue
          - Structural entropy (global + per-residue)
          - Hydrogen bond network (DSSP electrostatic criterion)

        Parameters
        ----------
        pdb_path             : path to ESMFold PDB file
        contact_cutoff       : CA-CA contact distance threshold (A)
        packing_radius       : neighbour-count sphere radius (A)
        hbond_energy_cutoff  : DSSP H-bond energy threshold (kcal/mol)

        Returns
        -------
        StructureMetrics
        """
        pdb_path  = Path(pdb_path)
        structure = load_pdb(pdb_path)
        ca_coords, _plddt, _res_names = extract_ca_data(structure, chain_id=self.chain_id)

        return run_structure_analysis(
            structure=structure,
            ca_coords=ca_coords,
            pdb_path=str(pdb_path),
            chain_id=self.chain_id,
            contact_cutoff=contact_cutoff,
            packing_radius=packing_radius,
            hbond_energy_cutoff=hbond_energy_cutoff,
        )

    def save_structure_metrics(
        self,
        metrics: StructureMetrics,
        label: str = "structure",
    ) -> Tuple[Path, Path]:
        """
        Write StructureMetrics as JSON and CSV to self.output_dir.

        Returns
        -------
        (json_path, csv_path)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{label}_geometry.json"
        csv_path  = self.output_dir / f"{label}_geometry.csv"
        save_json(structure_metrics_to_dict(metrics), json_path)
        save_csv(structure_metrics_to_dataframe(metrics), csv_path)
        return json_path, csv_path
