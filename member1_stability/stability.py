"""
Stability analysis: RMSD, RMSF, and per-residue stability scoring.

Handles two regimes:
  - Single structure  → pLDDT-to-RMSF approximation via B-factor physics
  - Ensemble (≥2)    → true RMSF from Kabsch-aligned superpositions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

from .alignment import align, trim_to_common_length


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class StabilityClass(str, Enum):
    STABLE   = "stable"    # RMSF ≤ 0.8 Å  — packed secondary structure
    MODERATE = "moderate"  # 0.8 < RMSF ≤ 2.0 Å — loops, surface helices
    FLEXIBLE = "flexible"  # RMSF > 2.0 Å  — disordered termini / linkers


@dataclass
class ResidueStability:
    residue_index:  int
    residue_name:   str
    plddt:          float  # ESMFold confidence, 0–100
    rmsf:           float  # Root-mean-square fluctuation, Å
    stability_score: float  # Exponential decay score ∈ [0, 1]; 1 = most stable
    classification: StabilityClass


@dataclass
class StabilityReport:
    """Full stability report for one protein (single or ensemble analysis)."""
    pdb_paths:            List[str]
    n_structures:         int
    pairwise_rmsd_matrix: Optional[np.ndarray]  # None when n_structures == 1
    mean_rmsd:            Optional[float]        # None when n_structures == 1
    residues:             List[ResidueStability]
    # Populated in __post_init__
    n_stable:   int = field(init=False)
    n_moderate: int = field(init=False)
    n_flexible: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_stable   = sum(1 for r in self.residues if r.classification == StabilityClass.STABLE)
        self.n_moderate = sum(1 for r in self.residues if r.classification == StabilityClass.MODERATE)
        self.n_flexible = sum(1 for r in self.residues if r.classification == StabilityClass.FLEXIBLE)

    def fraction_stable(self) -> float:
        n = len(self.residues)
        return self.n_stable / n if n else 0.0

    def mean_plddt(self) -> float:
        if not self.residues:
            return 0.0
        return float(np.mean([r.plddt for r in self.residues]))

    def mean_rmsf(self) -> float:
        if not self.residues:
            return 0.0
        return float(np.mean([r.rmsf for r in self.residues]))


# ---------------------------------------------------------------------------
# Low-level RMSD / RMSF computation
# ---------------------------------------------------------------------------

def compute_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """
    RMSD between two pre-aligned (N, 3) coordinate arrays (Å).
    Alignment is assumed to have been performed; use alignment.align() first.
    """
    diff = coords_a - coords_b
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


def compute_pairwise_rmsd(ca_coord_list: List[np.ndarray]) -> np.ndarray:
    """
    Build an (N × N) symmetric pairwise RMSD matrix for a list of structures.

    Each pair is:
      1. Trimmed to the shorter residue count.
      2. Aligned with Kabsch superposition.
      3. RMSD is computed on the aligned coordinates.

    Returns
    -------
    np.ndarray of shape (n_structures, n_structures), diagonal = 0.
    """
    n = len(ca_coord_list)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = trim_to_common_length(ca_coord_list[i], ca_coord_list[j])
            result = align(a, b)
            matrix[i, j] = result.rmsd
            matrix[j, i] = result.rmsd
    return matrix


def compute_rmsf_ensemble(ca_coord_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute per-residue RMSF from a structural ensemble.

    Algorithm
    ---------
    1. Trim all structures to the minimum residue count.
    2. Align every structure onto the first (reference) using Kabsch.
    3. Compute ensemble mean position per residue.
    4. RMSF_i = sqrt( mean_over_structures( ||r_i - <r_i>||² ) )

    Parameters
    ----------
    ca_coord_list : list of (N_k, 3) arrays, one per structure

    Returns
    -------
    np.ndarray of shape (M,) where M = min residue count; values in Å.
    """
    n_res = min(len(c) for c in ca_coord_list)
    trimmed = [c[:n_res] for c in ca_coord_list]

    ref = trimmed[0]
    aligned_ensemble = [ref]
    for mob in trimmed[1:]:
        result = align(mob, ref)
        aligned_ensemble.append(result.aligned_coords)

    stack    = np.stack(aligned_ensemble, axis=0)  # (n_struct, n_res, 3)
    mean_pos = stack.mean(axis=0)                  # (n_res, 3)
    sq_dev   = ((stack - mean_pos) ** 2).sum(axis=2)  # (n_struct, n_res)
    rmsf     = np.sqrt(sq_dev.mean(axis=0))        # (n_res,)
    return rmsf


def compute_rmsf_from_plddt(plddt: np.ndarray) -> np.ndarray:
    """
    Estimate per-residue RMSF from ESMFold pLDDT confidence scores.

    Physical basis
    --------------
    Crystallographic B-factors relate to atomic displacement via:
        B = (8π²/3) · RMSF²

    We estimate B from pLDDT with the empirical linear model:
        B_est = B_MAX · (1 - pLDDT/100)

    where B_MAX ≈ 80 Å² corresponds to a completely disordered residue.
    This gives:
        RMSF ≈ sqrt(3 · B_est / (8π²))

    Note: this is an approximation for single-structure analysis.
    Use compute_rmsf_ensemble() when ≥ 2 conformations are available.

    Parameters
    ----------
    plddt : (N,) array of pLDDT values in the 0–100 range

    Returns
    -------
    np.ndarray of shape (N,) with RMSF values in Å.
    """
    plddt_norm = np.clip(plddt, 0.0, 100.0) / 100.0
    B_MAX = 80.0  # Å²  — empirical upper bound for disordered residue
    B_est = B_MAX * (1.0 - plddt_norm)
    rmsf  = np.sqrt(np.maximum(3.0 * B_est / (8.0 * np.pi ** 2), 0.0))
    return rmsf


# ---------------------------------------------------------------------------
# Stability scoring and classification
# ---------------------------------------------------------------------------

# RMSF thresholds (Å) derived from typical MD simulation / experimental surveys
_RMSF_STABLE_CUTOFF   = 0.8   # Å — packed core / regular secondary structure
_RMSF_MODERATE_CUTOFF = 2.0   # Å — surface loops, termini


def _rmsf_to_stability_score(rmsf: np.ndarray) -> np.ndarray:
    """
    Map RMSF values to stability scores in [0, 1] (1 = most stable).

    Uses an exponential decay: score = exp(−rmsf / λ)
    where λ = 1.5 Å is the characteristic fluctuation length scale.
    """
    return np.clip(np.exp(-rmsf / 1.5), 0.0, 1.0)


def _classify_residue(rmsf_val: float) -> StabilityClass:
    if rmsf_val <= _RMSF_STABLE_CUTOFF:
        return StabilityClass.STABLE
    if rmsf_val <= _RMSF_MODERATE_CUTOFF:
        return StabilityClass.MODERATE
    return StabilityClass.FLEXIBLE


def build_residue_stability(
    plddt: np.ndarray,
    rmsf: np.ndarray,
    residue_names: Optional[List[str]] = None,
) -> List[ResidueStability]:
    """
    Combine pLDDT and RMSF arrays into a list of ResidueStability records.

    Parameters
    ----------
    plddt         : (N,) pLDDT values for up to N residues
    rmsf          : (M,) RMSF values; M may be ≤ N (e.g. after ensemble trim)
    residue_names : optional list of 3-letter residue names (length ≥ M)

    Returns
    -------
    List[ResidueStability] of length M.
    """
    n = len(rmsf)
    scores = _rmsf_to_stability_score(rmsf)
    names = (residue_names or [])[:n]

    records: List[ResidueStability] = []
    for i in range(n):
        records.append(ResidueStability(
            residue_index=i,
            residue_name=names[i] if i < len(names) else f"UNK",
            plddt=float(plddt[i]) if i < len(plddt) else 0.0,
            rmsf=float(rmsf[i]),
            stability_score=float(scores[i]),
            classification=_classify_residue(float(rmsf[i])),
        ))
    return records


# ---------------------------------------------------------------------------
# High-level analysis entry points
# ---------------------------------------------------------------------------

def analyze_single_structure(
    ca_coords: np.ndarray,
    plddt: np.ndarray,
    residue_names: Optional[List[str]] = None,
    pdb_path: str = "",
) -> StabilityReport:
    """
    Stability analysis for a single ESMFold structure.

    RMSF is approximated from pLDDT via the B-factor physics model
    (see compute_rmsf_from_plddt for derivation).

    Parameters
    ----------
    ca_coords     : (N, 3) C-alpha coordinates
    plddt         : (N,) per-residue pLDDT scores (0–100)
    residue_names : optional list of 3-letter residue codes
    pdb_path      : source PDB file path (for provenance)
    """
    rmsf = compute_rmsf_from_plddt(plddt)
    residues = build_residue_stability(plddt, rmsf, residue_names)
    return StabilityReport(
        pdb_paths=[pdb_path],
        n_structures=1,
        pairwise_rmsd_matrix=None,
        mean_rmsd=None,
        residues=residues,
    )


def analyze_ensemble(
    ca_coord_list: List[np.ndarray],
    plddt_list: List[np.ndarray],
    residue_names_list: Optional[List[List[str]]] = None,
    pdb_paths: Optional[List[str]] = None,
) -> StabilityReport:
    """
    Ensemble stability analysis for two or more structures of the same protein.

    Computes:
      - Pairwise RMSD matrix (all pairs, after Kabsch alignment)
      - Per-residue RMSF from aligned ensemble
      - Per-residue stability scores and classifications

    pLDDT is taken from the first (reference) structure for residue-level
    confidence annotation; RMSF drives the stability classification.

    Parameters
    ----------
    ca_coord_list      : list of (N_k, 3) CA coordinate arrays
    plddt_list         : list of (N_k,) pLDDT arrays, one per structure
    residue_names_list : optional list of residue-name lists, one per structure
    pdb_paths          : optional list of source PDB paths

    Returns
    -------
    StabilityReport
    """
    if len(ca_coord_list) < 2:
        raise ValueError("Ensemble analysis requires at least 2 structures.")

    rmsd_matrix = compute_pairwise_rmsd(ca_coord_list)
    upper_tri   = rmsd_matrix[np.triu_indices(len(ca_coord_list), k=1)]
    mean_rmsd   = float(upper_tri.mean()) if len(upper_tri) > 0 else 0.0

    rmsf    = compute_rmsf_ensemble(ca_coord_list)
    n_res   = len(rmsf)

    plddt_ref = plddt_list[0][:n_res]
    res_names = (residue_names_list[0][:n_res] if residue_names_list else None)

    residues = build_residue_stability(plddt_ref, rmsf, res_names)

    return StabilityReport(
        pdb_paths=pdb_paths or [],
        n_structures=len(ca_coord_list),
        pairwise_rmsd_matrix=rmsd_matrix,
        mean_rmsd=mean_rmsd,
        residues=residues,
    )
