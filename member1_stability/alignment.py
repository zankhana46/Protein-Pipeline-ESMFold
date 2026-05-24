"""
Structural alignment using the Kabsch algorithm.

Provides optimal superposition of CA-atom coordinate sets by minimising RMSD
via singular value decomposition. Handles proper rotations (det = +1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignmentResult:
    """Result of a Kabsch superposition."""
    aligned_coords: np.ndarray   # (N, 3) mobile coordinates after alignment
    rotation_matrix: np.ndarray  # (3, 3) optimal rotation R
    translation_vector: np.ndarray  # (3,) translation t such that aligned = R @ mobile + t
    rmsd: float                  # RMSD (Å) after alignment


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Compute the optimal rotation matrix R that minimises ||P @ R.T - Q||_F
    for two centred (N, 3) coordinate sets P and Q.

    Reflection-safe: the sign of the smallest singular value is corrected so
    det(R) = +1 (proper rotation, not improper).
    """
    H = P.T @ Q                          # (3, 3) cross-covariance matrix
    U, _S, Vt = np.linalg.svd(H)

    # Correct for reflections: ensure det(R) = +1
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, float(d)])

    R = (Vt.T @ D) @ U.T                # (3, 3)
    return R


def align(mobile: np.ndarray, reference: np.ndarray) -> AlignmentResult:
    """
    Superimpose *mobile* onto *reference* via Kabsch superposition.

    Both arrays must have shape (N, 3) and the same N.

    Parameters
    ----------
    mobile    : (N, 3) CA coordinates to be moved
    reference : (N, 3) CA coordinates treated as fixed

    Returns
    -------
    AlignmentResult
        aligned_coords  — mobile after optimal superposition onto reference
        rotation_matrix — 3×3 rotation R
        translation_vector — translation t (applied after rotation)
        rmsd            — post-alignment RMSD in Å
    """
    if mobile.shape != reference.shape:
        raise ValueError(
            f"Coordinate shape mismatch: mobile {mobile.shape} vs "
            f"reference {reference.shape}. Trim to equal length first."
        )

    mob_center = mobile.mean(axis=0)     # centroid of mobile
    ref_center = reference.mean(axis=0)  # centroid of reference

    P = mobile    - mob_center           # centred mobile
    Q = reference - ref_center           # centred reference

    R = _kabsch_rotation(P, Q)

    # Rotate centred mobile, then translate to reference centroid
    aligned = (R @ P.T).T + ref_center

    diff = aligned - reference
    rmsd = float(np.sqrt((diff ** 2).sum(axis=1).mean()))

    # Full translation: t such that aligned = R @ mobile + t
    t = ref_center - R @ mob_center

    return AlignmentResult(
        aligned_coords=aligned,
        rotation_matrix=R,
        translation_vector=t,
        rmsd=rmsd,
    )


def trim_to_common_length(
    coords_a: np.ndarray, coords_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trim both (N, 3) and (M, 3) arrays to the shorter length from the N-terminus.
    Required when comparing structures of differing residue counts.
    """
    n = min(len(coords_a), len(coords_b))
    return coords_a[:n], coords_b[:n]
