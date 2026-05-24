"""
Member 1 - Structural geometry and biophysics analysis.

Implements algorithms 2-10 from the analysis suite:
  2.  Distance Matrix        — NxN pairwise CA-CA distances
  3.  Contact Map            — binary residue contacts at 8 A cutoff
  4.  Secondary Structure    — DSSP when binary available, phi/psi fallback
  5.  B-factor / pLDDT       — already in stability.py; reused here for context
  6.  SASA                   — Shrake-Rupley solvent accessibility
  7.  Radius of Gyration     — compactness metric
  8.  Packing Density        — local neighbour count per residue
  9.  Structural Entropy     — Shannon entropy of contact distribution
  10. Hydrogen Bond Network  — DSSP electrostatic energy criterion (backbone)

Kabsch alignment (#1) lives in alignment.py.
pLDDT -> RMSF (#5 B-factor proxy) lives in stability.py.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio import PDB
from Bio.PDB.SASA import ShrakeRupley


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

SS_LABELS = {"H": "helix", "E": "sheet", "C": "coil"}


@dataclass
class HBond:
    donor_index:    int    # residue index of N-H donor
    acceptor_index: int    # residue index of C=O acceptor
    energy:         float  # DSSP energy (kcal/mol); more negative = stronger
    distance_NH_O:  float  # H...O distance in Angstroms (estimated)


@dataclass
class PerResidueMetrics:
    residue_index:         int
    residue_name:          str
    secondary_structure:   str   # 'H', 'E', or 'C'
    sasa:                  float  # Angstrom^2; -1 if unavailable
    packing_density:       int    # CA neighbours within 10 A (seq-non-local)
    contact_count:         int    # contacts within 8 A (|i-j| >= 4)
    mean_ca_distance:      float  # mean distance to all other CA atoms (A)
    hbond_count:           int    # backbone H-bonds this residue participates in
    local_entropy:         float  # Shannon entropy of this residue's contact row


@dataclass
class StructureMetrics:
    """All structural geometry and biophysics metrics for one structure."""
    pdb_path:            str
    n_residues:          int

    # Global scalars
    radius_of_gyration:  float   # Angstroms
    structural_entropy:  float   # Shannon entropy of full contact distribution
    total_hbonds:        int
    mean_sasa:           float   # Angstrom^2
    mean_packing_density: float
    fraction_helix:      float   # 0-1
    fraction_sheet:      float
    fraction_coil:       float
    ss_source:           str     # 'dssp' or 'dihedral_fallback'

    # Per-residue
    per_residue: List[PerResidueMetrics]
    hbonds:      List[HBond]

    # Arrays (NxN)
    distance_matrix: np.ndarray   # shape (N, N)
    contact_map:     np.ndarray   # shape (N, N), binary

    # Computed in __post_init__
    n_helix: int = field(init=False)
    n_sheet: int = field(init=False)
    n_coil:  int = field(init=False)

    def __post_init__(self) -> None:
        self.n_helix = sum(1 for r in self.per_residue if r.secondary_structure == "H")
        self.n_sheet = sum(1 for r in self.per_residue if r.secondary_structure == "E")
        self.n_coil  = sum(1 for r in self.per_residue if r.secondary_structure == "C")


# ---------------------------------------------------------------------------
# 2. Distance Matrix
# ---------------------------------------------------------------------------

def compute_distance_matrix(ca_coords: np.ndarray) -> np.ndarray:
    """
    Build the NxN pairwise Euclidean distance matrix for CA coordinates.
    Uses broadcasting for efficiency; O(N^2) memory.

    Returns
    -------
    np.ndarray of shape (N, N), symmetric, diagonal = 0.
    """
    # diff[i,j] = ca_coords[i] - ca_coords[j]
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]  # (N, N, 3)
    return np.sqrt((diff ** 2).sum(axis=2))                            # (N, N)


# ---------------------------------------------------------------------------
# 3. Contact Map
# ---------------------------------------------------------------------------

def compute_contact_map(
    ca_coords: np.ndarray,
    cutoff: float = 8.0,
    min_seq_separation: int = 4,
) -> np.ndarray:
    """
    Binary NxN contact map using CA-CA distances.

    contacts[i, j] = 1  iff  dist(i, j) <= cutoff  AND  |i-j| >= min_seq_separation

    The sequence-separation filter removes trivially bonded neighbours
    (i+1, i+2, i+3) so the map reflects non-local structural contacts only.

    Parameters
    ----------
    ca_coords           : (N, 3) CA coordinates
    cutoff              : distance threshold in Angstroms (default 8.0)
    min_seq_separation  : minimum |i-j| to include (default 4)

    Returns
    -------
    np.ndarray of shape (N, N), dtype int8.
    """
    dist = compute_distance_matrix(ca_coords)
    within_cutoff = dist <= cutoff

    # Mask close sequence neighbours
    n = len(ca_coords)
    seq_sep = np.abs(np.arange(n)[:, np.newaxis] - np.arange(n)[np.newaxis, :])
    mask = seq_sep >= min_seq_separation

    contacts = (within_cutoff & mask).astype(np.int8)
    return contacts


# ---------------------------------------------------------------------------
# 4. Secondary Structure Assignment
# ---------------------------------------------------------------------------

def _compute_phi_psi(
    chain: PDB.Chain.Chain,
) -> List[Tuple[Optional[float], Optional[float]]]:
    """
    Compute backbone phi and psi dihedral angles for all residues in a chain.
    Returns a list of (phi, psi) in degrees; None where not computable
    (first/last residues or missing atoms).
    """
    residues = [r for r in chain.get_residues() if r.get_id()[0] == " "]
    phi_psi  = []

    for i, res in enumerate(residues):
        phi = psi = None
        try:
            # phi: C(i-1) - N(i) - CA(i) - C(i)
            if i > 0:
                prev = residues[i - 1]
                if all(a in prev for a in ("C",)) and all(a in res for a in ("N", "CA", "C")):
                    v1 = prev["C"].get_vector()
                    v2 = res["N"].get_vector()
                    v3 = res["CA"].get_vector()
                    v4 = res["C"].get_vector()
                    phi = float(PDB.calc_dihedral(v1, v2, v3, v4)) * 180.0 / np.pi

            # psi: N(i) - CA(i) - C(i) - N(i+1)
            if i < len(residues) - 1:
                nxt = residues[i + 1]
                if all(a in res for a in ("N", "CA", "C")) and "N" in nxt:
                    v1 = res["N"].get_vector()
                    v2 = res["CA"].get_vector()
                    v3 = res["C"].get_vector()
                    v4 = nxt["N"].get_vector()
                    psi = float(PDB.calc_dihedral(v1, v2, v3, v4)) * 180.0 / np.pi
        except Exception:
            pass

        phi_psi.append((phi, psi))

    return phi_psi


def _classify_ramachandran(phi: Optional[float], psi: Optional[float]) -> str:
    """
    Assign secondary structure from phi/psi angles using Ramachandran regions.

    Regions (conservative, based on Hollingsworth & Karplus 2010):
      Helix  (H): phi in [-100, -30],  psi in [-70,  30]
      Sheet  (E): phi in [-180, -90],  psi in [ 90, 180] or [-180, -150]
      Coil   (C): all other regions
    """
    if phi is None or psi is None:
        return "C"
    if -100.0 <= phi <= -30.0 and -70.0 <= psi <= 30.0:
        return "H"
    if phi <= -90.0 and (psi >= 90.0 or psi <= -150.0):
        return "E"
    return "C"


def assign_secondary_structure(
    structure: PDB.Structure.Structure,
    chain_id: Optional[str] = None,
) -> Tuple[List[str], str]:
    """
    Assign per-residue secondary structure.

    Tries BioPython's DSSP wrapper first (requires the `mkdssp` binary).
    Falls back to phi/psi Ramachandran classification if the binary is absent.

    Parameters
    ----------
    structure : BioPython Structure
    chain_id  : chain to analyse (default: first chain)

    Returns
    -------
    ss_list : list of 'H', 'E', or 'C' for each residue
    source  : 'dssp' or 'dihedral_fallback'
    """
    model = structure[0]
    chain = (
        model[chain_id]
        if chain_id is not None
        else list(model.get_chains())[0]
    )

    residues = [r for r in chain.get_residues() if r.get_id()[0] == " " and "CA" in r]

    # -- Attempt DSSP --------------------------------------------------------
    try:
        dssp = PDB.DSSP(model, structure.header.get("name", "tmp") or "tmp")
        ss_list = []
        for res in residues:
            key = (chain.get_id(), res.get_id())
            if key in dssp:
                raw = dssp[key][2]          # 1-letter SS code
                # Normalise DSSP's 8-state to 3-state: H/G/I -> H; E/B -> E; rest -> C
                if raw in ("H", "G", "I"):
                    ss_list.append("H")
                elif raw in ("E", "B"):
                    ss_list.append("E")
                else:
                    ss_list.append("C")
            else:
                ss_list.append("C")
        return ss_list, "dssp"
    except Exception:
        pass

    # -- Fallback: dihedral-based Ramachandran classification ----------------
    phi_psi = _compute_phi_psi(chain)
    n_res_chain = len([r for r in chain.get_residues() if r.get_id()[0] == " "])

    # Align phi_psi length to residues with CA
    # (some residues may lack CA and were excluded)
    ss_list = []
    pp_iter = iter(phi_psi)
    for res in residues:
        try:
            phi, psi = next(pp_iter)
        except StopIteration:
            phi, psi = None, None
        ss_list.append(_classify_ramachandran(phi, psi))

    return ss_list, "dihedral_fallback"


# ---------------------------------------------------------------------------
# 6. SASA — Solvent Accessible Surface Area
# ---------------------------------------------------------------------------

def compute_sasa(
    structure: PDB.Structure.Structure,
    chain_id: Optional[str] = None,
    probe_radius: float = 1.4,
) -> np.ndarray:
    """
    Compute per-residue Solvent Accessible Surface Area (A^2) using the
    Shrake-Rupley rolling sphere algorithm (BioPython 1.80+).

    Parameters
    ----------
    structure    : BioPython Structure
    chain_id     : chain to extract (default: first chain)
    probe_radius : solvent probe radius in A (default 1.4 A = water)

    Returns
    -------
    np.ndarray of shape (N,) with per-residue SASA in A^2.
    """
    sr = ShrakeRupley(probe_radius=probe_radius)
    sr.compute(structure, level="R")

    model = structure[0]
    chain = (
        model[chain_id]
        if chain_id is not None
        else list(model.get_chains())[0]
    )

    sasa_vals = []
    for res in chain.get_residues():
        if res.get_id()[0] != " " or "CA" not in res:
            continue
        sasa_vals.append(float(res.sasa))

    return np.array(sasa_vals, dtype=float)


# ---------------------------------------------------------------------------
# 7. Radius of Gyration
# ---------------------------------------------------------------------------

def compute_radius_of_gyration(ca_coords: np.ndarray) -> float:
    """
    Mass-weighted radius of gyration using CA atoms as equal-mass pseudoatoms.

    Rg = sqrt( mean( |r_i - r_centroid|^2 ) )

    Returns
    -------
    float in Angstroms.
    """
    centroid = ca_coords.mean(axis=0)
    sq_dist  = ((ca_coords - centroid) ** 2).sum(axis=1)
    return float(np.sqrt(sq_dist.mean()))


# ---------------------------------------------------------------------------
# 8. Packing Density
# ---------------------------------------------------------------------------

def compute_packing_density(
    ca_coords: np.ndarray,
    radius: float = 10.0,
    min_seq_separation: int = 4,
) -> np.ndarray:
    """
    Per-residue local packing density: number of CA neighbours within *radius*,
    excluding trivially bonded neighbours (|i-j| < min_seq_separation).

    Parameters
    ----------
    ca_coords           : (N, 3) array
    radius              : sphere radius in Angstroms (default 10.0)
    min_seq_separation  : exclude close sequence neighbours (default 4)

    Returns
    -------
    np.ndarray of shape (N,) with integer neighbour counts.
    """
    dist  = compute_distance_matrix(ca_coords)
    n     = len(ca_coords)
    seq_sep = np.abs(np.arange(n)[:, np.newaxis] - np.arange(n)[np.newaxis, :])
    within  = (dist < radius) & (seq_sep >= min_seq_separation)
    return within.sum(axis=1).astype(float)


# ---------------------------------------------------------------------------
# 9. Structural Entropy
# ---------------------------------------------------------------------------

def compute_structural_entropy(
    contact_map: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Compute global and per-residue structural entropy from the contact map.

    Global entropy: Shannon entropy of the contact probability distribution
    over all (i,j) pairs (upper triangle).

    Per-residue entropy: for each row i, treat contact_map[i, :] as a
    probability vector and compute its Shannon entropy.

    Parameters
    ----------
    contact_map : (N, N) binary array (output of compute_contact_map)

    Returns
    -------
    global_entropy   : float — overall structural disorder metric
    per_res_entropy  : (N,) float array — local disorder per residue
    """
    n = contact_map.shape[0]

    # -- Global entropy ------------------------------------------------------
    # Use the contact map's row-wise contact probabilities
    row_sums = contact_map.sum(axis=1).astype(float)
    total = row_sums.sum()
    if total > 0:
        prob = row_sums / total
        prob = prob[prob > 0]
        global_entropy = float(-np.sum(prob * np.log2(prob)))
    else:
        global_entropy = 0.0

    # -- Per-residue entropy -------------------------------------------------
    per_res_entropy = np.zeros(n, dtype=float)
    for i in range(n):
        row = contact_map[i].astype(float)
        s   = row.sum()
        if s > 0:
            p = row[row > 0] / s
            # abs() removes IEEE -0.0 (occurs when p=[1.0], log2(1)=0)
            per_res_entropy[i] = abs(float(-np.sum(p * np.log2(p))))

    return global_entropy, per_res_entropy


# ---------------------------------------------------------------------------
# 10. Hydrogen Bond Network (DSSP electrostatic criterion)
# ---------------------------------------------------------------------------

def _estimate_h_position(
    n_coord: np.ndarray,
    c_prev_coord: np.ndarray,
    bond_length: float = 1.0,
) -> np.ndarray:
    """
    Estimate amide hydrogen position from N and the preceding carbonyl carbon.
    H ≈ N + unit(N - C_prev) * bond_length
    """
    direction = n_coord - c_prev_coord
    norm      = np.linalg.norm(direction)
    if norm < 1e-6:
        return n_coord + np.array([0.0, 0.0, bond_length])
    return n_coord + (direction / norm) * bond_length


def _dssp_hbond_energy(
    n_i: np.ndarray,   # donor nitrogen
    h_i: np.ndarray,   # estimated amide H
    c_j: np.ndarray,   # acceptor carbonyl C
    o_j: np.ndarray,   # acceptor carbonyl O
) -> float:
    """
    DSSP backbone H-bond electrostatic energy (kcal/mol).

    E = 0.084 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 332

    A bond is accepted if E < -0.5 kcal/mol.
    All distances must be > 0.5 A to avoid numerical singularities.
    """
    FACTOR = 0.084 * 332.0  # kcal/mol·A

    def safe_inv(a: np.ndarray, b: np.ndarray) -> float:
        d = float(np.linalg.norm(a - b))
        return 1.0 / d if d > 0.5 else 0.0

    e = FACTOR * (
        safe_inv(o_j, n_i) +
        safe_inv(c_j, h_i) -
        safe_inv(o_j, h_i) -
        safe_inv(c_j, n_i)
    )
    return e


def detect_hydrogen_bonds(
    structure: PDB.Structure.Structure,
    chain_id: Optional[str] = None,
    energy_cutoff: float = -0.5,
) -> List[HBond]:
    """
    Detect backbone N-H...O=C hydrogen bonds using the DSSP electrostatic
    criterion. No external binary required — uses backbone atom coordinates.

    A bond is recorded when E(i -> j) < energy_cutoff (default -0.5 kcal/mol),
    with minimum sequence separation |i-j| >= 2 (DSSP standard).

    Parameters
    ----------
    structure      : BioPython Structure
    chain_id       : chain to analyse (default: first chain)
    energy_cutoff  : DSSP energy threshold in kcal/mol (default -0.5)

    Returns
    -------
    List[HBond] sorted by energy (strongest first).
    """
    model  = structure[0]
    chain  = (
        model[chain_id]
        if chain_id is not None
        else list(model.get_chains())[0]
    )

    # Collect backbone atoms per residue (standard AA only)
    residues = [r for r in chain.get_residues() if r.get_id()[0] == " "]
    bb: List[Optional[Dict[str, np.ndarray]]] = []

    for res in residues:
        atoms = {}
        for name in ("N", "CA", "C", "O"):
            if name in res:
                atoms[name] = res[name].get_coord().astype(float)
        bb.append(atoms if len(atoms) == 4 else None)

    hbonds: List[HBond] = []

    for i, donor_bb in enumerate(bb):
        if donor_bb is None:
            continue
        # Estimate H position using preceding C
        if i > 0 and bb[i - 1] is not None:
            h_i = _estimate_h_position(donor_bb["N"], bb[i - 1]["C"])
        else:
            continue  # cannot estimate H for first residue

        for j, acc_bb in enumerate(bb):
            if acc_bb is None or abs(i - j) < 2:
                continue  # skip self and immediate neighbours

            e = _dssp_hbond_energy(
                donor_bb["N"], h_i,
                acc_bb["C"],   acc_bb["O"],
            )
            if e < energy_cutoff:
                h_to_o = float(np.linalg.norm(h_i - acc_bb["O"]))
                hbonds.append(HBond(
                    donor_index=i,
                    acceptor_index=j,
                    energy=round(e, 4),
                    distance_NH_O=round(h_to_o, 3),
                ))

    hbonds.sort(key=lambda b: b.energy)
    return hbonds


# ---------------------------------------------------------------------------
# Orchestrator: run all analyses
# ---------------------------------------------------------------------------

def run_structure_analysis(
    structure: PDB.Structure.Structure,
    ca_coords: np.ndarray,
    pdb_path: str = "",
    chain_id: Optional[str] = None,
    contact_cutoff: float = 8.0,
    packing_radius: float = 10.0,
    sasa_probe: float = 1.4,
    hbond_energy_cutoff: float = -0.5,
) -> StructureMetrics:
    """
    Run the full Member 1 structural analysis suite on one ESMFold structure.

    Executes (in order):
      2.  Distance matrix
      3.  Contact map
      4.  Secondary structure (DSSP or dihedral fallback)
      6.  SASA (Shrake-Rupley)
      7.  Radius of gyration
      8.  Packing density
      9.  Structural entropy
      10. Hydrogen bond network

    Parameters
    ----------
    structure       : BioPython Structure object
    ca_coords       : (N, 3) CA coordinates (already extracted)
    pdb_path        : source file path (for provenance)
    chain_id        : chain selector
    contact_cutoff  : CA-CA distance threshold for contacts (A)
    packing_radius  : radius for packing density count (A)
    sasa_probe      : solvent probe radius for SASA (A)
    hbond_energy_cutoff : DSSP H-bond energy threshold (kcal/mol)

    Returns
    -------
    StructureMetrics
    """
    n = len(ca_coords)

    # 2. Distance matrix
    dist_matrix = compute_distance_matrix(ca_coords)

    # 3. Contact map
    contact_map = compute_contact_map(ca_coords, cutoff=contact_cutoff)

    # 4. Secondary structure
    ss_list, ss_source = assign_secondary_structure(structure, chain_id=chain_id)
    # Pad / trim to match CA count
    if len(ss_list) < n:
        ss_list += ["C"] * (n - len(ss_list))
    ss_list = ss_list[:n]

    # 6. SASA
    try:
        sasa_arr = compute_sasa(structure, chain_id=chain_id, probe_radius=sasa_probe)
        if len(sasa_arr) != n:
            sasa_arr = np.full(n, -1.0)
    except Exception:
        sasa_arr = np.full(n, -1.0)

    # 7. Radius of gyration
    rg = compute_radius_of_gyration(ca_coords)

    # 8. Packing density
    packing = compute_packing_density(ca_coords, radius=packing_radius)

    # 9. Structural entropy
    global_entropy, per_res_entropy = compute_structural_entropy(contact_map)

    # 10. H-bond network
    hbonds = detect_hydrogen_bonds(
        structure, chain_id=chain_id, energy_cutoff=hbond_energy_cutoff
    )

    # Per-residue H-bond participation count
    hbond_counts = np.zeros(n, dtype=int)
    for hb in hbonds:
        if hb.donor_index < n:
            hbond_counts[hb.donor_index] += 1
        if hb.acceptor_index < n:
            hbond_counts[hb.acceptor_index] += 1

    # Mean CA distance per residue (off-diagonal mean)
    mean_ca_dist = (dist_matrix.sum(axis=1) - np.diag(dist_matrix)) / max(n - 1, 1)

    # Contact count per residue (already filtered by seq sep in contact_map)
    contact_counts = contact_map.sum(axis=1).astype(int)

    # Assemble per-residue records
    from .utils import _PARSER  # reuse parser registration; avoid circular import
    model = structure[0]
    chain = (
        model[chain_id]
        if chain_id is not None
        else list(model.get_chains())[0]
    )
    res_names = [
        r.get_resname()
        for r in chain.get_residues()
        if r.get_id()[0] == " " and "CA" in r
    ][:n]
    if len(res_names) < n:
        res_names += ["UNK"] * (n - len(res_names))

    per_residue = [
        PerResidueMetrics(
            residue_index=i,
            residue_name=res_names[i],
            secondary_structure=ss_list[i],
            sasa=float(sasa_arr[i]),
            packing_density=int(packing[i]),
            contact_count=int(contact_counts[i]),
            mean_ca_distance=float(mean_ca_dist[i]),
            hbond_count=int(hbond_counts[i]),
            local_entropy=float(per_res_entropy[i]),
        )
        for i in range(n)
    ]

    # Global aggregates
    valid_sasa  = sasa_arr[sasa_arr >= 0]
    mean_sasa   = float(valid_sasa.mean()) if len(valid_sasa) > 0 else 0.0
    n_h = ss_list.count("H")
    n_e = ss_list.count("E")
    n_c = ss_list.count("C")

    return StructureMetrics(
        pdb_path=pdb_path,
        n_residues=n,
        radius_of_gyration=round(rg, 3),
        structural_entropy=round(global_entropy, 4),
        total_hbonds=len(hbonds),
        mean_sasa=round(mean_sasa, 2),
        mean_packing_density=round(float(packing.mean()), 2),
        fraction_helix=round(n_h / n, 4) if n else 0.0,
        fraction_sheet=round(n_e / n, 4) if n else 0.0,
        fraction_coil=round(n_c / n, 4) if n else 0.0,
        ss_source=ss_source,
        per_residue=per_residue,
        hbonds=hbonds,
        distance_matrix=dist_matrix,
        contact_map=contact_map,
    )
