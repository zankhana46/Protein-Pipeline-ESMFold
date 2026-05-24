"""
Shared data structures used by all team members
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class ProteinStructure:
    """
    Shared ESMFold output that all members use
    """
    sequence: str
    pdb_string: str
    pdb_path: str
    plddt_scores: np.ndarray
    coordinates: np.ndarray
    residue_names: List[str]
    atom_names: List[str]
    
    def get_high_confidence_residues(self, threshold: float = 70.0) -> List[int]:
        return [i for i, score in enumerate(self.plddt_scores) if score > threshold]
    
    def summary(self) -> str:
        avg_plddt = np.mean(self.plddt_scores)
        return f"Protein: {len(self.sequence)} residues, avg pLDDT: {avg_plddt:.1f}"


@dataclass
class PocketCandidate:
    """
    Generic pocket representation - all members produce these
    """
    pocket_id: int
    center: np.ndarray
    residue_indices: List[int]
    score: float
    method: str
    volume: Optional[float] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
