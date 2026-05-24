"""
Fusion Engine - Combines all member results
Final weighted scoring for druggable pockets
"""
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class FusionResult:
    """Final fused pocket prediction"""
    pocket_id: int
    center: np.ndarray
    residue_indices: List[int]
    ml_score: float
    geometry_score: float
    stability_score: float
    chemistry_score: float
    final_score: float
    druggability: float
    features: Dict

class PocketFusionEngine:
    """
    Combines predictions from all members
    """
    
    def __init__(self, structure):
        self.structure = structure
        
        # Fusion weights (from your spec)
        self.weights = {
            'ml': 0.40,
            'geometry': 0.30,
            'stability': 0.20,
            'chemistry': 0.10
        }
    
    def fuse_predictions(
        self,
        ml_scores: np.ndarray,
        geometry_scores: np.ndarray = None,
        stability_scores: np.ndarray = None,
        chemistry_features: Dict = None
    ) -> List[FusionResult]:
        """
        Fuse all predictions into final ranked pockets
        
        Args:
            ml_scores: Per-residue ML pocket scores (from YOUR GNN)
            geometry_scores: Per-residue geometry scores (from Member 2)
            stability_scores: Per-residue stability scores (from Member 1)
            chemistry_features: Chemistry analysis results
        
        Returns:
            List of FusionResult objects, sorted by final score
        """
        n_residues = len(ml_scores)
        
        # Default values if other members haven't provided data yet
        if geometry_scores is None:
            geometry_scores = np.ones(n_residues) * 0.5  # Neutral
        
        if stability_scores is None:
            stability_scores = np.ones(n_residues) * 0.5  # Neutral
        
        # Normalize all scores to 0-1
        ml_norm = self._normalize(ml_scores)
        geo_norm = self._normalize(geometry_scores)
        stab_norm = self._normalize(stability_scores)
        
        # Calculate final scores
        final_scores = (
            ml_norm * self.weights['ml'] +
            geo_norm * self.weights['geometry'] +
            stab_norm * self.weights['stability']
        )
        
        # Add chemistry boost
        if chemistry_features:
            chem_score = chemistry_features.get('druggability_score', 0.5)
            final_scores = final_scores * (1.0 + chem_score * self.weights['chemistry'])
        
        # Cluster high-scoring residues into pockets
        pockets = self._cluster_pockets(final_scores, threshold=0.6)
        
        # Create FusionResult objects
        results = []
        for pocket_id, residue_indices in enumerate(pockets):
            if len(residue_indices) == 0:
                continue
            
            # Calculate pocket center
            ca_coords = self._get_ca_coordinates()
            pocket_coords = ca_coords[residue_indices]
            center = np.mean(pocket_coords, axis=0)
            
            # Average scores for this pocket
            avg_ml = np.mean(ml_norm[residue_indices])
            avg_geo = np.mean(geo_norm[residue_indices])
            avg_stab = np.mean(stab_norm[residue_indices])
            avg_final = np.mean(final_scores[residue_indices])
            
            result = FusionResult(
                pocket_id=pocket_id,
                center=center,
                residue_indices=residue_indices.tolist(),
                ml_score=avg_ml,
                geometry_score=avg_geo,
                stability_score=avg_stab,
                chemistry_score=chemistry_features.get('druggability_score', 0.5) if chemistry_features else 0.5,
                final_score=avg_final,
                druggability=chemistry_features.get('druggability_score', 0.5) if chemistry_features else 0.5,
                features=chemistry_features if chemistry_features else {}
            )
            
            results.append(result)
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalization to 0-1"""
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-6:
            return np.ones_like(scores) * 0.5
        return (scores - min_val) / (max_val - min_val)
    
    def _cluster_pockets(self, scores: np.ndarray, threshold: float = 0.6) -> List[np.ndarray]:
        """
        Cluster high-scoring residues into discrete pockets
        Uses spatial proximity
        """
        # Get high-scoring residues
        high_score_indices = np.where(scores > threshold)[0]
        
        if len(high_score_indices) == 0:
            return []
        
        # Get CA coordinates
        ca_coords = self._get_ca_coordinates()
        
        # Simple clustering: group residues within 10Å
        pockets = []
        visited = set()
        
        for idx in high_score_indices:
            if idx in visited:
                continue
            
            # Start new pocket
            pocket = [idx]
            visited.add(idx)
            
            # Add nearby high-scoring residues
            for other_idx in high_score_indices:
                if other_idx in visited:
                    continue
                
                dist = np.linalg.norm(ca_coords[idx] - ca_coords[other_idx])
                if dist < 10.0:
                    pocket.append(other_idx)
                    visited.add(other_idx)
            
            pockets.append(np.array(pocket))
        
        return pockets
    
    def _get_ca_coordinates(self) -> np.ndarray:
        """Extract CA atom coordinates"""
        ca_coords = []
        for i, atom in enumerate(self.structure.atom_names):
            if atom == 'CA':
                ca_coords.append(self.structure.coordinates[i])
        return np.array(ca_coords)
    
    def generate_report(self, results: List[FusionResult]) -> str:
        """
        Generate final report
        """
        report = []
        report.append("=" * 70)
        report.append("FINAL POCKET PREDICTION REPORT")
        report.append("=" * 70)
        report.append(f"\nProtein: {self.structure.pdb_path}")
        report.append(f"Total residues: {len(self.structure.plddt_scores)}")
        report.append(f"Pockets identified: {len(results)}\n")
        
        for i, result in enumerate(results[:5], 1):  # Top 5
            report.append(f"\n{'='*70}")
            report.append(f"POCKET #{i} (ID: {result.pocket_id})")
            report.append(f"{'='*70}")
            report.append(f"Final Score: {result.final_score:.3f}")
            report.append(f"Druggability: {result.druggability:.3f}")
            report.append(f"\nComponent Scores:")
            report.append(f"  ML Score:        {result.ml_score:.3f} (weight: {self.weights['ml']:.0%})")
            report.append(f"  Geometry Score:  {result.geometry_score:.3f} (weight: {self.weights['geometry']:.0%})")
            report.append(f"  Stability Score: {result.stability_score:.3f} (weight: {self.weights['stability']:.0%})")
            report.append(f"  Chemistry Score: {result.chemistry_score:.3f} (weight: {self.weights['chemistry']:.0%})")
            report.append(f"\nPocket Properties:")
            report.append(f"  Center: ({result.center[0]:.1f}, {result.center[1]:.1f}, {result.center[2]:.1f})")
            report.append(f"  Residues: {len(result.residue_indices)}")
            report.append(f"  Residue indices: {result.residue_indices[:10]}{'...' if len(result.residue_indices) > 10 else ''}")
        
        report.append(f"\n{'='*70}")
        report.append("RECOMMENDATION")
        report.append(f"{'='*70}")
        if len(results) > 0:
            best = results[0]
            report.append(f"Best binding site: Pocket #{best.pocket_id}")
            report.append(f"Confidence: {best.final_score:.1%}")
            report.append(f"Druggability: {best.druggability:.1%}")
        else:
            report.append("No significant pockets identified.")
        
        return "\n".join(report)
