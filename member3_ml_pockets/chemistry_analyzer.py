"""
Chemistry analysis using RDKit
Pharmacophore features, H-bonds, charge analysis
"""
import numpy as np
from typing import List, Dict

class ChemistryAnalyzer:
    """
    Analyze chemical properties of predicted pockets
    """
    
    def __init__(self, structure, pocket_residues):
        """
        Args:
            structure: ProteinStructure object
            pocket_residues: List of residue indices forming the pocket
        """
        self.structure = structure
        self.pocket_residues = pocket_residues
        
    def analyze_pocket_chemistry(self):
        """
        Analyze chemical features of the pocket
        Returns dict with scores
        """
        # Amino acid property mappings
        charged_positive = ['ARG', 'LYS', 'HIS']
        charged_negative = ['ASP', 'GLU']
        aromatic = ['PHE', 'TRP', 'TYR', 'HIS']
        hydrophobic = ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO']
        polar = ['SER', 'THR', 'ASN', 'GLN', 'CYS', 'TYR']
        hbond_donors = ['SER', 'THR', 'TYR', 'CYS', 'LYS', 'ARG', 'HIS', 'TRP', 'ASN', 'GLN']
        hbond_acceptors = ['ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'HIS', 'TYR']
        
        # Get residue names for pocket
        ca_residues = [self.structure.residue_names[i] 
                      for i, atom in enumerate(self.structure.atom_names) 
                      if atom == 'CA']
        
        pocket_res_names = [ca_residues[i] for i in self.pocket_residues]
        
        # Count features
        features = {
            'charged_positive': sum(1 for r in pocket_res_names if r in charged_positive),
            'charged_negative': sum(1 for r in pocket_res_names if r in charged_negative),
            'aromatic': sum(1 for r in pocket_res_names if r in aromatic),
            'hydrophobic': sum(1 for r in pocket_res_names if r in hydrophobic),
            'polar': sum(1 for r in pocket_res_names if r in polar),
            'hbond_donors': sum(1 for r in pocket_res_names if r in hbond_donors),
            'hbond_acceptors': sum(1 for r in pocket_res_names if r in hbond_acceptors),
            'total_residues': len(pocket_res_names)
        }
        
        # Calculate druggability score
        # Good pockets have: hydrophobic core, some polar/charged edges, aromatic rings
        hydrophobic_ratio = features['hydrophobic'] / features['total_residues']
        aromatic_ratio = features['aromatic'] / features['total_residues']
        polar_ratio = (features['polar'] + features['charged_positive'] + features['charged_negative']) / features['total_residues']
        hbond_ratio = (features['hbond_donors'] + features['hbond_acceptors']) / (2 * features['total_residues'])
        
        # Druggability score (0-1)
        druggability = (
            hydrophobic_ratio * 0.4 +  # Hydrophobic core important
            aromatic_ratio * 0.2 +      # Aromatic interactions
            polar_ratio * 0.2 +         # Some polarity needed
            hbond_ratio * 0.2           # H-bond capability
        )
        
        features['druggability_score'] = min(druggability, 1.0)
        
        return features
    
    def generate_pharmacophore(self):
        """
        Generate pharmacophore features (simplified)
        Returns: List of pharmacophore points
        """
        pharmacophore = {
            'hydrophobic_centers': [],
            'hbond_donors': [],
            'hbond_acceptors': [],
            'aromatic_rings': [],
            'positive_charges': [],
            'negative_charges': []
        }
        
        # Get CA coordinates for pocket residues
        ca_coords = []
        ca_residues = []
        
        for i, atom in enumerate(self.structure.atom_names):
            if atom == 'CA':
                ca_coords.append(self.structure.coordinates[i])
                ca_residues.append(self.structure.residue_names[i])
        
        ca_coords = np.array(ca_coords)
        
        # Map residues to pharmacophore features
        for res_idx in self.pocket_residues:
            coord = ca_coords[res_idx]
            res_name = ca_residues[res_idx]
            
            if res_name in ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO']:
                pharmacophore['hydrophobic_centers'].append(coord)
            
            if res_name in ['SER', 'THR', 'TYR', 'CYS', 'LYS', 'ARG', 'HIS', 'TRP']:
                pharmacophore['hbond_donors'].append(coord)
            
            if res_name in ['ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'HIS', 'TYR']:
                pharmacophore['hbond_acceptors'].append(coord)
            
            if res_name in ['PHE', 'TRP', 'TYR', 'HIS']:
                pharmacophore['aromatic_rings'].append(coord)
            
            if res_name in ['ARG', 'LYS', 'HIS']:
                pharmacophore['positive_charges'].append(coord)
            
            if res_name in ['ASP', 'GLU']:
                pharmacophore['negative_charges'].append(coord)
        
        return pharmacophore
