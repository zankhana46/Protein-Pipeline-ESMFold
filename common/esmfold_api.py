"""
Simple ESMFold using API - no complex dependencies
"""
import requests
from pathlib import Path
from .data_models import ProteinStructure
import numpy as np

class ESMFoldAPI:
    """Use ESMFold via web API - much simpler"""
    
    @classmethod
    def predict(cls, sequence: str, output_dir: Path = Path("outputs/structures")):
        """
        Predict structure using ESMFold API
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Predicting structure for {len(sequence)} residues via API...")
        
        # Call ESMFold API
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        response = requests.post(url, data=sequence)
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
        
        pdb_string = response.text
        
        # Parse pLDDT from PDB
        plddt_scores = []
        coords = []
        residue_names = []
        atom_names = []
        
        for line in pdb_string.split('\n'):
            if line.startswith('ATOM'):
                b_factor = float(line[60:66])
                plddt_scores.append(b_factor)
                
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                atom_names.append(atom_name)
                residue_names.append(res_name)
                coords.append([x, y, z])
        
        # Get per-residue pLDDT (use CA atoms)
        per_residue_plddt = [plddt_scores[i] for i, atom in enumerate(atom_names) if atom == 'CA']
        
        # Save PDB
        pdb_path = output_dir / f"structure_{len(sequence)}res.pdb"
        with open(pdb_path, 'w') as f:
            f.write(pdb_string)
        
        avg_plddt = np.mean(per_residue_plddt)
        print(f"✓ Structure saved to {pdb_path}")
        print(f"  Average pLDDT: {avg_plddt:.1f}")
        
        return ProteinStructure(
            sequence=sequence,
            pdb_string=pdb_string,
            pdb_path=str(pdb_path),
            plddt_scores=np.array(per_residue_plddt),
            coordinates=np.array(coords),
            residue_names=residue_names,
            atom_names=atom_names
        )
