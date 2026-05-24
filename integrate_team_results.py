"""
FULL TEAM INTEGRATION - All 3 Members Combined
"""
import sys
sys.path.insert(0, '/root/protein_pipeline')

import json
import numpy as np
from common.data_models import ProteinStructure
from member3_ml_pockets.graph_builder import ProteinGraph
from member3_ml_pockets.gnn_model import GeometricPocketPredictor
from member3_ml_pockets.chemistry_analyzer import ChemistryAnalyzer
from member3_ml_pockets.fusion_engine import PocketFusionEngine

# Load structure (lysozyme - 129 residues)
pdb_path = "outputs/structures/structure_129res.pdb"
with open(pdb_path) as f:
    pdb_string = f.read()

coords, residue_names, atom_names, plddt_scores = [], [], [], []

for line in pdb_string.split('\n'):
    if line.startswith('ATOM'):
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        b_factor = float(line[60:66])
        atom_names.append(atom_name)
        residue_names.append(res_name)
        coords.append([x, y, z])
        plddt_scores.append(b_factor)

per_residue_plddt = [plddt_scores[i] for i, atom in enumerate(atom_names) if atom == 'CA']

structure = ProteinStructure(
    sequence="", pdb_string=pdb_string, pdb_path=pdb_path,
    plddt_scores=np.array(per_residue_plddt),
    coordinates=np.array(coords),
    residue_names=residue_names, atom_names=atom_names
)

print("=" * 70)
print("FULL TEAM INTEGRATION - ALL 3 MEMBERS")
print("=" * 70)

# YOUR ML + Chemistry (Member 3)
print("\n[Member 3] Running ML pocket prediction...")
graph_builder = ProteinGraph(structure)
G = graph_builder.build_residue_graph()
predictor = GeometricPocketPredictor(graph_builder)
ml_scores = predictor.predict_pockets()

top_k = 15
top_indices = np.argsort(ml_scores)[-top_k:][::-1]
analyzer = ChemistryAnalyzer(structure, top_indices)
chem_features = analyzer.analyze_pocket_chemistry()

# Load Member 1 data (Stability)
print("\n[Member 1] Loading stability scores...")
try:
    with open('outputs/stability/structure_129res_stability.json') as f:
        stability_data = json.load(f)
    
    stability_scores = np.array([stability_data['residues'][i]['stability_score'] 
                                 for i in range(len(stability_data['residues']))])
    print(f"✓ Loaded {len(stability_scores)} stability scores")
    print(f"  Range: {stability_scores.min():.3f} - {stability_scores.max():.3f}")
except Exception as e:
    print(f"⚠ Could not load Member 1 data: {e}")
    stability_scores = None

# Load Member 2 data (Geometry)
print("\n[Member 2] Loading geometry scores...")
try:
    geometry_scores = np.load('outputs/member2/lysozyme_geometry_scores.npy')
    print(f"✓ Loaded {len(geometry_scores)} geometry scores")
    print(f"  Range: {geometry_scores.min():.3f} - {geometry_scores.max():.3f}")
    print(f"  Non-zero: {np.count_nonzero(geometry_scores)}")
except Exception as e:
    print(f"⚠ Could not load Member 2 data: {e}")
    geometry_scores = None

# FUSION
print("\n[Fusion] Combining all predictions...")
fusion = PocketFusionEngine(structure)
results = fusion.fuse_predictions(
    ml_scores=ml_scores,
    geometry_scores=geometry_scores,
    stability_scores=stability_scores,
    chemistry_features=chem_features
)

# Generate report
print("\n" + fusion.generate_report(results))

# Save integrated report
report_path = "outputs/final/integrated_team_report.txt"
import os
os.makedirs("outputs/final", exist_ok=True)
with open(report_path, 'w') as f:
    f.write(fusion.generate_report(results))

print(f"\n{'='*70}")
print(f"✓ INTEGRATION COMPLETE!")
print(f"{'='*70}")
print(f"Final report saved to: {report_path}")
