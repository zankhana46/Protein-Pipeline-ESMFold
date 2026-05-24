import sys
sys.path.insert(0, '/root/protein_pipeline')

from common.data_models import ProteinStructure
from member3_ml_pockets.graph_builder import ProteinGraph
from member3_ml_pockets.gnn_model import GeometricPocketPredictor
from member3_ml_pockets.chemistry_analyzer import ChemistryAnalyzer
from member3_ml_pockets.fusion_engine import PocketFusionEngine
import numpy as np

# Load structure
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
print("RUNNING FULL PIPELINE - MEMBER 3")
print("=" * 70)

# Step 1: Build graph
print("\n[1/4] Building protein graph...")
graph_builder = ProteinGraph(structure)
G = graph_builder.build_residue_graph()

# Step 2: ML prediction
print("\n[2/4] Running ML pocket prediction...")
predictor = GeometricPocketPredictor(graph_builder)
ml_scores = predictor.predict_pockets()

# Step 3: Chemistry analysis
print("\n[3/4] Analyzing chemistry...")
top_k = 15
top_indices = np.argsort(ml_scores)[-top_k:][::-1]
analyzer = ChemistryAnalyzer(structure, top_indices)
chem_features = analyzer.analyze_pocket_chemistry()

# Step 4: Fusion
print("\n[4/4] Fusing predictions...")
fusion = PocketFusionEngine(structure)
results = fusion.fuse_predictions(
    ml_scores=ml_scores,
    chemistry_features=chem_features
)

# Generate report
print("\n" + fusion.generate_report(results))

# Save report
report_path = "outputs/member3/final_report.txt"
import os
os.makedirs("outputs/member3", exist_ok=True)
with open(report_path, 'w') as f:
    f.write(fusion.generate_report(results))

print(f"\n✓ Report saved to: {report_path}")
