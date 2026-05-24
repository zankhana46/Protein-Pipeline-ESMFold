import sys
sys.path.insert(0, '/root/protein_pipeline')

from common.data_models import ProteinStructure
from member3_ml_pockets.graph_builder import ProteinGraph
from member3_ml_pockets.gnn_model import GeometricPocketPredictor
from member3_ml_pockets.chemistry_analyzer import ChemistryAnalyzer
import numpy as np

# Load structure
pdb_path = "outputs/structures/structure_129res.pdb"
with open(pdb_path) as f:
    pdb_string = f.read()

coords = []
residue_names = []
atom_names = []
plddt_scores = []

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

# Build graph and predict pockets
graph_builder = ProteinGraph(structure)
G = graph_builder.build_residue_graph()
predictor = GeometricPocketPredictor(graph_builder)
pocket_scores = predictor.predict_pockets()

# Get top pocket residues
top_k = 15
top_indices = np.argsort(pocket_scores)[-top_k:][::-1]

print("=" * 60)
print("CHEMISTRY ANALYSIS")
print("=" * 60)

# Analyze chemistry
analyzer = ChemistryAnalyzer(structure, top_indices)
chem_features = analyzer.analyze_pocket_chemistry()

print("\nPocket Chemical Features:")
for key, value in chem_features.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("PHARMACOPHORE")
print("=" * 60)

pharmacophore = analyzer.generate_pharmacophore()
print("\nPharmacophore features:")
for key, coords in pharmacophore.items():
    print(f"  {key}: {len(coords)} points")

print("\n✓ Chemistry analysis complete!")
