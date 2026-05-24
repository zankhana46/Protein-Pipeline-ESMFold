import sys
sys.path.insert(0, '/root/protein_pipeline')

from common.data_models import ProteinStructure
from member3_ml_pockets.graph_builder import ProteinGraph
from member3_ml_pockets.gnn_model import GeometricPocketPredictor
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
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        b_factor = float(line[60:66])
        
        atom_names.append(atom_name)
        residue_names.append(res_name)
        coords.append([x, y, z])
        plddt_scores.append(b_factor)

per_residue_plddt = [plddt_scores[i] for i, atom in enumerate(atom_names) if atom == 'CA']

structure = ProteinStructure(
    sequence="",
    pdb_string=pdb_string,
    pdb_path=pdb_path,
    plddt_scores=np.array(per_residue_plddt),
    coordinates=np.array(coords),
    residue_names=residue_names,
    atom_names=atom_names
)

# Build graph
print("Building graph...")
graph_builder = ProteinGraph(structure)
G = graph_builder.build_residue_graph(distance_threshold=8.0)

# Predict pockets
print("\nPredicting pockets...")
predictor = GeometricPocketPredictor(graph_builder)
pocket_scores = predictor.predict_pockets()

print(f"\nPocket prediction complete!")
print(f"  Mean score: {pocket_scores.mean():.3f}")
print(f"  Max score: {pocket_scores.max():.3f}")
print(f"  Min score: {pocket_scores.min():.3f}")

# Find top pocket residues
top_k = 10
top_indices = np.argsort(pocket_scores)[-top_k:][::-1]

print(f"\nTop {top_k} pocket residues:")
for idx in top_indices:
    print(f"  Residue {idx}: {G.nodes[idx]['residue']} (score: {pocket_scores[idx]:.3f})")

print("\n✓ Pocket prediction successful!")
