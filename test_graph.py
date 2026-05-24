import sys
sys.path.insert(0, '/root/protein_pipeline')

from common.esmfold_api import ESMFoldAPI
from member3_ml_pockets.graph_builder import ProteinGraph

# Load a structure
print("Loading structure...")
from pathlib import Path
import pickle

# Use the lysozyme structure we generated
pdb_path = "outputs/structures/structure_129res.pdb"

# Read PDB and create structure object (simplified for testing)
from common.data_models import ProteinStructure
import numpy as np

# Parse PDB file
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

# Get per-residue pLDDT
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

print(f"Structure loaded: {len(per_residue_plddt)} residues")

# Build graph
print("\nBuilding protein graph...")
graph_builder = ProteinGraph(structure)
G = graph_builder.build_residue_graph(distance_threshold=8.0)

print(f"\nGraph statistics:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Node features shape: {graph_builder.node_features.shape}")

print("\n✓ Graph construction successful!")
