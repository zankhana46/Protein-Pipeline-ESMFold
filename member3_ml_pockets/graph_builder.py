"""
Graph construction from protein structure
Converts PDB structure into a graph for GNN processing
"""
import numpy as np
import networkx as nx
from typing import List, Tuple
from pathlib import Path

class ProteinGraph:
    """
    Build a graph representation of protein structure
    Nodes = residues (or atoms)
    Edges = spatial proximity or chemical bonds
    """
    
    def __init__(self, structure):
        """
        Args:
            structure: ProteinStructure object from common.data_models
        """
        self.structure = structure
        self.graph = None
        self.node_features = None
        self.edge_index = None
        
    def build_residue_graph(self, distance_threshold: float = 8.0):
        """
        Build graph where nodes = residues, edges = spatial proximity
        
        Args:
            distance_threshold: Max distance (Å) between CA atoms to form edge
        """
        # Extract CA (alpha carbon) atoms - one per residue
        ca_coords = []
        ca_indices = []
        
        for i, atom_name in enumerate(self.structure.atom_names):
            if atom_name == 'CA':
                ca_coords.append(self.structure.coordinates[i])
                ca_indices.append(i)
        
        ca_coords = np.array(ca_coords)
        n_residues = len(ca_coords)
        
        print(f"Building graph: {n_residues} residues")
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (residues)
        for i in range(n_residues):
            G.add_node(i, 
                      pos=ca_coords[i],
                      residue=self.structure.residue_names[ca_indices[i]],
                      plddt=self.structure.plddt_scores[i])
        
        # Add edges based on distance
        edge_count = 0
        for i in range(n_residues):
            for j in range(i+1, n_residues):
                dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
                if dist < distance_threshold:
                    G.add_edge(i, j, distance=dist)
                    edge_count += 1
        
        print(f"✓ Graph built: {n_residues} nodes, {edge_count} edges")
        
        self.graph = G
        self._extract_node_features()
        self._extract_edge_index()
        
        return G
    
    def _extract_node_features(self):
        """
        Extract features for each residue node
        Features: [x, y, z, plddt, residue_type_encoded]
        """
        # Amino acid encoding (simple)
        aa_types = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                   'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                   'THR', 'TRP', 'TYR', 'VAL']
        aa_to_idx = {aa: i for i, aa in enumerate(aa_types)}
        
        features = []
        for node in self.graph.nodes():
            pos = self.graph.nodes[node]['pos']
            plddt = self.graph.nodes[node]['plddt']
            residue = self.graph.nodes[node]['residue']
            
            # One-hot encode residue type
            res_encoding = np.zeros(20)
            if residue in aa_to_idx:
                res_encoding[aa_to_idx[residue]] = 1
            
            # Combine features: [x, y, z, plddt, one-hot(20)]
            node_feat = np.concatenate([pos, [plddt], res_encoding])
            features.append(node_feat)
        
        self.node_features = np.array(features)
        print(f"✓ Node features: {self.node_features.shape}")
    
    def _extract_edge_index(self):
        """
        Extract edge connectivity in PyTorch Geometric format
        edge_index: [2, num_edges] array
        """
        edges = list(self.graph.edges())
        if len(edges) == 0:
            self.edge_index = np.array([[], []])
            return
        
        edge_index = np.array(edges).T  # Shape: [2, num_edges]
        
        # Make undirected (add reverse edges)
        edge_index = np.concatenate([edge_index, edge_index[[1,0]]], axis=1)
        
        self.edge_index = edge_index
        print(f"✓ Edge index: {self.edge_index.shape}")
    
    def visualize(self, output_path: str = "graph_viz.png"):
        """
        Visualize the graph (optional - for debugging)
        """
        import matplotlib.pyplot as plt
        
        pos_dict = {i: self.graph.nodes[i]['pos'][:2] for i in self.graph.nodes()}
        
        plt.figure(figsize=(10, 10))
        nx.draw(self.graph, pos_dict, node_size=50, node_color='lightblue', 
                edge_color='gray', alpha=0.6)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Graph visualization saved to {output_path}")
    
    def get_pytorch_geometric_data(self):
        """
        Return data in PyTorch Geometric format
        """
        import torch
        from torch_geometric.data import Data
        
        x = torch.tensor(self.node_features, dtype=torch.float)
        edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
        return data
