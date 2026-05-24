"""
Graph Neural Network for pocket prediction
Predicts pocket probability for each residue
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class PocketGNN(nn.Module):
    """
    GNN for pocket prediction
    Outputs: probability per residue (0-1)
    """
    
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=1):
        """
        Args:
            input_dim: Node feature dimension (24 in our case)
            hidden_dim: Hidden layer size
            output_dim: Output per node (1 = pocket probability)
        """
        super(PocketGNN, self).__init__()
        
        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Final prediction layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        print(f"✓ GNN initialized: {input_dim} → {hidden_dim} → {output_dim}")
    
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyTorch Geometric Data object
        
        Returns:
            pocket_probs: [num_nodes, 1] pocket probabilities
        """
        x, edge_index = data.x, data.edge_index
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Predict pocket probability per node
        pocket_probs = torch.sigmoid(self.fc(x))
        
        return pocket_probs


class GeometricPocketPredictor:
    """
    Rule-based pocket prediction using graph properties
    (No training needed - uses geometry + chemistry heuristics)
    """
    
    def __init__(self, graph_builder):
        self.graph_builder = graph_builder
        self.graph = graph_builder.graph
    
    def predict_pockets(self):
        """
        Predict pocket residues using geometric rules:
        - High curvature regions
        - Buried residues
        - Hydrophobic clusters
        """
        import numpy as np
        
        scores = []
        
        for node in self.graph.nodes():
            # Feature 1: Local density (number of neighbors)
            neighbors = len(list(self.graph.neighbors(node)))
            density_score = min(neighbors / 15.0, 1.0)  # Normalize
            
            # Feature 2: Distance from center (buried residues)
            pos = self.graph.nodes[node]['pos']
            center = np.mean([self.graph.nodes[n]['pos'] for n in self.graph.nodes()], axis=0)
            dist_from_center = np.linalg.norm(pos - center)
            burial_score = 1.0 / (1.0 + dist_from_center / 20.0)
            
            # Feature 3: Residue type (hydrophobic = higher pocket score)
            residue = self.graph.nodes[node]['residue']
            hydrophobic = ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO']
            hydrophobic_score = 1.0 if residue in hydrophobic else 0.3
            
            # Feature 4: Confidence
            plddt = self.graph.nodes[node]['plddt']
            confidence_score = plddt
            
            # Combine scores
            final_score = (
                density_score * 0.3 +
                burial_score * 0.2 +
                hydrophobic_score * 0.3 +
                confidence_score * 0.2
            )
            
            scores.append(final_score)
        
        return np.array(scores)
