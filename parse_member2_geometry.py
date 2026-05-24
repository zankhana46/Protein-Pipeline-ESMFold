"""
Parse Member 2's geometry output into per-residue scores
"""
import re
import numpy as np

def parse_geometry_file(filepath, n_residues):
    """
    Parse Member 2's geometry output
    Returns: per-residue geometry scores
    """
    with open(filepath) as f:
        content = f.read()
    
    # Initialize all residues to 0
    geometry_scores = np.zeros(n_residues)
    
    # Find all pockets
    pocket_pattern = r'Pocket #\d+\s+geometry_score = ([\d.]+).*?Binding pocket residues.*?:\s+(.*?)(?=\n\s*───|$)'
    
    matches = re.findall(pocket_pattern, content, re.DOTALL)
    
    for score_str, residues_str in matches:
        score = float(score_str)
        
        # Extract residue numbers (e.g., "PHE-34" -> 34)
        # Note: Residue numbers are 1-indexed in PDB, 0-indexed in our arrays
        residue_matches = re.findall(r'([A-Z]{3})-(\d+)', residues_str)
        
        for _, res_num in residue_matches:
            res_idx = int(res_num) - 1  # Convert to 0-indexed
            if 0 <= res_idx < n_residues:
                # Use max score if residue appears in multiple pockets
                geometry_scores[res_idx] = max(geometry_scores[res_idx], score)
    
    return geometry_scores

# Test it
geometry_scores = parse_geometry_file('outputs/output_129res_geometry.txt', 129)

print(f"Parsed geometry scores for {len(geometry_scores)} residues")
print(f"Range: {geometry_scores.min():.3f} - {geometry_scores.max():.3f}")
print(f"Non-zero residues: {np.count_nonzero(geometry_scores)}")

# Save for fusion
np.save('outputs/member2/lysozyme_geometry_scores.npy', geometry_scores)
print("✓ Saved to outputs/member2/lysozyme_geometry_scores.npy")
