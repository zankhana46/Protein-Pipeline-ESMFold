# Protein Pipeline - Team Division Project

## Team Structure

**Shared Base**: ESMFold structure prediction (all members use same output)

### Member 1 - Stability Analysis
- RMSD/RMSF calculations
- Flexible region detection
- pLDDT confidence mapping

### Member 2 - Geometric Pocket Detection
- DBSCAN clustering
- Surface curvature analysis
- Cavity detection

### Member 3 - GNN ML Pocket Prediction
- Graph Neural Network
- Per-residue pocket probability
- ML-based binding site ranking

### Member 4 - Fusion Engine
- Weighted score combination
- Pharmacophore generation
- Final druggable site prediction

---

## Setup

```bash
# Clone repo
git clone <your-repo-url>
cd protein_pipeline

# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Workflow

Each member works in their assigned folder:
- `member1_stability/`
- `member2_geometry/`
- `member3_ml_pockets/` ← Member 3 (GNN)
- `member4_fusion/`

All members use the shared `common/` and `base/` modules.

---

## Branch Strategy

```bash
# Create your branch
git checkout -b member1-stability  # or member2-geometry, member3-gnn, member4-fusion

# Work on your code
git add .
git commit -m "Add module implementation"
git push origin <branch-name>
```
