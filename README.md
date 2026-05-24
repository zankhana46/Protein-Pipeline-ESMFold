# Protein Binding Site Prediction Pipeline

**Team Collaboration Project**: Multi-method pocket detection and druggability assessment using ESMFold structure prediction.

---

## 🎯 Project Overview

This pipeline predicts drug-binding sites (pockets) in proteins by combining three independent analysis methods:

1. **Stability Analysis** (Member 1)
2. **Geometric Pocket Detection** (Member 2)  
3. **ML Prediction + Chemistry + Fusion** (Member 3)

All methods analyze the **same ESMFold-predicted structure** and results are fused using weighted scoring.

---

## 🏗️ Architecture

```
Input: Amino Acid Sequence
         ↓
    ESMFold API (shared structure prediction)
         ↓
    3D Protein Structure (PDB)
         ↓
    ┌────────────┬──────────────┬─────────────────┐
    ↓            ↓              ↓                 ↓
Member 1     Member 2       Member 3          Member 4
Stability    Geometry       ML Pockets        (Integrated
Analysis     Detection      + Chemistry        into Member 3)
RMSD/RMSF    fpocket        GNN               
pLDDT        DBSCAN         RDKit             
             Curvature      Pharmacophore     
    ↓            ↓              ↓                 
    └────────────┴──────────────┘
                 ↓
           Fusion Engine (Member 3)
       Weighted Score Combination
                 ↓
        Final Ranked Pockets
     (Druggable Binding Sites)
```

---

## 📊 Fusion Scoring Formula

```
Final Score = 
    (ML Score        × 40%) +
    (Geometry Score  × 30%) +
    (Stability Score × 20%) +
    (Chemistry Score × 10%)
```

---

## 🧬 Test Results (Lysozyme - 129 residues)

### **Best Binding Site Identified:**
- **Pocket #1**: 74.4% confidence
- **Residues**: 57, 58, 61, 62, 93, 97, 106
- **Contains TRP-62** (known active site residue!)
- **Validation**: Matches published lysozyme active site

### **Performance:**
- **Member 3 alone**: 63.6% confidence
- **All 3 members**: 74.4% confidence
- **Improvement**: +11% with team integration

---

## 🚀 Quick Start

### **1. Setup Environment**

```bash
# Clone repository
git clone https://github.com/zankhana46/Protein-Pipeline-ESMFold.git
cd Protein-Pipeline-ESMFold

# Create virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Generate Structures**

```bash
# Generate 3D structures from sequences
python3 generate_structures.py
```

**Output**: PDB files in `outputs/structures/`

### **3. Run Full Pipeline**

```bash
# Integrate all member predictions
python3 integrate_team_results.py
```

**Output**: Final report in `outputs/final/integrated_team_report.txt`

---

## 📁 Project Structure

```
protein_pipeline/
├── common/
│   ├── data_models.py          # Shared data structures
│   └── esmfold_api.py          # Structure prediction API
│
├── base/
│   └── analyzer.py             # Base class for all analyzers
│
├── member1_stability/
│   └── analyzer.py             # Stability & flexibility analysis
│
├── member2_geometry/
│   └── analyzer.py             # Geometric pocket detection
│
├── member3_ml_pockets/
│   ├── graph_builder.py        # Protein → Graph conversion
│   ├── gnn_model.py            # GNN pocket prediction
│   ├── chemistry_analyzer.py   # RDKit druggability scoring
│   └── fusion_engine.py        # Combines all predictions
│
├── outputs/
│   ├── structures/             # ESMFold PDB files
│   ├── stability/              # Member 1 results
│   ├── member2/                # Member 2 results
│   ├── member3/                # Member 3 results
│   └── final/                  # Integrated reports
│
├── generate_structures.py      # Generate PDB files
├── integrate_team_results.py   # Full pipeline integration
└── README.md
```

---

## 🔬 Member Responsibilities

### **Member 1: Stability Analysis**
- **Input**: PDB structure
- **Methods**: RMSD, RMSF, pLDDT mapping
- **Output**: `outputs/stability/structure_*_stability.json`
- **Format**: Per-residue stability scores (0-1)

### **Member 2: Geometric Pocket Detection**
- **Input**: PDB structure
- **Methods**: fpocket, DBSCAN, surface curvature, SASA
- **Output**: `outputs/output_*_geometry.txt`
- **Format**: Pocket list with geometry scores

### **Member 3: ML + Chemistry + Fusion**
- **Input**: PDB structure + Member 1 & 2 results
- **Methods**: 
  - Graph Neural Network (pocket prediction)
  - RDKit (pharmacophore, H-bonds, hydrophobicity)
  - Weighted fusion scoring
- **Output**: `outputs/final/integrated_team_report.txt`
- **Format**: Ranked druggable pockets

---

## 📈 Results Interpretation

### **Final Score Ranges:**
- **>0.70**: High-confidence binding site (investigate further)
- **0.60-0.70**: Moderate confidence (potential site)
- **<0.60**: Low confidence (likely not druggable)

### **Druggability Score:**
- **>0.60**: Good drug-binding properties
- **0.40-0.60**: Moderate (may require optimization)
- **<0.40**: Poor druggability

### **Component Scores:**
- **ML Score**: Confidence from graph neural network
- **Geometry Score**: Pocket depth, volume, curvature
- **Stability Score**: Structural rigidity (stable = better)
- **Chemistry Score**: Hydrophobic, aromatic, H-bond features

---

## 🧪 Tested Proteins

| Protein | Residues | Pockets Found | Best Score | Validation |
|---------|----------|---------------|------------|------------|
| Lysozyme | 129 | 5 | 74.4% | ✅ Matches known active site (TRP-62) |
| Insulin | 110 | - | - | Pending |
| Small Peptide | 16 | - | - | Pending |

---

## 🛠️ Dependencies

```
torch>=2.0.0
fair-esm
biopython
numpy
pandas
scikit-learn
scipy
torch-geometric
matplotlib
seaborn
py3Dmol
rdkit-pypi
requests
```

---

## 📝 Workflow for New Proteins

1. **Add sequence** to `generate_structures.py`
2. **Run ESMFold**: `python3 generate_structures.py`
3. **Member 1**: Analyze stability → save to `outputs/stability/`
4. **Member 2**: Detect pockets → save to `outputs/member2/`
5. **Member 3**: Run fusion → `python3 integrate_team_results.py`
6. **Review**: Check `outputs/final/integrated_team_report.txt`

---

## 🤝 Team Collaboration

### **Branch Strategy:**
- `main`: Stable integrated code
- `member1-stability`: Member 1 development
- `member2-geometry`: Member 2 development
- `member3-gnn-fusion`: Member 3 development

### **Integration Points:**
Each member outputs standardized formats that the fusion engine expects:
- Member 1: JSON with `stability_score` per residue
- Member 2: TXT with pocket geometry scores
- Member 3: Fusion engine loads both and combines

---

## 🎓 Citation & References

### **Methods Used:**
- **ESMFold**: Lin et al. (2023) - Structure prediction
- **fpocket**: Le Guilloux et al. (2009) - Pocket detection
- **RDKit**: Open-source cheminformatics
- **Graph Neural Networks**: Custom implementation

### **Test Case:**
- **Lysozyme** (PDB: 1LYZ) - Classic antibacterial enzyme

---

## 📧 Contact

- **Team Lead**: Member 3 (zankhana46)
- **Repository**: https://github.com/zankhana46/Protein-Pipeline-ESMFold

---

## ✅ Project Status

- [x] ESMFold API integration
- [x] Member 1: Stability analysis
- [x] Member 2: Geometry detection
- [x] Member 3: ML + Chemistry + Fusion
- [x] Full pipeline integration
- [x] Lysozyme validation
- [ ] Additional protein testing
- [ ] Web interface (future work)

---

## 🏆 Key Achievements

✅ **Successfully predicted lysozyme active site** with 74.4% confidence  
✅ **Validated against known structures** (matches TRP-62 active site)  
✅ **Modular design** - each member works independently  
✅ **Scalable fusion** - easy to add new prediction methods  
✅ **Production-ready** - generates publication-quality reports  

---

*Last Updated: May 24, 2026*