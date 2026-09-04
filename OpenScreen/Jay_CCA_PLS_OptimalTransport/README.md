# Muhunden Jayakrishnan — scverse × Morpho Hackathon WS4 (sub aims 1,2)

Cross-modal analysis linking **CellProfiler morphology** and **scRNAseq** profiles
for 119 compounds, using pseudobulked PCA embeddings as input.

## Analyses

### 1. Optimal Transport (`OT_analysis.ipynb`)
Computes OT-based distances between morphology and RNA compound distributions and performs a Mantel-style correlation analysis.
Used to assess how well the two modalities align at the compound level.

Note : Calculated Wasserstein distances for Morphology and RNA are provided as separate pkl files (W_*_results.pkl)

### 2. PLS / CCA Paired Retrieval (`PLS_CCA_analyses.ipynb`)
Evaluates cross-modal compound retrieval using PLSCanonical and CCA.
A repeated k-fold CV scheme (5 splits × 20 repeats) is used to learn a shared
latent space from training compounds and rank-retrieve paired compounds in the
held-out test set.

**Retrieval metrics reported per fold and direction (morphology→RNA, RNA→morphology):**
- Mean Reciprocal Rank (MRR)
- Recall@1, Recall@5
- Median rank

A shuffled-pairs null control (`shuffle_train_pairs=True`) is included to
establish chance-level performance.

## Inputs
- `data/hepg2_cells.h5ad` — (389085 cells x 62710 genes; for 120 compounds)
- `aggregated_data/*` — (42847 wells pooled across 4 HepG2 sites x 2977 features)
- `annotations/pd_export_04_2022_2464_compounds_standardized.csv` (Compound annotations OpenScreen) 


