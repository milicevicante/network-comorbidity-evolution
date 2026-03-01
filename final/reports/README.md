# Final Deliverable — Comorbidity Embedding Pipeline

## What This Does

`run_all_combinations.py` executes the complete comorbidity-embedding pipeline for all 6 (sex, variant) combinations:

1. **Graph inspection** — loads GEXF comorbidity networks, computes stats, draws gallery
2. **Node2Vec embeddings** — trains embeddings per age group
3. **PCA sanity plots** — 2D scatter of raw embeddings coloured by degree
4. **Procrustes alignment** — aligns all ages to a reference age
5. **Age-shift metrics** — drift (L2) and kNN stability (Jaccard) between consecutive ages
6. **Structure preservation** — link-prediction AUC and weight–similarity Spearman r
7. **Robustness** *(optional)* — retrains with varied (p,q) and checks drift-ranking consistency

## Combinations

| Sex    | Variant  | Ages available |
|--------|----------|----------------|
| male   | icd      | 1–8            |
| male   | blocks   | 1–8            |
| male   | chronic  | 2–8            |
| female | icd      | 1–8            |
| female | blocks   | 1–8            |
| female | chronic  | 2–8            |

Chronic variant graphs start at age group 2 (10–19). The script auto-detects the minimum available age and falls back when `--reference-age 1` is unavailable.

## How to Reproduce

```bash
cd /path/to/csh-tuwien-project

# Default run (uses cached results if available)
python final/run_all_combinations.py

# Force-retrain everything from scratch
python final/run_all_combinations.py --force

# Include robustness analysis (adds ~3x training time)
python final/run_all_combinations.py --robustness

# Full fresh run with robustness
python final/run_all_combinations.py --force --robustness

# Custom embedding parameters
python final/run_all_combinations.py --dim 64 --walk-length 40 --num-walks 5

# Use a different reference age
python final/run_all_combinations.py --reference-age 2
```

### Environment

- Python 3.9+
- Dependencies: `networkx`, `numpy`, `scipy`, `pandas`, `gensim`, `matplotlib`, `pyyaml`, `scikit-learn`
- Data must be at `../Data/` (relative to repo root) or set `DATA_ROOT` env var

## Output Directory Map

```
final/outputs/
├── logs/
│   ├── run_all.log              # Full debug log
│   ├── summary.csv              # One row per combination (status + time)
│   ├── male_icd.log             # Per-combo logs
│   ├── male_blocks.log
│   ├── male_chronic.log
│   ├── female_icd.log
│   ├── female_blocks.log
│   └── female_chronic.log
├── tables/{sex}/{node_type}/
│   ├── graph_stats.csv          # Nodes, edges, density per age
│   ├── alignment_residuals.csv  # Procrustes residuals per age
│   ├── drift.csv                # Per-node drift (long format)
│   ├── drift_summary.csv        # Mean/median/max drift per transition
│   ├── stability.csv            # Per-node kNN stability (long format)
│   ├── stability_summary.csv    # Mean/median/min stability per transition
│   ├── top_drifters.csv         # Top 20 nodes by mean drift
│   ├── structure_preservation.csv  # AUC + Spearman per age
│   └── robustness_summary.csv   # (if --robustness) Spearman rank correlations
├── figures/{sex}/{node_type}/
│   ├── graphs_gallery.png       # Network layouts for all ages
│   ├── pca_raw_age_{1..8}.png   # PCA scatter per age
│   ├── alignment_residuals.png  # Bar chart of Procrustes residuals
│   ├── drift_distribution.png   # Histogram of all drift values
│   ├── drift_by_transition.png  # Boxplot per age transition
│   ├── stability_distribution.png   # Histogram of stability
│   ├── stability_by_transition.png  # Boxplot per transition
│   └── structure_preservation.png   # AUC + Spearman bar charts
└── embeddings/{sex}/{node_type}/
    ├── raw/age_{1..8}.parquet       # Node2Vec embeddings
    ├── aligned/age_{1..8}.parquet   # After Procrustes alignment
    └── robustness/p=*_q=*/...      # (if --robustness)
```

## How to Interpret Key Outputs

### Graph Gallery (`graphs_gallery.png`)
Spring-layout visualization of each age group's comorbidity network. Node size reflects degree; edge width reflects weight. Compare visual density and structure across ages.

### Alignment Residuals (`alignment_residuals.png`)
Frobenius norm of residuals after Procrustes alignment to the reference age. Lower = better alignment. The reference age itself has residual 0. Large residuals suggest the embedding space changed structurally.

### Drift (`drift_distribution.png`, `drift_by_transition.png`)
L2 distance each disease node moved between consecutive aligned embeddings. High drift = the node's comorbidity neighbourhood changed substantially across that age transition. The boxplot reveals which transitions show the most change globally.

### Stability (`stability_distribution.png`, `stability_by_transition.png`)
Jaccard overlap of each node's k-nearest neighbours between consecutive ages. Values near 1 = stable neighbourhood; near 0 = complete turnover. Complements drift with a neighbourhood-level view.

### Top Drifters (`top_drifters.csv`)
The 20 disease codes with highest mean drift across all transitions. These are candidates for clinically meaningful age-dependent comorbidity shifts.

### Structure Preservation (`structure_preservation.csv`)
- **AUC**: How well cosine similarity in embedding space separates real edges from random non-edges (>0.5 = better than chance)
- **Spearman r**: Rank correlation between edge weights and embedding cosine similarity (positive = embeddings capture weight structure)

### Robustness (`robustness_summary.csv`)
Spearman rank correlation of per-node mean drift rankings across three (p,q) configurations. High correlation (>0.8) means drift findings are stable regardless of walk bias.

## Important Notes

- Results for each (sex, variant) combination are independent. Do not directly compare embedding coordinates across different combinations.
- Chronic variant lacks age group 1, so the reference age automatically falls back to 2.
- The `--force` flag only affects outputs under `final/outputs/`. It does not touch anything in the main `outputs/` directory.
- Second runs without `--force` are fast — the script loads cached embeddings and metrics.
