# Comorbidity Network Analysis Across Age Groups

## What This Project Does

This project investigates how disease comorbidity patterns change with patient age. Given population-level health data from Austria, it builds comorbidity networks (where nodes are diseases and edges represent statistically significant co-occurrence), embeds those networks into a continuous vector space, aligns the vector spaces across age groups, and measures which diseases experience the greatest age-related shifts in their comorbidity profiles.

The central question: **Which diseases have comorbidity patterns that change the most as patients age, and at which life stages do the biggest transitions occur?**

---

## The Data

The project consumes four layers of pre-processed epidemiological data, each derived from the previous:

### Layer 1 — Prevalence Data (`1.Prevalence/`)

A CSV file (`Prevalence_Sex_Age_Year_ICD.csv`) containing 135,708 records of per-disease prevalence rates. Each row records the prevalence of one ICD code for a given sex, age group, and year.

| Column | Description | Example Values |
|--------|-------------|----------------|
| `sex` | Patient sex | Female, Male |
| `Age_Group` | Decade bin | 0-9, 10-19, ..., 70-79 |
| `year` | Calendar year | 2003–2014 |
| `icd_code` | ICD-10 code | A01, E11, M54 |
| `p` | Prevalence rate | 0.00005 – 0.09266 |

Coverage: 2 sexes, 8 age groups, 12 years, ~1,060 unique ICD codes.

### Layer 2 — Contingency Tables (`2.ContingencyTables/`)

Six RDS files (R binary format) storing disease-pair co-occurrence counts in a 2x2 table layout (a/b/c/d cells). One file per sex x variant combination. These counts are the basis for computing association measures (relative risk, phi coefficient) that become edge weights.

| File | Size |
|------|------|
| `Blocks_ContingencyTables_{Sex}_Final.rds` | ~4.7 MB |
| `Chronic_ContingencyTables_{Sex}_Final.rds` | ~0.7 MB |
| `ICD_ContingencyTables_{Sex}_Final.rds` | ~108 MB |

### Layer 3 — Adjacency Matrices (`3.AdjacencyMatrices/`)

84 space-delimited CSV files, one per sex x variant x age group. Each file is a square matrix where rows and columns represent diseases, and cell values are the computed association weight. Most entries are zero (98%+ sparsity); non-zero entries represent statistically significant comorbidity associations.

Filename pattern: `Adj_Matrix_{Sex}_{Variant}_age_{N}.csv`

### Layer 4 — GEXF Graphs (`4.Graphs-gexffiles/`)

82 pre-built graph files in GEXF format (XML-based graph exchange). These are the primary input to the analysis pipeline. Each graph represents one sex x variant x age-group stratum.

Filename pattern: `Graph_{Sex}_{Variant}_Age_{N}.gexf`

| Property | Description |
|----------|-------------|
| **Nodes** | Diseases (keyed by ICD-10 codes or block ranges) |
| **Edges** | Comorbidity associations (undirected) |
| **Edge weight** | Association strength (higher = stronger comorbidity link) |

The graphs come in three granularity variants:

| Variant | Description | Node Count | Example Node |
|---------|-------------|------------|--------------|
| **blocks** | ICD-10 block-level aggregation | 131 | A00-A09 |
| **chronic** | Chronic disease categories | ~80 | (category labels) |
| **icd** | Individual ICD-10 codes | 1,080 | M54, E11 |

**Key observation**: Graph density increases dramatically with age. For Male/Blocks, edge count grows from 115 (age 10–19) to 1,290 (age 70–79) — older patients have far more comorbidity associations.

### Data Location

Data resides **outside the repository** at `../Data/` (relative to the repo root). Set the `DATA_ROOT` environment variable to override:

```bash
export DATA_ROOT="/path/to/Data"
```

### Stratification

All data is stratified by three dimensions:

| Dimension | Values |
|-----------|--------|
| **Sex** | `male`, `female` |
| **Variant** | `blocks`, `chronic`, `icd` |
| **Age group** | 1 (0–9), 2 (10–19), 3 (20–29), 4 (30–39), 5 (40–49), 6 (50–59), 7 (60–69), 8 (70–79) |

---

## How the Solution Is Built

The analysis pipeline has four stages. Each stage builds on the outputs of the previous one.

### Pipeline Workflow

```
                         ┌─────────────────────┐
                         │   GEXF Graph Files   │
                         │  (per sex/variant/   │
                         │     age group)        │
                         └──────────┬────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  STAGE 1: Train Embeddings    │
                    │  (node2vec → 128-d vectors)   │
                    │                               │
                    │  For each age group, run       │
                    │  random walks on the graph     │
                    │  and train Word2Vec to learn   │
                    │  a 128-d embedding per node.   │
                    └───────────────┬───────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Raw Embeddings      │
                         │  (one per age group, │
                         │   INCOMPARABLE across│
                         │   ages)              │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  STAGE 2: Procrustes Align    │
                    │  (rotate to shared space)     │
                    │                               │
                    │  Find the optimal rotation     │
                    │  matrix R that maps each       │
                    │  age's embeddings onto the     │
                    │  reference age (age 1).        │
                    └───────────────┬───────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Aligned Embeddings  │
                         │  (all ages in the    │
                         │   same coordinate    │
                         │   system — NOW       │
                         │   comparable)        │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  STAGE 3: Compute Metrics     │
                    │  (drift + kNN stability)      │
                    │                               │
                    │  For each disease, measure:    │
                    │  • Drift: L2 distance between  │
                    │    consecutive ages             │
                    │  • kNN stability: Jaccard       │
                    │    overlap of 25 nearest        │
                    │    neighbors between ages       │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  STAGE 4: Evaluate & Report   │
                    │                               │
                    │  • Link prediction AUC         │
                    │  • Robustness to (p,q) params  │
                    │  • Disease rankings             │
                    │  • Age-transition patterns      │
                    └───────────────────────────────┘
```

### Stage 1 — Node2Vec Embeddings

**What**: For each age group's comorbidity graph, run biased random walks and train a Word2Vec model to learn a 128-dimensional vector for each disease node.

**Why**: Graph embeddings convert the discrete graph structure into a continuous vector space where proximity reflects structural similarity. Two diseases that share many comorbidity partners will have nearby vectors.

**Parameters** (from `configs/default.yaml`):
- Dimensionality: 128
- Walk length: 120 steps
- Walks per node: 20
- Context window: 12
- p = 2.0, q = 0.5 (BFS-biased local exploration)

**Problem**: Each age group's embedding is trained independently, so the coordinate systems are arbitrary. Age 1's vectors and Age 2's vectors point in completely different directions — you cannot compare them directly.

### Stage 2 — Procrustes Alignment

**What**: Apply orthogonal Procrustes analysis to rotate each age's embeddings into the reference age's coordinate system.

**Why**: To compare how a disease's vector changes across ages, all vectors must live in the same coordinate space. Procrustes finds the rotation matrix R that minimizes ||X_ref - X_target * R|| using only orthogonal transformations (rotations and reflections), which preserve distances within each embedding.

**Anchor nodes**: Diseases present in both the reference and target age serve as correspondences. For the blocks variant, all 131 diseases appear at every age, giving 131 anchors — the maximum possible.

**Residual error**: Reported as mean Frobenius error per anchor disease (proposal §5.2: ε_a = (1/|S|) · ||X_a(S) R*_a − X_{a0}(S)||_F). This per-anchor normalization makes residuals comparable across variants with different numbers of anchors. Values range from 0.047 (ICD, 1,080 anchors) to 0.253 (Chronic, 46 anchors).

### Stage 3 — Age-Shift Metrics

Two complementary metrics quantify how disease comorbidity profiles change between consecutive age groups:

**Drift (L2 distance)**: For each disease, compute the Euclidean distance between its aligned embedding at age _t_ and age _t+1_. High drift means the disease's position in comorbidity space moved substantially. This follows the proposal definition: `drift_{a→a'}(v) = || x̃_{a'}(v) − x̃_a(v) ||_2`.

| Drift Range | Interpretation |
|-------------|----------------|
| Low (~0–1) | Stable comorbidity pattern |
| Medium (~1–3) | Typical age-related changes |
| High (>3) | Dramatic comorbidity shift |

**kNN Stability (Jaccard similarity)**: For each disease, find its k=25 nearest neighbors (by cosine similarity) at age _t_ and age _t+1_, then compute the Jaccard index (intersection / union). High stability means the disease's comorbidity neighborhood is preserved.

| Stability | Interpretation |
|-----------|----------------|
| 1.0 | Identical neighbors (perfect stability) |
| 0.5 | Half of neighbors change |
| 0.0 | Complete neighborhood turnover |

Drift magnitude uses Euclidean (L2) distance as defined in the proposal. kNN neighborhoods use cosine similarity as suggested in the proposal.

These metrics are complementary: drift measures magnitude of movement, stability measures structural preservation. They are typically negatively correlated.

### Stage 4 — Evaluation & Reporting

**Structure preservation**: Link prediction AUC (0.35–0.78 with hard 2-hop negatives and held-out retrained evaluation) and Spearman correlation between edge weight and cosine similarity confirm embeddings capture meaningful graph structure. AUC improves with network density (older age groups).

**Robustness**: Re-running the pipeline with different (p, q) parameters yields highly correlated disease rankings (Spearman rho > 0.83, mean > 0.91), confirming results are not artifacts of parameter choice.

---

## Project Structure

```
csh-tuwien-project/
├── configs/
│   ├── default.yaml              # All hyperparameters and paths
│   └── age_groups.yaml           # Age group 1–8 → decade ranges
├── src/                          # Library code (importable modules)
│   ├── inventory.py              # Discover and catalog GEXF files
│   ├── loader.py                 # Load GEXF → NetworkX with validation
│   ├── embeddings.py             # Node2vec training, save/load
│   ├── alignment.py              # Orthogonal Procrustes alignment (normalized residual)
│   ├── metrics.py                # Drift (L2) and kNN stability (cosine) computation
│   └── evaluation.py             # Link-prediction AUC + Spearman r evaluation
├── final/
│   ├── run_all_combinations.py   # Single entry point: runs all 6 pipeline combos
│   └── outputs/                  # Generated outputs (gitignored)
├── reports/
│   ├── report.md                 # Findings report
│   └── evaluation_report.md      # Embedding evaluation report
├── .gitignore
├── requirements.txt
└── README.md                     # This file
```

### Role of Each Component

| Component | Role |
|-----------|------|
| `configs/` | Single source of truth for all parameters (embedding dimensions, walk parameters, reference age, k for kNN). Change settings here, not in code. |
| `src/` | Reusable library code. Each module handles one pipeline stage. The pipeline runner imports from `src/`. |
| `final/` | Single entry point (`run_all_combinations.py`) that runs the full pipeline for all 6 sex×variant combinations. Outputs are stored under `final/outputs/` (gitignored). |
| `reports/` | Analysis reports in markdown: findings and evaluation. |

---

## Setup

### Requirements

Python 3.9+ with these packages:

- **Core**: `networkx`, `numpy`, `scipy`, `pandas`, `pyyaml`
- **Embeddings**: `gensim`
- **Storage**: `pyarrow`
- **Visualization**: `matplotlib`
- **Evaluation**: `scikit-learn` (for AUC computation)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Setup

Place the data directory at `../Data/` relative to the project root, or set:

```bash
export DATA_ROOT="/path/to/Data"
```

---

## Usage

### Run the Pipeline

```bash
# Standard run (uses cached outputs where possible):
python final/run_all_combinations.py

# Regenerate all outputs from scratch:
python final/run_all_combinations.py --force

# Include (p,q) robustness analysis:
python final/run_all_combinations.py --robustness

# Deterministic training (single worker):
python final/run_all_combinations.py --reproducible

# Include edge-weight threshold robustness:
python final/run_all_combinations.py --threshold-robustness

# Full run with all options:
python final/run_all_combinations.py --force --robustness --reproducible --threshold-robustness
```

**Note**: Chronic variant has ages 2–8 only; use `--reference-age 2` for chronic.

---

## Key Findings

### Single-stratum results (Male/Blocks)

- **Graph density increases with age**: Edge count grows from 115 (teens) to 1,290 (70s) — comorbidity networks become 10x denser.
- **Young-to-middle-age transitions are most volatile**: Drift peaks at the 10–19 → 20–29 transition (mean drift 1.41); kNN stability is lowest for the youngest transitions (0.19–0.21).
- **Older ages stabilize**: Drift drops to 0.76 and stability rises to 0.35–0.41 for the oldest transitions.
- **Top drifting diseases** (blocks): M95-M99 (musculoskeletal), N20-N23 (kidney), N40-N51 (prostate), H60-H62 (ear) — conditions whose comorbidity profiles are especially age-dependent.
- **Results are robust**: Disease-drift rankings are consistent across different node2vec parameter settings (Spearman rho > 0.91).

### Cross-sex and cross-variant comparison

- **Drift rankings are moderately correlated across sexes**: Spearman rho ~0.57 (blocks) and ~0.58 (ICD) between male and female drift rankings — the same diseases tend to drift in both sexes, but with meaningful sex-specific differences.
- **Blocks and ICD variants use disjoint node sets**: Block-level codes (e.g., "A00-A09") and individual ICD codes (e.g., "A00") have zero overlap, so cross-variant rank correlation is not directly computable.
- **16 diseases appear in the top-20 drifters across multiple strata**, suggesting these are genuinely age-sensitive conditions regardless of sex or granularity level.
- **Age-transition profiles are broadly consistent**: Both sexes and both variants show the same qualitative pattern of higher drift at younger transitions and stabilization at older ages.

---

## Configuration Reference

All parameters live in `configs/default.yaml`:

```yaml
embeddings:
  dim: 128            # Embedding dimensionality
  walk_length: 120    # Random walk length
  num_walks: 20       # Walks per node
  window: 12          # Word2Vec context window
  p: 2.0              # Return parameter (>1 = BFS-biased local exploration)
  q: 0.5              # In-out parameter (<1 = BFS-biased local exploration)
  workers: 4          # Parallel workers
  seed: 42            # Random seed

alignment:
  reference_age: 1    # Align all ages to this one

metrics:
  knn_k: 25           # Neighbors for kNN stability
```

### Age Groups

| Group | Range | Group | Range |
|-------|-------|-------|-------|
| 1 | 0–9 | 5 | 40–49 |
| 2 | 10–19 | 6 | 50–59 |
| 3 | 20–29 | 7 | 60–69 |
| 4 | 30–39 | 8 | 70–79 |

---

## Requirements vs Implementation

### Pipeline requirements

| Requirement | What had to be done | Where it's implemented |
|---|---|---|
| **Graph loading** | Parse GEXF files into NetworkX, handle GEXF version incompatibilities, validate node/edge structure | `src/loader.py` — `load_gexf_graph()` converts GEXF 1.3→1.2draft for NetworkX 3.x, returns `(nx.Graph, GraphStats)` |
| **Data inventory** | Discover and catalog all available GEXF files across sex/variant/age strata | `src/inventory.py` — `find_graph_file()` locates files by stratum |
| **Node2vec embeddings** | Train 128-d embeddings per age group using biased random walks on the comorbidity graph | `src/embeddings.py` — `train_node2vec()` wraps gensim Word2Vec; `save/load_embeddings()` handle parquet I/O |
| **Procrustes alignment** | Rotate each age's embedding space onto a shared reference so cross-age comparison is valid; residual = mean Frobenius error per anchor (§5.2) | `src/alignment.py` — `align_all_to_reference()` applies scipy's `orthogonal_procrustes` using all common nodes as anchors |
| **Drift metric** | Measure L2 distance per disease between consecutive aligned ages | `src/metrics.py` — `compute_all_drifts()` produces `DriftResult` with per-node and aggregate statistics |
| **kNN stability metric** | Measure Jaccard overlap of cosine-based k-nearest-neighbor sets between consecutive ages | `src/metrics.py` — `compute_all_knn_stability()` produces `StabilityResult` (k=25, cosine neighbors) |
| **Embedding quality validation** | Confirm embeddings faithfully represent graph structure (link prediction AUC, weight correlation) | `src/evaluation.py` + `final/run_all_combinations.py` step F — AUC 0.35–0.78 (hard 2-hop negatives, held-out retrain) and Spearman r across ages |
| **Robustness check** | Verify drift rankings are not artifacts of node2vec (p,q) parameter choices | `final/run_all_combinations.py` step G — re-trains with 3 (p,q) configs, all pairwise Spearman rho > 0.83 |
| **Cross-sex comparison** | Compare male vs female age-shift dynamics for the same variant | Cross-comparison analysis — overlay drift/stability curves, scatter plots with Spearman correlation |
| **Cross-variant comparison** | Compare blocks vs ICD granularity for the same sex | Cross-comparison analysis — side-by-side bar charts per sex; note: node sets are disjoint so direct rank correlation is N/A |
| **Top-drifter identification** | Rank diseases by mean drift, find which are consistently high-drift across strata | Per-stratum top-20 lists, heatmap, 16 diseases shared across multiple strata |

### Infrastructure requirements

| Requirement | What had to be done | Where it's implemented |
|---|---|---|
| **Reproducibility** | All parameters in config files, not hardcoded; outputs regenerated from source data; `--reproducible` flag for deterministic training | `configs/default.yaml`, `configs/age_groups.yaml`; all outputs are gitignored; `run_metadata.json` tracks parameters |
| **Caching** | Avoid recomputation when outputs already exist; detect parameter changes | `cached()` function + `run_metadata.json` parameter mismatch warnings |
| **Output verification** | Confirm all expected files and columns are present after each run | `verify_outputs()` checks columns, file existence, and AUC sanity |

