# Data Directory

This directory is a placeholder. **No data files are committed to the repository.**

All data resides outside the repo at `../Data/` relative to the repository root.

## Data Structure

The external data directory contains:

```
../Data/
├── 1.Prevalence/           # Prevalence statistics by sex, age, year, ICD
├── 2.ContingencyTables/    # Contingency tables for comorbidity analysis
├── 3.AdjacencyMatrices/    # CSV + RDS adjacency matrices
└── 4.Graphs-gexffiles/     # GEXF graph files (primary input for this project)
```

## Setting DATA_ROOT

Set the `DATA_ROOT` environment variable to point to the data directory:

```bash
export DATA_ROOT="/path/to/Data"
```

If not set, scripts default to `../Data` relative to the repo root.
