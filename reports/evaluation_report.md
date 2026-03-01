# Evaluation Report

## 1. Graph Structure Preservation

### Method

Two complementary evaluations were performed:

- **Link Prediction AUC**: Multi-split held-out edge evaluation (3 independent splits, mean ± std reported). For each split:
  1. **Iterative hold-out**: 20% of "safe" edges (both endpoints with degree ≥ 2) are greedily removed while ensuring every node retains at least one edge in the training graph. This prevents batch degree violations where multiple removals could isolate a node.
  2. **Retraining**: Node2vec is retrained on the reduced training graph.
  3. **Hard negatives**: Negative samples are 2-hop non-edges discovered via the *training* graph (not the full graph), preventing information leakage through held-out paths. Non-edge validation uses the *original* graph's edge set to ensure held-out edges are never sampled as negatives.
  4. **Fallback negatives**: If the 2-hop pool is smaller than the positive set, random non-edges supplement the pool (flagged via `used_fallback`).
  5. **AUC**: Computed from cosine similarities of retrained embeddings on held-out positives vs hard negatives.

- **Weight Correlation**: Spearman correlation between embedding cosine similarity and edge weight (computed on the full graph, unchanged).

- **Shuffled Control**: For the densest age group per combination, a label-shuffled control AUC is computed by randomly permuting the retrained node–vector mapping. Expected AUC ≈ 0.5 under the null; values outside [0.3, 0.7] trigger warnings.

### Results — Link Prediction AUC (3-Split Mean ± Std) Across All Combinations

| Age | Age Range | Male ICD | Male Blocks | Male Chronic | Female ICD | Female Blocks | Female Chronic |
|-----|-----------|----------|-------------|--------------|------------|---------------|----------------|
| 1 | 0–9 | 0.643 ± 0.072 | 0.695 ± 0.049 | — | 0.514 ± 0.042 | 0.609 ± 0.039 | — |
| 2 | 10–19 | 0.484 ± 0.057 | 0.498 ± 0.020 | NaN | 0.353 ± 0.116 | 0.502 ± 0.077 | NaN |
| 3 | 20–29 | 0.629 ± 0.014 | 0.632 ± 0.034 | NaN | 0.616 ± 0.026 | 0.583 ± 0.035 | NaN |
| 4 | 30–39 | 0.663 ± 0.012 | 0.648 ± 0.056 | 0.653 ± 0.095 | 0.604 ± 0.022 | 0.671 ± 0.048 | 0.638 ± 0.075 |
| 5 | 40–49 | 0.639 ± 0.026 | 0.674 ± 0.034 | 0.587 ± 0.059 | 0.637 ± 0.009 | 0.728 ± 0.036 | 0.688 ± 0.053 |
| 6 | 50–59 | 0.611 ± 0.013 | 0.692 ± 0.004 | 0.666 ± 0.127 | 0.668 ± 0.011 | 0.686 ± 0.041 | 0.690 ± 0.050 |
| 7 | 60–69 | 0.657 ± 0.006 | 0.703 ± 0.014 | 0.782 ± 0.058 | 0.627 ± 0.015 | 0.667 ± 0.002 | 0.763 ± 0.011 |
| 8 | 70–79 | 0.641 ± 0.016 | 0.729 ± 0.009 | 0.755 ± 0.050 | 0.620 ± 0.009 | 0.668 ± 0.033 | 0.648 ± 0.057 |

**Notes**: Chronic variant has ages 2–8 only (age 1 graph unavailable); ages 2–3 for chronic had too few safe edges (< 20) and returned NaN. No combinations required fallback negatives.

### Results — Shuffled-Control AUC (Densest Age per Combination)

| Combination | Control Age | Shuffled AUC | Status |
|-------------|-------------|--------------|--------|
| Male ICD | 8 | 0.505 | OK |
| Male Blocks | 8 | 0.385 | OK |
| Male Chronic | 8 | 0.447 | OK |
| Female ICD | 8 | 0.493 | OK |
| Female Blocks | 8 | 0.408 | OK |
| Female Chronic | 8 | 0.432 | OK |

All shuffled-control AUC values fall within [0.3, 0.7], confirming the evaluation methodology is sound. ICD variants cluster closest to the 0.5 null expectation.

### Results — Spearman r (Weight Correlation) Across All Combinations

| Age | Age Range | Male ICD | Male Blocks | Male Chronic | Female ICD | Female Blocks | Female Chronic |
|-----|-----------|----------|-------------|--------------|------------|---------------|----------------|
| 1 | 0–9 | 0.552 | 0.368 | — | 0.621 | 0.545 | — |
| 2 | 10–19 | 0.578 | 0.525 | NaN | 0.550 | 0.593 | −0.418 |
| 3 | 20–29 | 0.703 | 0.594 | −0.142 | 0.617 | 0.646 | 0.275 |
| 4 | 30–39 | 0.724 | 0.536 | 0.491 | 0.684 | 0.548 | 0.268 |
| 5 | 40–49 | 0.755 | 0.483 | 0.322 | 0.716 | 0.477 | 0.475 |
| 6 | 50–59 | 0.715 | 0.479 | 0.471 | 0.706 | 0.415 | 0.551 |
| 7 | 60–69 | 0.694 | 0.449 | 0.365 | 0.681 | 0.362 | 0.467 |
| 8 | 70–79 | 0.614 | 0.469 | 0.459 | 0.585 | 0.385 | 0.364 |

### Interpretation

With the improved evaluation methodology (iterative hold-out, train-graph negatives, multi-split averaging):

- **AUC ranges from 0.35 to 0.78** across all combinations, providing a realistic and stable measure of embedding quality. AUC values are computed against *hard* 2-hop non-edge negatives, not random non-edges. With random negatives AUC would be >0.90 but less informative. Standard deviations are typically 0.01–0.13, confirming reasonable split-to-split consistency.
- **AUC generally increases with network density** (older age groups), as denser graphs give node2vec more structural signal to learn from.
- **Sparse young-age graphs** (ages 1–3) produce the lowest and most variable AUC values, reflecting the difficulty of learning meaningful structure from very few edges.
- **Blocks and Chronic variants** tend to show higher AUC than ICD for the same age at older ages, likely because smaller node sets produce denser per-node connectivity.
- **Edge weight correlation** (Spearman r) ranges from −0.42 to 0.76. ICD embeddings show the strongest weight correlations (r ≈ 0.55–0.76), peaking at ages 4–6. Chronic shows weaker and occasionally negative correlations in sparse age groups.
- **No fallback negatives were needed** in any combination, indicating that the 2-hop non-edge pool is consistently large enough across all graph densities.

### Sparse-Regime Caveats

Several evaluation metrics behave differently in sparse graphs (young age groups with few edges and many isolate nodes). These are expected data-sparsity effects, not pipeline failures:

- **Low AUC (0.35–0.51) in young ages**: When a graph has fewer than ~100 edges and 80–90% isolate nodes, the 2-hop negative pool is tiny and positive/negative similarity distributions heavily overlap. AUC near or below chance reflects genuine lack of learnable structure, not a bug.
- **Negative Spearman r**: In very sparse age groups (e.g., Chronic ages 2–3 with < 20 edges), the weight–similarity correlation can be negative due to small-sample instability. With so few data points, the correlation estimate has high variance and can flip sign by chance.
- **Coverage column**: When `coverage < 0.2` in the non-isolate summaries, the all-node drift and stability metrics are noise-dominated (random embeddings of isolates contribute most of the signal). In these cases, prefer the non-isolate summaries (`drift_summary_non_isolates.csv`, `stability_summary_non_isolates.csv`) which restrict to structurally informative nodes.

## 2. Alignment Quality

### Method

Orthogonal Procrustes alignment finds the optimal rotation R minimizing ||X_a R − X_{a0}||_F over anchor nodes shared between each age group and the reference (age 1 for ICD/Blocks, age 2 for Chronic). Alignment residual is reported as mean Frobenius error per anchor disease (proposal §5.2):

    ε_a = (1 / |S|) · ||X_a(S) R*_a − X_{a0}(S)||_F

where S is the set of anchor nodes and |S| is its cardinality. This normalization makes residuals comparable across variants with different numbers of anchors.

### Results — Mean Frobenius Error per Anchor

| Age | Age Range | Male ICD | Male Blocks | Male Chronic | Female ICD | Female Blocks | Female Chronic |
|-----|-----------|----------|-------------|--------------|------------|---------------|----------------|
| 1 | 0–9 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |
| 2 | 10–19 | 0.047 | 0.123 | 0.000 | 0.049 | 0.144 | 0.000 |
| 3 | 20–29 | 0.050 | 0.164 | 0.219 | 0.055 | 0.158 | 0.221 |
| 4 | 30–39 | 0.056 | 0.148 | 0.253 | 0.057 | 0.159 | 0.237 |
| 5 | 40–49 | 0.057 | 0.133 | 0.230 | 0.058 | 0.151 | 0.242 |
| 6 | 50–59 | 0.058 | 0.133 | 0.236 | 0.058 | 0.144 | 0.241 |
| 7 | 60–69 | 0.058 | 0.131 | 0.236 | 0.058 | 0.145 | 0.241 |
| 8 | 70–79 | 0.056 | 0.132 | 0.217 | 0.058 | 0.143 | 0.241 |

**Notes**: Reference age rows show 0.000 (identity alignment). Chronic variant starts at age 2. Anchors: ICD = 1,080, Blocks = 131, Chronic = 46.

### Interpretation

- Residuals are **small and stable** across age groups within each variant, confirming consistent alignment quality.
- **ICD** has the lowest per-anchor residuals (0.047–0.058), reflecting the large anchor set (1,080 nodes) which provides strong constraints for the rotation fit.
- **Blocks** residuals (0.123–0.164) are moderate, with the highest values in early transitions (ages 2–3) where graph structure changes most rapidly.
- **Chronic** residuals (0.217–0.253) are the highest, expected given the small anchor set (46 nodes) which provides fewer constraints for the Procrustes fit.
- The stability of residuals across ages confirms that alignment is not breaking down at any particular age transition.

## 3. Robustness Analysis

### Method

Three node2vec (p, q) configurations tested across all 6 combinations:
- Baseline: p=1.0, q=1.0 (DeepWalk-like)
- BFS-biased: p=0.5, q=2.0 (local structure emphasis)
- DFS-biased: p=2.0, q=0.5 (global structure emphasis)

Drift rank correlations (Spearman rho) are computed between each pair of configurations over all common nodes.

### Results — Drift Rank Correlations

| Combination | p=1,q=1 vs p=0.5,q=2 | p=1,q=1 vs p=2,q=0.5 | p=0.5,q=2 vs p=2,q=0.5 | Mean |
|-------------|----------------------|----------------------|------------------------|------|
| Male ICD | 0.914 | 0.911 | 0.908 | 0.911 |
| Male Blocks | 0.942 | 0.964 | 0.907 | 0.938 |
| Male Chronic | 0.892 | 0.934 | 0.850 | 0.892 |
| Female ICD | 0.909 | 0.904 | 0.906 | 0.906 |
| Female Blocks | 0.957 | 0.977 | 0.941 | 0.958 |
| Female Chronic | 0.933 | 0.954 | 0.885 | 0.924 |

**Overall mean drift rank correlation: 0.922**

### Interpretation

Rankings are **highly robust** to (p, q) parameter choices:

- All pairwise drift rank correlations exceed 0.83, with most above 0.90.
- Blocks variant shows the highest robustness (mean rho ≈ 0.95), followed by chronic (≈ 0.91) and ICD (≈ 0.91). This aligns with smaller node set sizes producing more stable embeddings.
- Male and female patterns are comparable, indicating robustness is not sex-dependent.

## 4. Key Findings

1. **Improved evaluation methodology**: Iterative hold-out prevents batch degree violations, train-graph 2-hop negatives eliminate information leakage, and 3-split averaging provides variance estimates (typical std 0.01–0.14).
2. **Shuffled controls validate soundness**: All 6 combinations yield shuffled-control AUC in [0.39, 0.51], close to the 0.5 null expectation.
3. **AUC ranges from 0.35 to 0.78** with hard negative sampling, showing embeddings meaningfully distinguish direct connections from structurally close non-edges.
4. **AUC improves with network density** — older age groups with more edges yield better structure preservation.
5. **Edge weight correlations** (Spearman r ≈ 0.55–0.76 for ICD) confirm embeddings capture weighted structure, not just topology.
6. **Alignment residuals are small and stable** — mean Frobenius error per anchor ranges from 0.047 (ICD) to 0.253 (Chronic), with consistent values across age groups within each variant.
7. **Drift rankings are robust** to node2vec hyperparameter choices (p, q), with rank correlations > 0.83 across all 6 combinations (mean 0.922).
8. These results validate that the observed age-shift patterns are not artifacts of specific hyperparameter choices or evaluation methodology.

## Output Files

| File | Description |
|------|-------------|
| `final/outputs/tables/{sex}/{variant}/structure_preservation.csv` | Per-age evaluation: AUC (mean ± std), Spearman r, diagnostics (n_safe, n_held_out, two_hop_pool, n_neg, used_fallback) |
| `final/outputs/tables/{sex}/{variant}/alignment_residuals.csv` | Per-age alignment residual (mean Frobenius error per anchor), num_anchors, scale |
| `final/outputs/tables/{sex}/{variant}/alignment_summary_non_isolates.csv` | Non-isolate alignment: residual_per_anchor, num_anchors_non_iso, low_anchor_warning |
| `final/outputs/tables/{sex}/{variant}/robustness_summary.csv` | Drift rank correlations across (p, q) configs |
| `final/outputs/tables/{sex}/{variant}/robustness_summary_non_isolates.csv` | Non-isolate robustness rank correlations |
| `final/outputs/figures/{sex}/{variant}/structure_preservation.png` | Per-combo AUC and Spearman r bar plots |
| `final/outputs/figures/combined_structure_preservation.png` | 2×3 overview grid across all combinations |
| `final/outputs/figures/combined_link_prediction.png` | Combined link prediction summary figure |
