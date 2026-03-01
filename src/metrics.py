"""
Age-shift metrics module.

Computes metrics to quantify how disease embeddings change across age groups:
- Drift: Euclidean (L2) distance between a node's aligned embeddings in
  consecutive ages (proposal Section 3.4 definition).  Cosine distance is
  available as an optional alternative.
- kNN stability: Jaccard overlap of k-nearest neighbor sets (cosine-based by
  default) across ages

These metrics help identify diseases whose comorbidity patterns shift
significantly with age versus those that remain stable.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Cosine geometry helpers
# ---------------------------------------------------------------------------

def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix (1 - cosine_similarity).

    Args:
        X: 2-D array of shape (n, d) — each row is an embedding vector.

    Returns:
        Symmetric (n, n) array of cosine distances in [0, 2].
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    X_normed = X / norms
    sim = X_normed @ X_normed.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def _cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine distance between two vectors: 1 - cosine_similarity."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 1.0
    sim = float(np.dot(u, v) / (nu * nv))
    return 1.0 - max(-1.0, min(1.0, sim))


@dataclass
class DriftResult:
    """
    Drift metrics between consecutive age groups.

    Attributes:
        age_from: Starting age group
        age_to: Ending age group
        node_drifts: Dictionary mapping node -> drift distance (L2 by default)
        mean_drift: Mean drift across all nodes
        median_drift: Median drift across all nodes
        max_drift: Maximum drift
        num_nodes: Number of nodes with drift computed
    """
    age_from: int
    age_to: int
    node_drifts: Dict[str, float]
    mean_drift: float
    median_drift: float
    max_drift: float
    num_nodes: int


@dataclass
class StabilityResult:
    """
    kNN stability metrics between consecutive age groups.

    Attributes:
        age_from: Starting age group
        age_to: Ending age group
        node_stability: Dictionary mapping node -> Jaccard similarity of kNN sets
        mean_stability: Mean stability across all nodes
        median_stability: Median stability across all nodes
        min_stability: Minimum stability
        num_nodes: Number of nodes with stability computed
        k: Number of neighbors used
    """
    age_from: int
    age_to: int
    node_stability: Dict[str, float]
    mean_stability: float
    median_stability: float
    min_stability: float
    num_nodes: int
    k: int


def compute_drift(
    embeddings_from: Dict[str, np.ndarray],
    embeddings_to: Dict[str, np.ndarray],
    age_from: int,
    age_to: int,
    metric: str = "euclidean",
) -> DriftResult:
    """
    Compute drift for each node between two age groups.

    By default uses Euclidean (L2) distance as defined in the proposal
    (Section 3.4).  Pass ``metric="cosine"`` for cosine distance.

    Args:
        embeddings_from: Embeddings at starting age (aligned)
        embeddings_to: Embeddings at ending age (aligned)
        age_from: Starting age group number
        age_to: Ending age group number
        metric: Distance metric — ``"euclidean"`` (default) or ``"cosine"``

    Returns:
        DriftResult with per-node and aggregate drift metrics
    """
    # Find common nodes
    common_nodes = set(embeddings_from.keys()) & set(embeddings_to.keys())

    node_drifts = {}
    for node in common_nodes:
        vec_from = embeddings_from[node]
        vec_to = embeddings_to[node]
        if metric == "cosine":
            drift = _cosine_distance(vec_from, vec_to)
        else:
            drift = np.linalg.norm(vec_to - vec_from)
        node_drifts[node] = float(drift)

    drifts = list(node_drifts.values())

    return DriftResult(
        age_from=age_from,
        age_to=age_to,
        node_drifts=node_drifts,
        mean_drift=float(np.mean(drifts)) if drifts else 0.0,
        median_drift=float(np.median(drifts)) if drifts else 0.0,
        max_drift=float(np.max(drifts)) if drifts else 0.0,
        num_nodes=len(drifts),
    )


def _get_knn(
    node: str,
    embeddings: Dict[str, np.ndarray],
    k: int,
    metric: str = "cosine",
) -> Set[str]:
    """
    Get k-nearest neighbors of a node in embedding space.

    By default uses cosine similarity (highest similarity = nearest).
    Pass ``metric="euclidean"`` for L2-based neighbors.

    Args:
        node: Target node
        embeddings: All embeddings
        k: Number of neighbors
        metric: ``"cosine"`` (default) or ``"euclidean"``

    Returns:
        Set of k nearest neighbor node IDs (excluding the node itself)
    """
    if node not in embeddings:
        return set()

    target_vec = embeddings[node].reshape(1, -1)
    other_nodes = [n for n in embeddings.keys() if n != node]

    if len(other_nodes) < k:
        return set(other_nodes)

    other_vecs = np.array([embeddings[n] for n in other_nodes])

    if metric == "cosine":
        # Cosine similarity: largest = nearest → negate for argsort
        dists = cosine_distance_matrix(
            np.vstack([target_vec, other_vecs])
        )[0, 1:]
    else:
        dists = cdist(target_vec, other_vecs, metric="euclidean")[0]

    # Get indices of k smallest distances
    knn_indices = np.argsort(dists)[:k]
    return {other_nodes[i] for i in knn_indices}


def compute_knn_stability(
    embeddings_from: Dict[str, np.ndarray],
    embeddings_to: Dict[str, np.ndarray],
    age_from: int,
    age_to: int,
    k: int = 25,
    metric: str = "cosine",
) -> StabilityResult:
    """
    Compute kNN stability (Jaccard similarity) for each node between two ages.

    For each node, compares its k-nearest neighbors in both embedding spaces
    and computes Jaccard similarity: |kNN_from ∩ kNN_to| / |kNN_from ∪ kNN_to|.

    By default, neighbors are determined by cosine similarity (largest
    similarity = nearest), consistent with Word2Vec embedding geometry.
    Pass ``metric="euclidean"`` for L2-based neighbors.

    Args:
        embeddings_from: Embeddings at starting age (aligned)
        embeddings_to: Embeddings at ending age (aligned)
        age_from: Starting age group number
        age_to: Ending age group number
        k: Number of nearest neighbors
        metric: ``"cosine"`` (default) or ``"euclidean"``

    Returns:
        StabilityResult with per-node and aggregate stability metrics
    """
    # Find common nodes (need node present in both for kNN comparison)
    common_nodes = set(embeddings_from.keys()) & set(embeddings_to.keys())

    node_stability = {}
    for node in common_nodes:
        knn_from = _get_knn(node, embeddings_from, k, metric=metric)
        knn_to = _get_knn(node, embeddings_to, k, metric=metric)

        # Jaccard similarity
        intersection = len(knn_from & knn_to)
        union = len(knn_from | knn_to)
        jaccard = intersection / union if union > 0 else 0.0
        node_stability[node] = float(jaccard)

    stabilities = list(node_stability.values())

    return StabilityResult(
        age_from=age_from,
        age_to=age_to,
        node_stability=node_stability,
        mean_stability=float(np.mean(stabilities)) if stabilities else 0.0,
        median_stability=float(np.median(stabilities)) if stabilities else 0.0,
        min_stability=float(np.min(stabilities)) if stabilities else 0.0,
        num_nodes=len(stabilities),
        k=k,
    )


def compute_all_drifts(
    aligned_embeddings: Dict[int, Dict[str, np.ndarray]],
) -> List[DriftResult]:
    """
    Compute drift between all consecutive age groups.

    Args:
        aligned_embeddings: Dictionary mapping age -> embeddings (must be aligned)

    Returns:
        List of DriftResult for each consecutive pair (1->2, 2->3, ..., 7->8)
    """
    ages = sorted(aligned_embeddings.keys())
    results = []

    for i in range(len(ages) - 1):
        age_from, age_to = ages[i], ages[i + 1]
        result = compute_drift(
            aligned_embeddings[age_from],
            aligned_embeddings[age_to],
            age_from,
            age_to,
        )
        results.append(result)

    return results


def compute_all_knn_stability(
    aligned_embeddings: Dict[int, Dict[str, np.ndarray]],
    k: int = 25,
) -> List[StabilityResult]:
    """
    Compute kNN stability between all consecutive age groups.

    Args:
        aligned_embeddings: Dictionary mapping age -> embeddings (must be aligned)
        k: Number of nearest neighbors

    Returns:
        List of StabilityResult for each consecutive pair
    """
    ages = sorted(aligned_embeddings.keys())
    results = []

    for i in range(len(ages) - 1):
        age_from, age_to = ages[i], ages[i + 1]
        result = compute_knn_stability(
            aligned_embeddings[age_from],
            aligned_embeddings[age_to],
            age_from,
            age_to,
            k=k,
        )
        results.append(result)

    return results


def save_drift_results(
    path: Union[str, Path],
    results: List[DriftResult],
) -> None:
    """
    Save drift results to CSV.

    Creates two files:
    - {path}: Per-node drift values (long format)
    - {path.stem}_summary.csv: Aggregate statistics per age transition

    Args:
        path: Output file path for per-node results
        results: List of DriftResult objects
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Per-node results (long format)
    rows = []
    for r in results:
        for node, drift in r.node_drifts.items():
            rows.append({
                "age_from": r.age_from,
                "age_to": r.age_to,
                "node": node,
                "drift": drift,
            })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    # Summary statistics
    summary_rows = []
    for r in results:
        summary_rows.append({
            "age_from": r.age_from,
            "age_to": r.age_to,
            "mean_drift": r.mean_drift,
            "median_drift": r.median_drift,
            "max_drift": r.max_drift,
            "num_nodes": r.num_nodes,
        })

    summary_path = path.parent / f"{path.stem}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)


def save_stability_results(
    path: Union[str, Path],
    results: List[StabilityResult],
) -> None:
    """
    Save kNN stability results to CSV.

    Creates two files:
    - {path}: Per-node stability values (long format)
    - {path.stem}_summary.csv: Aggregate statistics per age transition

    Args:
        path: Output file path for per-node results
        results: List of StabilityResult objects
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Per-node results (long format)
    rows = []
    for r in results:
        for node, stability in r.node_stability.items():
            rows.append({
                "age_from": r.age_from,
                "age_to": r.age_to,
                "node": node,
                "knn_stability": stability,
            })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    # Summary statistics
    summary_rows = []
    for r in results:
        summary_rows.append({
            "age_from": r.age_from,
            "age_to": r.age_to,
            "k": r.k,
            "mean_stability": r.mean_stability,
            "median_stability": r.median_stability,
            "min_stability": r.min_stability,
            "num_nodes": r.num_nodes,
        })

    summary_path = path.parent / f"{path.stem}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
