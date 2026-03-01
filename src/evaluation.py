"""Canonical link-prediction evaluation for comorbidity embeddings.

This module is the single source of truth for ``cosine_sim`` and
``evaluate_structure``.  The pipeline runner (``final/run_all_combinations.py``)
imports from here.

Key design choices implemented here:
- **Iterative hold-out**: edges are greedily removed so every node keeps
  degree >= 1 in the training graph (no batch degree violations).
- **Train-graph negatives**: 2-hop non-edges are discovered via the training
  graph, avoiding information leakage through held-out paths.  Non-edge
  validation uses the *original* graph's edge set so held-out edges are never
  sampled as negatives.
- **Multi-split AUC**: ``n_splits`` independent hold-out splits are run;
  mean +/- std is reported.
- **Shuffled control**: optional label-shuffled control AUC on the first
  successful split.
- **Fallback negatives**: if the 2-hop pool is smaller than the positive set,
  random non-edges supplement the pool (flagged via ``used_fallback``).
- **Diagnostics columns**: per-split bookkeeping for downstream transparency.
"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score

from src.embeddings import train_node2vec


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(np.dot(u, v) / (nu * nv)) if nu > 0 and nv > 0 else 0.0


# ---------------------------------------------------------------------------
# Structure preservation evaluation
# ---------------------------------------------------------------------------

def evaluate_structure(
    graph: nx.Graph,
    emb_dict: Dict[str, np.ndarray],
    emb_params: dict,
    held_out_fraction: float = 0.2,
    seed: int = 42,
    compute_shuffled_control: bool = False,
    n_splits: int = 3,
) -> Dict[str, object]:
    """Link-prediction AUC (held-out edges) and weight-similarity Spearman r.

    Spearman r is computed on the full graph using the aligned embeddings.
    AUC is computed by holding out edges, retraining node2vec on the reduced
    graph, and scoring the held-out edges vs random non-edges.

    Multi-split: runs *n_splits* independent hold-out splits (varying the
    random seed) and reports mean +/- std AUC.  Shuffled control runs on the
    first successful split only.

    Args:
        graph: Original graph.
        emb_dict: Node label -> embedding vector mapping.
        emb_params: Parameters for node2vec retraining.
        held_out_fraction: Fraction of safe edges to hold out.
        seed: Base random seed (each split uses seed + split_index).
        compute_shuffled_control: Whether to compute shuffled-control AUC.
        n_splits: Number of independent hold-out splits.
    """
    emb_nodes = set(emb_dict.keys())

    empty_diagnostics = {
        "n_eligible": 0, "n_safe": 0, "n_held_out": 0,
        "train_nodes": 0, "train_edges": 0,
        "two_hop_pool_size": 0, "n_neg_sampled": 0,
        "used_fallback": False,
    }

    def _early(spearman_r_val, n_edges_val, diag=None):
        return {
            "spearman_r": spearman_r_val, "auc": np.nan, "auc_std": np.nan,
            "n_splits": 0, "auc_shuffled": np.nan, "n_edges": n_edges_val,
            "diagnostics": diag if diag is not None else empty_diagnostics,
        }

    # --- Spearman r on full graph (unchanged) ---
    sims, weights = [], []
    for u, v, d in graph.edges(data=True):
        if u in emb_nodes and v in emb_nodes:
            sims.append(cosine_sim(emb_dict[u], emb_dict[v]))
            weights.append(d.get("weight", 1.0))

    if len(sims) < 10:
        return _early(np.nan, len(sims))

    spearman_r, _ = sp_stats.spearmanr(sims, weights)

    # --- AUC with held-out edges (multi-split) ---

    # Collect eligible edges (both endpoints embedded)
    eligible = [
        (u, v) for u, v in graph.edges()
        if u in emb_nodes and v in emb_nodes
    ]
    if len(eligible) < 20:
        return _early(spearman_r, len(eligible),
                       {**empty_diagnostics, "n_eligible": len(eligible)})

    # IMPORTANT: full_edge_set is from the ORIGINAL graph, not train_graph.
    # This prevents held-out test edges from being sampled as negatives.
    full_edge_set = {(min(u, v), max(u, v)) for u, v in graph.edges()}

    # Pre-filter safe edges (degree >= 2); iterative selection refines further
    degree_orig = dict(graph.degree())
    safe = [(u, v) for u, v in eligible
            if degree_orig[u] >= 2 and degree_orig[v] >= 2]

    if len(safe) < 20:
        return _early(spearman_r, len(eligible),
                       {**empty_diagnostics, "n_eligible": len(eligible),
                        "n_safe": len(safe)})

    auc_splits: List[float] = []
    auc_shuffled = np.nan
    first_diagnostics = {**empty_diagnostics, "n_eligible": len(eligible),
                         "n_safe": len(safe)}

    for split_i in range(n_splits):
        split_seed = seed + split_i
        rng = np.random.default_rng(split_seed)

        # Iteratively select hold-out edges, updating degree counts
        # so every node retains degree >= 1 in the training graph.
        degree = dict(graph.degree())
        n_target = int(len(safe) * held_out_fraction)
        n_target = min(n_target, len(safe))

        candidates = list(safe)
        rng.shuffle(candidates)
        test_edges: List[Tuple[str, str]] = []
        for u, v in candidates:
            if len(test_edges) >= n_target:
                break
            if degree[u] >= 2 and degree[v] >= 2:
                test_edges.append((u, v))
                degree[u] -= 1
                degree[v] -= 1

        if len(test_edges) < 10:
            continue

        # Build training graph = full graph minus test edges
        train_graph = graph.copy()
        train_graph.remove_edges_from(test_edges)

        # Retrain node2vec on training graph
        retrained = train_node2vec(train_graph, **emb_params)
        retrained_set = set(retrained.keys())
        retrained_nodes = list(retrained_set & set(graph.nodes()))

        # Positive samples: cosine similarity of test edges (retrained embs)
        pos_sims_auc: List[float] = []
        for u, v in test_edges:
            if u in retrained_set and v in retrained_set:
                pos_sims_auc.append(cosine_sim(retrained[u], retrained[v]))

        if len(pos_sims_auc) < 10:
            continue

        # Hard negatives: 2-hop non-edges in TRAINING graph.
        # Non-edge check uses ORIGINAL graph (full_edge_set) to avoid
        # accidentally sampling held-out positives as negatives.
        two_hop_non_edges: set = set()
        for node in retrained_nodes:
            neighbors = set(train_graph.neighbors(node))
            for nbr in neighbors:
                for hop2 in train_graph.neighbors(nbr):
                    if hop2 != node and hop2 in retrained_set:
                        pair = (min(node, hop2), max(node, hop2))
                        if pair not in full_edge_set:
                            two_hop_non_edges.add(pair)

        hard_neg_list = list(two_hop_non_edges)
        rng.shuffle(hard_neg_list)
        n_neg = min(len(pos_sims_auc), len(hard_neg_list))

        # Fallback: supplement with random non-edges if 2-hop pool too small
        used_fallback = False
        if len(hard_neg_list) < len(pos_sims_auc):
            all_nodes = list(retrained_set)
            fallback_negs: set = set()
            max_attempts = len(pos_sims_auc) * 20
            for _ in range(max_attempts):
                i_r, j_r = rng.choice(len(all_nodes), size=2, replace=False)
                u_r, v_r = all_nodes[i_r], all_nodes[j_r]
                pair = (min(u_r, v_r), max(u_r, v_r))
                if pair not in full_edge_set and pair not in two_hop_non_edges:
                    fallback_negs.add(pair)
                    if len(fallback_negs) + len(hard_neg_list) >= len(pos_sims_auc):
                        break
            hard_neg_list.extend(list(fallback_negs))
            n_neg = min(len(pos_sims_auc), len(hard_neg_list))
            used_fallback = True

        neg_sims_auc: List[float] = [
            cosine_sim(retrained[u], retrained[v])
            for u, v in hard_neg_list[:n_neg]
        ]

        if len(neg_sims_auc) < 10:
            continue

        y_true = [1] * len(pos_sims_auc) + [0] * len(neg_sims_auc)
        y_score = pos_sims_auc + neg_sims_auc
        split_auc = roc_auc_score(y_true, y_score)
        auc_splits.append(split_auc)

        # Record diagnostics from first successful split
        if len(auc_splits) == 1:
            first_diagnostics = {
                "n_eligible": len(eligible),
                "n_safe": len(safe),
                "n_held_out": len(test_edges),
                "train_nodes": train_graph.number_of_nodes(),
                "train_edges": train_graph.number_of_edges(),
                "two_hop_pool_size": len(two_hop_non_edges),
                "n_neg_sampled": n_neg,
                "used_fallback": used_fallback,
            }

        # Shuffled control: only on first successful split
        if compute_shuffled_control and len(auc_splits) == 1:
            ctrl_rng = np.random.default_rng(split_seed + 999)
            nodes_list = list(retrained.keys())
            vecs = [retrained[n] for n in nodes_list]
            ctrl_rng.shuffle(vecs)
            shuffled = dict(zip(nodes_list, vecs))

            pos_sims_shuf = [
                cosine_sim(shuffled[u], shuffled[v])
                for u, v in test_edges
                if u in shuffled and v in shuffled
            ]
            neg_sims_shuf = [
                cosine_sim(shuffled[u], shuffled[v])
                for u, v in hard_neg_list[:n_neg]
            ]
            if len(pos_sims_shuf) >= 5 and len(neg_sims_shuf) >= 5:
                y_true_shuf = [1] * len(pos_sims_shuf) + [0] * len(neg_sims_shuf)
                y_score_shuf = pos_sims_shuf + neg_sims_shuf
                auc_shuffled = roc_auc_score(y_true_shuf, y_score_shuf)

    if not auc_splits:
        return _early(spearman_r, len(eligible), first_diagnostics)

    auc_mean = float(np.mean(auc_splits))
    auc_std = float(np.std(auc_splits)) if len(auc_splits) > 1 else 0.0

    return {
        "spearman_r": spearman_r,
        "auc": auc_mean,
        "auc_std": auc_std,
        "n_splits": len(auc_splits),
        "auc_shuffled": auc_shuffled,
        "n_edges": len(eligible),
        "diagnostics": first_diagnostics,
    }
