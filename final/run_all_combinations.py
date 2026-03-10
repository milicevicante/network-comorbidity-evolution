#!/usr/bin/env python3
"""
Run the full comorbidity-embedding pipeline for all (sex, node_type) combinations.

Produces a self-contained output tree under final/outputs/ with figures, tables,
embeddings, and logs for each of the 6 combinations (2 sexes × 3 variants).

Usage:
    python final/run_all_combinations.py
    python final/run_all_combinations.py --force                       # retrain everything
    python final/run_all_combinations.py --robustness                  # include robustness check
    python final/run_all_combinations.py --reproducible                # deterministic (workers=1)
    python final/run_all_combinations.py --threshold-robustness        # edge-weight filtering robustness
    python final/run_all_combinations.py --force --robustness --reproducible --threshold-robustness
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from scipy import stats as sp_stats
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.inventory import find_graph_file
from src.loader import load_gexf_graph
from src.embeddings import train_node2vec, save_embeddings, load_embeddings
from src.alignment import align_all_to_reference, save_alignment_results
from src.metrics import (
    compute_drift,
    compute_knn_stability,
    compute_all_drifts,
    compute_all_knn_stability,
    save_drift_results,
    save_stability_results,
)
from src.evaluation import cosine_sim, evaluate_structure

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEXES = ["male", "female"]
NODE_TYPES = ["icd", "blocks", "chronic"]
ALL_AGES = list(range(1, 9))

OUTPUT_ROOT_DEFAULT = Path(__file__).resolve().parent / "outputs"
OUTPUT_ROOT = OUTPUT_ROOT_DEFAULT  # may be overridden by --output-dir

# cosine_sim and evaluate_structure imported from src.evaluation


def plot_graph_on_ax(ax: plt.Axes, G: nx.Graph, title: str) -> None:
    """Draw non-isolate subgraph on a matplotlib axes."""
    non_iso = [n for n, d in G.degree() if d > 0]
    if len(non_iso) == 0:
        ax.text(0.5, 0.5, "No edges", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        return

    H = G.subgraph(non_iso).copy()
    degrees = dict(H.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [8 + 60 * (degrees[n] / max_deg) for n in H.nodes()]

    ws = [d.get("weight", 1.0) for _, _, d in H.edges(data=True)]
    max_w = max(ws) if ws else 1.0
    edge_widths = [0.2 + 1.5 * (w / max_w) for w in ws]

    pos = nx.spring_layout(H, k=1.5 / np.sqrt(max(len(H), 1)), seed=42, iterations=50)
    nx.draw_networkx_edges(H, pos, ax=ax, width=edge_widths, alpha=0.3, edge_color="gray")
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=node_sizes, node_color="steelblue",
                           alpha=0.7, edgecolors="k", linewidths=0.3)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def cached(path: Path, force: bool) -> bool:
    """Return True if *path* exists and we are NOT forcing regeneration."""
    return path.exists() and not force


# ---------------------------------------------------------------------------
# Run metadata helpers (cache safety)
# ---------------------------------------------------------------------------

def _get_git_hash() -> str:
    """Return short git hash of HEAD, or 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _build_run_metadata(emb_params: dict, reference_age: int, knn_k: int) -> dict:
    """Capture all reproducibility-relevant parameters."""
    return {
        "dim": emb_params["dim"],
        "walk_length": emb_params["walk_length"],
        "num_walks": emb_params["num_walks"],
        "window": emb_params["window"],
        "p": emb_params["p"],
        "q": emb_params["q"],
        "seed": emb_params["seed"],
        "workers": emb_params["workers"],
        "reference_age": reference_age,
        "knn_k": knn_k,
        "git_hash": _get_git_hash(),
    }


def _check_metadata(
    meta_path: Path, current_meta: dict, log: logging.Logger
) -> None:
    """Warn if cached metadata differs from current parameters."""
    if not meta_path.exists():
        return
    try:
        with open(meta_path) as f:
            old_meta = json.load(f)
    except Exception:
        return
    for key in current_meta:
        if key == "git_hash":
            continue
        if key in old_meta and old_meta[key] != current_meta[key]:
            log.warning(
                "  PARAM MISMATCH: %s was %s, now %s. "
                "Cached outputs may be stale — consider --force.",
                key, old_meta[key], current_meta[key],
            )


def _write_metadata(meta_path: Path, meta: dict) -> None:
    """Write run metadata JSON."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------------

def verify_outputs(
    sex: str, node_type: str, log: logging.Logger
) -> List[str]:
    """Check that expected output files and columns exist. Returns warnings."""
    tbl_dir = OUTPUT_ROOT / "tables" / sex / node_type
    warnings: List[str] = []

    def _check_csv_cols(filename: str, required_cols: List[str]) -> None:
        path = tbl_dir / filename
        if not path.exists():
            warnings.append(f"Missing file: {filename}")
            return
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        for col in required_cols:
            if col not in cols:
                warnings.append(f"{filename}: missing column '{col}'")

    # 1. graph_stats.csv
    _check_csv_cols("graph_stats.csv", ["n_isolates", "isolate_frac"])

    # 2. alignment_summary_non_isolates.csv
    _check_csv_cols("alignment_summary_non_isolates.csv", ["low_anchor_warning"])

    # 3. stability_summary_non_isolates.csv
    _check_csv_cols("stability_summary_non_isolates.csv", ["low_n_warning"])

    # 4. structure_preservation.csv
    _check_csv_cols("structure_preservation.csv", [
        "auc_std", "n_splits", "n_safe", "n_held_out",
        "two_hop_pool", "n_neg", "used_fallback",
    ])

    # 5. auc_shuffled sanity check
    sp_path = tbl_dir / "structure_preservation.csv"
    if sp_path.exists():
        sp_df = pd.read_csv(sp_path)
        if "auc_shuffled" in sp_df.columns:
            valid = sp_df["auc_shuffled"].dropna()
            bad = valid[(valid < 0.3) | (valid > 0.7)]
            if len(bad) > 0:
                warnings.append(
                    f"structure_preservation.csv: {len(bad)} auc_shuffled "
                    f"values outside [0.3, 0.7]"
                )

    # 6. run_metadata.json
    if not (tbl_dir / "run_metadata.json").exists():
        warnings.append("Missing file: run_metadata.json")

    if warnings:
        for w in warnings:
            log.warning("  VERIFY: %s", w)
    else:
        log.info("  VERIFY: all output checks passed")

    return warnings


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure root logger with console + global file handler."""
    log_dir = OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    # Console (INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                      datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    # Global file (DEBUG)
    fh = logging.FileHandler(log_dir / "run_all.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(fh)

    return logger


def add_combo_file_handler(
    logger: logging.Logger, sex: str, node_type: str
) -> logging.FileHandler:
    """Add a per-combo log file handler and return it (for later removal)."""
    log_dir = OUTPUT_ROOT / "logs"
    fh = logging.FileHandler(log_dir / f"{sex}_{node_type}.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(fh)
    return fh


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_a_graph_inspection(
    sex: str,
    node_type: str,
    data_root: Path,
    age_labels: Dict[int, str],
    reference_age: int,
    force: bool,
    log: logging.Logger,
) -> Tuple[Dict[int, nx.Graph], int]:
    """Load graphs, save stats CSV and gallery figure. Returns (graphs, ref_age)."""
    fig_dir = OUTPUT_ROOT / "figures" / sex / node_type
    tbl_dir = OUTPUT_ROOT / "tables" / sex / node_type
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    graphs: Dict[int, nx.Graph] = {}
    stats_rows = []

    for age in ALL_AGES:
        gpath = find_graph_file(data_root, sex, node_type, age_group=age)
        if gpath is None:
            log.warning("  Graph missing: sex=%s variant=%s age=%d", sex, node_type, age)
            continue
        G, st = load_gexf_graph(gpath)
        graphs[age] = G
        density = (2 * st.edge_count) / (st.node_count * (st.node_count - 1)) if st.node_count > 1 else 0
        mean_deg = 2 * st.edge_count / st.node_count if st.node_count > 0 else 0
        n_isolates = sum(1 for _, d in G.degree() if d == 0)
        stats_rows.append({
            "age": age,
            "age_range": age_labels.get(age, "?"),
            "nodes": st.node_count,
            "edges": st.edge_count,
            "density": round(density, 6),
            "mean_degree": round(mean_deg, 2),
            "components": st.num_components,
            "n_isolates": n_isolates,
            "isolate_frac": round(n_isolates / st.node_count, 4) if st.node_count > 0 else 0,
        })

    if not graphs:
        raise RuntimeError(f"No graphs found for {sex}/{node_type}")

    # Save stats CSV
    pd.DataFrame(stats_rows).to_csv(tbl_dir / "graph_stats.csv", index=False)
    log.info("  [A] Loaded %d graphs, saved graph_stats.csv", len(graphs))

    # Auto-detect reference age
    available = sorted(graphs.keys())
    if reference_age in available:
        ref_age = reference_age
    else:
        ref_age = available[0]
        log.warning("  Reference age %d unavailable, falling back to %d", reference_age, ref_age)

    # Gallery figure
    gallery_path = fig_dir / "graphs_gallery.png"
    if not cached(gallery_path, force):
        ncols = len(available)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        if ncols == 1:
            axes = [axes]
        for ax, age in zip(axes, available):
            G = graphs[age]
            n_e = G.number_of_edges()
            non_iso = sum(1 for _, d in G.degree() if d > 0)
            label = age_labels.get(age, str(age))
            title = f"Age {label}\n{non_iso} nodes, {n_e} edges"
            plot_graph_on_ax(ax, G, title)
        fig.suptitle(f"{sex.title()} / {node_type.upper()} — Graph Gallery", fontsize=12)
        fig.tight_layout()
        fig.savefig(gallery_path, dpi=150)
        plt.close(fig)
        log.info("  [A] Saved graphs_gallery.png")

    # Individual graph panels per age
    for age in available:
        ind_path = fig_dir / f"graph_age_{age}.png"
        if not cached(ind_path, force):
            G = graphs[age]
            n_e = G.number_of_edges()
            non_iso = sum(1 for _, d in G.degree() if d > 0)
            label = age_labels.get(age, str(age))
            title = f"Age {label}\n{non_iso} nodes, {n_e} edges"
            fig_i, ax_i = plt.subplots(figsize=(6, 5))
            plot_graph_on_ax(ax_i, G, title)
            fig_i.suptitle(f"{sex.title()} / {node_type.upper()}", fontsize=12)
            fig_i.tight_layout()
            fig_i.savefig(ind_path, dpi=150, bbox_inches="tight")
            plt.close(fig_i)
    log.info("  [A] Saved individual graph panels")

    return graphs, ref_age


def step_b_embeddings(
    sex: str,
    node_type: str,
    graphs: Dict[int, nx.Graph],
    emb_params: dict,
    force: bool,
    log: logging.Logger,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Train or load node2vec embeddings per age."""
    emb_dir = OUTPUT_ROOT / "embeddings" / sex / node_type / "raw"
    emb_dir.mkdir(parents=True, exist_ok=True)

    embeddings_by_age: Dict[int, Dict[str, np.ndarray]] = {}

    for age, G in sorted(graphs.items()):
        emb_path = emb_dir / f"age_{age}.parquet"
        if cached(emb_path, force):
            embeddings_by_age[age] = load_embeddings(emb_path)
            log.debug("  [B] Loaded cached embeddings age %d", age)
        else:
            log.info("  [B] Training embeddings age %d (%d nodes) ...", age, G.number_of_nodes())
            emb = train_node2vec(
                G,
                dim=emb_params["dim"],
                walk_length=emb_params["walk_length"],
                num_walks=emb_params["num_walks"],
                window=emb_params["window"],
                p=emb_params["p"],
                q=emb_params["q"],
                workers=emb_params["workers"],
                seed=emb_params["seed"],
            )
            save_embeddings(emb_path, emb)
            embeddings_by_age[age] = emb
            log.info("  [B] Saved embeddings age %d → %s", age, emb_path.name)

    return embeddings_by_age


def step_c_pca_plots(
    sex: str,
    node_type: str,
    graphs: Dict[int, nx.Graph],
    embeddings_by_age: Dict[int, Dict[str, np.ndarray]],
    age_labels: Dict[int, str],
    force: bool,
    log: logging.Logger,
) -> None:
    """PCA scatter of raw embeddings coloured by node degree."""
    fig_dir = OUTPUT_ROOT / "figures" / sex / node_type
    fig_dir.mkdir(parents=True, exist_ok=True)

    for age in sorted(embeddings_by_age.keys()):
        out_path = fig_dir / f"pca_raw_age_{age}.png"
        if cached(out_path, force):
            continue
        emb = embeddings_by_age[age]
        G = graphs[age]
        nodes = sorted(emb.keys())
        X = np.array([emb[n] for n in nodes])
        degrees = np.array([G.degree(n) if n in G else 0 for n in nodes])

        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(X2[:, 0], X2[:, 1], c=degrees, cmap="viridis", s=15, alpha=0.7)
        fig.colorbar(sc, ax=ax, label="Node degree")
        label = age_labels.get(age, str(age))
        ax.set_title(f"PCA of raw embeddings — Age {label} ({sex}/{node_type})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

    log.info("  [C] Saved PCA plots")


def _compute_non_isolate_anchors(
    G_ref: nx.Graph,
    G_target: nx.Graph,
    embeddings_ref: Dict[str, np.ndarray],
    embeddings_target: Dict[str, np.ndarray],
) -> Set[str]:
    """Return nodes with degree > 0 in both graphs that also exist in both embeddings."""
    non_iso_ref = {n for n, d in G_ref.degree() if d > 0}
    non_iso_target = {n for n, d in G_target.degree() if d > 0}
    emb_common = set(embeddings_ref.keys()) & set(embeddings_target.keys())
    return non_iso_ref & non_iso_target & emb_common


def step_d_alignment(
    sex: str,
    node_type: str,
    embeddings_by_age: Dict[int, Dict[str, np.ndarray]],
    graphs: Dict[int, nx.Graph],
    ref_age: int,
    force: bool,
    log: logging.Logger,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, np.ndarray]]]:
    """Procrustes alignment, save aligned embeddings + residuals.

    Returns:
        (aligned, aligned_non_iso): aligned uses all anchors (backwards-compat),
        aligned_non_iso uses only non-isolate anchors (for isolate-aware metrics).
    """
    from src.alignment import procrustes_align, AlignmentResult

    aligned_dir = OUTPUT_ROOT / "embeddings" / sex / node_type / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    ni_aligned_dir = OUTPUT_ROOT / "embeddings" / sex / node_type / "aligned_non_iso"
    ni_aligned_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir = OUTPUT_ROOT / "tables" / sex / node_type
    fig_dir = OUTPUT_ROOT / "figures" / sex / node_type

    # ---- All-anchor alignment (existing, backwards-compatible) ----
    all_cached = all(
        cached(aligned_dir / f"age_{a}.parquet", force)
        for a in embeddings_by_age
    )

    if all_cached:
        aligned = {}
        for a in sorted(embeddings_by_age.keys()):
            aligned[a] = load_embeddings(aligned_dir / f"age_{a}.parquet")
        log.info("  [D] Loaded cached aligned embeddings (ref=%d)", ref_age)
    else:
        log.info("  [D] Aligning to reference age %d ...", ref_age)
        aligned, results = align_all_to_reference(embeddings_by_age, reference_age=ref_age)

        # Save aligned embeddings
        for a, emb in sorted(aligned.items()):
            save_embeddings(aligned_dir / f"age_{a}.parquet", emb)

        # Save residuals CSV
        result_tuples = [(a, results[a]) for a in sorted(results.keys())]
        save_alignment_results(tbl_dir / "alignment_residuals.csv", result_tuples)

        # Residuals plot
        res_path = fig_dir / "alignment_residuals.png"
        ages_sorted = sorted(results.keys())
        residuals = [results[a].residual_error for a in ages_sorted]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([str(a) for a in ages_sorted], residuals, color="steelblue")
        ax.set_xlabel("Age group")
        ax.set_ylabel("Mean Frobenius error per anchor")
        ax.set_title(f"Alignment residuals ({sex}/{node_type}, ref={ref_age})")
        fig.tight_layout()
        fig.savefig(res_path, dpi=120)
        plt.close(fig)

        log.info("  [D] Alignment complete, saved residuals")

    # ---- Non-isolate alignment (fit Procrustes using only degree>0 anchors) ----
    ni_all_cached = all(
        cached(ni_aligned_dir / f"age_{a}.parquet", force)
        for a in embeddings_by_age
    )
    ni_summary_path = tbl_dir / "alignment_summary_non_isolates.csv"

    if ni_all_cached and cached(ni_summary_path, force):
        aligned_non_iso = {}
        for a in sorted(embeddings_by_age.keys()):
            aligned_non_iso[a] = load_embeddings(ni_aligned_dir / f"age_{a}.parquet")
        log.info("  [D] Loaded cached non-isolate aligned embeddings")
    else:
        log.info("  [D] Aligning with non-isolate anchors ...")
        ref_emb = embeddings_by_age[ref_age]
        G_ref = graphs.get(ref_age)
        aligned_non_iso = {}
        ni_rows = []

        for age in sorted(embeddings_by_age.keys()):
            if age == ref_age:
                # Reference is not transformed
                aligned_non_iso[age] = ref_emb.copy()
                ni_rows.append({
                    "age_group": age,
                    "num_anchors_non_iso": sum(1 for _, d in G_ref.degree() if d > 0) if G_ref else 0,
                    "residual_per_anchor": 0.0,
                    "low_anchor_warning": False,
                })
                continue

            G_target = graphs.get(age)
            if G_ref is None or G_target is None:
                # Fallback: use all-anchor alignment
                aligned_non_iso[age] = aligned[age]
                continue

            non_iso = _compute_non_isolate_anchors(
                G_ref, G_target, ref_emb, embeddings_by_age[age]
            )

            if len(non_iso) < 2:
                log.warning(
                    "  [D] Non-isolate anchors < 2 for age %d (%d found), "
                    "falling back to all-anchor alignment",
                    age, len(non_iso),
                )
                aligned_non_iso[age] = aligned[age]
                ni_rows.append({
                    "age_group": age,
                    "num_anchors_non_iso": len(non_iso),
                    "residual_per_anchor": np.nan,
                    "low_anchor_warning": True,
                })
                continue

            # Fit Procrustes on non-isolate anchors, apply R to ALL target nodes
            ni_aligned_emb, ni_result = procrustes_align(
                ref_emb, embeddings_by_age[age], anchor_nodes=non_iso
            )
            aligned_non_iso[age] = ni_aligned_emb
            ni_rows.append({
                "age_group": age,
                "num_anchors_non_iso": ni_result.num_anchors,
                "residual_per_anchor": ni_result.residual_error,
                "low_anchor_warning": ni_result.num_anchors < 25,
            })

        # Save non-isolate aligned embeddings
        for a, emb in sorted(aligned_non_iso.items()):
            save_embeddings(ni_aligned_dir / f"age_{a}.parquet", emb)

        pd.DataFrame(ni_rows).to_csv(ni_summary_path, index=False)
        log.info("  [D] Non-isolate alignment complete, saved to aligned_non_iso/")

    return aligned, aligned_non_iso


def _get_isolate_info(
    graphs: Dict[int, nx.Graph], a_from: int, a_to: int
) -> Tuple[int, int, float, float]:
    """Return (n_nodes_union, n_nodes_from, isolate_frac_from, isolate_frac_to)."""
    G_from, G_to = graphs.get(a_from), graphs.get(a_to)
    if G_from is None or G_to is None:
        return 0, 0, 0.0, 0.0
    n_all = len(set(G_from.nodes()) | set(G_to.nodes()))
    n_from = G_from.number_of_nodes()
    n_to = G_to.number_of_nodes()
    iso_from = sum(1 for _, d in G_from.degree() if d == 0)
    iso_to = sum(1 for _, d in G_to.degree() if d == 0)
    frac_from = iso_from / n_from if n_from > 0 else 0.0
    frac_to = iso_to / n_to if n_to > 0 else 0.0
    return n_all, n_from, frac_from, frac_to


def step_e_metrics(
    sex: str,
    node_type: str,
    aligned: Dict[int, Dict[str, np.ndarray]],
    aligned_non_iso: Dict[int, Dict[str, np.ndarray]],
    graphs: Dict[int, nx.Graph],
    knn_k: int,
    age_labels: Dict[int, str],
    force: bool,
    log: logging.Logger,
) -> None:
    """Drift and kNN stability metrics + plots."""
    tbl_dir = OUTPUT_ROOT / "tables" / sex / node_type
    fig_dir = OUTPUT_ROOT / "figures" / sex / node_type

    # --- Drift ---
    drift_path = tbl_dir / "drift.csv"
    if cached(drift_path, force):
        drift_df = pd.read_csv(drift_path)
        log.info("  [E] Loaded cached drift.csv")
    else:
        drift_results = compute_all_drifts(aligned)
        save_drift_results(drift_path, drift_results)
        drift_df = pd.read_csv(drift_path)
        log.info("  [E] Computed and saved drift")

    # --- Stability ---
    stab_path = tbl_dir / "stability.csv"
    if cached(stab_path, force):
        stab_df = pd.read_csv(stab_path)
        log.info("  [E] Loaded cached stability.csv")
    else:
        stab_results = compute_all_knn_stability(aligned, k=knn_k)
        save_stability_results(stab_path, stab_results)
        stab_df = pd.read_csv(stab_path)
        log.info("  [E] Computed and saved stability")

    # --- Top drifters ---
    top_path = tbl_dir / "top_drifters.csv"
    if not cached(top_path, force):
        mean_drift = drift_df.groupby("node")["drift"].mean().sort_values(ascending=False)
        top = mean_drift.head(20).reset_index()
        top.columns = ["node", "mean_drift"]
        top.to_csv(top_path, index=False)
        log.info("  [E] Saved top_drifters.csv")

    # ===================================================================
    # Non-isolate drift / stability
    # ===================================================================
    ni_drift_summary_path = tbl_dir / "drift_summary_non_isolates.csv"
    ni_stab_summary_path = tbl_dir / "stability_summary_non_isolates.csv"

    if not cached(ni_drift_summary_path, force) or not cached(ni_stab_summary_path, force):
        ages = sorted(aligned_non_iso.keys())
        ni_drift_rows = []
        ni_stab_rows = []

        for i in range(len(ages) - 1):
            a_from, a_to = ages[i], ages[i + 1]
            G_from, G_to = graphs.get(a_from), graphs.get(a_to)
            if G_from is None or G_to is None:
                continue

            # Nodes with degree > 0 in both endpoint graphs
            non_iso = (
                {n for n, d in G_from.degree() if d > 0}
                & {n for n, d in G_to.degree() if d > 0}
            )

            # Must be present in non-isolate-aligned embeddings
            non_iso = non_iso & set(aligned_non_iso[a_from].keys()) & set(aligned_non_iso[a_to].keys())

            # Coverage info
            all_common = set(aligned[a_from].keys()) & set(aligned[a_to].keys())
            num_nodes_all = len(all_common)
            num_nodes_non_iso = len(non_iso)
            coverage = num_nodes_non_iso / num_nodes_all if num_nodes_all > 0 else 0.0
            _, _, iso_frac_from, iso_frac_to = _get_isolate_info(graphs, a_from, a_to)

            # --- Non-isolate drift (using non-isolate-aligned embeddings) ---
            if len(non_iso) < 2:
                log.warning(
                    "  [E] Non-isolate nodes < 2 for transition %d\u2192%d (%d nodes), "
                    "recording NaN drift",
                    a_from, a_to, len(non_iso),
                )
                ni_drift_rows.append({
                    "age_from": a_from, "age_to": a_to,
                    "mean_drift": np.nan, "median_drift": np.nan, "max_drift": np.nan,
                    "num_nodes_all": num_nodes_all,
                    "num_nodes_non_iso": num_nodes_non_iso,
                    "coverage": round(coverage, 4),
                    "isolate_frac_from": round(iso_frac_from, 4),
                    "isolate_frac_to": round(iso_frac_to, 4),
                })
            else:
                emb_from = {n: v for n, v in aligned_non_iso[a_from].items() if n in non_iso}
                emb_to = {n: v for n, v in aligned_non_iso[a_to].items() if n in non_iso}
                dr = compute_drift(emb_from, emb_to, a_from, a_to)
                ni_drift_rows.append({
                    "age_from": a_from, "age_to": a_to,
                    "mean_drift": dr.mean_drift, "median_drift": dr.median_drift,
                    "max_drift": dr.max_drift,
                    "num_nodes_all": num_nodes_all,
                    "num_nodes_non_iso": dr.num_nodes,
                    "coverage": round(coverage, 4),
                    "isolate_frac_from": round(iso_frac_from, 4),
                    "isolate_frac_to": round(iso_frac_to, 4),
                })

            # --- Non-isolate stability (using non-isolate-aligned embeddings) ---
            k_eff = knn_k
            if len(non_iso) <= knn_k:
                k_eff = len(non_iso) - 1
            if k_eff < 1:
                log.warning(
                    "  [E] Non-isolate nodes (%d) too few for kNN at transition %d\u2192%d, "
                    "recording NaN stability",
                    len(non_iso), a_from, a_to,
                )
                ni_stab_rows.append({
                    "age_from": a_from, "age_to": a_to,
                    "k": knn_k, "k_eff": 0,
                    "mean_stability": np.nan, "median_stability": np.nan,
                    "min_stability": np.nan,
                    "num_nodes_all": num_nodes_all,
                    "num_nodes_non_iso": num_nodes_non_iso,
                    "coverage": round(coverage, 4),
                    "isolate_frac_from": round(iso_frac_from, 4),
                    "isolate_frac_to": round(iso_frac_to, 4),
                    "low_n_warning": True,
                })
            else:
                emb_from = {n: v for n, v in aligned_non_iso[a_from].items() if n in non_iso}
                emb_to = {n: v for n, v in aligned_non_iso[a_to].items() if n in non_iso}
                sr = compute_knn_stability(emb_from, emb_to, a_from, a_to, k=k_eff)
                ni_stab_rows.append({
                    "age_from": a_from, "age_to": a_to,
                    "k": knn_k, "k_eff": k_eff,
                    "mean_stability": sr.mean_stability,
                    "median_stability": sr.median_stability,
                    "min_stability": sr.min_stability,
                    "num_nodes_all": num_nodes_all,
                    "num_nodes_non_iso": sr.num_nodes,
                    "coverage": round(coverage, 4),
                    "isolate_frac_from": round(iso_frac_from, 4),
                    "isolate_frac_to": round(iso_frac_to, 4),
                    "low_n_warning": sr.num_nodes < knn_k + 2,
                })

        pd.DataFrame(ni_drift_rows).to_csv(ni_drift_summary_path, index=False)
        pd.DataFrame(ni_stab_rows).to_csv(ni_stab_summary_path, index=False)
        log.info("  [E] Saved non-isolate drift/stability summaries")

    # ===================================================================
    # Plots
    # ===================================================================

    # Load non-isolate summaries for overlay plots
    ni_drift_df = pd.read_csv(ni_drift_summary_path) if ni_drift_summary_path.exists() else None
    ni_stab_df = pd.read_csv(ni_stab_summary_path) if ni_stab_summary_path.exists() else None

    # --- Drift distribution (all nodes) ---
    if not cached(fig_dir / "drift_distribution.png", force):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(drift_df["drift"], bins=50, color="steelblue", edgecolor="white")
        ax.set_xlabel("Drift (L2)")
        ax.set_ylabel("Count")
        ax.set_title(f"Drift distribution ({sex}/{node_type})")
        fig.tight_layout()
        fig.savefig(fig_dir / "drift_distribution.png", dpi=120)
        plt.close(fig)

    # --- Drift by transition: paired boxplots (all nodes + non-isolates) ---
    if not cached(fig_dir / "drift_by_transition.png", force):
        drift_df["transition"] = (
            drift_df["age_from"].astype(str) + "\u2192" + drift_df["age_to"].astype(str)
        )
        transitions = sorted(drift_df["transition"].unique())

        # Build non-isolate per-node data for boxplots (using non-iso aligned embeddings)
        ages = sorted(aligned_non_iso.keys())
        ni_drift_per_node: Dict[str, List[float]] = {}  # transition -> drifts
        for i in range(len(ages) - 1):
            a_from, a_to = ages[i], ages[i + 1]
            G_from, G_to = graphs.get(a_from), graphs.get(a_to)
            t_label = f"{a_from}\u2192{a_to}"
            if G_from is None or G_to is None:
                ni_drift_per_node[t_label] = []
                continue
            non_iso = (
                {n for n, d in G_from.degree() if d > 0}
                & {n for n, d in G_to.degree() if d > 0}
                & set(aligned_non_iso[a_from].keys()) & set(aligned_non_iso[a_to].keys())
            )
            if len(non_iso) < 2:
                ni_drift_per_node[t_label] = []
                continue
            emb_f = {n: v for n, v in aligned_non_iso[a_from].items() if n in non_iso}
            emb_t = {n: v for n, v in aligned_non_iso[a_to].items() if n in non_iso}
            dr = compute_drift(emb_f, emb_t, a_from, a_to)
            ni_drift_per_node[t_label] = list(dr.node_drifts.values())

        fig, ax = plt.subplots(figsize=(max(9, len(transitions) * 1.8), 5))
        positions_all = []
        positions_ni = []
        data_all = []
        data_ni = []
        x_ticks = []

        for i, t in enumerate(transitions):
            data_all.append(drift_df.loc[drift_df["transition"] == t, "drift"].values)
            data_ni.append(np.array(ni_drift_per_node.get(t, [])))
            positions_all.append(i * 3)
            positions_ni.append(i * 3 + 1)
            x_ticks.append(i * 3 + 0.5)

        bp_all = ax.boxplot(data_all, positions=positions_all, widths=0.8,
                            patch_artist=True, manage_ticks=False)
        for patch in bp_all["boxes"]:
            patch.set_facecolor("lightgray")
            patch.set_alpha(0.7)

        # Only plot non-isolate boxes for transitions that have data
        ni_data_filtered = [d for d in data_ni if len(d) > 0]
        ni_pos_filtered = [positions_ni[j] for j, d in enumerate(data_ni) if len(d) > 0]
        bp_ni = None
        if ni_data_filtered:
            bp_ni = ax.boxplot(
                ni_data_filtered, positions=ni_pos_filtered,
                widths=0.8, patch_artist=True, manage_ticks=False,
            )
            for patch in bp_ni["boxes"]:
                patch.set_facecolor("steelblue")
                patch.set_alpha(0.7)

        # Annotate with isolate fraction and coverage
        if ni_drift_df is not None:
            for i, t in enumerate(transitions):
                parts = t.split("\u2192")
                if len(parts) == 2:
                    row = ni_drift_df[
                        (ni_drift_df["age_from"].astype(str) == parts[0])
                        & (ni_drift_df["age_to"].astype(str) == parts[1])
                    ]
                    if not row.empty:
                        r = row.iloc[0]
                        ann = f"iso {r['isolate_frac_from']:.2f}\u2192{r['isolate_frac_to']:.2f}\ncov {r['coverage']:.2f}"
                        ax.text(
                            x_ticks[i], ax.get_ylim()[1] * 0.98, ann,
                            ha="center", va="top", fontsize=5.5, color="dimgray",
                        )

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(transitions, fontsize=8)
        ax.set_xlabel("Age transition")
        ax.set_ylabel("Drift (L2)")
        ax.set_title(f"Drift by transition ({sex}/{node_type})")
        if bp_ni is not None:
            ax.legend(
                [bp_all["boxes"][0], bp_ni["boxes"][0]],
                ["All nodes", "Non-isolates"],
                loc="upper left", fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(fig_dir / "drift_by_transition.png", dpi=120)
        plt.close(fig)

    # --- Stability distribution (all nodes) ---
    if not cached(fig_dir / "stability_distribution.png", force):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(stab_df["knn_stability"], bins=50, color="coral", edgecolor="white")
        ax.set_xlabel("kNN Jaccard stability")
        ax.set_ylabel("Count")
        ax.set_title(f"kNN stability distribution ({sex}/{node_type})")
        fig.tight_layout()
        fig.savefig(fig_dir / "stability_distribution.png", dpi=120)
        plt.close(fig)

    # --- Stability by transition: paired boxplots (all nodes + non-isolates) ---
    if not cached(fig_dir / "stability_by_transition.png", force):
        stab_df["transition"] = (
            stab_df["age_from"].astype(str) + "\u2192" + stab_df["age_to"].astype(str)
        )
        transitions_s = sorted(stab_df["transition"].unique())

        # Build non-isolate per-node stability data (using non-iso aligned embeddings)
        ni_stab_per_node: Dict[str, List[float]] = {}
        for i in range(len(ages) - 1):
            a_from, a_to = ages[i], ages[i + 1]
            G_from, G_to = graphs.get(a_from), graphs.get(a_to)
            t_label = f"{a_from}\u2192{a_to}"
            if G_from is None or G_to is None:
                ni_stab_per_node[t_label] = []
                continue
            non_iso = (
                {n for n, d in G_from.degree() if d > 0}
                & {n for n, d in G_to.degree() if d > 0}
                & set(aligned_non_iso[a_from].keys()) & set(aligned_non_iso[a_to].keys())
            )
            k_eff = min(knn_k, len(non_iso) - 1) if len(non_iso) > 1 else 0
            if k_eff < 1:
                ni_stab_per_node[t_label] = []
                continue
            emb_f = {n: v for n, v in aligned_non_iso[a_from].items() if n in non_iso}
            emb_t = {n: v for n, v in aligned_non_iso[a_to].items() if n in non_iso}
            sr = compute_knn_stability(emb_f, emb_t, a_from, a_to, k=k_eff)
            ni_stab_per_node[t_label] = list(sr.node_stability.values())

        fig, ax = plt.subplots(figsize=(max(9, len(transitions_s) * 1.8), 5))
        positions_all_s = []
        positions_ni_s = []
        data_all_s = []
        data_ni_s = []
        x_ticks_s = []

        for i, t in enumerate(transitions_s):
            data_all_s.append(stab_df.loc[stab_df["transition"] == t, "knn_stability"].values)
            data_ni_s.append(np.array(ni_stab_per_node.get(t, [])))
            positions_all_s.append(i * 3)
            positions_ni_s.append(i * 3 + 1)
            x_ticks_s.append(i * 3 + 0.5)

        bp_all_s = ax.boxplot(data_all_s, positions=positions_all_s, widths=0.8,
                              patch_artist=True, manage_ticks=False)
        for patch in bp_all_s["boxes"]:
            patch.set_facecolor("lightgray")
            patch.set_alpha(0.7)

        # Only plot non-isolate boxes for transitions that have data
        ni_data_s_filtered = [d for d in data_ni_s if len(d) > 0]
        ni_pos_s_filtered = [positions_ni_s[j] for j, d in enumerate(data_ni_s) if len(d) > 0]
        bp_ni_s = None
        if ni_data_s_filtered:
            bp_ni_s = ax.boxplot(
                ni_data_s_filtered, positions=ni_pos_s_filtered,
                widths=0.8, patch_artist=True, manage_ticks=False,
            )
            for patch in bp_ni_s["boxes"]:
                patch.set_facecolor("coral")
                patch.set_alpha(0.7)

        # Annotate with isolate fraction and coverage
        if ni_stab_df is not None:
            for i, t in enumerate(transitions_s):
                parts = t.split("\u2192")
                if len(parts) == 2:
                    row = ni_stab_df[
                        (ni_stab_df["age_from"].astype(str) == parts[0])
                        & (ni_stab_df["age_to"].astype(str) == parts[1])
                    ]
                    if not row.empty:
                        r = row.iloc[0]
                        ann = f"iso {r['isolate_frac_from']:.2f}\u2192{r['isolate_frac_to']:.2f}\ncov {r['coverage']:.2f}"
                        ax.text(
                            x_ticks_s[i], ax.get_ylim()[1] * 0.98, ann,
                            ha="center", va="top", fontsize=5.5, color="dimgray",
                        )

        ax.set_xticks(x_ticks_s)
        ax.set_xticklabels(transitions_s, fontsize=8)
        ax.set_xlabel("Age transition")
        ax.set_ylabel("kNN Jaccard stability")
        ax.set_title(f"Stability by transition ({sex}/{node_type})")
        if bp_ni_s is not None:
            ax.legend(
                [bp_all_s["boxes"][0], bp_ni_s["boxes"][0]],
                ["All nodes", "Non-isolates"],
                loc="upper left", fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(fig_dir / "stability_by_transition.png", dpi=120)
        plt.close(fig)

    log.info("  [E] Metrics plots saved")


def step_f_structure(
    sex: str,
    node_type: str,
    graphs: Dict[int, nx.Graph],
    aligned: Dict[int, Dict[str, np.ndarray]],
    age_labels: Dict[int, str],
    emb_params: dict,
    force: bool,
    log: logging.Logger,
) -> None:
    """Structure preservation evaluation (link-pred AUC + Spearman)."""
    tbl_dir = OUTPUT_ROOT / "tables" / sex / node_type
    fig_dir = OUTPUT_ROOT / "figures" / sex / node_type
    csv_path = tbl_dir / "structure_preservation.csv"

    if cached(csv_path, force):
        df = pd.read_csv(csv_path)
        log.info("  [F] Loaded cached structure_preservation.csv")
    else:
        # Pick the densest age for the shuffled control (only compute once)
        best_age = max(
            (age for age in sorted(aligned.keys()) if age in graphs),
            key=lambda a: graphs[a].number_of_edges(),
            default=None,
        )

        rows = []
        for age in sorted(aligned.keys()):
            if age not in graphs:
                continue
            do_ctrl = (age == best_age)
            result = evaluate_structure(
                graphs[age], aligned[age], emb_params,
                compute_shuffled_control=do_ctrl,
            )
            diag = result.get("diagnostics", {})
            row = {
                "age": age,
                "age_range": age_labels.get(age, "?"),
                "auc": result["auc"],
                "auc_std": result.get("auc_std", np.nan),
                "n_splits": result.get("n_splits", 0),
                "spearman_r": result["spearman_r"],
                "n_edges": result["n_edges"],
                "n_safe": diag.get("n_safe", np.nan),
                "n_held_out": diag.get("n_held_out", np.nan),
                "two_hop_pool": diag.get("two_hop_pool_size", np.nan),
                "n_neg": diag.get("n_neg_sampled", np.nan),
                "used_fallback": diag.get("used_fallback", False),
            }
            if do_ctrl:
                row["auc_shuffled"] = result.get("auc_shuffled", np.nan)

            # Log diagnostics at DEBUG level
            log.debug(
                "  [F] age %s diag: eligible=%s safe=%s held_out=%s "
                "train=%s/%s 2hop=%s neg=%s fallback=%s",
                age, diag.get("n_eligible"), diag.get("n_safe"),
                diag.get("n_held_out"), diag.get("train_nodes"),
                diag.get("train_edges"), diag.get("two_hop_pool_size"),
                diag.get("n_neg_sampled"), diag.get("used_fallback"),
            )
            if diag.get("used_fallback"):
                log.warning(
                    "  [F] WARNING: age %s (%s/%s) used random-fallback "
                    "negatives (2-hop pool too small).",
                    age, sex, node_type,
                )

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        # Plot
        ages_str = [str(r["age"]) for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(ages_str, df["auc"], color="steelblue")
        ax1.set_xlabel("Age group")
        ax1.set_ylabel("Link-prediction AUC")
        ax1.set_title("AUC")
        ax1.set_ylim(0, 1)
        ax2.bar(ages_str, df["spearman_r"], color="coral")
        ax2.set_xlabel("Age group")
        ax2.set_ylabel("Spearman r")
        ax2.set_title("Weight\u2013similarity correlation")
        fig.suptitle(f"Structure preservation ({sex}/{node_type})", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_dir / "structure_preservation.png", dpi=120)
        plt.close(fig)

        # Individual AUC figure
        auc_path = fig_dir / "structure_preservation_auc.png"
        if not cached(auc_path, force):
            fig_a, ax_a = plt.subplots(figsize=(6, 4))
            ax_a.bar(ages_str, df["auc"], color="steelblue")
            ax_a.set_xlabel("Age group")
            ax_a.set_ylabel("Link-prediction AUC")
            ax_a.set_title(f"AUC ({sex.title()} / {node_type.upper()})")
            ax_a.set_ylim(0, 1)
            fig_a.tight_layout()
            fig_a.savefig(auc_path, dpi=150, bbox_inches="tight")
            plt.close(fig_a)

        # Individual Spearman r figure
        spearman_path = fig_dir / "structure_preservation_spearman.png"
        if not cached(spearman_path, force):
            fig_s, ax_s = plt.subplots(figsize=(6, 4))
            ax_s.bar(ages_str, df["spearman_r"], color="coral")
            ax_s.set_xlabel("Age group")
            ax_s.set_ylabel("Spearman r")
            ax_s.set_title(f"Weight\u2013similarity correlation ({sex.title()} / {node_type.upper()})")
            fig_s.tight_layout()
            fig_s.savefig(spearman_path, dpi=150, bbox_inches="tight")
            plt.close(fig_s)

        log.info("  [F] Structure preservation computed and saved")

    # --- Sanity-check warnings (always run, even on cached data) ---
    for _, r in df.iterrows():
        auc_val = r["auc"]
        if not np.isnan(auc_val) and auc_val > 0.95:
            log.warning(
                "  [F] WARNING: AUC=%.3f for age %s (%s/%s) is suspiciously high "
                "(>0.95). Verify negative sampling and graph sparsity.",
                auc_val, r["age"], sex, node_type,
            )

    # Log shuffled-control AUC if present
    if "auc_shuffled" in df.columns:
        ctrl_rows = df.dropna(subset=["auc_shuffled"])
        for _, r in ctrl_rows.iterrows():
            auc_shuf = r["auc_shuffled"]
            log.info(
                "  [F] Control AUC (shuffled retrained embeddings, age %s): %.3f "
                "(expected ~0.5 under null)",
                r["age"], auc_shuf,
            )
            if auc_shuf > 0.7:
                log.warning(
                    "  [F] WARNING: Shuffled-control AUC=%.3f is unexpectedly high. "
                    "Evaluation methodology may be unreliable.",
                    auc_shuf,
                )
            if auc_shuf < 0.3:
                log.warning(
                    "  [F] WARNING: Shuffled-control AUC=%.3f is unexpectedly low "
                    "(systematic anti-correlation). Evaluation may be unreliable.",
                    auc_shuf,
                )

    # Isolate fraction warning
    stats_path = tbl_dir / "graph_stats.csv"
    ni_drift_path = tbl_dir / "drift_summary_non_isolates.csv"
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        if "isolate_frac" in stats_df.columns:
            high_iso = stats_df[stats_df["isolate_frac"] > 0.8]
            if not high_iso.empty and not ni_drift_path.exists():
                log.warning(
                    "  [F] WARNING: %d age groups have isolate_frac > 0.80 but "
                    "non-isolate drift/stability summaries are missing. "
                    "Rerun with --force to generate them.",
                    len(high_iso),
                )


def step_g_robustness(
    sex: str,
    node_type: str,
    graphs: Dict[int, nx.Graph],
    ref_age: int,
    emb_params: dict,
    force: bool,
    log: logging.Logger,
) -> None:
    """Robustness check: retrain with different (p,q), compare drift rankings."""
    tbl_dir = OUTPUT_ROOT / "tables" / sex / node_type
    csv_path = tbl_dir / "robustness_summary.csv"
    ni_csv_path = tbl_dir / "robustness_summary_non_isolates.csv"
    if cached(csv_path, force) and cached(ni_csv_path, force):
        log.info("  [G] Loaded cached robustness summaries")
        return

    configs = [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)]
    per_config_mean_drift: Dict[str, pd.Series] = {}
    per_config_mean_drift_ni: Dict[str, pd.Series] = {}

    # Precompute non-isolate node sets per transition
    ages = sorted(graphs.keys())
    non_iso_per_transition: Dict[Tuple[int, int], Set[str]] = {}
    for i in range(len(ages) - 1):
        a_from, a_to = ages[i], ages[i + 1]
        G_from, G_to = graphs.get(a_from), graphs.get(a_to)
        if G_from is not None and G_to is not None:
            non_iso_per_transition[(a_from, a_to)] = (
                {n for n, d in G_from.degree() if d > 0}
                & {n for n, d in G_to.degree() if d > 0}
            )

    for p, q in configs:
        label = f"p={p}_q={q}"
        log.info("  [G] Robustness config %s ...", label)
        rob_dir = OUTPUT_ROOT / "embeddings" / sex / node_type / "robustness" / label
        rob_dir.mkdir(parents=True, exist_ok=True)

        # Train / load
        emb_by_age: Dict[int, Dict[str, np.ndarray]] = {}
        for age, G in sorted(graphs.items()):
            emb_path = rob_dir / f"age_{age}.parquet"
            if cached(emb_path, force):
                emb_by_age[age] = load_embeddings(emb_path)
            else:
                emb = train_node2vec(
                    G,
                    dim=emb_params["dim"],
                    walk_length=emb_params["walk_length"],
                    num_walks=emb_params["num_walks"],
                    window=emb_params["window"],
                    p=p,
                    q=q,
                    workers=emb_params["workers"],
                    seed=emb_params["seed"],
                )
                save_embeddings(emb_path, emb)
                emb_by_age[age] = emb

        # Align (all anchors)
        aligned_rob, _ = align_all_to_reference(emb_by_age, reference_age=ref_age)

        # Align (non-isolate anchors)
        from src.alignment import procrustes_align as _pa
        ref_emb_rob = emb_by_age[ref_age]
        G_ref_rob = graphs.get(ref_age)
        aligned_rob_ni: Dict[int, Dict[str, np.ndarray]] = {}
        for age in sorted(emb_by_age.keys()):
            if age == ref_age:
                aligned_rob_ni[age] = ref_emb_rob.copy()
                continue
            G_target = graphs.get(age)
            if G_ref_rob is None or G_target is None:
                aligned_rob_ni[age] = aligned_rob[age]
                continue
            ni_anchors = _compute_non_isolate_anchors(
                G_ref_rob, G_target, ref_emb_rob, emb_by_age[age]
            )
            if len(ni_anchors) < 2:
                aligned_rob_ni[age] = aligned_rob[age]
            else:
                ni_emb, _ = _pa(ref_emb_rob, emb_by_age[age], anchor_nodes=ni_anchors)
                aligned_rob_ni[age] = ni_emb

        # Drift — all nodes
        drift_results = compute_all_drifts(aligned_rob)
        all_drifts: Dict[str, List[float]] = {}
        for dr in drift_results:
            for node, d in dr.node_drifts.items():
                all_drifts.setdefault(node, []).append(d)
        mean_series = pd.Series({n: np.mean(ds) for n, ds in all_drifts.items()})
        per_config_mean_drift[label] = mean_series

        # Drift — non-isolates only (using non-isolate aligned embeddings)
        ni_drifts: Dict[str, List[float]] = {}
        rob_ages = sorted(aligned_rob_ni.keys())
        for i in range(len(rob_ages) - 1):
            a_from, a_to = rob_ages[i], rob_ages[i + 1]
            non_iso = non_iso_per_transition.get((a_from, a_to), set())
            non_iso = non_iso & set(aligned_rob_ni[a_from].keys()) & set(aligned_rob_ni[a_to].keys())
            if len(non_iso) < 2:
                continue
            emb_f = {n: v for n, v in aligned_rob_ni[a_from].items() if n in non_iso}
            emb_t = {n: v for n, v in aligned_rob_ni[a_to].items() if n in non_iso}
            dr = compute_drift(emb_f, emb_t, a_from, a_to)
            for node, d in dr.node_drifts.items():
                ni_drifts.setdefault(node, []).append(d)
        ni_mean_series = pd.Series({n: np.mean(ds) for n, ds in ni_drifts.items()})
        per_config_mean_drift_ni[label] = ni_mean_series

    # Pairwise Spearman rank correlations — all nodes
    labels = list(per_config_mean_drift.keys())
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            s1 = per_config_mean_drift[labels[i]]
            s2 = per_config_mean_drift[labels[j]]
            common = sorted(set(s1.index) & set(s2.index))
            rho = np.nan
            if len(common) >= 5:
                rho, _ = sp_stats.spearmanr(s1[common], s2[common])
            rows.append({
                "config_a": labels[i],
                "config_b": labels[j],
                "mode": "all",
                "spearman_rho": round(rho, 4) if not np.isnan(rho) else np.nan,
                "n_common_nodes": len(common),
            })

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info("  [G] Robustness summary saved (all nodes)")

    # Pairwise Spearman rank correlations — non-isolates
    ni_labels = list(per_config_mean_drift_ni.keys())
    ni_rows = []
    for i in range(len(ni_labels)):
        for j in range(i + 1, len(ni_labels)):
            s1 = per_config_mean_drift_ni[ni_labels[i]]
            s2 = per_config_mean_drift_ni[ni_labels[j]]
            common = sorted(set(s1.index) & set(s2.index))
            rho = np.nan
            if len(common) >= 5:
                rho, _ = sp_stats.spearmanr(s1[common], s2[common])
            ni_rows.append({
                "config_a": ni_labels[i],
                "config_b": ni_labels[j],
                "mode": "non_isolates",
                "spearman_rho": round(rho, 4) if not np.isnan(rho) else np.nan,
                "n_common_nodes": len(common),
            })

    pd.DataFrame(ni_rows).to_csv(ni_csv_path, index=False)
    log.info("  [G] Robustness summary saved (non-isolates)")


# ---------------------------------------------------------------------------
# Combined summary figure
# ---------------------------------------------------------------------------

AGE_CMAP = plt.cm.viridis  # noqa: module-level for consistency


def plot_combined_embeddings(
    age_labels: Dict[int, str],
    force: bool,
    log: logging.Logger,
) -> None:
    """Create a 2×3 grid showing PCA of aligned embeddings for all combos.

    Each subplot overlays all age groups coloured by age, so you can
    visually compare embedding structure across sex and variant at a glance.
    """
    out_path = OUTPUT_ROOT / "figures" / "combined_embeddings.png"
    if cached(out_path, force):
        log.info("Combined embeddings figure already exists, skipping")
        return

    fig, axes = plt.subplots(
        len(SEXES), len(NODE_TYPES), figsize=(6 * len(NODE_TYPES), 5 * len(SEXES))
    )

    all_ages_union = sorted({a for a in ALL_AGES})
    norm = plt.Normalize(vmin=min(all_ages_union), vmax=max(all_ages_union))

    for row, sex in enumerate(SEXES):
        for col, node_type in enumerate(NODE_TYPES):
            ax = axes[row, col]
            aligned_dir = OUTPUT_ROOT / "embeddings" / sex / node_type / "aligned"

            # Load all available aligned embeddings for this combo
            emb_by_age: Dict[int, Dict[str, np.ndarray]] = {}
            for age in ALL_AGES:
                p = aligned_dir / f"age_{age}.parquet"
                if p.exists():
                    emb_by_age[age] = load_embeddings(p)

            if not emb_by_age:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")
                ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=11)
                ax.axis("off")
                continue

            # Stack all nodes across ages, fit a single PCA
            all_nodes_vecs = []
            all_ages_list = []
            node_lists_per_age = {}
            for age in sorted(emb_by_age.keys()):
                emb = emb_by_age[age]
                nodes = sorted(emb.keys())
                vecs = np.array([emb[n] for n in nodes])
                node_lists_per_age[age] = nodes
                all_nodes_vecs.append(vecs)
                all_ages_list.extend([age] * len(nodes))

            X_all = np.vstack(all_nodes_vecs)
            ages_arr = np.array(all_ages_list)

            pca = PCA(n_components=2)
            X2 = pca.fit_transform(X_all)

            # Plot each age group
            for age in sorted(emb_by_age.keys()):
                mask = ages_arr == age
                color = AGE_CMAP(norm(age))
                label_str = age_labels.get(age, str(age))
                ax.scatter(
                    X2[mask, 0], X2[mask, 1],
                    c=[color], s=8, alpha=0.5, label=f"{age}: {label_str}",
                    edgecolors="none",
                )

            ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=11)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", fontsize=8)
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", fontsize=8)
            ax.tick_params(labelsize=7)

            # Legend only on the first subplot
            if row == 0 and col == len(NODE_TYPES) - 1:
                ax.legend(fontsize=7, title="Age", title_fontsize=8,
                          loc="upper left", bbox_to_anchor=(1.02, 1.0),
                          borderaxespad=0)

    fig.suptitle(
        "Aligned Embeddings — PCA overview (all ages overlaid)",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved combined embeddings figure → %s", out_path)

    # --- Individual panels ---
    split_dir = OUTPUT_ROOT / "figures" / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    for sex in SEXES:
        for node_type in NODE_TYPES:
            ind_path = split_dir / f"embeddings_{sex}_{node_type}.png"
            if cached(ind_path, force):
                continue
            aligned_dir = OUTPUT_ROOT / "embeddings" / sex / node_type / "aligned"
            emb_by_age: Dict[int, Dict[str, np.ndarray]] = {}
            for age in ALL_AGES:
                p = aligned_dir / f"age_{age}.parquet"
                if p.exists():
                    emb_by_age[age] = load_embeddings(p)
            if not emb_by_age:
                continue
            all_nodes_vecs = []
            all_ages_list = []
            for age in sorted(emb_by_age.keys()):
                emb = emb_by_age[age]
                nodes = sorted(emb.keys())
                vecs = np.array([emb[n] for n in nodes])
                all_nodes_vecs.append(vecs)
                all_ages_list.extend([age] * len(nodes))
            X_all = np.vstack(all_nodes_vecs)
            ages_arr = np.array(all_ages_list)
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(X_all)
            all_ages_union = sorted({a for a in ALL_AGES})
            norm = plt.Normalize(vmin=min(all_ages_union), vmax=max(all_ages_union))
            fig_i, ax_i = plt.subplots(figsize=(7, 6))
            for age in sorted(emb_by_age.keys()):
                mask = ages_arr == age
                color = AGE_CMAP(norm(age))
                label_str = age_labels.get(age, str(age))
                ax_i.scatter(
                    X2[mask, 0], X2[mask, 1],
                    c=[color], s=8, alpha=0.5, label=f"{age}: {label_str}",
                    edgecolors="none",
                )
            ax_i.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=12)
            ax_i.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", fontsize=9)
            ax_i.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", fontsize=9)
            ax_i.legend(fontsize=7, title="Age", title_fontsize=8)
            fig_i.tight_layout()
            fig_i.savefig(ind_path, dpi=150, bbox_inches="tight")
            plt.close(fig_i)
    log.info("Saved split embeddings panels → %s", split_dir)


def plot_combined_alignment_residuals(
    force: bool, log: logging.Logger,
) -> None:
    """2×3 grid of alignment residual bar charts."""
    out_path = OUTPUT_ROOT / "figures" / "combined_alignment_residuals.png"
    if cached(out_path, force):
        log.info("Combined alignment residuals figure already exists, skipping")
        return

    fig, axes = plt.subplots(len(SEXES), len(NODE_TYPES),
                             figsize=(5 * len(NODE_TYPES), 4 * len(SEXES)))
    for row, sex in enumerate(SEXES):
        for col, node_type in enumerate(NODE_TYPES):
            ax = axes[row, col]
            csv_path = OUTPUT_ROOT / "tables" / sex / node_type / "alignment_residuals.csv"
            if not csv_path.exists():
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
                ax.axis("off")
                continue
            df = pd.read_csv(csv_path)
            ax.bar(df["age_group"].astype(str), df["residual_error"], color="steelblue")
            ax.set_xlabel("Age group", fontsize=8)
            ax.set_ylabel("Frobenius residual", fontsize=8)
            ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
            ax.tick_params(labelsize=7)

    fig.suptitle("Alignment Residuals — all combinations", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved combined alignment residuals → %s", out_path)

    # --- Individual panels ---
    split_dir = OUTPUT_ROOT / "figures" / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    for sex in SEXES:
        for node_type in NODE_TYPES:
            ind_path = split_dir / f"alignment_residuals_{sex}_{node_type}.png"
            if cached(ind_path, force):
                continue
            csv_path = OUTPUT_ROOT / "tables" / sex / node_type / "alignment_residuals.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            fig_i, ax_i = plt.subplots(figsize=(6, 4))
            ax_i.bar(df["age_group"].astype(str), df["residual_error"], color="steelblue")
            ax_i.set_xlabel("Age group")
            ax_i.set_ylabel("Frobenius residual")
            ax_i.set_title(f"Alignment Residuals — {sex.title()} / {node_type.upper()}")
            fig_i.tight_layout()
            fig_i.savefig(ind_path, dpi=150, bbox_inches="tight")
            plt.close(fig_i)
    log.info("Saved split alignment residual panels → %s", split_dir)


def plot_combined_drift(
    force: bool, log: logging.Logger,
) -> None:
    """2×3 grids for drift distribution and drift by transition."""
    # --- Distribution ---
    out1 = OUTPUT_ROOT / "figures" / "combined_drift_distribution.png"
    if not cached(out1, force):
        fig, axes = plt.subplots(len(SEXES), len(NODE_TYPES),
                                 figsize=(5 * len(NODE_TYPES), 4 * len(SEXES)))
        for row, sex in enumerate(SEXES):
            for col, node_type in enumerate(NODE_TYPES):
                ax = axes[row, col]
                csv_path = OUTPUT_ROOT / "tables" / sex / node_type / "drift.csv"
                if not csv_path.exists():
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
                    ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
                    ax.axis("off")
                    continue
                df = pd.read_csv(csv_path)
                ax.hist(df["drift"], bins=50, color="steelblue", edgecolor="white")
                ax.set_xlabel("Drift (L2)", fontsize=8)
                ax.set_ylabel("Count", fontsize=8)
                ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
                ax.tick_params(labelsize=7)

        fig.suptitle("Drift Distribution — all combinations", fontsize=14)
        fig.tight_layout()
        fig.savefig(out1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved combined drift distribution → %s", out1)

        # --- Individual drift distribution panels ---
        split_dir = OUTPUT_ROOT / "figures" / "split"
        split_dir.mkdir(parents=True, exist_ok=True)
        for sex in SEXES:
            for node_type in NODE_TYPES:
                ind_path = split_dir / f"drift_distribution_{sex}_{node_type}.png"
                if cached(ind_path, force):
                    continue
                csv_path = OUTPUT_ROOT / "tables" / sex / node_type / "drift.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                fig_i, ax_i = plt.subplots(figsize=(6, 4))
                ax_i.hist(df["drift"], bins=50, color="steelblue", edgecolor="white")
                ax_i.set_xlabel("Drift (L2)")
                ax_i.set_ylabel("Count")
                ax_i.set_title(f"Drift Distribution — {sex.title()} / {node_type.upper()}")
                fig_i.tight_layout()
                fig_i.savefig(ind_path, dpi=150, bbox_inches="tight")
                plt.close(fig_i)
        log.info("Saved split drift distribution panels → %s", split_dir)

    # --- By transition (with non-isolate overlay) ---
    out2 = OUTPUT_ROOT / "figures" / "combined_drift_by_transition.png"
    if not cached(out2, force):
        fig, axes = plt.subplots(len(SEXES), len(NODE_TYPES),
                                 figsize=(6 * len(NODE_TYPES), 4.5 * len(SEXES)))
        for row, sex in enumerate(SEXES):
            for col, node_type in enumerate(NODE_TYPES):
                ax = axes[row, col]
                csv_path = OUTPUT_ROOT / "tables" / sex / node_type / "drift.csv"
                ni_path = OUTPUT_ROOT / "tables" / sex / node_type / "drift_summary_non_isolates.csv"
                if not csv_path.exists():
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
                    ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
                    ax.axis("off")
                    continue
                df = pd.read_csv(csv_path)
                # Read all-nodes summary for median line
                all_summary_path = OUTPUT_ROOT / "tables" / sex / node_type / "drift_summary.csv"
                all_summary = pd.read_csv(all_summary_path) if all_summary_path.exists() else None
                ni_summary = pd.read_csv(ni_path) if ni_path.exists() else None

                df["transition"] = df["age_from"].astype(str) + "\u2192" + df["age_to"].astype(str)
                transitions = sorted(df["transition"].unique())
                x = np.arange(len(transitions))

                # Plot two lines: all-nodes median vs non-isolate median
                if all_summary is not None:
                    all_summary["transition"] = all_summary["age_from"].astype(str) + "\u2192" + all_summary["age_to"].astype(str)
                    med_all = [
                        all_summary.loc[all_summary["transition"] == t, "median_drift"].values[0]
                        if t in all_summary["transition"].values else np.nan
                        for t in transitions
                    ]
                    ax.plot(x, med_all, "o--", color="gray", markersize=4, label="All (median)")

                if ni_summary is not None:
                    ni_summary["transition"] = ni_summary["age_from"].astype(str) + "\u2192" + ni_summary["age_to"].astype(str)
                    med_ni = [
                        ni_summary.loc[ni_summary["transition"] == t, "median_drift"].values[0]
                        if t in ni_summary["transition"].values else np.nan
                        for t in transitions
                    ]
                    ax.plot(x, med_ni, "s-", color="steelblue", markersize=4, label="Non-iso (median)")

                    # Annotate with isolate fraction range
                    iso_fracs = []
                    for _, r in ni_summary.iterrows():
                        iso_fracs.extend([r.get("isolate_frac_from", 0), r.get("isolate_frac_to", 0)])
                    if iso_fracs:
                        iso_min, iso_max = min(iso_fracs), max(iso_fracs)
                        ax.set_title(
                            f"{sex.title()} / {node_type.upper()} (iso {iso_min:.0%}\u2013{iso_max:.0%})",
                            fontsize=10,
                        )
                    else:
                        ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
                else:
                    ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)

                ax.set_xticks(x)
                ax.set_xticklabels(transitions, fontsize=7)
                ax.set_xlabel("Age transition", fontsize=8)
                ax.set_ylabel("Drift (L2)", fontsize=8)
                ax.tick_params(labelsize=7)
                if row == 0 and col == 0:
                    ax.legend(fontsize=6, loc="upper right")

        fig.suptitle("Drift by Transition — all combinations", fontsize=14)
        fig.tight_layout()
        fig.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved combined drift by transition → %s", out2)

        # --- Individual drift by-transition panels ---
        split_dir = OUTPUT_ROOT / "figures" / "split"
        split_dir.mkdir(parents=True, exist_ok=True)
        for sex in SEXES:
            for node_type in NODE_TYPES:
                ind_path = split_dir / f"drift_by_transition_{sex}_{node_type}.png"
                if cached(ind_path, force):
                    continue
                csv_path = OUTPUT_ROOT / "tables" / sex / node_type / "drift.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                all_summary_path = OUTPUT_ROOT / "tables" / sex / node_type / "drift_summary.csv"
                ni_path = OUTPUT_ROOT / "tables" / sex / node_type / "drift_summary_non_isolates.csv"
                all_summary = pd.read_csv(all_summary_path) if all_summary_path.exists() else None
                ni_summary = pd.read_csv(ni_path) if ni_path.exists() else None

                df["transition"] = df["age_from"].astype(str) + "\u2192" + df["age_to"].astype(str)
                transitions = sorted(df["transition"].unique())
                x = np.arange(len(transitions))

                fig_i, ax_i = plt.subplots(figsize=(7, 5))
                if all_summary is not None:
                    all_summary["transition"] = all_summary["age_from"].astype(str) + "\u2192" + all_summary["age_to"].astype(str)
                    med_all = [
                        all_summary.loc[all_summary["transition"] == t, "median_drift"].values[0]
                        if t in all_summary["transition"].values else np.nan
                        for t in transitions
                    ]
                    ax_i.plot(x, med_all, "o--", color="gray", markersize=4, label="All (median)")
                if ni_summary is not None:
                    ni_summary["transition"] = ni_summary["age_from"].astype(str) + "\u2192" + ni_summary["age_to"].astype(str)
                    med_ni = [
                        ni_summary.loc[ni_summary["transition"] == t, "median_drift"].values[0]
                        if t in ni_summary["transition"].values else np.nan
                        for t in transitions
                    ]
                    ax_i.plot(x, med_ni, "s-", color="steelblue", markersize=4, label="Non-iso (median)")
                ax_i.set_xticks(x)
                ax_i.set_xticklabels(transitions, fontsize=8)
                ax_i.set_xlabel("Age transition")
                ax_i.set_ylabel("Drift (L2)")
                ax_i.set_title(f"Drift by Transition — {sex.title()} / {node_type.upper()}")
                ax_i.legend(fontsize=7)
                fig_i.tight_layout()
                fig_i.savefig(ind_path, dpi=150, bbox_inches="tight")
                plt.close(fig_i)
        log.info("Saved split drift by-transition panels → %s", split_dir)


def plot_combined_stability(
    force: bool, log: logging.Logger,
) -> None:
    """2×3 grids for stability distribution and stability by transition."""
    # --- Distribution ---
    out1 = OUTPUT_ROOT / "figures" / "combined_stability_distribution.png"
    if not cached(out1, force):
        fig, axes = plt.subplots(len(SEXES), len(NODE_TYPES),
                                 figsize=(5 * len(NODE_TYPES), 4 * len(SEXES)))
        for row, sex in enumerate(SEXES):
            for col, node_type in enumerate(NODE_TYPES):
                ax = axes[row, col]
                csv_path = OUTPUT_ROOT / "tables" / sex / node_type / "stability.csv"
                if not csv_path.exists():
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
                    ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
                    ax.axis("off")
                    continue
                df = pd.read_csv(csv_path)
                ax.hist(df["knn_stability"], bins=50, color="coral", edgecolor="white")
                ax.set_xlabel("kNN Jaccard stability", fontsize=8)
                ax.set_ylabel("Count", fontsize=8)
                ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
                ax.tick_params(labelsize=7)

        fig.suptitle("kNN Stability Distribution — all combinations", fontsize=14)
        fig.tight_layout()
        fig.savefig(out1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved combined stability distribution → %s", out1)

        # --- Individual stability distribution panels ---
        split_dir = OUTPUT_ROOT / "figures" / "split"
        split_dir.mkdir(parents=True, exist_ok=True)
        for sex in SEXES:
            for node_type in NODE_TYPES:
                ind_path = split_dir / f"stability_distribution_{sex}_{node_type}.png"
                if cached(ind_path, force):
                    continue
                csv_path = OUTPUT_ROOT / "tables" / sex / node_type / "stability.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                fig_i, ax_i = plt.subplots(figsize=(6, 4))
                ax_i.hist(df["knn_stability"], bins=50, color="coral", edgecolor="white")
                ax_i.set_xlabel("kNN Jaccard stability")
                ax_i.set_ylabel("Count")
                ax_i.set_title(f"kNN Stability Distribution — {sex.title()} / {node_type.upper()}")
                fig_i.tight_layout()
                fig_i.savefig(ind_path, dpi=150, bbox_inches="tight")
                plt.close(fig_i)
        log.info("Saved split stability distribution panels → %s", split_dir)

    # --- By transition (with non-isolate overlay) ---
    out2 = OUTPUT_ROOT / "figures" / "combined_stability_by_transition.png"
    if not cached(out2, force):
        fig, axes = plt.subplots(len(SEXES), len(NODE_TYPES),
                                 figsize=(6 * len(NODE_TYPES), 4.5 * len(SEXES)))
        for row, sex in enumerate(SEXES):
            for col, node_type in enumerate(NODE_TYPES):
                ax = axes[row, col]
                csv_path = OUTPUT_ROOT / "tables" / sex / node_type / "stability.csv"
                ni_path = OUTPUT_ROOT / "tables" / sex / node_type / "stability_summary_non_isolates.csv"
                if not csv_path.exists():
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
                    ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
                    ax.axis("off")
                    continue

                all_summary_path = OUTPUT_ROOT / "tables" / sex / node_type / "stability_summary.csv"
                all_summary = pd.read_csv(all_summary_path) if all_summary_path.exists() else None
                ni_summary = pd.read_csv(ni_path) if ni_path.exists() else None

                df = pd.read_csv(csv_path)
                df["transition"] = df["age_from"].astype(str) + "\u2192" + df["age_to"].astype(str)
                transitions = sorted(df["transition"].unique())
                x = np.arange(len(transitions))

                if all_summary is not None:
                    all_summary["transition"] = all_summary["age_from"].astype(str) + "\u2192" + all_summary["age_to"].astype(str)
                    med_all = [
                        all_summary.loc[all_summary["transition"] == t, "median_stability"].values[0]
                        if t in all_summary["transition"].values else np.nan
                        for t in transitions
                    ]
                    ax.plot(x, med_all, "o--", color="gray", markersize=4, label="All (median)")

                if ni_summary is not None:
                    ni_summary["transition"] = ni_summary["age_from"].astype(str) + "\u2192" + ni_summary["age_to"].astype(str)
                    med_ni = [
                        ni_summary.loc[ni_summary["transition"] == t, "median_stability"].values[0]
                        if t in ni_summary["transition"].values else np.nan
                        for t in transitions
                    ]
                    ax.plot(x, med_ni, "s-", color="coral", markersize=4, label="Non-iso (median)")

                    iso_fracs = []
                    for _, r in ni_summary.iterrows():
                        iso_fracs.extend([r.get("isolate_frac_from", 0), r.get("isolate_frac_to", 0)])
                    if iso_fracs:
                        iso_min, iso_max = min(iso_fracs), max(iso_fracs)
                        ax.set_title(
                            f"{sex.title()} / {node_type.upper()} (iso {iso_min:.0%}\u2013{iso_max:.0%})",
                            fontsize=10,
                        )
                    else:
                        ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
                else:
                    ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)

                ax.set_xticks(x)
                ax.set_xticklabels(transitions, fontsize=7)
                ax.set_xlabel("Age transition", fontsize=8)
                ax.set_ylabel("kNN Jaccard stability", fontsize=8)
                ax.tick_params(labelsize=7)
                if row == 0 and col == 0:
                    ax.legend(fontsize=6, loc="lower right")

        fig.suptitle("kNN Stability by Transition — all combinations", fontsize=14)
        fig.tight_layout()
        fig.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved combined stability by transition → %s", out2)

        # --- Individual stability by-transition panels ---
        split_dir = OUTPUT_ROOT / "figures" / "split"
        split_dir.mkdir(parents=True, exist_ok=True)
        for sex in SEXES:
            for node_type in NODE_TYPES:
                ind_path = split_dir / f"stability_by_transition_{sex}_{node_type}.png"
                if cached(ind_path, force):
                    continue
                csv_path = OUTPUT_ROOT / "tables" / sex / node_type / "stability.csv"
                if not csv_path.exists():
                    continue
                all_summary_path = OUTPUT_ROOT / "tables" / sex / node_type / "stability_summary.csv"
                ni_path = OUTPUT_ROOT / "tables" / sex / node_type / "stability_summary_non_isolates.csv"
                all_summary = pd.read_csv(all_summary_path) if all_summary_path.exists() else None
                ni_summary = pd.read_csv(ni_path) if ni_path.exists() else None

                df = pd.read_csv(csv_path)
                df["transition"] = df["age_from"].astype(str) + "\u2192" + df["age_to"].astype(str)
                transitions = sorted(df["transition"].unique())
                x = np.arange(len(transitions))

                fig_i, ax_i = plt.subplots(figsize=(7, 5))
                if all_summary is not None:
                    all_summary["transition"] = all_summary["age_from"].astype(str) + "\u2192" + all_summary["age_to"].astype(str)
                    med_all = [
                        all_summary.loc[all_summary["transition"] == t, "median_stability"].values[0]
                        if t in all_summary["transition"].values else np.nan
                        for t in transitions
                    ]
                    ax_i.plot(x, med_all, "o--", color="gray", markersize=4, label="All (median)")
                if ni_summary is not None:
                    ni_summary["transition"] = ni_summary["age_from"].astype(str) + "\u2192" + ni_summary["age_to"].astype(str)
                    med_ni = [
                        ni_summary.loc[ni_summary["transition"] == t, "median_stability"].values[0]
                        if t in ni_summary["transition"].values else np.nan
                        for t in transitions
                    ]
                    ax_i.plot(x, med_ni, "s-", color="coral", markersize=4, label="Non-iso (median)")
                ax_i.set_xticks(x)
                ax_i.set_xticklabels(transitions, fontsize=8)
                ax_i.set_xlabel("Age transition")
                ax_i.set_ylabel("kNN Jaccard stability")
                ax_i.set_title(f"kNN Stability by Transition — {sex.title()} / {node_type.upper()}")
                ax_i.legend(fontsize=7)
                fig_i.tight_layout()
                fig_i.savefig(ind_path, dpi=150, bbox_inches="tight")
                plt.close(fig_i)
        log.info("Saved split stability by-transition panels → %s", split_dir)


def plot_combined_structure_preservation(
    force: bool, log: logging.Logger,
) -> None:
    """2×3 grid with AUC and Spearman r side by side per combo."""
    out_path = OUTPUT_ROOT / "figures" / "combined_structure_preservation.png"
    if cached(out_path, force):
        log.info("Combined structure preservation figure already exists, skipping")
        return

    fig, axes = plt.subplots(len(SEXES), len(NODE_TYPES),
                             figsize=(6 * len(NODE_TYPES), 4 * len(SEXES)))
    for row, sex in enumerate(SEXES):
        for col, node_type in enumerate(NODE_TYPES):
            ax = axes[row, col]
            csv_path = OUTPUT_ROOT / "tables" / sex / node_type / "structure_preservation.csv"
            if not csv_path.exists():
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
                ax.axis("off")
                continue
            df = pd.read_csv(csv_path)
            ages_str = df["age"].astype(str)
            x = np.arange(len(ages_str))
            w = 0.35
            ax.bar(x - w / 2, df["auc"], w, label="AUC", color="steelblue")
            ax.bar(x + w / 2, df["spearman_r"], w, label="Spearman r", color="coral")
            ax.set_xticks(x)
            ax.set_xticklabels(ages_str)
            ax.set_xlabel("Age group", fontsize=8)
            ax.set_ylabel("Score", fontsize=8)
            ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Structure Preservation — all combinations", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved combined structure preservation → %s", out_path)

    # --- Individual panels ---
    split_dir = OUTPUT_ROOT / "figures" / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    for sex in SEXES:
        for node_type in NODE_TYPES:
            ind_path = split_dir / f"structure_preservation_{sex}_{node_type}.png"
            if cached(ind_path, force):
                continue
            csv_path = OUTPUT_ROOT / "tables" / sex / node_type / "structure_preservation.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            ages_str = df["age"].astype(str)
            x = np.arange(len(ages_str))
            w = 0.35
            fig_i, ax_i = plt.subplots(figsize=(7, 4))
            ax_i.bar(x - w / 2, df["auc"], w, label="AUC", color="steelblue")
            ax_i.bar(x + w / 2, df["spearman_r"], w, label="Spearman r", color="coral")
            ax_i.set_xticks(x)
            ax_i.set_xticklabels(ages_str)
            ax_i.set_xlabel("Age group")
            ax_i.set_ylabel("Score")
            ax_i.set_title(f"Structure Preservation — {sex.title()} / {node_type.upper()}")
            ax_i.set_ylim(0, 1)
            ax_i.legend(fontsize=8)
            fig_i.tight_layout()
            fig_i.savefig(ind_path, dpi=150, bbox_inches="tight")
            plt.close(fig_i)
    log.info("Saved split structure preservation panels → %s", split_dir)


def plot_combined_graphs_gallery(
    data_root: Path, age_labels: Dict[int, str],
    force: bool, log: logging.Logger,
) -> None:
    """2×3 grid showing one representative graph per combo (largest age group)."""
    out_path = OUTPUT_ROOT / "figures" / "combined_graphs_gallery.png"
    if cached(out_path, force):
        log.info("Combined graphs gallery already exists, skipping")
        return

    fig, axes = plt.subplots(len(SEXES), len(NODE_TYPES),
                             figsize=(6 * len(NODE_TYPES), 5 * len(SEXES)))
    for row, sex in enumerate(SEXES):
        for col, node_type in enumerate(NODE_TYPES):
            ax = axes[row, col]
            # Pick age with most edges
            best_age, best_G, best_edges = None, None, -1
            for age in ALL_AGES:
                gpath = find_graph_file(data_root, sex, node_type, age_group=age)
                if gpath is None:
                    continue
                G, st = load_gexf_graph(gpath)
                if st.edge_count > best_edges:
                    best_age, best_G, best_edges = age, G, st.edge_count
            if best_G is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
                ax.axis("off")
                continue
            non_iso = sum(1 for _, d in best_G.degree() if d > 0)
            label = age_labels.get(best_age, str(best_age))
            title = (f"{sex.title()} / {node_type.upper()}\n"
                     f"Age {label} — {non_iso} nodes, {best_edges} edges")
            plot_graph_on_ax(ax, best_G, title)

    fig.suptitle("Graph Gallery — densest age per combination", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved combined graphs gallery → %s", out_path)


def plot_individual_graphs_gallery(
    data_root: Path, age_labels: Dict[int, str],
    force: bool, log: logging.Logger,
) -> None:
    """One PNG per (sex, variant) showing the densest-age graph."""
    out_dir = OUTPUT_ROOT / "figures" / "graphs_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)

    for sex in SEXES:
        for node_type in NODE_TYPES:
            out_path = out_dir / f"graph_{sex}_{node_type}.png"
            if cached(out_path, force):
                continue

            best_age, best_G, best_edges = None, None, -1
            for age in ALL_AGES:
                gpath = find_graph_file(data_root, sex, node_type, age_group=age)
                if gpath is None:
                    continue
                G, st = load_gexf_graph(gpath)
                if st.edge_count > best_edges:
                    best_age, best_G, best_edges = age, G, st.edge_count

            fig, ax = plt.subplots(figsize=(8, 7))
            if best_G is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14, color="gray")
                ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=13)
                ax.axis("off")
            else:
                non_iso = sum(1 for _, d in best_G.degree() if d > 0)
                label = age_labels.get(best_age, str(best_age))
                title = (f"{sex.title()} / {node_type.upper()}\n"
                         f"Age {label} — {non_iso} nodes, {best_edges} edges")
                plot_graph_on_ax(ax, best_G, title)
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

    log.info("Saved individual graph gallery PNGs → %s", out_dir)


def plot_all_combined(
    data_root: Path, age_labels: Dict[int, str],
    force: bool, log: logging.Logger,
) -> None:
    """Generate all combined summary figures."""
    log.info("Generating combined summary figures ...")
    plot_combined_embeddings(age_labels, force, log)
    plot_combined_graphs_gallery(data_root, age_labels, force, log)
    plot_individual_graphs_gallery(data_root, age_labels, force, log)
    plot_combined_alignment_residuals(force, log)
    plot_combined_drift(force, log)
    plot_combined_stability(force, log)
    plot_combined_structure_preservation(force, log)


# ---------------------------------------------------------------------------
# Step H — Chronic analysis and node trajectory distances
# ---------------------------------------------------------------------------

# Nodes of clinical interest for trajectory analysis
TRAJECTORY_NODES = ["E11", "I10", "I50", "E66", "F32"]


def expand_icd_range(icd_range: str, valid_nodes: Set[str]) -> Set[str]:
    """Expand an ICD range string into individual codes present in the graph.

    Handles patterns from Chronic_All.csv:
      - Single code: 'E66' → {'E66'}
      - Same-letter range: 'E10-E14' → {'E10','E11','E12','E13','E14'}
      - Double-hyphen same-letter: 'M40--M54' → M40..M54
      - Cross-letter: 'I60-G45' → match both endpoints + everything between
        that exists in valid_nodes
    """
    raw = icd_range.strip()

    # Split on '--' first (double-hyphen), else '-'
    if "--" in raw:
        parts = raw.split("--", 1)
    elif "-" in raw:
        parts = raw.split("-", 1)
    else:
        parts = [raw]

    # Single code
    if len(parts) == 1:
        code = parts[0]
        return {code} & valid_nodes

    start, end = parts[0].strip(), parts[1].strip()

    # Parse letter + numeric
    m_start = re.match(r"^([A-Z])(\d+)$", start)
    m_end = re.match(r"^([A-Z])(\d+)$", end)
    if not m_start or not m_end:
        # Can't parse; return both endpoints
        return {start, end} & valid_nodes

    letter_s, num_s = m_start.group(1), int(m_start.group(2))
    letter_e, num_e = m_end.group(1), int(m_end.group(2))

    if letter_s == letter_e:
        # Same-letter range: expand numerically
        codes = set()
        for n in range(num_s, num_e + 1):
            codes.add(f"{letter_s}{n:02d}")
        return codes & valid_nodes
    else:
        # Cross-letter range (e.g. E79--M10, I60-G45, F00--R54).
        # These represent composite chronic conditions whose ICD codes span
        # multiple chapters.  Just include the two endpoint codes rather than
        # sweeping entire chapters.
        return {start, end} & valid_nodes


def load_chronic_icd_codes(data_root: Path, valid_nodes: Set[str]) -> Set[str]:
    """Read Chronic_All.csv and return set of individual ICD codes.

    Expands all ICD ranges and intersects with valid_nodes.
    """
    csv_path = data_root / "Chronic_All.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Chronic_All.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    chronic_codes: Set[str] = set()
    for _, row in df.iterrows():
        expanded = expand_icd_range(row["icd_code"], valid_nodes)
        chronic_codes |= expanded
    return chronic_codes


def step_i_threshold_robustness(
    sex: str,
    node_type: str,
    graphs: Dict[int, nx.Graph],
    ref_age: int,
    emb_params: dict,
    knn_k: int,
    force: bool,
    log: logging.Logger,
) -> None:
    """Edge-weight filtering robustness: compare drift rankings across thresholds."""
    tbl_dir = OUTPUT_ROOT / "tables" / sex / node_type
    csv_path = tbl_dir / "threshold_robustness_summary.csv"
    if cached(csv_path, force):
        log.info("  [I] Loaded cached threshold robustness summary")
        return

    thresholds = [1.0, 0.8, 0.6]  # keep top 100%, 80%, 60% of edges
    ages = sorted(graphs.keys())
    drift_by_threshold: Dict[float, pd.Series] = {}

    for thr in thresholds:
        log.info("  [I] Threshold %.0f%%: training + aligning ...", thr * 100)
        # Filter edges: keep top thr fraction by weight
        filtered_graphs: Dict[int, nx.Graph] = {}
        for age, G in graphs.items():
            if thr >= 1.0:
                filtered_graphs[age] = G
                continue
            weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
            if not weights:
                filtered_graphs[age] = G
                continue
            cutoff = np.percentile(weights, (1.0 - thr) * 100)
            H = G.copy()
            remove = [(u, v) for u, v, d in H.edges(data=True)
                       if d.get("weight", 1.0) < cutoff]
            H.remove_edges_from(remove)
            filtered_graphs[age] = H

        # Train embeddings on filtered graphs
        embeddings: Dict[int, Dict[str, np.ndarray]] = {}
        for age in ages:
            if age not in filtered_graphs:
                continue
            emb = train_node2vec(filtered_graphs[age], **emb_params)
            embeddings[age] = emb

        if ref_age not in embeddings or len(embeddings) < 2:
            log.warning("  [I] Threshold %.0f%%: not enough embeddings", thr * 100)
            continue

        # Align to reference
        from src.alignment import align_all_to_reference
        aligned, _ = align_all_to_reference(embeddings, ref_age)

        # Compute mean drift per node across all transitions
        drift_records: Dict[str, List[float]] = {}
        for i in range(len(ages) - 1):
            a_from, a_to = ages[i], ages[i + 1]
            if a_from not in aligned or a_to not in aligned:
                continue
            emb_from, emb_to = aligned[a_from], aligned[a_to]
            common = set(emb_from.keys()) & set(emb_to.keys())
            for node in common:
                d = float(np.linalg.norm(emb_from[node] - emb_to[node]))
                drift_records.setdefault(node, []).append(d)

        mean_drift = pd.Series({n: np.mean(ds) for n, ds in drift_records.items()})
        drift_by_threshold[thr] = mean_drift

    # Compare drift rankings across thresholds
    rows = []
    thr_keys = sorted(drift_by_threshold.keys(), reverse=True)
    for i in range(len(thr_keys)):
        for j in range(i + 1, len(thr_keys)):
            ta, tb = thr_keys[i], thr_keys[j]
            sa, sb = drift_by_threshold[ta], drift_by_threshold[tb]
            common_nodes = sorted(set(sa.index) & set(sb.index))
            if len(common_nodes) < 10:
                continue
            rho, _ = sp_stats.spearmanr(
                sa.loc[common_nodes].values, sb.loc[common_nodes].values
            )
            # Jaccard of top-20 drifters
            top_a = set(sa.loc[common_nodes].nlargest(20).index)
            top_b = set(sb.loc[common_nodes].nlargest(20).index)
            jaccard = len(top_a & top_b) / len(top_a | top_b) if top_a | top_b else 0.0
            rows.append({
                "threshold_a": f"{ta:.0%}",
                "threshold_b": f"{tb:.0%}",
                "spearman_rho": round(rho, 4),
                "jaccard_at_20": round(jaccard, 4),
                "n_common": len(common_nodes),
            })

    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        log.info("  [I] Threshold robustness: %d comparisons saved", len(rows))
    else:
        log.warning("  [I] No threshold comparisons could be computed")


def step_h_chronic_analysis(
    data_root: Path,
    age_labels: Dict[int, str],
    force: bool,
    log: logging.Logger,
) -> None:
    """H: Chronic vs non-chronic drift + per-node trajectory analysis (ICD only)."""
    log.info("=" * 60)
    log.info("STEP H — Chronic analysis & node trajectories")
    log.info("=" * 60)

    fig_dir = OUTPUT_ROOT / "figures" / "chronic_analysis"
    tbl_dir = OUTPUT_ROOT / "tables" / "chronic_analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    for sex in SEXES:
        log.info("  [H] Processing %s / icd ...", sex)

        # Load drift data
        drift_path = OUTPUT_ROOT / "tables" / sex / "icd" / "drift.csv"
        if not drift_path.exists():
            log.warning("  [H] drift.csv missing for %s/icd, skipping", sex)
            continue
        drift_df = pd.read_csv(drift_path)
        all_nodes = set(drift_df["node"].unique())

        # Load chronic codes
        chronic_codes = load_chronic_icd_codes(data_root, all_nodes)
        log.info("  [H] %d chronic codes matched out of %d graph nodes",
                 len(chronic_codes), len(all_nodes))

        # --- H1: Chronic vs non-chronic drift comparison ---
        _step_h1_chronic_drift(sex, drift_df, chronic_codes, age_labels,
                               fig_dir, tbl_dir, force, log)

        # --- H2: Per-node trajectories ---
        _step_h2_node_trajectories(sex, drift_df, all_nodes, age_labels,
                                   fig_dir, tbl_dir, force, log)


def _step_h1_chronic_drift(
    sex: str,
    drift_df: pd.DataFrame,
    chronic_codes: Set[str],
    age_labels: Dict[int, str],
    fig_dir: Path,
    tbl_dir: Path,
    force: bool,
    log: logging.Logger,
) -> None:
    """H1: Compare drift distributions for chronic vs non-chronic ICD codes."""
    # Tag nodes
    drift_df = drift_df.copy()
    drift_df["chronic"] = drift_df["node"].isin(chronic_codes)

    # --- CSV: per-node mean drift with chronic flag ---
    csv_path = tbl_dir / f"drift_chronic_vs_other_{sex}.csv"
    if not cached(csv_path, force):
        mean_drift = (
            drift_df.groupby("node")["drift"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        mean_drift.columns = ["node", "mean_drift", "std_drift", "n_transitions"]
        mean_drift["chronic"] = mean_drift["node"].isin(chronic_codes)
        mean_drift = mean_drift.sort_values("mean_drift", ascending=False)
        mean_drift.to_csv(csv_path, index=False)
        log.info("  [H1] Saved %s", csv_path.name)

    # --- Boxplot: side-by-side per transition ---
    box_path = fig_dir / f"drift_chronic_vs_other_{sex}.png"
    if not cached(box_path, force):
        drift_df["transition"] = (
            drift_df["age_from"].astype(str) + "\u2192" + drift_df["age_to"].astype(str)
        )
        transitions = sorted(drift_df["transition"].unique())

        fig, ax = plt.subplots(figsize=(max(10, len(transitions) * 1.5), 5))
        positions_chronic = []
        positions_other = []
        data_chronic = []
        data_other = []
        x_ticks = []

        for i, t in enumerate(transitions):
            t_df = drift_df[drift_df["transition"] == t]
            data_chronic.append(t_df.loc[t_df["chronic"], "drift"].values)
            data_other.append(t_df.loc[~t_df["chronic"], "drift"].values)
            positions_chronic.append(i * 3)
            positions_other.append(i * 3 + 1)
            x_ticks.append(i * 3 + 0.5)

        bp_c = ax.boxplot(data_chronic, positions=positions_chronic, widths=0.8,
                          patch_artist=True, manage_ticks=False)
        bp_o = ax.boxplot(data_other, positions=positions_other, widths=0.8,
                          patch_artist=True, manage_ticks=False)

        for patch in bp_c["boxes"]:
            patch.set_facecolor("#e74c3c")
            patch.set_alpha(0.7)
        for patch in bp_o["boxes"]:
            patch.set_facecolor("#3498db")
            patch.set_alpha(0.7)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(transitions, fontsize=9)
        ax.set_xlabel("Age transition")
        ax.set_ylabel("Drift (L2)")
        ax.set_title(f"Drift: Chronic vs Non-Chronic ICD codes ({sex.title()})")
        ax.legend([bp_c["boxes"][0], bp_o["boxes"][0]],
                  ["Chronic", "Non-chronic"], loc="upper left")
        fig.tight_layout()
        fig.savefig(box_path, dpi=150)
        plt.close(fig)
        log.info("  [H1] Saved %s", box_path.name)

    # --- Violin plot: overall chronic vs non-chronic ---
    violin_path = fig_dir / f"drift_violin_{sex}.png"
    if not cached(violin_path, force):
        fig, ax = plt.subplots(figsize=(6, 5))
        chronic_drifts = drift_df.loc[drift_df["chronic"], "drift"].values
        other_drifts = drift_df.loc[~drift_df["chronic"], "drift"].values

        vp = ax.violinplot([chronic_drifts, other_drifts], positions=[1, 2],
                           showmeans=True, showmedians=True)
        vp["bodies"][0].set_facecolor("#e74c3c")
        vp["bodies"][0].set_alpha(0.7)
        vp["bodies"][1].set_facecolor("#3498db")
        vp["bodies"][1].set_alpha(0.7)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Chronic", "Non-chronic"])
        ax.set_ylabel("Drift (L2)")
        ax.set_title(f"Drift Distribution: Chronic vs Non-Chronic ({sex.title()})")

        # Add summary stats as text
        c_med = np.median(chronic_drifts)
        o_med = np.median(other_drifts)
        c_mean = np.mean(chronic_drifts)
        o_mean = np.mean(other_drifts)
        ax.text(0.02, 0.98,
                f"Chronic: median={c_med:.3f}, mean={c_mean:.3f}, n={len(chronic_drifts)}\n"
                f"Other:   median={o_med:.3f}, mean={o_mean:.3f}, n={len(other_drifts)}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        fig.tight_layout()
        fig.savefig(violin_path, dpi=150)
        plt.close(fig)
        log.info("  [H1] Saved %s", violin_path.name)


def _step_h2_node_trajectories(
    sex: str,
    drift_df: pd.DataFrame,
    all_nodes: Set[str],
    age_labels: Dict[int, str],
    fig_dir: Path,
    tbl_dir: Path,
    force: bool,
    log: logging.Logger,
) -> None:
    """H2: Per-node age trajectory distances for selected nodes."""
    # Determine selected nodes: clinical interest + top 5 drifters
    top_path = OUTPUT_ROOT / "tables" / sex / "icd" / "top_drifters.csv"
    selected = [n for n in TRAJECTORY_NODES if n in all_nodes]
    if top_path.exists():
        top_df = pd.read_csv(top_path)
        for node in top_df["node"].head(5):
            if node not in selected and node in all_nodes:
                selected.append(node)

    if not selected:
        log.warning("  [H2] No trajectory nodes available for %s, skipping", sex)
        return

    # Load all aligned embeddings
    aligned_dir = OUTPUT_ROOT / "embeddings" / sex / "icd" / "aligned"
    emb_by_age: Dict[int, Dict[str, np.ndarray]] = {}
    for age in ALL_AGES:
        p = aligned_dir / f"age_{age}.parquet"
        if p.exists():
            emb_by_age[age] = load_embeddings(p)

    if len(emb_by_age) < 2:
        log.warning("  [H2] Not enough aligned embeddings for %s, skipping", sex)
        return

    available_ages = sorted(emb_by_age.keys())
    ref_age = available_ages[0]

    # --- CSV: distance-from-reference per node per age ---
    traj_csv_path = tbl_dir / f"node_trajectory_{sex}.csv"
    if not cached(traj_csv_path, force):
        rows = []
        for node in selected:
            if node not in emb_by_age[ref_age]:
                continue
            ref_vec = emb_by_age[ref_age][node]
            for age in available_ages:
                if node not in emb_by_age[age]:
                    continue
                vec = emb_by_age[age][node]
                dist = float(np.linalg.norm(vec - ref_vec))
                rows.append({
                    "node": node,
                    "age": age,
                    "age_range": age_labels.get(age, str(age)),
                    "distance_from_ref": dist,
                })
        traj_df = pd.DataFrame(rows)
        traj_df.to_csv(traj_csv_path, index=False)
        log.info("  [H2] Saved %s", traj_csv_path.name)
    else:
        traj_df = pd.read_csv(traj_csv_path)

    # --- Line plot: drift trajectory for selected nodes ---
    traj_fig_path = fig_dir / f"node_trajectories_{sex}.png"
    if not cached(traj_fig_path, force):
        fig, ax = plt.subplots(figsize=(9, 5))
        for node in selected:
            node_df = traj_df[traj_df["node"] == node].sort_values("age")
            if node_df.empty:
                continue
            label = node
            if node in TRAJECTORY_NODES:
                label = f"{node} *"  # mark clinical nodes
            ax.plot(node_df["age"], node_df["distance_from_ref"],
                    marker="o", markersize=5, linewidth=1.5, label=label)

        ax.set_xlabel("Age group")
        ax.set_ylabel(f"L2 distance from age {ref_age} embedding")
        ax.set_title(f"Node Embedding Trajectories ({sex.title()} / ICD)")
        ax.set_xticks(available_ages)
        ax.set_xticklabels([age_labels.get(a, str(a)) for a in available_ages],
                           fontsize=8, rotation=45, ha="right")
        ax.legend(fontsize=7, loc="upper left", ncol=2,
                  title="* = clinical interest", title_fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(traj_fig_path, dpi=150)
        plt.close(fig)
        log.info("  [H2] Saved %s", traj_fig_path.name)

    # --- Heatmap: age×age pairwise distance for E11 (or first available) ---
    heatmap_node = "E11" if "E11" in all_nodes else selected[0]
    heatmap_path = fig_dir / f"distance_heatmap_{sex}.png"
    if not cached(heatmap_path, force):
        # Build pairwise distance matrix
        ages_with_node = [a for a in available_ages if heatmap_node in emb_by_age[a]]
        n = len(ages_with_node)
        dist_matrix = np.zeros((n, n))
        for i, a1 in enumerate(ages_with_node):
            for j, a2 in enumerate(ages_with_node):
                v1 = emb_by_age[a1][heatmap_node]
                v2 = emb_by_age[a2][heatmap_node]
                dist_matrix[i, j] = float(np.linalg.norm(v1 - v2))

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(dist_matrix, cmap="YlOrRd", aspect="equal")
        age_strs = [age_labels.get(a, str(a)) for a in ages_with_node]
        ax.set_xticks(range(n))
        ax.set_xticklabels(age_strs, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(age_strs, fontsize=8)
        ax.set_xlabel("Age group")
        ax.set_ylabel("Age group")
        ax.set_title(f"Pairwise Embedding Distance — {heatmap_node} ({sex.title()})")

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = dist_matrix[i, j]
                color = "white" if val > dist_matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

        fig.colorbar(im, ax=ax, label="L2 distance", shrink=0.8)
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        log.info("  [H2] Saved %s", heatmap_path.name)


def plot_combined_link_prediction(
    force: bool, log: logging.Logger,
) -> None:
    """H3: Combined link prediction summary — relabeled structure preservation figure."""
    out_path = OUTPUT_ROOT / "figures" / "combined_link_prediction.png"
    if cached(out_path, force):
        log.info("Combined link prediction figure already exists, skipping")
        return

    fig, axes = plt.subplots(len(SEXES), len(NODE_TYPES),
                             figsize=(6 * len(NODE_TYPES), 4 * len(SEXES)))
    for row, sex in enumerate(SEXES):
        for col, node_type in enumerate(NODE_TYPES):
            ax = axes[row, col]
            csv_path = OUTPUT_ROOT / "tables" / sex / node_type / "structure_preservation.csv"
            if not csv_path.exists():
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
                ax.axis("off")
                continue
            df = pd.read_csv(csv_path)
            ages_str = df["age"].astype(str)
            x = np.arange(len(ages_str))
            w = 0.35
            ax.bar(x - w / 2, df["auc"], w, label="Link-pred AUC", color="steelblue")
            ax.bar(x + w / 2, df["spearman_r"], w, label="Weight–sim Spearman r",
                   color="coral")
            ax.set_xticks(x)
            ax.set_xticklabels(ages_str)
            ax.set_xlabel("Age group", fontsize=8)
            ax.set_ylabel("Score", fontsize=8)
            ax.set_title(f"{sex.title()} / {node_type.upper()}", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.tick_params(labelsize=7)

            # Add value labels on bars
            for xi, auc_val in zip(x, df["auc"]):
                if not np.isnan(auc_val):
                    ax.text(xi - w / 2, auc_val + 0.02, f"{auc_val:.2f}",
                            ha="center", fontsize=6, color="steelblue")
            for xi, sr_val in zip(x, df["spearman_r"]):
                if not np.isnan(sr_val):
                    ax.text(xi + w / 2, sr_val + 0.02, f"{sr_val:.2f}",
                            ha="center", fontsize=6, color="coral")

            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Link Prediction Validation — Can We Trust the Embeddings?",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved combined link prediction figure → %s", out_path)

    # --- Individual panels ---
    split_dir = OUTPUT_ROOT / "figures" / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    for sex in SEXES:
        for node_type in NODE_TYPES:
            ind_path = split_dir / f"link_prediction_{sex}_{node_type}.png"
            if cached(ind_path, force):
                continue
            csv_path = OUTPUT_ROOT / "tables" / sex / node_type / "structure_preservation.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            ages_str = df["age"].astype(str)
            x = np.arange(len(ages_str))
            w = 0.35
            fig_i, ax_i = plt.subplots(figsize=(7, 4))
            ax_i.bar(x - w / 2, df["auc"], w, label="Link-pred AUC", color="steelblue")
            ax_i.bar(x + w / 2, df["spearman_r"], w, label="Weight\u2013sim Spearman r",
                     color="coral")
            ax_i.set_xticks(x)
            ax_i.set_xticklabels(ages_str)
            ax_i.set_xlabel("Age group")
            ax_i.set_ylabel("Score")
            ax_i.set_title(f"Link Prediction — {sex.title()} / {node_type.upper()}")
            ax_i.set_ylim(0, 1.05)
            for xi, auc_val in zip(x, df["auc"]):
                if not np.isnan(auc_val):
                    ax_i.text(xi - w / 2, auc_val + 0.02, f"{auc_val:.2f}",
                              ha="center", fontsize=7, color="steelblue")
            for xi, sr_val in zip(x, df["spearman_r"]):
                if not np.isnan(sr_val):
                    ax_i.text(xi + w / 2, sr_val + 0.02, f"{sr_val:.2f}",
                              ha="center", fontsize=7, color="coral")
            ax_i.legend(fontsize=8)
            fig_i.tight_layout()
            fig_i.savefig(ind_path, dpi=150, bbox_inches="tight")
            plt.close(fig_i)
    log.info("Saved split link prediction panels → %s", split_dir)


# ---------------------------------------------------------------------------
# Per-combination runner
# ---------------------------------------------------------------------------

def run_combination(
    sex: str,
    node_type: str,
    data_root: Path,
    age_labels: Dict[int, str],
    reference_age: int,
    emb_params: dict,
    knn_k: int,
    force: bool,
    robustness: bool,
    threshold_robustness: bool,
    log: logging.Logger,
) -> dict:
    """Run the full pipeline for one (sex, node_type) combination."""
    t0 = time.time()
    log.info("=" * 60)
    log.info("START  %s / %s", sex, node_type)
    log.info("=" * 60)

    # Run metadata (cache safety)
    tbl_dir_meta = OUTPUT_ROOT / "tables" / sex / node_type
    tbl_dir_meta.mkdir(parents=True, exist_ok=True)
    meta_path = tbl_dir_meta / "run_metadata.json"
    current_meta = _build_run_metadata(emb_params, reference_age, knn_k)
    if not force:
        _check_metadata(meta_path, current_meta, log)
    _write_metadata(meta_path, current_meta)

    # A — Graph inspection
    graphs, ref_age = step_a_graph_inspection(
        sex, node_type, data_root, age_labels, reference_age, force, log
    )

    # Staleness check: warn if cached outputs are missing expected new columns/files
    if not force:
        tbl_dir = OUTPUT_ROOT / "tables" / sex / node_type
        _stats_path = tbl_dir / "graph_stats.csv"
        if _stats_path.exists():
            _cols = pd.read_csv(_stats_path, nrows=0).columns.tolist()
            if "n_isolates" not in _cols or "isolate_frac" not in _cols:
                log.warning(
                    "  STALE OUTPUTS: graph_stats.csv is missing isolate columns. "
                    "Rerun with --force to regenerate."
                )
        _ni_aligned_dir = OUTPUT_ROOT / "embeddings" / sex / node_type / "aligned_non_iso"
        if not _ni_aligned_dir.exists() or not any(_ni_aligned_dir.glob("*.parquet")):
            _aligned_dir = OUTPUT_ROOT / "embeddings" / sex / node_type / "aligned"
            if _aligned_dir.exists() and any(_aligned_dir.glob("*.parquet")):
                log.warning(
                    "  STALE OUTPUTS: aligned_non_iso/ is missing but aligned/ exists. "
                    "Rerun with --force to generate non-isolate aligned embeddings."
                )
        _sp_path = tbl_dir / "structure_preservation.csv"
        if _sp_path.exists():
            _sp_cols = pd.read_csv(_sp_path, nrows=0).columns.tolist()
            expected_sp = {"auc_std", "n_splits", "n_safe", "n_held_out",
                           "two_hop_pool", "n_neg", "used_fallback"}
            missing_sp = expected_sp - set(_sp_cols)
            if missing_sp:
                log.warning(
                    "  STALE OUTPUTS: structure_preservation.csv missing columns %s. "
                    "Rerun with --force to regenerate.", sorted(missing_sp),
                )

    # B — Embeddings
    embeddings_by_age = step_b_embeddings(
        sex, node_type, graphs, emb_params, force, log
    )

    # C — PCA plots
    step_c_pca_plots(sex, node_type, graphs, embeddings_by_age, age_labels, force, log)

    # D — Alignment
    aligned, aligned_non_iso = step_d_alignment(
        sex, node_type, embeddings_by_age, graphs, ref_age, force, log
    )

    # E — Metrics
    step_e_metrics(sex, node_type, aligned, aligned_non_iso, graphs, knn_k, age_labels, force, log)

    # F — Structure preservation
    step_f_structure(sex, node_type, graphs, aligned, age_labels, emb_params, force, log)

    # G — Robustness (optional)
    if robustness:
        step_g_robustness(sex, node_type, graphs, ref_age, emb_params, force, log)

    # I — Threshold robustness (optional)
    if threshold_robustness:
        step_i_threshold_robustness(
            sex, node_type, graphs, ref_age, emb_params, knn_k, force, log
        )

    # Post-run output verification
    verify_outputs(sex, node_type, log)

    elapsed = time.time() - t0
    log.info("DONE   %s / %s  (%.1f s)", sex, node_type, elapsed)
    return {"sex": sex, "node_type": node_type, "status": "OK", "seconds": round(elapsed, 1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full comorbidity-embedding pipeline for all combinations."
    )
    p.add_argument("--force", action="store_true",
                   help="Regenerate all outputs (ignore cache)")
    p.add_argument("--robustness", action="store_true",
                   help="Include robustness analysis (slower)")
    p.add_argument("--reproducible", action="store_true",
                   help="Force workers=1 for deterministic embedding training")
    p.add_argument("--threshold-robustness", action="store_true",
                   help="Include edge-weight threshold robustness (step I)")
    p.add_argument("--dim", type=int, default=128,
                   help="Embedding dimensionality (default 128)")
    p.add_argument("--walk-length", type=int, default=80,
                   help="Random walk length (default 80)")
    p.add_argument("--num-walks", type=int, default=10,
                   help="Walks per node (default 10)")
    p.add_argument("--window", type=int, default=10,
                   help="Word2Vec window (default 10)")
    p.add_argument("--p", type=float, default=1.0,
                   help="Node2vec return parameter (default 1.0)")
    p.add_argument("--q", type=float, default=1.0,
                   help="Node2vec in-out parameter (default 1.0)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default 42)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers (default 4)")
    p.add_argument("--knn-k", type=int, default=25,
                   help="k for kNN stability (default 25)")
    p.add_argument("--reference-age", type=int, default=1,
                   help="Reference age group for alignment (default 1)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Custom output directory (default: final/outputs)")
    return p.parse_args()


def main() -> None:
    global OUTPUT_ROOT
    args = parse_args()
    if args.output_dir:
        OUTPUT_ROOT = Path(args.output_dir).resolve()
    log = setup_logging()

    # Data root
    data_root = Path(os.environ.get("DATA_ROOT", str(REPO_ROOT / ".." / "Data")))
    if not data_root.exists():
        log.error("Data root not found: %s", data_root)
        sys.exit(1)

    # Age labels
    age_groups_path = REPO_ROOT / "configs" / "age_groups.yaml"
    with open(age_groups_path) as f:
        age_labels: Dict[int, str] = yaml.safe_load(f)["age_groups"]

    emb_params = {
        "dim": args.dim,
        "walk_length": args.walk_length,
        "num_walks": args.num_walks,
        "window": args.window,
        "p": args.p,
        "q": args.q,
        "seed": args.seed,
        "workers": args.workers,
    }

    if args.reproducible:
        emb_params["workers"] = 1
        log.info("--reproducible: forcing workers=1 for deterministic training")

    log.info(
        "Pipeline starting  (force=%s, robustness=%s, reproducible=%s, threshold_robustness=%s)",
        args.force, args.robustness, args.reproducible, args.threshold_robustness,
    )
    log.info("Data root: %s", data_root.resolve())
    log.info("Output root: %s", OUTPUT_ROOT.resolve())

    summary_rows = []

    for sex in SEXES:
        for node_type in NODE_TYPES:
            combo_handler = add_combo_file_handler(log, sex, node_type)
            try:
                row = run_combination(
                    sex=sex,
                    node_type=node_type,
                    data_root=data_root,
                    age_labels=age_labels,
                    reference_age=args.reference_age,
                    emb_params=emb_params,
                    knn_k=args.knn_k,
                    force=args.force,
                    robustness=args.robustness,
                    threshold_robustness=args.threshold_robustness,
                    log=log,
                )
            except Exception as e:
                log.error("FAILED  %s / %s: %s", sex, node_type, e, exc_info=True)
                row = {"sex": sex, "node_type": node_type, "status": f"FAIL: {e}", "seconds": 0}
            finally:
                log.removeHandler(combo_handler)
                combo_handler.close()

            summary_rows.append(row)

    # Write summary CSV
    summary_path = OUTPUT_ROOT / "logs" / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sex", "node_type", "status", "seconds"])
        writer.writeheader()
        writer.writerows(summary_rows)

    # Combined summary figures
    plot_all_combined(data_root, age_labels, args.force, log)

    # H — Chronic analysis & node trajectories (ICD variant only)
    try:
        step_h_chronic_analysis(data_root, age_labels, args.force, log)
        plot_combined_link_prediction(args.force, log)
    except Exception as e:
        log.error("STEP H failed: %s", e, exc_info=True)

    log.info("-" * 60)
    log.info("SUMMARY")
    for r in summary_rows:
        log.info("  %s / %-8s  %s  (%.1f s)", r["sex"], r["node_type"], r["status"], r["seconds"])
    log.info("Summary written to %s", summary_path)
    log.info("All done.")


if __name__ == "__main__":
    main()
