"""
Orthogonal Procrustes alignment module.

Aligns embedding spaces across age groups using orthogonal Procrustes analysis.
This enables meaningful comparison of node positions across different age strata.

The alignment finds the optimal orthogonal transformation R that minimizes:
    ||E_target @ R - E_reference||_F

where the Frobenius norm is computed over anchor nodes present in both spaces.
Residual error is reported as mean Frobenius error per anchor: ||·||_F / |S|.

References:
- Schönemann (1966): A generalized solution of the orthogonal Procrustes problem
- Hamilton et al. (2016): Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes


@dataclass
class AlignmentResult:
    """
    Result of Procrustes alignment.

    Attributes:
        transform: Orthogonal transformation matrix R (dim x dim)
        scale: Scaling factor (always 1.0 for orthogonal Procrustes)
        residual_error: Mean Frobenius error per anchor (||XR - X_ref||_F / |S|)
        num_anchors: Number of anchor nodes used for alignment
        anchor_nodes: Set of node IDs used as anchors
    """
    transform: np.ndarray
    scale: float
    residual_error: float
    num_anchors: int
    anchor_nodes: Set[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "residual_error": float(self.residual_error),
            "num_anchors": self.num_anchors,
            "scale": float(self.scale),
        }


def find_anchor_nodes(
    embeddings_ref: Dict[str, np.ndarray],
    embeddings_target: Dict[str, np.ndarray],
) -> Set[str]:
    """
    Find anchor nodes present in both embedding spaces.

    Args:
        embeddings_ref: Reference embeddings (node -> vector)
        embeddings_target: Target embeddings (node -> vector)

    Returns:
        Set of node IDs present in both spaces
    """
    return set(embeddings_ref.keys()) & set(embeddings_target.keys())


def procrustes_align(
    embeddings_ref: Dict[str, np.ndarray],
    embeddings_target: Dict[str, np.ndarray],
    anchor_nodes: Optional[Set[str]] = None,
) -> Tuple[Dict[str, np.ndarray], AlignmentResult]:
    """
    Align target embeddings to reference using orthogonal Procrustes.

    Finds orthogonal matrix R that minimizes ||E_target @ R - E_ref||_F
    over the anchor nodes, then applies R to all target embeddings.

    Args:
        embeddings_ref: Reference embeddings (node -> vector)
        embeddings_target: Target embeddings to align (node -> vector)
        anchor_nodes: Optional set of anchor nodes. If None, uses intersection.

    Returns:
        Tuple of:
        - aligned_embeddings: Target embeddings after transformation
        - result: AlignmentResult with transform matrix and diagnostics

    Raises:
        ValueError: If fewer than 2 anchor nodes available
    """
    # Determine anchor nodes
    if anchor_nodes is None:
        anchor_nodes = find_anchor_nodes(embeddings_ref, embeddings_target)

    anchor_list = sorted(anchor_nodes)  # Deterministic ordering

    if len(anchor_list) < 2:
        raise ValueError(
            f"Need at least 2 anchor nodes for alignment, got {len(anchor_list)}"
        )

    # Build matrices for anchor nodes
    # Rows = nodes, columns = dimensions
    E_ref = np.array([embeddings_ref[n] for n in anchor_list])
    E_target = np.array([embeddings_target[n] for n in anchor_list])

    # Compute orthogonal Procrustes: find R minimizing ||E_target @ R - E_ref||
    # scipy returns (R, scale) where R is orthogonal
    R, scale = orthogonal_procrustes(E_target, E_ref)

    # Compute residual error: mean Frobenius error per anchor (proposal §5.2)
    aligned_anchors = E_target @ R
    fro_norm = np.linalg.norm(aligned_anchors - E_ref, "fro")
    num_anchors = len(anchor_list)
    residual = fro_norm / num_anchors if num_anchors > 0 else float("nan")

    # Apply transformation to all target embeddings
    aligned_embeddings = {}
    for node, vec in embeddings_target.items():
        aligned_embeddings[node] = vec @ R

    result = AlignmentResult(
        transform=R,
        scale=scale,
        residual_error=residual,
        num_anchors=len(anchor_list),
        anchor_nodes=set(anchor_list),
    )

    return aligned_embeddings, result


def save_alignment_results(
    path: Union[str, Path],
    results: List[Tuple[int, AlignmentResult]],
) -> None:
    """
    Save alignment results to CSV.

    Args:
        path: Output file path
        results: List of (age_group, AlignmentResult) tuples
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for age, result in results:
        rows.append({
            "age_group": age,
            "residual_error": result.residual_error,
            "num_anchors": result.num_anchors,
            "scale": result.scale,
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def align_all_to_reference(
    embeddings_by_age: Dict[int, Dict[str, np.ndarray]],
    reference_age: int = 1,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, AlignmentResult]]:
    """
    Align all age groups to a reference age.

    Args:
        embeddings_by_age: Dictionary mapping age group -> embeddings dict
        reference_age: Age group to use as reference (default: 1)

    Returns:
        Tuple of:
        - aligned: Dictionary mapping age -> aligned embeddings
        - results: Dictionary mapping age -> AlignmentResult

    Note:
        The reference age embeddings are returned unchanged.
    """
    if reference_age not in embeddings_by_age:
        raise ValueError(f"Reference age {reference_age} not in embeddings")

    ref_emb = embeddings_by_age[reference_age]

    aligned = {}
    results = {}

    for age, emb in embeddings_by_age.items():
        if age == reference_age:
            # Reference is not transformed
            aligned[age] = emb.copy()
            results[age] = AlignmentResult(
                transform=np.eye(len(next(iter(emb.values())))),
                scale=1.0,
                residual_error=0.0,
                num_anchors=len(emb),
                anchor_nodes=set(emb.keys()),
            )
        else:
            aligned_emb, result = procrustes_align(ref_emb, emb)
            aligned[age] = aligned_emb
            results[age] = result

    return aligned, results
