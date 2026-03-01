"""
Node2Vec embedding module.

Trains node2vec embeddings on comorbidity graphs using random walks
and Word2Vec. Supports saving/loading embeddings in Parquet format.

References:
- Grover & Leskovec (2016): node2vec: Scalable Feature Learning for Networks
"""

from pathlib import Path
from typing import Dict, Optional, Union

import networkx as nx
import numpy as np

import pandas as pd
from gensim.models import Word2Vec


class Node2VecWalker:
    """
    Generates biased random walks for node2vec.

    The walk bias is controlled by parameters p and q:
    - p: Return parameter. Higher p = less likely to revisit previous node.
    - q: In-out parameter. Higher q = biased toward BFS-like exploration.

    When p=q=1, walks are equivalent to unbiased random walks (DeepWalk).
    """

    def __init__(
        self,
        graph: nx.Graph,
        p: float = 1.0,
        q: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the walker.

        Args:
            graph: NetworkX graph (undirected)
            p: Return parameter
            q: In-out parameter
            seed: Random seed for reproducibility
        """
        self.graph = graph
        self.p = p
        self.q = q
        self.rng = np.random.default_rng(seed)

        # Precompute alias tables for efficient sampling if p != 1 or q != 1
        self._precompute_transition_probs()

    def _precompute_transition_probs(self) -> None:
        """Precompute transition probabilities for biased walks."""
        self.alias_nodes = {}
        self.alias_edges = {}

        # For each node, compute normalized neighbor weights
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if not neighbors:
                continue
            # Use edge weights if available
            weights = [
                self.graph[node][nbr].get("weight", 1.0)
                for nbr in neighbors
            ]
            norm = sum(weights)
            self.alias_nodes[node] = (neighbors, [w / norm for w in weights])

        # For biased walks (p != 1 or q != 1), precompute edge transition probs
        if self.p != 1.0 or self.q != 1.0:
            for src, dst in self.graph.edges():
                self._precompute_edge_probs(src, dst)
                self._precompute_edge_probs(dst, src)

    def _precompute_edge_probs(self, src: str, dst: str) -> None:
        """Compute transition probabilities for edge (src -> dst -> ?)."""
        neighbors = list(self.graph.neighbors(dst))
        if not neighbors:
            return

        weights = []
        for nbr in neighbors:
            weight = self.graph[dst][nbr].get("weight", 1.0)
            if nbr == src:
                # Return to previous node
                weights.append(weight / self.p)
            elif self.graph.has_edge(nbr, src):
                # Neighbor of both src and dst (distance 1)
                weights.append(weight)
            else:
                # Neighbor of dst only (distance 2 from src)
                weights.append(weight / self.q)

        norm = sum(weights)
        self.alias_edges[(src, dst)] = (neighbors, [w / norm for w in weights])

    def walk(self, start_node: str, walk_length: int) -> list:
        """
        Perform a single biased random walk starting from start_node.

        Args:
            start_node: Starting node for the walk
            walk_length: Number of steps in the walk

        Returns:
            List of node IDs visited during the walk
        """
        walk = [start_node]

        if start_node not in self.alias_nodes:
            return walk

        while len(walk) < walk_length:
            cur = walk[-1]

            if cur not in self.alias_nodes:
                break

            if len(walk) == 1:
                # First step: sample from node's neighbors
                neighbors, probs = self.alias_nodes[cur]
                next_node = self.rng.choice(neighbors, p=probs)
            else:
                # Subsequent steps: use edge transition probabilities
                prev = walk[-2]
                if (prev, cur) in self.alias_edges:
                    neighbors, probs = self.alias_edges[(prev, cur)]
                    next_node = self.rng.choice(neighbors, p=probs)
                elif cur in self.alias_nodes:
                    neighbors, probs = self.alias_nodes[cur]
                    next_node = self.rng.choice(neighbors, p=probs)
                else:
                    break

            walk.append(next_node)

        return walk

    def generate_walks(
        self,
        num_walks: int,
        walk_length: int,
    ) -> list:
        """
        Generate random walks for all nodes.

        Args:
            num_walks: Number of walks per node
            walk_length: Length of each walk

        Returns:
            List of walks, where each walk is a list of node IDs
        """
        walks = []
        nodes = list(self.graph.nodes())

        for _ in range(num_walks):
            self.rng.shuffle(nodes)
            for node in nodes:
                walk = self.walk(node, walk_length)
                walks.append(walk)

        return walks


def train_node2vec(
    graph: nx.Graph,
    dim: int = 128,
    walk_length: int = 80,
    num_walks: int = 10,
    window: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    workers: int = 4,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Train node2vec embeddings on a graph.

    Args:
        graph: NetworkX graph with nodes as ICD codes
        dim: Embedding dimensionality
        walk_length: Length of each random walk
        num_walks: Number of walks per node
        window: Context window for Word2Vec
        p: Return parameter (1.0 = unbiased)
        q: In-out parameter (1.0 = unbiased)
        workers: Number of parallel workers
        seed: Random seed

    Returns:
        Dictionary mapping node IDs to embedding vectors (numpy arrays)
    """

    # Generate random walks
    walker = Node2VecWalker(graph, p=p, q=q, seed=seed)
    walks = walker.generate_walks(num_walks=num_walks, walk_length=walk_length)

    # Train Word2Vec on walks
    # Note: walks are lists of node IDs (strings)
    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=window,
        min_count=0,  # Include all nodes
        sg=1,  # Skip-gram
        workers=workers,
        seed=seed,
    )

    # Extract embeddings as dictionary
    embeddings = {}
    for node in graph.nodes():
        if node in model.wv:
            embeddings[node] = model.wv[node]

    return embeddings


def save_embeddings(
    path: Union[str, Path],
    embeddings: Dict[str, np.ndarray],
    format: str = "parquet",
) -> None:
    """
    Save embeddings to file.

    Args:
        path: Output file path
        embeddings: Dictionary mapping node IDs to vectors
        format: Output format ('parquet' or 'csv')

    Raises:
        ValueError: If format is not supported
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame: rows = nodes, columns = dimensions
    nodes = list(embeddings.keys())
    vectors = np.array([embeddings[n] for n in nodes])
    dim = vectors.shape[1]

    df = pd.DataFrame(
        vectors,
        index=nodes,
        columns=[f"dim_{i}" for i in range(dim)],
    )
    df.index.name = "node"

    if format == "parquet":
        df.to_parquet(path)
    elif format == "csv":
        df.to_csv(path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'csv'.")


def load_embeddings(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load embeddings from file.

    Args:
        path: Path to embedding file (parquet or csv)

    Returns:
        Dictionary mapping node IDs to vectors
    """
    path = Path(path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, index_col=0)
    else:
        # Try parquet first
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = pd.read_csv(path, index_col=0)

    # Convert to dictionary
    embeddings = {}
    for node in df.index:
        embeddings[str(node)] = df.loc[node].values.astype(np.float32)

    return embeddings
