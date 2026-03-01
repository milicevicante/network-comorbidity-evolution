"""
GEXF graph loader module.

Loads comorbidity graphs from GEXF files into NetworkX graph objects,
preserving node labels (ICD codes) and edge weights.

The loader validates the graph structure and provides statistics about
nodes, edges, and weight distributions.

Note: NetworkX 3.x has namespace issues with GEXF 1.3 files. This module
includes a workaround that temporarily strips the namespace before parsing.
"""

import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import networkx as nx


@dataclass
class GraphStats:
    """
    Statistics about a loaded graph.

    Attributes:
        node_count: Number of nodes in the graph
        edge_count: Number of edges in the graph
        is_directed: Whether the graph is directed
        num_components: Number of connected components
        largest_component_size: Size of the largest connected component
        weight_attr: Name of the weight attribute used
        weights_missing: Number of edges missing weight attribute
        weight_min: Minimum edge weight (None if no weights)
        weight_median: Median edge weight (None if no weights)
        weight_max: Maximum edge weight (None if no weights)
    """
    node_count: int
    edge_count: int
    is_directed: bool
    num_components: int
    largest_component_size: int
    weight_attr: str
    weights_missing: int
    weight_min: Optional[float]
    weight_median: Optional[float]
    weight_max: Optional[float]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "is_directed": self.is_directed,
            "num_components": self.num_components,
            "largest_component_size": self.largest_component_size,
            "weight_attr": self.weight_attr,
            "weights_missing": self.weights_missing,
            "weight_min": self.weight_min,
            "weight_median": self.weight_median,
            "weight_max": self.weight_max,
        }


def load_gexf_graph(
    path: Union[str, Path],
    weight_attr: str = "weight",
) -> Tuple[nx.Graph, GraphStats]:
    """
    Load a GEXF file into a NetworkX graph.

    The loader:
    1. Reads the GEXF file using NetworkX's native parser
    2. Converts node IDs to their label attributes (ICD codes)
    3. Extracts and validates edge weights
    4. Computes graph statistics

    Args:
        path: Path to the GEXF file
        weight_attr: Preferred name for edge weight attribute.
                     Falls back to first numeric edge attribute if not found.

    Returns:
        Tuple of (graph, stats):
        - graph: NetworkX Graph (or DiGraph if directed) with:
            - Nodes keyed by ICD code labels
            - Edges with 'weight' attribute (numeric)
        - stats: GraphStats object with summary statistics

    Raises:
        FileNotFoundError: If the GEXF file doesn't exist
        ValueError: If the file cannot be parsed as valid GEXF
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GEXF file not found: {path}")

    # Load the GEXF file
    # NetworkX 3.x has issues with GEXF 1.3 namespace declarations.
    # Workaround: strip namespace from XML before parsing.
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # NetworkX 3.x only supports GEXF 1.2draft namespace.
        # Convert 1.3 namespaces to 1.2draft equivalents.
        content = re.sub(
            r'xmlns="http://www\.gexf\.net/1\.3"',
            'xmlns="http://www.gexf.net/1.2draft"',
            content
        )
        content = re.sub(
            r'xmlns:viz="http://www\.gexf\.net/1\.3/viz"',
            'xmlns:viz="http://www.gexf.net/1.1draft/viz"',
            content
        )
        content = re.sub(r'version="1\.3"', 'version="1.2"', content)

        # Write to temporary file and parse
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gexf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            G_raw = nx.read_gexf(tmp_path)
        finally:
            Path(tmp_path).unlink()  # Clean up temp file

    except Exception as e:
        raise ValueError(f"Failed to parse GEXF file: {e}")

    # Determine if directed
    is_directed = G_raw.is_directed()

    # Create new graph with labels as node IDs
    # This makes nodes addressable by ICD code (e.g., "A00-A09") instead of numeric IDs
    G = nx.DiGraph() if is_directed else nx.Graph()

    # Build node ID -> label mapping
    id_to_label = {}
    for node_id, attrs in G_raw.nodes(data=True):
        label = attrs.get("label", str(node_id))
        id_to_label[node_id] = label
        # Copy node to new graph with label as ID, preserving other attributes
        node_attrs = {k: v for k, v in attrs.items() if k != "label"}
        G.add_node(label, **node_attrs)

    # Determine which weight attribute to use
    # Check first edge for available numeric attributes
    actual_weight_attr = weight_attr
    sample_edge_attrs = None
    for _, _, attrs in G_raw.edges(data=True):
        sample_edge_attrs = attrs
        break

    if sample_edge_attrs:
        if weight_attr not in sample_edge_attrs:
            # Find first numeric attribute
            for attr_name, attr_val in sample_edge_attrs.items():
                if isinstance(attr_val, (int, float)):
                    actual_weight_attr = attr_name
                    break

    # Copy edges with weight attribute
    weights_missing = 0
    weights = []

    for src, dst, attrs in G_raw.edges(data=True):
        src_label = id_to_label[src]
        dst_label = id_to_label[dst]

        # Extract weight
        edge_weight = attrs.get(actual_weight_attr)
        if edge_weight is None:
            weights_missing += 1
            edge_weight = 1.0  # Default weight if missing
        else:
            try:
                edge_weight = float(edge_weight)
                weights.append(edge_weight)
            except (ValueError, TypeError):
                weights_missing += 1
                edge_weight = 1.0

        G.add_edge(src_label, dst_label, weight=edge_weight)

    # Compute statistics
    if is_directed:
        # For directed graphs, use weakly connected components
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    num_components = len(components)
    largest_component_size = max(len(c) for c in components) if components else 0

    # Weight statistics
    if weights:
        weight_min = min(weights)
        weight_max = max(weights)
        weight_median = statistics.median(weights)
    else:
        weight_min = weight_max = weight_median = None

    stats = GraphStats(
        node_count=G.number_of_nodes(),
        edge_count=G.number_of_edges(),
        is_directed=is_directed,
        num_components=num_components,
        largest_component_size=largest_component_size,
        weight_attr=actual_weight_attr,
        weights_missing=weights_missing,
        weight_min=weight_min,
        weight_median=weight_median,
        weight_max=weight_max,
    )

    return G, stats


def print_graph_stats(stats: GraphStats, path: Optional[Path] = None) -> None:
    """
    Print graph statistics in a human-readable format.

    Args:
        stats: GraphStats object to print
        path: Optional path to the source file (for display)
    """
    if path:
        print(f"Graph: {path.name}")
        print("-" * 50)

    print(f"Nodes:              {stats.node_count}")
    print(f"Edges:              {stats.edge_count}")
    print(f"Directed:           {stats.is_directed}")
    print(f"Components:         {stats.num_components}")
    print(f"Largest component:  {stats.largest_component_size} nodes")
    print()
    print(f"Weight attribute:   '{stats.weight_attr}'")
    print(f"Missing weights:    {stats.weights_missing}")

    if stats.weight_min is not None:
        print(f"Weight min:         {stats.weight_min:.4f}")
        print(f"Weight median:      {stats.weight_median:.4f}")
        print(f"Weight max:         {stats.weight_max:.4f}")
    else:
        print("Weight stats:       N/A (no valid weights)")
