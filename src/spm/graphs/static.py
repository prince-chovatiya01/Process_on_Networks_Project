"""
Static graph representations and operations.
"""

from pathlib import Path
from typing import Optional, Union
import networkx as nx


def load_edgelist(filepath: Union[str, Path], weighted: bool = False) -> nx.Graph:
    """Load an edge list from disk into an undirected NetworkX Graph."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Edge list file not found: {path}")
    if weighted:
        return nx.read_weighted_edgelist(path)
    return nx.read_edgelist(path, nodetype=int)
