"""
Structural and spectral statistics calculation, including adjacency spectrum and NIMFA threshold.
"""

import logging
from typing import Any, Dict, Optional, Tuple
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from spm.utils.logging_config import setup_logger

logger = setup_logger("spm.graphs.stats")


def compute_largest_eigenvalue(
    G: nx.Graph,
    sparse_threshold: int = 500,
    tol: float = 1e-9
) -> float:
    """
    Compute the largest eigenvalue lambda_1 of the adjacency matrix A.
    Handles disconnected graphs by computing lambda_1 over the entire matrix or components.
    For unweighted/weighted undirected graphs, A is symmetric and real, so eigenvalues are real.
    """
    n = G.number_of_nodes()
    if n == 0:
        raise ValueError("Cannot compute eigenvalues for an empty graph (n=0).")
    if n == 1 or G.number_of_edges() == 0:
        return 0.0

    # Adjacency matrix
    if n <= sparse_threshold:
        A = nx.to_numpy_array(G, dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(A)
        return float(np.max(eigenvalues))
    else:
        A_sparse = nx.to_scipy_sparse_array(G, dtype=np.float64, format="csr")
        try:
            # For symmetric real matrices, eigsh with which='LA' finds largest algebraic eigenvalue
            vals, _ = spla.eigsh(A_sparse, k=1, which="LA", tol=tol, maxiter=2000)
            return float(vals[0])
        except Exception as e:
            logger.warning("Sparse eigensolver failed or did not converge (%s); falling back to dense solver.", e)
            A = nx.to_numpy_array(G, dtype=np.float64)
            return float(np.max(np.linalg.eigvalsh(A)))


def compute_spectral_threshold(
    G: nx.Graph,
    sparse_threshold: int = 500
) -> Tuple[float, Optional[float]]:
    """
    Compute lambda_max and the NIMFA epidemic threshold tau_c = 1 / lambda_max.
    Returns (lambda_max, tau_c). If lambda_max <= 0, tau_c is None (infinite threshold).
    """
    lam1 = compute_largest_eigenvalue(G, sparse_threshold=sparse_threshold)
    if lam1 > 1e-12:
        tau_c = 1.0 / lam1
    else:
        tau_c = None
    return lam1, tau_c


def compute_graph_statistics(
    G: nx.Graph,
    include_centrality: bool = True,
    sparse_threshold: int = 500
) -> Dict[str, Any]:
    """
    Compute comprehensive structural and spectral statistics for a graph.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "is_connected": False,
            "lambda_1": 0.0,
            "tau_c": None
        }

    degrees = [d for _, d in G.degree()]
    mean_degree = float(np.mean(degrees)) if degrees else 0.0
    max_degree = int(np.max(degrees)) if degrees else 0
    min_degree = int(np.min(degrees)) if degrees else 0

    is_connected = nx.is_connected(G)
    num_components = nx.number_connected_components(G)
    lcc_size = len(max(nx.connected_components(G), key=len)) if n > 0 else 0

    # Spectral properties
    lam1, tau_c = compute_spectral_threshold(G, sparse_threshold=sparse_threshold)

    # Core and clustering
    # Note: remove self-loops if any exist for clustering calculation
    if nx.number_of_selfloops(G) > 0:
        G_simple = G.copy()
        G_simple.remove_edges_from(nx.selfloop_edges(G_simple))
    else:
        G_simple = G

    avg_clustering = float(nx.average_clustering(G_simple))
    
    # K-core max number
    core_numbers = nx.core_number(G_simple)
    max_k_core = max(core_numbers.values()) if core_numbers else 0

    stats: Dict[str, Any] = {
        "nodes": n,
        "edges": m,
        "is_connected": is_connected,
        "num_components": num_components,
        "lcc_size": lcc_size,
        "mean_degree": mean_degree,
        "max_degree": max_degree,
        "min_degree": min_degree,
        "avg_clustering": avg_clustering,
        "max_k_core": max_k_core,
        "lambda_1": lam1,
        "tau_c": tau_c
    }

    if include_centrality:
        deg_cent = nx.degree_centrality(G)
        stats["max_degree_centrality"] = float(np.max(list(deg_cent.values())))
        # Eigenvector centrality on LCC if connected
        if is_connected:
            try:
                eig_cent = nx.eigenvector_centrality(G, max_iter=1000)
                stats["max_eigenvector_centrality"] = float(np.max(list(eig_cent.values())))
            except Exception:
                stats["max_eigenvector_centrality"] = None
        else:
            stats["max_eigenvector_centrality"] = None

    return stats
