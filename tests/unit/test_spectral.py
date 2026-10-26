"""
Analytical validation of spectral calculations and structural statistics against exact mathematical results.
"""

import math
import networkx as nx
import numpy as np
import pytest

from spm.graphs.stats import (
    compute_graph_statistics,
    compute_largest_eigenvalue,
    compute_spectral_threshold,
)


def test_complete_graph_spectral_exact():
    """
    Analytical validation on complete graphs K_n:
    Eigenvalues of A(K_n) are (n-1) with multiplicity 1, and -1 with multiplicity (n-1).
    Therefore, lambda_1 = n - 1 and tau_c = 1 / (n - 1).
    """
    for n in [5, 10, 25]:
        K_n = nx.complete_graph(n)
        lam1, tau_c = compute_spectral_threshold(K_n)
        
        expected_lam1 = float(n - 1)
        expected_tau_c = 1.0 / (n - 1)
        
        assert math.isclose(lam1, expected_lam1, rel_tol=1e-7, abs_tol=1e-7), (
            f"K_{n}: expected lambda_1={expected_lam1}, got {lam1}"
        )
        assert math.isclose(tau_c, expected_tau_c, rel_tol=1e-7, abs_tol=1e-7), (
            f"K_{n}: expected tau_c={expected_tau_c}, got {tau_c}"
        )


def test_star_graph_spectral_exact():
    """
    Analytical validation on star graphs S_n (1 center hub + (n-1) leaves):
    Non-zero eigenvalues of A(S_n) are +/- sqrt(n-1).
    Therefore, lambda_1 = sqrt(n-1) and tau_c = 1 / sqrt(n-1).
    """
    for n in [10, 17, 26]:  # n-1 = 9, 16, 25 (perfect squares for clean verification)
        S_n = nx.star_graph(n - 1)  # nx.star_graph(k) produces graph with k+1 nodes
        assert S_n.number_of_nodes() == n
        
        lam1, tau_c = compute_spectral_threshold(S_n)
        expected_lam1 = math.sqrt(n - 1)
        expected_tau_c = 1.0 / expected_lam1
        
        assert math.isclose(lam1, expected_lam1, rel_tol=1e-7, abs_tol=1e-7), (
            f"S_{n}: expected lambda_1={expected_lam1}, got {lam1}"
        )
        assert math.isclose(tau_c, expected_tau_c, rel_tol=1e-7, abs_tol=1e-7), (
            f"S_{n}: expected tau_c={expected_tau_c}, got {tau_c}"
        )


def test_sparse_vs_dense_eigensolver_agreement():
    """Verify that sparse and dense solvers agree on a larger graph (n=550)."""
    G = nx.erdos_renyi_graph(n=550, p=0.02, seed=123)
    val_dense = compute_largest_eigenvalue(G, sparse_threshold=1000)
    val_sparse = compute_largest_eigenvalue(G, sparse_threshold=100)
    assert math.isclose(val_dense, val_sparse, rel_tol=1e-4, abs_tol=1e-4)


def test_disconnected_and_edge_cases():
    """Verify handling of disconnected and minimal graphs."""
    # Two disconnected cliques: K_4 (lam1=3) and K_10 (lam1=9)
    G = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(10))
    lam1 = compute_largest_eigenvalue(G)
    assert math.isclose(lam1, 9.0, rel_tol=1e-6)

    # Graph with no edges (isolated nodes)
    G_empty_edges = nx.empty_graph(5)
    lam1_empty = compute_largest_eigenvalue(G_empty_edges)
    assert lam1_empty == 0.0

    # Empty graph
    with pytest.raises(ValueError):
        compute_largest_eigenvalue(nx.empty_graph(0))


def test_structural_statistics_consistency():
    """Verify structural statistics calculations on a known graph."""
    K_5 = nx.complete_graph(5)
    stats = compute_graph_statistics(K_5, include_centrality=True)
    assert stats["nodes"] == 5
    assert stats["edges"] == 10
    assert stats["mean_degree"] == 4.0
    assert stats["max_degree"] == 4
    assert stats["avg_clustering"] == 1.0
    assert stats["max_k_core"] == 4
    assert math.isclose(stats["lambda_1"], 4.0)
    assert math.isclose(stats["tau_c"], 0.25)
    assert stats["is_connected"] is True
