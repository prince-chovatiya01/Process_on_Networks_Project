"""
Unit tests for node removal interventions (degree, k-core, time-aware, random).
"""

import networkx as nx
import pandas as pd
import pytest

from spm.interventions.node_removal import (
    apply_node_removal_static,
    apply_node_removal_temporal,
    get_removal_nodes,
    run_removal_comparison,
)


def test_get_removal_nodes_degree():
    """Verify degree-based targeted selection picks the highest degree nodes."""
    # Star graph: hub is node 0 with degree 9, leaves 1-9 with degree 1
    G = nx.star_graph(9)
    # Remove fraction ~0.1 of 10 nodes => exactly 1 node
    removed = get_removal_nodes(G, events_df=None, strategy="degree", fraction=0.1)
    assert len(removed) == 1
    assert 0 in removed  # Hub must be targeted first


def test_get_removal_nodes_k_core():
    """Verify k-core targeted selection prioritizes high coreness nodes."""
    # Complete graph K_4 (core 3) attached to a chain (core 1)
    K4 = nx.complete_graph(4)
    chain = nx.path_graph([3, 4, 5, 6])
    G = nx.compose(K4, chain)

    removed = get_removal_nodes(G, events_df=None, strategy="k_core", fraction=0.3)
    assert len(removed) == 2
    # Removed nodes must belong to the dense K4 core (nodes 0, 1, 2, 3)
    for node in removed:
        assert node in [0, 1, 2, 3]


def test_get_removal_nodes_time_aware():
    """Verify time-aware targeted selection picks nodes with highest temporal activity."""
    G = nx.path_graph(4)  # nodes 0, 1, 2, 3
    # Node 2 participates in 5 events, node 0 in 1 event
    events = pd.DataFrame({
        "src_id": [2, 2, 2, 0, 2],
        "dst_id": [1, 3, 1, 1, 3],
        "timestamp": [10, 20, 30, 40, 50]
    })
    removed = get_removal_nodes(G, events_df=events, strategy="time_aware", fraction=0.25)
    assert len(removed) == 1
    assert 2 in removed  # Node 2 has highest event frequency


def test_apply_node_removal_static_and_temporal():
    """Verify graph and event filtering upon node removal."""
    G = nx.complete_graph(5)
    events = pd.DataFrame({
        "src_id": [0, 1, 2, 3],
        "dst_id": [1, 2, 3, 4],
        "timestamp": [100, 200, 300, 400]
    })

    to_remove = {1, 3}
    G_sub = apply_node_removal_static(G, to_remove)
    assert G_sub.number_of_nodes() == 3
    assert not G_sub.has_node(1)
    assert not G_sub.has_node(3)

    events_sub = apply_node_removal_temporal(events, to_remove)
    # Remaining edges cannot contain node 1 or 3: only none of the original 4 rows remain if all touch 1 or 3!
    # Row 0 has 0->1 (touch 1)
    # Row 1 has 1->2 (touch 1)
    # Row 2 has 2->3 (touch 3)
    # Row 3 has 3->4 (touch 3)
    assert len(events_sub) == 0


def test_run_removal_comparison_structure():
    """Verify end-to-end intervention experiment execution and metrics table."""
    G = nx.erdos_renyi_graph(n=20, p=0.3, seed=42)
    events = pd.DataFrame({
        "src_id": [0, 1, 2, 3],
        "dst_id": [1, 2, 3, 0],
        "timestamp": [10, 20, 30, 40]
    })
    df = run_removal_comparison(
        dataset_name="test_fixture",
        G_static=G,
        events_df=events,
        strategies=["degree", "random"],
        fractions=[0.0, 0.2],
        tau=0.2,
        repetitions=2,
        t_max=10.0
    )
    assert len(df) == 4  # 2 strategies * 2 fractions
    assert "static_mean_impact" in df.columns
    assert "temporal_mean_impact" in df.columns
    assert "lambda_1_remaining" in df.columns
