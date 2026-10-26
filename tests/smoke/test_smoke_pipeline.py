"""
End-to-end smoke test validating the synthetic pipeline from generation to spectral metrics.
"""

from pathlib import Path
import networkx as nx
import pandas as pd
import pytest

from spm.data.preprocess import build_static_graphs
from spm.data.validate import validate_event_dataframe
from spm.graphs.stats import compute_graph_statistics
from spm.graphs.synthetic import generate_erdos_renyi


@pytest.mark.smoke
def test_synthetic_pipeline_smoke_n20(tmp_path: Path):
    """
    Smoke test:
    1. Generate N=20 ER(p=0.3) graph with seed=0.
    2. Compute stats; assert lambda_1 > 0; assert N==20.
    3. Save and reload graph; assert edge sets identical (reproducibility).
    4. Parse a tiny synthetic event table; assert validation runs without error.
    """
    n = 20
    p = 0.3
    seed = 0

    # 1. Generate
    G = generate_erdos_renyi(n=n, p=p, seed=seed)
    assert G.number_of_nodes() == 20
    assert G.number_of_edges() > 0

    # 2. Compute statistics
    stats = compute_graph_statistics(G, include_centrality=True)
    assert stats["nodes"] == 20
    assert stats["lambda_1"] > 0.0
    assert stats["tau_c"] is not None and stats["tau_c"] > 0.0

    # 3. Save and reload
    save_file = tmp_path / "smoke_graph_edgelist.txt"
    nx.write_edgelist(G, save_file, data=False)
    G_reloaded = nx.read_edgelist(save_file, nodetype=int)
    edges_orig = {tuple(sorted(e)) for e in G.edges()}
    edges_reloaded = {tuple(sorted(e)) for e in G_reloaded.edges()}
    assert edges_orig == edges_reloaded

    # 4. Process tiny synthetic event table
    events_raw = pd.DataFrame({
        "src": ["node_1", "node_2", "node_1", "node_3", "node_3"],
        "dst": ["node_2", "node_3", "node_2", "node_1", "node_3"],  # includes 1 duplicate and 1 self loop
        "timestamp": [1000, 1020, 1000, 1050, 1080]
    })
    clean_events, val_report = validate_event_dataframe(
        events_raw,
        dataset_name="smoke_test_fixture",
        duration_semantic="instantaneous"
    )
    assert val_report.is_valid is True
    assert val_report.self_loop_rows == 1
    assert val_report.duplicate_rows == 1

    # Static graphs from events
    clean_events["src_id"] = [0, 1, 2]
    clean_events["dst_id"] = [1, 2, 0]
    G_bin, G_wt = build_static_graphs(clean_events)
    assert G_bin.number_of_nodes() == 3
    assert G_bin.number_of_edges() == 3
