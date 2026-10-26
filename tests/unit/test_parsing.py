"""
Unit tests for event-table validation, cleaning, and graph reconstruction.
"""

from pathlib import Path
import networkx as nx
import pandas as pd
import pytest

from spm.data.preprocess import build_static_graphs
from spm.data.validate import validate_event_dataframe


def test_event_dataframe_cleaning():
    """Verify handling of missing values, self-loops, and duplicate timestamps."""
    raw_data = {
        "src": ["A", "B", "C", "D", "D", None, "E"],
        "dst": ["B", "C", "C", "A", "A", "E", "A"],  # "C"->"C" is a self-loop
        "timestamp": [10, 20, 30, 40, 40, 50, 60]     # Row 3 and Row 4 are duplicates
    }
    df = pd.DataFrame(raw_data)
    
    clean_df, report = validate_event_dataframe(
        df,
        dataset_name="synthetic_test",
        duration_semantic="instantaneous",
        time_col="timestamp",
        src_col="src",
        dst_col="dst"
    )

    assert report.total_raw_rows == 7
    assert report.missing_value_rows == 1  # Row with None dropped
    assert report.self_loop_rows == 1      # C->C dropped
    assert report.duplicate_rows == 1      # Second C->C or duplicate dropped
    assert report.clean_event_count == 4   # (A,B,10), (B,C,20), (D,A,40), (E,A,60)
    assert report.unique_nodes == 5        # A, B, C, D, E


def test_build_static_graphs_binary_and_weighted():
    """Verify that multiple events between same pair increment weights correctly."""
    events = pd.DataFrame({
        "src_id": [0, 0, 1, 2],
        "dst_id": [1, 1, 2, 0],
        "timestamp": [100, 200, 300, 400]
    })

    G_bin, G_weight = build_static_graphs(events, src_col="src_id", dst_col="dst_id", undirected=True)

    assert G_bin.number_of_nodes() == 3
    assert G_bin.number_of_edges() == 3  # (0,1), (1,2), (2,0)
    
    # Check binary edge presence
    assert G_bin.has_edge(0, 1)

    # Check weighted edge count: edge (0,1) occurred twice
    assert G_weight[0][1]["weight"] == 2
    assert G_weight[1][2]["weight"] == 1
    assert G_weight[0][2]["weight"] == 1


def test_sociopatterns_parser_3col(tmp_path: Path):
    """Verify that 3-column SocioPatterns format parses timestamp, src, dst with 20s interval duration."""
    import gzip
    from spm.data.preprocess import parse_sociopatterns_dataset

    test_file = tmp_path / "test_sp.dat.gz"
    with gzip.open(test_file, "wt", encoding="utf-8") as f:
        f.write("28840 574 1362\n")
        f.write("28840 164 779\n")
        f.write("28860 447 763\n")

    df = parse_sociopatterns_dataset(test_file)
    assert len(df) == 3
    assert list(df["timestamp"]) == [28840, 28840, 28860]
    assert list(df["src"]) == ["574", "164", "447"]
    assert list(df["dst"]) == ["1362", "779", "763"]
    assert list(df["duration"]) == [20, 20, 20]
    assert list(df["end_time"]) == [28860, 28860, 28880]

