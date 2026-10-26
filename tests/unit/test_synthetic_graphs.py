"""
Unit tests for synthetic graph generation, parameter constraints, and reproducibility.
"""

from pathlib import Path
import networkx as nx
import pytest

from spm.graphs.synthetic import (
    generate_all_synthetic_graphs,
    generate_barabasi_albert,
    generate_erdos_renyi,
    generate_watts_strogatz,
)


def test_reproducibility_with_seed():
    """Verify that calling generators with identical seeds produces identical graphs."""
    g1 = generate_erdos_renyi(n=100, p=0.05, seed=42)
    g2 = generate_erdos_renyi(n=100, p=0.05, seed=42)
    assert set(g1.edges()) == set(g2.edges())

    g3 = generate_erdos_renyi(n=100, p=0.05, seed=43)
    assert set(g1.edges()) != set(g3.edges())


def test_watts_strogatz_constraints():
    """Verify that Watts-Strogatz rejects invalid parameters according to graph theory constraints."""
    # k must be even
    with pytest.raises(ValueError, match="must be even"):
        generate_watts_strogatz(n=100, k=5, p=0.1, seed=42)

    # k must be strictly less than n
    with pytest.raises(ValueError, match="strictly less than n"):
        generate_watts_strogatz(n=50, k=50, p=0.1, seed=42)

    # p must be between 0 and 1
    with pytest.raises(ValueError, match="between 0 and 1"):
        generate_watts_strogatz(n=100, k=6, p=1.5, seed=42)


def test_all_nine_configurations_generation(tmp_path: Path):
    """Verify that all nine TA configurations generate successfully with valid properties."""
    config = {
        "seed_base": 42,
        "default_n": 100,  # Use N=100 for fast test suite execution
        "models": {
            "erdos_renyi": {
                "n": 100,
                "parameters": [{"p": 0.1}, {"p": 0.2}, {"p": 0.3}]
            },
            "barabasi_albert": {
                "n": 100,
                "parameters": [{"m": 4}, {"m": 6}, {"m": 8}]
            },
            "watts_strogatz": {
                "n": 100,
                "p_rewire": 0.1,
                "parameters": [{"k": 6}, {"k": 8}, {"k": 10}]
            }
        }
    }

    manifest = generate_all_synthetic_graphs(config, output_dir=tmp_path)
    assert len(manifest) == 9

    types_found = [m["parameters"]["type"] for m in manifest]
    assert types_found.count("ER") == 3
    assert types_found.count("BA") == 3
    assert types_found.count("WS") == 3

    for item in manifest:
        assert item["nodes"] == 100
        assert item["edges"] > 0
        edge_file = Path(item["file_path"])
        assert edge_file.exists()
