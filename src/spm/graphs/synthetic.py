"""
Synthetic network generator for Erdos-Renyi, Barabasi-Albert, and Watts-Strogatz ensembles.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import networkx as nx

from spm.utils.logging_config import setup_logger

logger = setup_logger("spm.graphs.synthetic")


def generate_erdos_renyi(n: int, p: float, seed: Optional[int] = None) -> nx.Graph:
    """Generate an Erdos-Renyi random graph G(n, p)."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Probability p must be between 0 and 1, got {p}")
    if n <= 0:
        raise ValueError(f"Node count n must be positive, got {n}")
    return nx.erdos_renyi_graph(n=n, p=p, seed=seed)


def generate_barabasi_albert(n: int, m: int, seed: Optional[int] = None) -> nx.Graph:
    """Generate a Barabasi-Albert preferential attachment graph."""
    if m < 1 or m >= n:
        raise ValueError(f"Attachment parameter m must satisfy 1 <= m < n, got m={m}, n={n}")
    if n <= 0:
        raise ValueError(f"Node count n must be positive, got {n}")
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)


def generate_watts_strogatz(n: int, k: int, p: float, seed: Optional[int] = None) -> nx.Graph:
    """
    Generate a Watts-Strogatz small-world graph.
    Validates parameter constraints: k < n, k must be even, 0 <= p <= 1.
    """
    if n <= 0:
        raise ValueError(f"Node count n must be positive, got {n}")
    if k >= n:
        raise ValueError(f"Watts-Strogatz constraint violated: k ({k}) must be strictly less than n ({n})")
    if k % 2 != 0:
        raise ValueError(f"Watts-Strogatz constraint violated: k ({k}) must be even")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Rewiring probability p must be between 0 and 1, got {p}")
    return nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)


def generate_all_synthetic_graphs(
    config: Dict[str, Any],
    output_dir: Path
) -> List[Dict[str, Any]]:
    """
    Generate all nine TA-specified synthetic graph configurations deterministically:
    - ER: p = 0.1, 0.2, 0.3
    - BA: m = 4, 6, 8
    - WS: k = 6, 8, 10 (p_rewire=0.1)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    master_seed = config.get("seed_base", 42)
    default_n = config.get("default_n", 1000)

    generated_manifest = []

    models_cfg = config.get("models", {})

    # 1. ER configurations
    er_cfg = models_cfg.get("erdos_renyi", {})
    er_n = er_cfg.get("n", default_n)
    for idx, param_entry in enumerate(er_cfg.get("parameters", [])):
        p = param_entry["p"]
        graph_id = param_entry.get("id", f"er_n{er_n}_p{p}")
        seed = master_seed + idx
        G = generate_erdos_renyi(n=er_n, p=p, seed=seed)
        meta = _save_synthetic_graph(output_dir, graph_id, G, {
            "type": "ER",
            "n": er_n,
            "p": p,
            "seed": seed,
            "is_provisional": p == 0.3
        })
        generated_manifest.append(meta)

    # 2. BA configurations
    ba_cfg = models_cfg.get("barabasi_albert", {})
    ba_n = ba_cfg.get("n", default_n)
    for idx, param_entry in enumerate(ba_cfg.get("parameters", [])):
        m = param_entry["m"]
        graph_id = param_entry.get("id", f"ba_n{ba_n}_m{m}")
        seed = master_seed + 10 + idx
        G = generate_barabasi_albert(n=ba_n, m=m, seed=seed)
        meta = _save_synthetic_graph(output_dir, graph_id, G, {
            "type": "BA",
            "n": ba_n,
            "m": m,
            "seed": seed
        })
        generated_manifest.append(meta)

    # 3. WS configurations
    ws_cfg = models_cfg.get("watts_strogatz", {})
    ws_n = ws_cfg.get("n", default_n)
    p_rewire = ws_cfg.get("p_rewire", 0.1)
    for idx, param_entry in enumerate(ws_cfg.get("parameters", [])):
        k = param_entry["k"]
        graph_id = param_entry.get("id", f"ws_n{ws_n}_k{k}")
        seed = master_seed + 20 + idx
        G = generate_watts_strogatz(n=ws_n, k=k, p=p_rewire, seed=seed)
        meta = _save_synthetic_graph(output_dir, graph_id, G, {
            "type": "WS",
            "n": ws_n,
            "k": k,
            "p_rewire": p_rewire,
            "seed": seed
        })
        generated_manifest.append(meta)

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(generated_manifest, f, indent=2)

    logger.info("Successfully generated %d synthetic graph configurations in %s", len(generated_manifest), output_dir)
    return generated_manifest


def _save_synthetic_graph(
    base_dir: Path,
    graph_id: str,
    G: nx.Graph,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Helper to save graph edge list and metadata."""
    sub_dir = base_dir / graph_id
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    edge_file = sub_dir / "edgelist.txt"
    nx.write_edgelist(G, edge_file, data=False)

    metadata = {
        "graph_id": graph_id,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "parameters": params,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_path": str(edge_file)
    }

    meta_file = sub_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
