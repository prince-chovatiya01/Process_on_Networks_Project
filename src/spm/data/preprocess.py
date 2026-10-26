"""
Data parsing, cleaning, node remapping, and graph aggregation pipeline.
"""

import gzip
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import networkx as nx
import pandas as pd

from spm.data.validate import (
    ValidationReport,
    generate_markdown_report,
    validate_event_dataframe,
)
from spm.utils.logging_config import setup_logger

logger = setup_logger("spm.data.preprocess")


def parse_snap_dataset(file_path: Path) -> pd.DataFrame:
    """
    Parse SNAP temporal datasets (e.g., email-Eu-core-temporal, CollegeMsg).
    Expected space/tab separated columns: src dst timestamp
    """
    logger.info("Parsing SNAP dataset from %s", file_path)
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["src", "dst", "timestamp"],
        dtype={"src": str, "dst": str, "timestamp": int},
        compression="infer"
    )
    return df


def parse_sociopatterns_dataset(file_path: Path) -> pd.DataFrame:
    """
    Parse SocioPatterns face-to-face datasets (e.g. InVS15).
    Standard formats:
    - 5 columns: t src dst dept_src dept_dst (20s contact intervals)
    - 4 columns: t_start t_end src dst
    """
    logger.info("Parsing SocioPatterns dataset from %s", file_path)
    # Peek first line to determine column structure
    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rt", encoding="utf-8") as f:
        first_line = f.readline().strip()
    
    parts = first_line.split()
    if len(parts) >= 5:
        # Format: t src dst c1 c2
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            usecols=[0, 1, 2],
            names=["timestamp", "src", "dst"],
            dtype={"timestamp": int, "src": str, "dst": str},
            compression="infer"
        )
        # In SocioPatterns 20s protocol, each record represents an active 20s contact window
        df["duration"] = 20
        df["end_time"] = df["timestamp"] + 20
    elif len(parts) == 4:
        # Format: t_start t_end src dst
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=["timestamp", "end_time", "src", "dst"],
            dtype={"timestamp": int, "end_time": int, "src": str, "dst": str},
            compression="infer"
        )
    else:
        # Standard 3-col SocioPatterns format: timestamp (t), src (i), dst (j)
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=["timestamp", "src", "dst"],
            dtype={"timestamp": int, "src": str, "dst": str},
            compression="infer"
        )
        # SocioPatterns 20s discrete face-to-face active interval
        df["duration"] = 20
        df["end_time"] = df["timestamp"] + 20
    return df


def build_static_graphs(
    events_df: pd.DataFrame,
    src_col: str = "src_id",
    dst_col: str = "dst_id",
    undirected: bool = True
) -> Tuple[nx.Graph, nx.Graph]:
    """
    Construct static binary and contact-count-weighted graphs from event list.
    """
    # Count contact interactions between node pairs
    if undirected:
        # Normalize pairs so u <= v for symmetric aggregation
        pairs = events_df[[src_col, dst_col]].apply(
            lambda r: (min(r[src_col], r[dst_col]), max(r[src_col], r[dst_col])),
            axis=1
        )
        edge_counts = pairs.value_counts()
        
        G_binary = nx.Graph()
        G_weighted = nx.Graph()
        
        # Ensure all unique nodes exist in graphs
        all_nodes = set(events_df[src_col]).union(set(events_df[dst_col]))
        G_binary.add_nodes_from(all_nodes)
        G_weighted.add_nodes_from(all_nodes)
        
        for (u, v), count in edge_counts.items():
            G_binary.add_edge(u, v)
            G_weighted.add_edge(u, v, weight=int(count))
            
        return G_binary, G_weighted
    else:
        G_binary = nx.DiGraph()
        G_weighted = nx.DiGraph()
        all_nodes = set(events_df[src_col]).union(set(events_df[dst_col]))
        G_binary.add_nodes_from(all_nodes)
        G_weighted.add_nodes_from(all_nodes)
        edge_counts = events_df.groupby([src_col, dst_col]).size()
        for (u, v), count in edge_counts.items():
            G_binary.add_edge(u, v)
            G_weighted.add_edge(u, v, weight=int(count))
        return G_binary, G_weighted


def process_dataset(
    dataset_key: str,
    raw_file_path: Path,
    output_dir: Path,
    dataset_cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, nx.Graph, nx.Graph, ValidationReport]:
    """
    Full preprocessing pipeline: Parse -> Validate -> Remap IDs -> Build Graphs -> Save Artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Processing dataset %s from raw path %s", dataset_key, raw_file_path)

    # 1. Parse raw file
    if "sociopatterns" in dataset_key.lower():
        raw_df = parse_sociopatterns_dataset(raw_file_path)
    else:
        raw_df = parse_snap_dataset(raw_file_path)

    # 2. Validate
    duration_semantic = dataset_cfg.get("duration_semantic", "instantaneous")
    end_col = "end_time" if "end_time" in raw_df.columns else None
    clean_df, val_report = validate_event_dataframe(
        raw_df,
        dataset_name=dataset_cfg.get("name", dataset_key),
        duration_semantic=duration_semantic,
        time_col="timestamp",
        src_col="src",
        dst_col="dst",
        end_time_col=end_col
    )

    # 3. Create dense 0-indexed node mapping
    all_unique_nodes = sorted(list(set(clean_df["src"].unique()).union(set(clean_df["dst"].unique()))))
    node_to_id = {node: idx for idx, node in enumerate(all_unique_nodes)}
    id_to_node = {idx: node for idx, node in enumerate(all_unique_nodes)}

    clean_df["src_id"] = clean_df["src"].map(node_to_id)
    clean_df["dst_id"] = clean_df["dst"].map(node_to_id)

    # 4. Build static graphs
    undirected = dataset_cfg.get("undirected_for_static", True)
    G_binary, G_weighted = build_static_graphs(
        clean_df,
        src_col="src_id",
        dst_col="dst_id",
        undirected=undirected
    )

    # 5. Save processed artifacts
    clean_events_file = output_dir / "clean_events.csv.gz"
    clean_df.to_csv(clean_events_file, index=False, compression="gzip")

    mapping_file = output_dir / "node_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump({"node_to_id": node_to_id, "id_to_node": id_to_node}, f, indent=2)

    # Save graphs as edge lists
    nx.write_edgelist(G_binary, output_dir / "static_binary_edgelist.txt", data=False)
    nx.write_weighted_edgelist(G_weighted, output_dir / "static_weighted_edgelist.txt")

    # Save reports
    report_md = output_dir / "validation_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(val_report))

    report_json = output_dir / "validation_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(asdict(val_report), f, indent=2)

    logger.info("Completed processing %s: %d nodes, %d binary edges", dataset_key, G_binary.number_of_nodes(), G_binary.number_of_edges())
    return clean_df, G_binary, G_weighted, val_report
