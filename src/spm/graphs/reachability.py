"""
Temporal reachability, Aggregation Inflation Ratio (AIR), and null models.

Implements:
1. Temporal Reachability Gap (TRG):
   TRG = 1 - (|P_temporal| / |P_static|)
2. Aggregation Inflation Ratio (AIR):
   AIR = Static SIR attack rate / Temporal SIR attack rate
3. Timestamp-shuffled temporal null model:
   Preserves static network structure, edge weights, and event volume,
   while destroying temporal correlations, burstiness, and daily cycles.
"""

from typing import Dict, List, Optional, Tuple, Union
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from spm.utils.logging_config import setup_logger

logger = setup_logger("spm.graphs.reachability")


def compute_static_reachability(G: nx.Graph) -> int:
    """
    Compute total number of ordered reachable node pairs (u, v) with u != v in G.
    For an undirected graph, sum over connected components C: |C| * (|C| - 1).
    """
    if G.is_directed():
        # Transitive closure / reachability count for directed graph
        reach_count = 0
        nodes = list(G.nodes())
        for u in nodes:
            visited = nx.descendants(G, u)
            reach_count += len(visited)
        return reach_count
    else:
        total = 0
        for comp in nx.connected_components(G):
            size = len(comp)
            total += size * (size - 1)
        return total


def compute_single_source_temporal_reachability(
    source: int,
    src_arr: np.ndarray,
    dst_arr: np.ndarray,
    time_arr: np.ndarray,
    n_nodes: int,
    directed: bool = False
) -> int:
    """
    Compute number of nodes reachable from source via time-respecting paths.
    Single linear pass over chronological event arrays: O(M).
    """
    T = np.full(n_nodes, np.inf, dtype=np.float64)
    T[source] = -np.inf

    n_events = len(src_arr)
    for k in range(n_events):
        u = src_arr[k]
        v = dst_arr[k]
        tk = time_arr[k]

        if not directed:
            # Undirected transmission
            if tk >= T[u] and tk < T[v]:
                T[v] = tk
            if tk >= T[v] and tk < T[u]:
                T[u] = tk
        else:
            # Directed transmission: u -> v only
            if tk >= T[u] and tk < T[v]:
                T[v] = tk

    # Count reachable nodes excluding the source itself
    reachable_mask = (T < np.inf)
    reachable_mask[source] = False
    return int(np.sum(reachable_mask))


def compute_temporal_reachability_gap(
    G_static: nx.Graph,
    events_df: pd.DataFrame,
    n_nodes: int,
    sample_sources: Optional[int] = None,
    seed: int = 42,
    directed: bool = False,
    n_jobs: int = -1,
    src_col: str = "src_id",
    dst_col: str = "dst_id",
    time_col: str = "timestamp"
) -> Dict[str, float]:
    """
    Compute Temporal Reachability Gap (TRG):
    TRG = 1 - (Temporally reachable pairs / Statically reachable pairs)
    """
    static_pairs = compute_static_reachability(G_static)

    src_arr = events_df[src_col].to_numpy(dtype=np.int32)
    dst_arr = events_df[dst_col].to_numpy(dtype=np.int32)
    time_arr = events_df[time_col].to_numpy(dtype=np.float64)

    if sample_sources is not None and sample_sources < n_nodes:
        rng = np.random.default_rng(seed)
        sources = rng.choice(n_nodes, size=sample_sources, replace=False)
        scale_factor = n_nodes / float(sample_sources)
    else:
        sources = np.arange(n_nodes)
        scale_factor = 1.0

    counts = Parallel(n_jobs=n_jobs)(
        delayed(compute_single_source_temporal_reachability)(
            int(s), src_arr, dst_arr, time_arr, n_nodes, directed=directed
        )
        for s in sources
    )

    est_temporal_pairs = float(sum(counts)) * scale_factor
    static_ref = max(1.0, float(static_pairs))
    reachability_ratio = min(1.0, est_temporal_pairs / static_ref)
    trg = max(0.0, 1.0 - reachability_ratio)

    return {
        "n_nodes": float(n_nodes),
        "n_events": float(len(events_df)),
        "static_reachable_pairs": float(static_pairs),
        "temporal_reachable_pairs": float(est_temporal_pairs),
        "temporal_reachability_ratio": float(reachability_ratio),
        "temporal_reachability_gap": float(trg),
        "directed": float(1.0 if directed else 0.0)
    }


def compute_aggregation_inflation_ratio(
    static_attack_rate: float,
    temporal_attack_rate: float,
    eps: float = 1e-6
) -> float:
    """
    Aggregation Inflation Ratio (AIR):
    AIR = Static SIR attack rate / Temporal SIR attack rate
    When AIR > 1, static aggregation artificially exaggerates outbreak reach.
    """
    denom = max(eps, temporal_attack_rate)
    return float(static_attack_rate / denom)


def generate_timestamp_shuffled_null_model(
    events_df: pd.DataFrame,
    seed: Optional[int] = None,
    time_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Construct a timestamp-shuffled temporal null model.
    Permutes event timestamps across all edges while preserving:
    - Exact static graph topology G
    - Number of events per edge (edge weights)
    - Total event volume
    Destroys:
    - Temporal burstiness, causality, and circadian intervals.
    """
    rng = np.random.default_rng(seed)
    shuffled_df = events_df.copy()
    timestamps = shuffled_df[time_col].to_numpy()
    rng.shuffle(timestamps)
    shuffled_df[time_col] = timestamps
    # Re-sort chronologically as required for temporal simulation
    shuffled_df = shuffled_df.sort_values(by=time_col).reset_index(drop=True)
    return shuffled_df
