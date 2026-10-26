"""
Node removal interventions: Random, Degree-based, K-core-based, and Time-aware strategies.
Measures structural degradation (lambda_1 reduction) and epidemiological impact (prevalence/attack rate reduction)
under matched experimental conditions for both static and temporal representations.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd

from spm.models.nimfa import compute_spectral_threshold_reference
from spm.models.static_sim import simulate_static_gillespie
from spm.models.temporal_sim import simulate_temporal_events
from spm.utils.logging_config import setup_logger

logger = setup_logger("spm.interventions.node_removal")


def get_removal_nodes(
    G: nx.Graph,
    events_df: Optional[pd.DataFrame],
    strategy: str,
    fraction: float,
    seed: Optional[int] = None
) -> Set[int]:
    """
    Select nodes for removal according to specified strategy:
    - 'random': Uniform random selection
    - 'degree': Highest degree nodes first
    - 'k_core': Highest core number nodes first (degree tie-breaking)
    - 'time_aware': Highest temporal event activity (contact frequency) nodes first
    """
    nodes = list(G.nodes())
    n = len(nodes)
    k_remove = int(round(fraction * n))
    if k_remove <= 0:
        return set()
    if k_remove >= n:
        return set(nodes)

    rng = np.random.default_rng(seed)

    if strategy == "random":
        chosen = rng.choice(nodes, size=k_remove, replace=False)
        return set(int(x) for x in chosen)

    elif strategy == "degree":
        deg = dict(G.degree())
        # Sort descending by degree; tie-break deterministically
        sorted_nodes = sorted(nodes, key=lambda u: (deg[u], u), reverse=True)
        return set(sorted_nodes[:k_remove])

    elif strategy == "k_core":
        # Remove self-loops if any
        G_simple = G.copy()
        if nx.number_of_selfloops(G_simple) > 0:
            G_simple.remove_edges_from(nx.selfloop_edges(G_simple))
        core_nums = nx.core_number(G_simple)
        deg = dict(G.degree())
        # Sort descending by coreness, tie-break by degree
        sorted_nodes = sorted(nodes, key=lambda u: (core_nums[u], deg[u], u), reverse=True)
        return set(sorted_nodes[:k_remove])

    elif strategy == "time_aware":
        if events_df is None or events_df.empty:
            logger.warning("No temporal events provided for time_aware strategy; falling back to degree.")
            return get_removal_nodes(G, None, strategy="degree", fraction=fraction, seed=seed)

        # Count activity per node across events
        activity = events_df["src_id"].value_counts().add(
            events_df["dst_id"].value_counts(), fill_value=0
        ).to_dict()

        sorted_nodes = sorted(nodes, key=lambda u: (activity.get(u, 0), G.degree(u), u), reverse=True)
        return set(sorted_nodes[:k_remove])

    else:
        raise ValueError(f"Unknown removal strategy: {strategy}")


def apply_node_removal_static(G: nx.Graph, removed_nodes: Set[int]) -> nx.Graph:
    """Return subgraph with removed nodes deleted."""
    G_sub = G.copy()
    G_sub.remove_nodes_from(removed_nodes)
    return G_sub


def apply_node_removal_temporal(events_df: pd.DataFrame, removed_nodes: Set[int]) -> pd.DataFrame:
    """Filter out events involving any removed nodes."""
    if not removed_nodes or events_df.empty:
        return events_df.copy()
    mask = ~events_df["src_id"].isin(removed_nodes) & ~events_df["dst_id"].isin(removed_nodes)
    return events_df[mask].reset_index(drop=True)


def run_removal_comparison(
    dataset_name: str,
    G_static: nx.Graph,
    events_df: pd.DataFrame,
    strategies: List[str],
    fractions: List[float],
    tau: float,
    gamma: float = 1.0,
    model: str = "SIS",
    repetitions: int = 5,
    seed_base: int = 42,
    t_max: float = 20.0
) -> pd.DataFrame:
    """
    Run systematic node removal comparison across strategies and removal fractions.
    Evaluates both static network SSA and temporal event simulations under matched conditions.
    """
    results = []
    n_nodes = G_static.number_of_nodes()
    beta = tau * gamma

    for strategy in strategies:
        logger.info("Evaluating node-removal strategy: %s on %s...", strategy, dataset_name)
        for frac in fractions:
            # Deterministic selection for targeted, seeded for random
            removed = get_removal_nodes(G_static, events_df, strategy=strategy, fraction=frac, seed=seed_base)
            n_removed = len(removed)

            # Modified static network
            G_sub = apply_node_removal_static(G_static, removed)
            lam1_sub, tau_c_sub = compute_spectral_threshold_reference(G_sub)

            # Modified temporal sequence
            events_sub = apply_node_removal_temporal(events_df, removed)

            # Evaluate static and temporal simulations over repetitions
            static_vals = []
            temporal_vals = []

            for rep in range(repetitions):
                seed = seed_base + int(frac * 10000) + rep
                # Static
                if G_sub.number_of_nodes() > 0 and G_sub.number_of_edges() > 0:
                    res_s = simulate_static_gillespie(
                        G=G_sub, beta=beta, gamma=gamma, model=model,
                        initial_infected_fraction=0.05, t_max=t_max, seed=seed
                    )
                    v_s = res_s.final_prevalence if model == "SIS" else res_s.attack_rate
                else:
                    v_s = 0.0
                static_vals.append(v_s)

                # Temporal
                if not events_sub.empty and G_sub.number_of_nodes() > 0:
                    res_t = simulate_temporal_events(
                        events_df=events_sub, n_nodes=n_nodes, beta=beta, gamma=gamma, model=model,
                        initial_infected_fraction=0.05, time_scale_total=t_max, seed=seed
                    )
                    v_t = res_t.final_prevalence if model == "SIS" else res_t.attack_rate
                else:
                    v_t = 0.0
                temporal_vals.append(v_t)

            results.append({
                "dataset": dataset_name,
                "strategy": strategy,
                "fraction_removed": frac,
                "nodes_removed": n_removed,
                "remaining_nodes": n_nodes - n_removed,
                "lambda_1_remaining": lam1_sub,
                "tau_c_remaining": tau_c_sub,
                "tau": tau,
                "model": model,
                "static_mean_impact": float(np.mean(static_vals)),
                "static_std_impact": float(np.std(static_vals, ddof=1)) if len(static_vals) > 1 else 0.0,
                "temporal_mean_impact": float(np.mean(temporal_vals)),
                "temporal_std_impact": float(np.std(temporal_vals, ddof=1)) if len(temporal_vals) > 1 else 0.0,
                "reps": repetitions
            })

    return pd.DataFrame(results)
