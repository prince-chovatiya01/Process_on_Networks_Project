"""
Node removal interventions: Random, Degree-based, K-core-based, and Time-aware strategies.

Measures structural degradation (lambda_1 reduction) and epidemiological impact
(prevalence/attack rate reduction) under matched experimental conditions for both
static and temporal representations.

Critical invariant: After removing nodes, the temporal event table is relabelled
to a contiguous 0..N_remaining-1 ID space before calling simulate_temporal_events().
This keeps the initial-infection sampling space and the prevalence denominator
consistent with the residual network.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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


def relabel_residual_temporal_system(
    G_sub: nx.Graph, events_sub: pd.DataFrame
) -> Tuple[pd.DataFrame, int]:
    """
    Relabel residual node IDs to contiguous integers 0..N_remaining-1.

    simulate_temporal_events() samples initially infected nodes from range(n_nodes).
    After node removal, using the original node count can accidentally select
    removed node IDs as seeds and uses the wrong prevalence denominator N.

    This function maps every remaining graph node to 0..N_remaining-1 and applies
    the same mapping to the filtered event sequence.

    Parameters
    ----------
    G_sub : nx.Graph
        Residual static graph after node removal.
    events_sub : pd.DataFrame
        Filtered temporal event table (no events touching removed nodes).

    Returns
    -------
    relabelled_events : pd.DataFrame
        Event table with src_id and dst_id remapped to 0..N_remaining-1.
    n_remaining : int
        Number of nodes in residual graph.
    """
    remaining_nodes = sorted(G_sub.nodes())
    mapping = {old: new for new, old in enumerate(remaining_nodes)}
    n_remaining = len(remaining_nodes)

    if events_sub.empty:
        return events_sub.copy(), n_remaining

    relabelled = events_sub.copy()
    relabelled["src_id"] = relabelled["src_id"].map(mapping)
    relabelled["dst_id"] = relabelled["dst_id"].map(mapping)

    # Verify no NaN mapping (all event nodes must be in G_sub)
    if relabelled[["src_id", "dst_id"]].isna().any().any():
        raise ValueError(
            "Filtered temporal events contain a node ID absent from the residual graph G_sub. "
            "Ensure apply_node_removal_temporal() was called before relabeling."
        )

    relabelled["src_id"] = relabelled["src_id"].astype(int)
    relabelled["dst_id"] = relabelled["dst_id"].astype(int)
    return relabelled, n_remaining


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

    Temporal simulation uses relabeled residual node IDs (0..N_remaining-1) so that
    initial infection sampling and prevalence denominator are consistent with the residual network.
    """
def _eval_single_removal_rep(
    strategy: str,
    frac: float,
    rep: int,
    G_static: nx.Graph,
    events_df: pd.DataFrame,
    beta: float,
    gamma: float,
    model: str,
    seed_base: int,
    t_max: float,
) -> dict:
    rem_seed = (
        (seed_base + rep * 1000 + int(frac * 100))
        if strategy == "random"
        else seed_base
    )
    sim_seed = seed_base + int(frac * 10000) + rep

    removed = get_removal_nodes(
        G_static, events_df, strategy=strategy, fraction=frac, seed=rem_seed
    )
    n_removed = len(removed)

    G_sub = apply_node_removal_static(G_static, removed)
    lam1_sub, tau_c_sub = compute_spectral_threshold_reference(G_sub)

    events_sub = apply_node_removal_temporal(events_df, removed)
    events_temporal, n_remaining = relabel_residual_temporal_system(G_sub, events_sub)

    # Static Gillespie SSA
    if G_sub.number_of_nodes() > 0 and G_sub.number_of_edges() > 0:
        res_s = simulate_static_gillespie(
            G=G_sub,
            beta=beta,
            gamma=gamma,
            model=model,
            initial_infected_fraction=0.05,
            t_max=t_max,
            seed=sim_seed,
        )
        v_s = res_s.final_prevalence if model == "SIS" else res_s.attack_rate
    else:
        v_s = 0.0

    # Temporal event simulation on relabeled residual system
    if not events_temporal.empty and n_remaining > 0:
        res_t = simulate_temporal_events(
            events_df=events_temporal,
            n_nodes=n_remaining,
            beta=beta,
            gamma=gamma,
            model=model,
            initial_infected_fraction=0.05,
            time_scale_total=t_max,
            seed=sim_seed,
        )
        v_t = res_t.final_prevalence if model == "SIS" else res_t.attack_rate
    else:
        v_t = 0.0

    return {
        "strategy": strategy,
        "fraction": frac,
        "rep": rep,
        "nodes_removed": n_removed,
        "remaining_nodes": n_remaining,
        "lambda_1_remaining": lam1_sub,
        "tau_c_remaining": tau_c_sub,
        "v_s": v_s,
        "v_t": v_t,
    }


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
    t_max: float = 20.0,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Run systematic node removal comparison across strategies and removal fractions.
    Evaluates both static network SSA and temporal event simulations under matched conditions.

    Temporal simulation uses relabeled residual node IDs (0..N_remaining-1) so that
    initial infection sampling and prevalence denominator are consistent with the residual network.
    Independent random removal sets are sampled across repetitions for the random strategy.
    """
    beta = tau * gamma

    tasks = [
        (strat, frac, rep)
        for strat in strategies
        for frac in fractions
        for rep in range(repetitions)
    ]

    logger.info(
        "Evaluating %d node-removal configurations on %s with n_jobs=%d...",
        len(tasks),
        dataset_name,
        n_jobs,
    )

    raw_results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_single_removal_rep)(
            strat, frac, rep, G_static, events_df, beta, gamma, model, seed_base, t_max
        )
        for (strat, frac, rep) in tasks
    )

    df_raw = pd.DataFrame(raw_results)
    aggregated = []

    for (strat, frac), group in df_raw.groupby(["strategy", "fraction"]):
        static_vals = group["v_s"].values
        temporal_vals = group["v_t"].values

        aggregated.append(
            {
                "dataset": dataset_name,
                "strategy": strat,
                "fraction_removed": frac,
                "nodes_removed": int(round(group["nodes_removed"].mean())),
                "remaining_nodes": int(round(group["remaining_nodes"].mean())),
                "lambda_1_remaining": float(group["lambda_1_remaining"].mean()),
                "tau_c_remaining": float(group["tau_c_remaining"].mean()),
                "tau": tau,
                "model": model,
                "static_mean_impact": float(np.mean(static_vals)),
                "static_std_impact": float(np.std(static_vals, ddof=1))
                if len(static_vals) > 1
                else 0.0,
                "temporal_mean_impact": float(np.mean(temporal_vals)),
                "temporal_std_impact": float(np.std(temporal_vals, ddof=1))
                if len(temporal_vals) > 1
                else 0.0,
                "reps": len(group),
            }
        )

    # Sort to preserve strategy and fraction ordering
    df_out = pd.DataFrame(aggregated)
    strategy_order = {s: i for i, s in enumerate(strategies)}
    df_out["order"] = df_out["strategy"].map(strategy_order)
    df_out = (
        df_out.sort_values(by=["order", "fraction_removed"])
        .drop(columns=["order"])
        .reset_index(drop=True)
    )
    return df_out
