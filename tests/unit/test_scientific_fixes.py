"""
Unit tests for critical scientific fixes:
1. Simultaneous timestamp batching (preventing intra-timestamp multi-hop cascades)
2. Row-permutation invariance within simultaneous event batches
3. NIMFA stability and analytical fixed-point verification
4. Empty network safety for SIS and SIR
5. Temporal reachability gap and Aggregation Inflation Ratio
6. Timestamp-shuffled temporal null model
7. Directed vs. undirected transmission semantics
8. Independent random node-removal sets
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from spm.graphs.reachability import (
    compute_aggregation_inflation_ratio,
    compute_static_reachability,
    compute_temporal_reachability_gap,
    generate_timestamp_shuffled_null_model,
)
from spm.interventions.node_removal import get_removal_nodes, relabel_residual_temporal_system
from spm.models.nimfa import (
    compute_nimfa_endemic_fixed_point,
    compute_nimfa_stability,
    compute_nimfa_trajectory,
    compute_spectral_threshold_reference,
)
from spm.models.static_sim import simulate_static_gillespie
from spm.models.temporal_sim import simulate_temporal_events


def test_simultaneous_batching_no_multihop():
    """
    Test that events sharing the exact same timestamp cannot create
    artificial instantaneous multi-hop spreading chains in zero elapsed time.
    """
    # Chain: 0 -> 1 -> 2 -> 3 all at timestamp t = 100
    events_data = [
        {"src_id": 0, "dst_id": 1, "timestamp": 100},
        {"src_id": 1, "dst_id": 2, "timestamp": 100},
        {"src_id": 2, "dst_id": 3, "timestamp": 100},
    ]
    df = pd.DataFrame(events_data)
    n_nodes = 4

    # Seed 0 initially infected (node 0)
    # beta very large so transmission probability p = 1 - exp(-beta) ≈ 1.0
    res = simulate_temporal_events(
        events_df=df,
        n_nodes=n_nodes,
        beta=100.0,
        gamma=0.0,  # No recovery
        model="SIS",
        initial_infected_fraction=0.25,
        min_initial_infected=1,
        time_scale_total=10.0,
        seed=1,  # Picks node 0 with this seed
        directed=True
    )

    # In a simultaneous batch:
    # Node 0 was infected at the start of t=100, so it infects Node 1.
    # But Node 1 was NOT infected at the start of t=100, so it CANNOT infect Node 2!
    # Therefore, final infected count MUST be exactly 2 (nodes 0 and 1), NOT 4!
    assert res.final_infected == 2, f"Expected 2 infected nodes, got {res.final_infected}"


def test_simultaneous_batching_row_permutation_invariance():
    """
    Test that the outcome of simultaneous events is invariant to row ordering in the DataFrame.
    """
    events_fwd = pd.DataFrame([
        {"src_id": 0, "dst_id": 1, "timestamp": 50},
        {"src_id": 1, "dst_id": 2, "timestamp": 50},
    ])
    events_rev = pd.DataFrame([
        {"src_id": 1, "dst_id": 2, "timestamp": 50},
        {"src_id": 0, "dst_id": 1, "timestamp": 50},
    ])

    res_fwd = simulate_temporal_events(
        events_df=events_fwd, n_nodes=3, beta=50.0, gamma=0.0, model="SIS",
        initial_infected_fraction=0.34, min_initial_infected=1, seed=0, directed=True
    )
    res_rev = simulate_temporal_events(
        events_df=events_rev, n_nodes=3, beta=50.0, gamma=0.0, model="SIS",
        initial_infected_fraction=0.34, min_initial_infected=1, seed=0, directed=True
    )

    assert res_fwd.final_infected == res_rev.final_infected


def test_nimfa_stability_and_fixed_point():
    """
    Test analytical NIMFA disease-free stability and endemic fixed-point convergence
    on Complete Graph K_5 where exact analytical solution is known:
    lambda_1(K_5) = 4, tau_c = 0.25.
    For tau = 0.5: p* = 1 - 1/(tau * (n-1)) = 1 - 1/(0.5 * 4) = 0.5.
    """
    G = nx.complete_graph(5)
    lam1, tau_c = compute_spectral_threshold_reference(G)
    assert np.isclose(lam1, 4.0)
    assert np.isclose(tau_c, 0.25)

    # Subcritical: tau = 0.1 < 0.25
    stab_sub = compute_nimfa_stability(G, beta=0.1, gamma=1.0)
    assert stab_sub["disease_free_stable"] is True
    assert stab_sub["jacobian_max_eigenvalue"] < 0.0

    fp_sub = compute_nimfa_endemic_fixed_point(G, beta=0.1, gamma=1.0)
    assert np.allclose(fp_sub["p_star"], 0.0)

    # Supercritical: tau = 0.5 > 0.25
    stab_super = compute_nimfa_stability(G, beta=0.5, gamma=1.0)
    assert stab_super["disease_free_stable"] is False
    assert stab_super["jacobian_max_eigenvalue"] > 0.0

    fp_super = compute_nimfa_endemic_fixed_point(G, beta=0.5, gamma=1.0)
    assert fp_super["converged"] is True
    assert np.allclose(fp_super["p_star"], 0.5, atol=1e-5)


def test_empty_networks_nimfa_and_simulators():
    """Test that empty graphs and event tables do not crash and return valid structures."""
    G_empty = nx.Graph()
    events_empty = pd.DataFrame(columns=["src_id", "dst_id", "timestamp"])

    res_static = simulate_static_gillespie(G_empty, beta=0.1, gamma=1.0, model="SIR")
    assert res_static.attack_rate == 0.0
    assert res_static.final_prevalence == 0.0

    res_temp = simulate_temporal_events(events_empty, n_nodes=0, beta=0.1, gamma=1.0, model="SIR")
    assert res_temp.attack_rate == 0.0
    assert res_temp.final_prevalence == 0.0

    traj_nimfa = compute_nimfa_trajectory(G_empty, beta=0.1, gamma=1.0, model="SIR")
    assert traj_nimfa["attack_rate"] == 0.0


def test_temporal_reachability_gap_and_air():
    """Test reachability gap and AIR calculation on a simple path network."""
    # Statically: 0 - 1 - 2
    G = nx.path_graph(3)
    static_pairs = compute_static_reachability(G)
    assert static_pairs == 6  # 3 * 2 ordered pairs

    # Temporal event 1: (1, 2) at t=10
    # Temporal event 2: (0, 1) at t=20
    # Causal path from 0 to 2 requires (0, 1) BEFORE (1, 2).
    # Since (1, 2) happened before (0, 1), 0 CANNOT reach 2!
    events = pd.DataFrame([
        {"src_id": 1, "dst_id": 2, "timestamp": 10.0},
        {"src_id": 0, "dst_id": 1, "timestamp": 20.0},
    ])

    trg_info = compute_temporal_reachability_gap(G, events, n_nodes=3, directed=False)
    # Statically 6 pairs reachable.
    # Temporally:
    # from 0: can reach 1 at t=20. Cannot reach 2. (1 pair)
    # from 1: can reach 2 at t=10, can reach 0 at t=20. (2 pairs)
    # from 2: can reach 1 at t=10, can reach 0 at t=20. (2 pairs)
    # Total temporally reachable = 1 + 2 + 2 = 5 pairs.
    # Gap = 1 - 5/6 = 1/6 ≈ 0.1667
    assert trg_info["temporal_reachable_pairs"] == 5.0
    assert np.isclose(trg_info["temporal_reachability_gap"], 1.0 / 6.0)

    # AIR calculation
    air = compute_aggregation_inflation_ratio(static_attack_rate=0.8, temporal_attack_rate=0.4)
    assert np.isclose(air, 2.0)


def test_timestamp_shuffled_null_model():
    """Test that timestamp shuffling preserves static edges and counts while changing sequence."""
    events = pd.DataFrame([
        {"src_id": 0, "dst_id": 1, "timestamp": 10.0},
        {"src_id": 1, "dst_id": 2, "timestamp": 20.0},
        {"src_id": 2, "dst_id": 3, "timestamp": 30.0},
        {"src_id": 3, "dst_id": 4, "timestamp": 40.0},
    ])
    shuffled = generate_timestamp_shuffled_null_model(events, seed=123)
    assert len(shuffled) == len(events)
    # Timestamps are sorted chronologically
    assert shuffled["timestamp"].is_monotonic_increasing
    # Same set of timestamps
    assert sorted(shuffled["timestamp"].tolist()) == sorted(events["timestamp"].tolist())


def test_directed_vs_undirected_transmission():
    """Test that directed option prevents backward transmission across directed stubs."""
    events = pd.DataFrame([
        {"src_id": 0, "dst_id": 1, "timestamp": 10.0}
    ])
    # Node 1 is infected, node 0 is susceptible.
    # Directed contact is 0 -> 1 (0 is sender, 1 is receiver).
    # Under directed=True, node 1 cannot infect node 0.
    res_dir = simulate_temporal_events(
        events_df=events, n_nodes=2, beta=10.0, gamma=0.0,
        initial_infected_fraction=0.5, min_initial_infected=1,
        seed=0,  # seed=0 picks node 1 as initially infected
        directed=True
    )
    assert res_dir.final_infected == 1  # 0 not infected

    # Under directed=False, node 1 can infect node 0.
    res_undir = simulate_temporal_events(
        events_df=events, n_nodes=2, beta=10.0, gamma=0.0,
        initial_infected_fraction=0.5, min_initial_infected=1,
        seed=0,
        directed=False
    )
    assert res_undir.final_infected == 2  # 0 infected


def test_node_removal_independent_random_sets():
    """Test that random strategy samples different nodes when different seeds are given."""
    G = nx.erdos_renyi_graph(100, 0.1, seed=42)
    nodes1 = get_removal_nodes(G, events_df=None, strategy="random", fraction=0.2, seed=10)
    nodes2 = get_removal_nodes(G, events_df=None, strategy="random", fraction=0.2, seed=20)
    assert len(nodes1) == 20
    assert len(nodes2) == 20
    assert nodes1 != nodes2  # Independent random subsets
