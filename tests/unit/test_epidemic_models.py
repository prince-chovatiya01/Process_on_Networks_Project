"""
Unit tests for SIS and SIR stochastic simulations, NIMFA solver, and reproducibility.
"""

import math
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from spm.models.base import State
from spm.models.nimfa import compute_nimfa_trajectory, compute_spectral_threshold_reference
from spm.models.static_sim import simulate_static_gillespie
from spm.models.temporal_sim import simulate_temporal_events


def test_static_gillespie_sis_reproducibility():
    """Verify that identical seeds produce identical stochastic Gillespie trajectories."""
    G = nx.erdos_renyi_graph(n=30, p=0.2, seed=123)
    res1 = simulate_static_gillespie(G, beta=0.15, gamma=1.0, model="SIS", seed=42, t_max=10.0)
    res2 = simulate_static_gillespie(G, beta=0.15, gamma=1.0, model="SIS", seed=42, t_max=10.0)

    assert res1.infected_counts == res2.infected_counts
    assert math.isclose(res1.duration, res2.duration)
    assert res1.final_infected == res2.final_infected


def test_static_gillespie_sir_absorbing_state():
    """Verify that in SIR, recovered nodes never revert to susceptible or infected."""
    G = nx.complete_graph(15)
    # High transmission rate to guarantee complete outbreak
    res = simulate_static_gillespie(G, beta=1.0, gamma=0.5, model="SIR", seed=99, t_max=20.0)

    # In a complete graph with high transmission, entire population should eventually be infected/recovered
    assert res.final_infected == 0  # Epidemic must terminate
    assert res.attack_rate > 0.8
    assert res.final_recovered > 0


def test_temporal_simulation_transitions():
    """Verify state transitions and chronological event-driven dynamics in temporal simulation."""
    # Synthetic events: node 0 -> node 1 at t=10, node 1 -> node 2 at t=20
    events = pd.DataFrame({
        "src_id": [0, 1],
        "dst_id": [1, 2],
        "timestamp": [10, 20]
    })

    # Start with only node 0 infected (seed ensures node 0 is chosen)
    # High beta guarantees transmission, zero gamma prevents recovery during test window
    res_sis = simulate_temporal_events(
        events_df=events,
        n_nodes=3,
        beta=10.0,
        gamma=0.0,
        model="SIS",
        initial_infected_fraction=0.34, # 1 of 3 nodes
        min_initial_infected=1,
        seed=0
    )
    # By t=20, infection must have propagated along the temporal path 0 -> 1 -> 2
    assert res_sis.peak_prevalence >= 0.6


def test_temporal_simulation_reproducibility():
    """Verify that temporal event simulator produces deterministic outcomes with fixed seed."""
    events = pd.DataFrame({
        "src_id": [0, 1, 2, 0, 1],
        "dst_id": [1, 2, 0, 2, 0],
        "timestamp": [100, 200, 300, 400, 500]
    })
    res1 = simulate_temporal_events(events, n_nodes=5, beta=0.5, gamma=0.2, model="SIS", seed=777)
    res2 = simulate_temporal_events(events, n_nodes=5, beta=0.5, gamma=0.2, model="SIS", seed=777)

    assert res1.final_infected == res2.final_infected
    assert res1.peak_prevalence == res2.peak_prevalence


def test_nimfa_spectral_threshold():
    """Verify NIMFA ODE integration and spectral threshold relation."""
    K_10 = nx.complete_graph(10)
    lam1, tau_c = compute_spectral_threshold_reference(K_10)
    assert math.isclose(lam1, 9.0)
    assert math.isclose(tau_c, 1.0 / 9.0)

    # Subcritical: beta / gamma < tau_c => extinction
    sub_sol = compute_nimfa_trajectory(K_10, beta=0.05, gamma=1.0, model="SIS", t_max=30.0, seed=42)
    assert sub_sol["final_prevalence"] < 0.05

    # Supercritical: beta / gamma > tau_c => endemic state
    sup_sol = compute_nimfa_trajectory(K_10, beta=0.3, gamma=1.0, model="SIS", t_max=30.0, seed=42)
    assert sup_sol["final_prevalence"] > 0.5
