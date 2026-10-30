"""
Temporal event-driven stochastic simulation for SIS and SIR on chronological contact sequences.
Optimized with exact Binomial inter-event Poisson recovery sampling and FastIndexedSet tracking.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from spm.models.base import SimulationRunResult, State
from spm.models.static_sim import FastIndexedSet


def simulate_temporal_events(
    events_df: pd.DataFrame,
    n_nodes: int,
    beta: float,
    gamma: float,
    model: str = "SIS",
    initial_infected_fraction: float = 0.05,
    min_initial_infected: int = 1,
    time_scale_total: float = 50.0,
    seed: Optional[int] = None,
    record_trajectory: bool = True,
    sample_dt: float = 0.5,
    src_col: str = "src_id",
    dst_col: str = "dst_id",
    time_col: str = "timestamp",
    directed: bool = False
) -> SimulationRunResult:
    """
    Event-driven simulation of SIS / SIR on a chronological temporal contact sequence.
    - Simultaneous timestamp batching: events occurring at the exact same timestamp
      are evaluated synchronously. Newly infected individuals in a batch cannot
      instantaneously transmit to others within the same timestamp, preventing
      artificial multi-hop spreading cascades and ensuring row-permutation invariance.
    - Inter-event Poisson recovery: n_rec ~ Binomial(n_I, 1 - exp(-gamma * dt_norm))
    - Contact transmission probability:
      * Instantaneous: p_inf = 1 - exp(-beta)
      * Interval (with duration): p_inf = 1 - exp(-beta * (duration / 20.0))
    - directed: if True, transmission only flows from src_id to dst_id.
    """
    rng = np.random.default_rng(seed)
    n = n_nodes
    if n == 0 or events_df.empty:
        return SimulationRunResult(
            model=model.upper(),
            network_type="temporal",
            beta=beta,
            gamma=gamma,
            tau=beta / gamma if gamma > 0 else float("inf"),
            seed=seed or 0,
            n_nodes=n,
            initial_infected=0
        )

    # Initial state
    state = np.full(n, State.SUSCEPTIBLE, dtype=np.int8)
    n_init = max(min_initial_infected, int(round(n * initial_infected_fraction)))
    init_infected = rng.choice(n, size=min(n_init, n), replace=False)

    infected_set = FastIndexedSet()
    for idx in init_infected:
        state[idx] = State.INFECTED
        infected_set.add(int(idx))

    # Time normalization mapping
    t_min = float(events_df[time_col].min())
    t_max = float(events_df[time_col].max())
    t_span = max(1.0, t_max - t_min)
    time_scaling = time_scale_total / t_span

    # Pre-extract numpy arrays
    src_arr = events_df[src_col].to_numpy(dtype=np.int32)
    dst_arr = events_df[dst_col].to_numpy(dtype=np.int32)
    time_arr = events_df[time_col].to_numpy(dtype=np.float64)
    has_duration = "duration" in events_df.columns
    dur_arr = events_df["duration"].to_numpy(dtype=np.float64) if has_duration else None

    # Trajectory tracking
    times: List[float] = [0.0]
    I_counts: List[int] = [len(infected_set)]
    R_counts: List[int] = [0]
    next_record_t = sample_dt

    current_t_norm = 0.0
    last_event_t = t_min
    peak_I = len(infected_set)
    peak_time = 0.0

    n_events = len(src_arr)
    idx = 0
    while idx < n_events:
        if len(infected_set) == 0:
            break

        event_raw_t = time_arr[idx]
        delta_raw_t = max(0.0, event_raw_t - last_event_t)
        delta_t_norm = delta_raw_t * time_scaling
        current_t_norm += delta_t_norm
        last_event_t = event_raw_t

        # 1. Exact Binomial Poisson recovery between distinct timestamps
        if delta_t_norm > 0.0 and len(infected_set) > 0 and gamma > 0.0:
            p_rec = min(1.0, 1.0 - np.exp(-gamma * delta_t_norm))
            n_I = len(infected_set)
            if p_rec >= 1.0:
                n_rec = n_I
            else:
                n_rec = int(rng.binomial(n_I, p_rec))

            if n_rec > 0:
                if n_rec >= n_I:
                    rec_nodes = list(infected_set.items)
                else:
                    rec_nodes = [infected_set.items[i] for i in rng.choice(n_I, size=n_rec, replace=False)]

                for u in rec_nodes:
                    infected_set.remove(u)
                    if model.upper() == "SIS":
                        state[u] = State.SUSCEPTIBLE
                    else:
                        state[u] = State.RECOVERED

        # Record trajectory snapshot
        if record_trajectory and current_t_norm >= next_record_t:
            while next_record_t <= current_t_norm and next_record_t <= time_scale_total:
                times.append(float(next_record_t))
                I_counts.append(len(infected_set))
                R_counts.append(int(np.sum(state == State.RECOVERED)))
                next_record_t += sample_dt

        if len(infected_set) == 0:
            break

        # 2. Batch all events occurring at this exact raw timestamp
        batch_end = idx
        while batch_end < n_events and time_arr[batch_end] == event_raw_t:
            batch_end += 1

        # Evaluate transmission across all contacts in this timestamp batch.
        # Nodes becoming infected during this batch are staged in newly_infected
        # and cannot infect other nodes during this exact same timestamp.
        newly_infected = set()
        for k in range(idx, batch_end):
            u = src_arr[k]
            v = dst_arr[k]

            if u < 0 or u >= n or v < 0 or v >= n:
                continue

            state_u = state[u]
            state_v = state[v]

            can_infect = False
            target = -1

            if not directed:
                # Undirected: u can infect v if u was already infected; or v can infect u
                if state_u == State.INFECTED and state_v == State.SUSCEPTIBLE and v not in newly_infected:
                    target = v
                    can_infect = True
                elif state_v == State.INFECTED and state_u == State.SUSCEPTIBLE and u not in newly_infected:
                    target = u
                    can_infect = True
            else:
                # Directed: only src (u) to dst (v)
                if state_u == State.INFECTED and state_v == State.SUSCEPTIBLE and v not in newly_infected:
                    target = v
                    can_infect = True

            if can_infect and target != -1:
                if dur_arr is not None:
                    duration_factor = max(0.1, dur_arr[k] / 20.0)
                    p_inf = 1.0 - np.exp(-beta * duration_factor)
                else:
                    p_inf = 1.0 - np.exp(-beta)

                if rng.random() < p_inf:
                    newly_infected.add(target)

        # Simultaneously apply all infections occurring in this timestamp batch
        for target in newly_infected:
            state[target] = State.INFECTED
            infected_set.add(target)

        if len(infected_set) > peak_I:
            peak_I = len(infected_set)
            peak_time = current_t_norm

        idx = batch_end
        if times[-1] < time_scale_total:
            times.append(float(time_scale_total))
            I_counts.append(len(infected_set))
            R_counts.append(int(np.sum(state == State.RECOVERED)))

    final_I = len(infected_set)
    final_R = int(np.sum(state == State.RECOVERED))
    attack_rate = float(final_R + final_I) / n if n > 0 else 0.0

    return SimulationRunResult(
        model=model.upper(),
        network_type="temporal",
        beta=beta,
        gamma=gamma,
        tau=beta / gamma if gamma > 0 else float("inf"),
        seed=seed or 0,
        n_nodes=n,
        initial_infected=len(init_infected),
        times=times,
        infected_counts=I_counts,
        recovered_counts=R_counts,
        final_infected=final_I,
        final_recovered=final_R,
        attack_rate=attack_rate,
        peak_prevalence=float(peak_I) / n if n > 0 else 0.0,
        peak_time=peak_time,
        extinction=(final_I == 0),
        duration=current_t_norm
    )
