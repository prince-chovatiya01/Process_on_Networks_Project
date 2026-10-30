"""
Exact stochastic Gillespie continuous-time Markov simulation for SIS and SIR on static graphs.
Optimized with O(1) active transmission stub and infected node indexing.
"""

from typing import Dict, List, Optional
import networkx as nx
import numpy as np

from spm.models.base import SimulationRunResult, State


class FastIndexedSet:
    """O(1) insertion, O(1) removal, O(1) random sampling set."""
    __slots__ = ("items", "pos")

    def __init__(self):
        self.items: List[int] = []
        self.pos: Dict[int, int] = {}

    def add(self, item: int) -> None:
        if item not in self.pos:
            self.pos[item] = len(self.items)
            self.items.append(item)

    def remove(self, item: int) -> None:
        idx = self.pos.get(item)
        if idx is not None:
            last_item = self.items[-1]
            self.items[idx] = last_item
            self.pos[last_item] = idx
            self.items.pop()
            del self.pos[item]

    def random_choice(self, rng: np.random.Generator) -> int:
        idx = rng.integers(0, len(self.items))
        return self.items[idx]

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, item: int) -> bool:
        return item in self.pos


def simulate_static_gillespie(
    G: nx.Graph,
    beta: float,
    gamma: float,
    model: str = "SIS",
    initial_infected_fraction: float = 0.05,
    min_initial_infected: int = 1,
    t_max: float = 50.0,
    max_steps: int = 300000,
    seed: Optional[int] = None,
    record_trajectory: bool = True,
    sample_dt: float = 0.5
) -> SimulationRunResult:
    """
    Exact Gillespie stochastic simulation of SIS or SIR on static graph G.
    - Infection rate per S-I edge: beta
    - Recovery rate per infected node: gamma
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    if n == 0:
        return SimulationRunResult(
            model=model,
            network_type="static",
            beta=beta,
            gamma=gamma,
            tau=beta / gamma if gamma > 0 else float("inf"),
            seed=seed or 0,
            n_nodes=0,
            initial_infected=0
        )

    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    is_directed = G.is_directed()
    if is_directed:
        out_adj = [np.array([node_to_idx[v] for v in G.successors(u)], dtype=np.int32) for u in nodes]
        in_adj = [np.array([node_to_idx[u] for u in G.predecessors(v)], dtype=np.int32) for v in nodes]
    else:
        out_adj = [np.array([node_to_idx[v] for v in G.neighbors(u)], dtype=np.int32) for u in nodes]
        in_adj = out_adj

    # Initial state
    state = np.full(n, State.SUSCEPTIBLE, dtype=np.int8)
    n_init = max(min_initial_infected, int(round(n * initial_infected_fraction)))
    init_infected = rng.choice(n, size=min(n_init, n), replace=False)

    infected_set = FastIndexedSet()
    for idx in init_infected:
        state[idx] = State.INFECTED
        infected_set.add(idx)

    # Active transmission stubs: directed pairs (u, v) where u is INFECTED, v is SUSCEPTIBLE
    # Stored as encoded integer: (u * n + v)
    si_stubs = FastIndexedSet()
    for u in infected_set.items:
        for v in out_adj[u]:
            if state[v] == State.SUSCEPTIBLE:
                si_stubs.add(int(u * n + v))

    t = 0.0
    times: List[float] = [0.0]
    I_counts: List[int] = [len(infected_set)]
    R_counts: List[int] = [0]

    next_record_t = sample_dt
    step = 0
    peak_I = len(infected_set)
    peak_time = 0.0

    while t < t_max and len(infected_set) > 0 and step < max_steps:
        step += 1
        n_I = len(infected_set)
        n_SI = len(si_stubs)

        w_inf = beta * n_SI
        w_rec = gamma * n_I
        w_tot = w_inf + w_rec

        if w_tot <= 0.0:
            break

        dt = rng.exponential(1.0 / w_tot)
        t += dt

        # Record trajectory snapshot
        if record_trajectory and t >= next_record_t:
            while next_record_t <= t and next_record_t <= t_max:
                times.append(float(next_record_t))
                I_counts.append(len(infected_set))
                R_counts.append(int(np.sum(state == State.RECOVERED)))
                next_record_t += sample_dt

        # Decide event: Infection vs Recovery
        if rng.random() < (w_inf / w_tot):
            # INFECTION EVENT: sample random active stub (u -> v) in O(1)
            stub = si_stubs.random_choice(rng)
            u = stub // n
            v = stub % n

            # Node v transitions S -> I
            state[v] = State.INFECTED
            infected_set.add(v)

            # Update active stubs around v:
            # 1. Any stub pointing to v (w -> v) is no longer active because v is no longer susceptible!
            for w in in_adj[v]:
                if state[w] == State.INFECTED:
                    si_stubs.remove(int(w * n + v))
            # 2. As newly infected, v can now infect its susceptible out-neighbors w (v -> w)
            for w in out_adj[v]:
                if state[w] == State.SUSCEPTIBLE:
                    si_stubs.add(int(v * n + w))

            if len(infected_set) > peak_I:
                peak_I = len(infected_set)
                peak_time = t

        else:
            # RECOVERY EVENT: sample random infected node v in O(1)
            v = infected_set.random_choice(rng)
            infected_set.remove(v)

            if model.upper() == "SIS":
                # Node v transitions I -> S
                state[v] = State.SUSCEPTIBLE
                # 1. v can no longer infect others: remove all (v -> w) stubs
                for w in out_adj[v]:
                    if state[w] == State.SUSCEPTIBLE:
                        si_stubs.remove(int(v * n + w))
                # 2. v is now susceptible, so infected in-neighbor w can infect v: add (w -> v) stub
                for w in in_adj[v]:
                    if state[w] == State.INFECTED:
                        si_stubs.add(int(w * n + v))
            else:
                # SIR: Node v transitions I -> R
                state[v] = State.RECOVERED
                # v can no longer infect others: remove all (v -> w) stubs
                for w in out_adj[v]:
                    if state[w] == State.SUSCEPTIBLE:
                        si_stubs.remove(int(v * n + w))

    # Ensure trajectory snapshots extend to t_max even if extinction happened early
    if record_trajectory:
        while next_record_t <= t_max:
            times.append(float(next_record_t))
            I_counts.append(len(infected_set))
            R_counts.append(int(np.sum(state == State.RECOVERED)))
            next_record_t += sample_dt
        if times[-1] < t_max:
            times.append(float(t_max))
            I_counts.append(len(infected_set))
            R_counts.append(int(np.sum(state == State.RECOVERED)))

    final_I = len(infected_set)
    final_R = int(np.sum(state == State.RECOVERED))
    attack_rate = float(final_R + final_I) / n if n > 0 else 0.0

    return SimulationRunResult(
        model=model.upper(),
        network_type="static",
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
        duration=t
    )
