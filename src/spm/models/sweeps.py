"""
Parameter sweep orchestrator for static vs. temporal epidemic spreading experiments.
"""

from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from spm.models.base import SimulationRunResult
from spm.models.nimfa import compute_spectral_threshold_reference
from spm.models.static_sim import simulate_static_gillespie
from spm.models.temporal_sim import simulate_temporal_events
from spm.utils.logging_config import setup_logger

logger = setup_logger("spm.models.sweeps")


def run_single_realization(
    sim_type: str,
    G_static: Optional[nx.Graph],
    events_df: Optional[pd.DataFrame],
    n_nodes: int,
    beta: float,
    gamma: float,
    model: str,
    seed: int,
    initial_infected_fraction: float = 0.05,
    t_max: float = 50.0
) -> SimulationRunResult:
    """Execute a single stochastic simulation run."""
    if sim_type == "static":
        assert G_static is not None
        return simulate_static_gillespie(
            G=G_static,
            beta=beta,
            gamma=gamma,
            model=model,
            initial_infected_fraction=initial_infected_fraction,
            t_max=t_max,
            seed=seed,
            record_trajectory=True
        )
    elif sim_type == "temporal":
        assert events_df is not None
        return simulate_temporal_events(
            events_df=events_df,
            n_nodes=n_nodes,
            beta=beta,
            gamma=gamma,
            model=model,
            initial_infected_fraction=initial_infected_fraction,
            time_scale_total=t_max,
            seed=seed,
            record_trajectory=True
        )
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}")


def run_parameter_sweep(
    dataset_name: str,
    G_static: nx.Graph,
    events_df: pd.DataFrame,
    tau_values: List[float],
    gamma: float = 1.0,
    model: str = "SIS",
    repetitions: int = 10,
    seed_base: int = 42,
    initial_infected_fraction: float = 0.05,
    t_max: float = 50.0,
    n_jobs: int = -1
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Run paired parameter sweep across static and temporal representations.
    Returns:
    - summary_df: aggregated mean, std, susceptibility per (sim_type, tau)
    - runs_df: detailed per-run metric entries
    - tau_c_nimfa: theoretical static NIMFA spectral threshold 1/lambda_1(A)
    """
    n_nodes = G_static.number_of_nodes()
    lam1, tau_c_nimfa = compute_spectral_threshold_reference(G_static)

    logger.info("[%s - %s] Static spectral reference: lambda_1=%.4f, tau_c=%.4f",
                dataset_name, model, lam1, tau_c_nimfa)

    tasks = []
    # Build task list
    for tau_idx, tau in enumerate(tau_values):
        beta = float(tau * gamma)
        for rep in range(repetitions):
            seed = seed_base + tau_idx * 1000 + rep
            # Static run
            tasks.append(("static", beta, gamma, tau, rep, seed))
            # Temporal run (using identical seed for fair paired comparison)
            tasks.append(("temporal", beta, gamma, tau, rep, seed))

    logger.info("Executing %d simulation runs with n_jobs=%d...", len(tasks), n_jobs)

    def _worker(sim_type, b, g, tau_val, rep_idx, s):
        res = run_single_realization(
            sim_type=sim_type,
            G_static=G_static if sim_type == "static" else None,
            events_df=events_df if sim_type == "temporal" else None,
            n_nodes=n_nodes,
            beta=b,
            gamma=g,
            model=model,
            seed=s,
            initial_infected_fraction=initial_infected_fraction,
            t_max=t_max
        )
        return {
            "dataset": dataset_name,
            "model": model,
            "sim_type": sim_type,
            "beta": b,
            "gamma": g,
            "tau": tau_val,
            "repetition": rep_idx,
            "seed": s,
            "final_infected": res.final_infected,
            "final_recovered": res.final_recovered,
            "final_prevalence": res.final_prevalence,
            "attack_rate": res.attack_rate,
            "peak_prevalence": res.peak_prevalence,
            "peak_time": res.peak_time,
            "extinction": res.extinction,
            "duration": res.duration
        }

    if n_jobs == 1:
        results_list = [_worker(st, b, g, t_val, r_idx, s) for (st, b, g, t_val, r_idx, s) in tasks]
    else:
        results_list = [_worker(st, b, g, t_val, r_idx, s) for (st, b, g, t_val, r_idx, s) in tasks]

    runs_df = pd.DataFrame(results_list)

    # Compute statistical aggregates
    summary_rows = []
    for (sim_type, tau_val), group in runs_df.groupby(["sim_type", "tau"]):
        final_prev = group["final_prevalence"].values
        attack_rates = group["attack_rate"].values

        # Primary order parameter depends on model
        primary_vals = final_prev if model == "SIS" else attack_rates
        n_reps = len(primary_vals)
        mean_val = float(np.mean(primary_vals))
        std_val = float(np.std(primary_vals, ddof=1)) if n_reps > 1 else 0.0
        stderr_val = (std_val / np.sqrt(n_reps)) if n_reps > 1 else 0.0
        ci95_val = 1.96 * stderr_val

        # Stable variance estimator: Var(X) = <X^2> - <X>^2
        mean_sq = float(np.mean(primary_vals ** 2))
        var_val = max(0.0, mean_sq - (mean_val ** 2))

        # Standard susceptibility estimator chi = N * Var(X) / <X>
        chi = (n_nodes * var_val / mean_val) if mean_val > 1e-4 else 0.0

        # Conditional statistics on non-extinct outbreaks (I > 0)
        surv_vals = primary_vals[primary_vals > 0.001]
        if len(surv_vals) > 0:
            cond_mean = float(np.mean(surv_vals))
            cond_var = max(0.0, float(np.mean(surv_vals ** 2)) - (cond_mean ** 2))
            cond_chi = (n_nodes * cond_var / cond_mean) if cond_mean > 1e-4 else 0.0
        else:
            cond_mean = 0.0
            cond_chi = 0.0

        extinction_prob = float(np.mean(group["extinction"].astype(float)))

        summary_rows.append({
            "dataset": dataset_name,
            "model": model,
            "sim_type": sim_type,
            "tau": tau_val,
            "tau_c_nimfa": tau_c_nimfa,
            "mean_order_param": mean_val,
            "std_order_param": std_val,
            "stderr_order_param": stderr_val,
            "ci95_order_param": ci95_val,
            "var_order_param": var_val,
            "mean_final_prevalence": float(np.mean(final_prev)),
            "std_final_prevalence": float(np.std(final_prev, ddof=1)) if n_reps > 1 else 0.0,
            "mean_attack_rate": float(np.mean(attack_rates)),
            "std_attack_rate": float(np.std(attack_rates, ddof=1)) if n_reps > 1 else 0.0,
            "conditional_mean": cond_mean,
            "conditional_chi": cond_chi,
            "susceptibility_chi": chi,
            "extinction_prob": extinction_prob,
            "reps": n_reps
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, runs_df, tau_c_nimfa
