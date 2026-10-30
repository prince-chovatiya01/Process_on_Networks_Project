"""
N-Intertwined Mean-Field Approximation (NIMFA) and analytical spectral threshold references.
"""

from typing import Any, Dict, Optional, Tuple
import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp

from spm.graphs.stats import compute_largest_eigenvalue


def compute_spectral_threshold_reference(G: nx.Graph) -> Tuple[float, float]:
    """
    Compute static NIMFA spectral threshold:
    tau_c = 1 / lambda_1(A)
    Returns (lambda_1, tau_c).
    """
    lam1 = compute_largest_eigenvalue(G)
    if lam1 <= 1e-12:
        return lam1, float("inf")
    return lam1, 1.0 / lam1


def compute_nimfa_stability(G: nx.Graph, beta: float, gamma: float) -> Dict[str, Any]:
    """
    Compute linearized Jacobian stability of the disease-free equilibrium p* = 0:
    J = beta * A - gamma * I_N
    Max eigenvalue: lambda_max(J) = beta * lambda_1(A) - gamma
    Disease-free state is asymptotically stable iff lambda_max(J) < 0.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {
            "lambda_1_A": 0.0,
            "tau_c_nimfa": float("inf"),
            "tau": beta / gamma if gamma > 0 else float("inf"),
            "jacobian_max_eigenvalue": -gamma,
            "disease_free_stable": True
        }

    lam1, tau_c = compute_spectral_threshold_reference(G)
    tau = beta / gamma if gamma > 0 else float("inf")
    jac_max_eig = beta * lam1 - gamma
    is_stable = bool(jac_max_eig < 0.0)

    return {
        "lambda_1_A": lam1,
        "tau_c_nimfa": tau_c,
        "tau": tau,
        "jacobian_max_eigenvalue": jac_max_eig,
        "disease_free_stable": is_stable
    }


def compute_nimfa_endemic_fixed_point(
    G: nx.Graph,
    beta: float,
    gamma: float,
    max_iter: int = 1000,
    tol: float = 1e-7
) -> Dict[str, Any]:
    """
    Compute the non-zero endemic fixed point p* for the SIS NIMFA system:
    p_i* = (tau * (A p*)_i) / (1 + tau * (A p*)_i)
    Converges to non-zero fixed point when tau > tau_c_nimfa.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {"p_star": np.array([]), "mean_prevalence": 0.0, "converged": True, "iterations": 0}

    tau = beta / gamma if gamma > 0 else float("inf")
    lam1, tau_c = compute_spectral_threshold_reference(G)

    if tau <= tau_c:
        # Below threshold: disease-free state is unique attractive fixed point
        return {
            "p_star": np.zeros(n, dtype=np.float64),
            "mean_prevalence": 0.0,
            "converged": True,
            "iterations": 0
        }

    A = nx.to_numpy_array(G, dtype=np.float64)
    p = np.full(n, 0.5, dtype=np.float64)

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        Ap = A @ p
        p_next = (tau * Ap) / (1.0 + tau * Ap)
        if np.max(np.abs(p_next - p)) < tol:
            p = p_next
            converged = True
            break
        p = p_next

    return {
        "p_star": p,
        "mean_prevalence": float(np.mean(p)),
        "converged": converged,
        "iterations": it
    }


def compute_nimfa_trajectory(
    G: nx.Graph,
    beta: float,
    gamma: float,
    model: str = "SIS",
    initial_infected_fraction: float = 0.05,
    t_max: float = 50.0,
    n_points: int = 100,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Integrate continuous-time NIMFA equations:
    SIS: dx_i/dt = -gamma * x_i + beta * (1 - x_i) * (A x)_i
    SIR: dx_i/dt = beta * (1 - x_i - z_i) * (A x)_i, dz_i/dt = gamma * x_i
    """
    n = G.number_of_nodes()
    if n == 0:
        return {
            "t": np.array([]),
            "prevalence": np.array([]),
            "recovered_fraction": np.array([]),
            "attack_rate": 0.0,
            "final_prevalence": 0.0,
            "peak_prevalence": 0.0
        }

    A = nx.to_numpy_array(G, dtype=np.float64)

    # Initial condition
    rng = np.random.default_rng(seed)
    n_init = max(1, int(round(n * initial_infected_fraction)))
    init_infected_indices = rng.choice(n, size=n_init, replace=False)
    
    x0 = np.zeros(n, dtype=np.float64)
    x0[init_infected_indices] = 1.0

    t_eval = np.linspace(0.0, t_max, n_points)

    if model.upper() == "SIS":
        def ode_sis(t, x):
            # x is length n vector in [0, 1]
            ax = A @ x
            dxdt = -gamma * x + beta * (1.0 - x) * ax
            return dxdt

        sol = solve_ivp(ode_sis, [0.0, t_max], x0, t_eval=t_eval, method="RK45")
        prevalence = np.mean(sol.y, axis=0)
        return {
            "t": sol.t,
            "prevalence": prevalence,
            "final_prevalence": float(prevalence[-1]) if len(prevalence) > 0 else 0.0,
            "peak_prevalence": float(np.max(prevalence)) if len(prevalence) > 0 else 0.0
        }

    elif model.upper() == "SIR":
        # State vector y = [x, z] of length 2n (infected x, recovered z)
        y0 = np.zeros(2 * n, dtype=np.float64)
        y0[:n] = x0

        def ode_sir(t, y):
            x = y[:n]
            z = y[n:]
            s = np.clip(1.0 - x - z, 0.0, 1.0)
            ax = A @ x
            dxdt = beta * s * ax - gamma * x
            dzdt = gamma * x
            return np.concatenate([dxdt, dzdt])

        sol = solve_ivp(ode_sir, [0.0, t_max], y0, t_eval=t_eval, method="RK45")
        x_sol = sol.y[:n, :]
        z_sol = sol.y[n:, :]
        prevalence = np.mean(x_sol, axis=0)
        recovered_fraction = np.mean(z_sol, axis=0)
        attack_rate = float(recovered_fraction[-1]) if len(recovered_fraction) > 0 else 0.0
        return {
            "t": sol.t,
            "prevalence": prevalence,
            "recovered_fraction": recovered_fraction,
            "attack_rate": attack_rate,
            "peak_prevalence": float(np.max(prevalence)) if len(prevalence) > 0 else 0.0
        }
    else:
        raise ValueError(f"Unsupported model: {model}")
