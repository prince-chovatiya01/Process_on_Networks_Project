"""
Synthetic Network Experiment Pipeline (SC452 Group 5 Sprint 3):
- Evaluates epidemic dynamics (SIS & SIR) across all 9 TA synthetic ensembles:
  * ER: p = 0.1, 0.2, 0.3
  * BA: m = 4, 6, 8
  * WS: k = 6, 8, 10
- Measures spectral threshold tau_c = 1 / lambda_1
- Runs degree vs k-core vs random node removal interventions
- Produces summary CSVs, heatmaps, and intervention plots
"""

import argparse
import json
import sys
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd

# Add src to python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from spm.interventions.node_removal import run_removal_comparison
from spm.models.nimfa import compute_spectral_threshold_reference
from spm.models.plotting import plot_node_removal_comparison, plot_sweep_heatmap
from spm.models.static_sim import simulate_static_gillespie
from spm.utils.config_loader import get_config
from spm.utils.logging_config import setup_logger

logger = setup_logger("spm.experiments.synthetic")


def main():
    parser = argparse.ArgumentParser(description="Synthetic Network Epidemic Experiments")
    parser.add_argument("--smoke", action="store_true", help="Quick run with reduced grid/reps")
    args = parser.parse_args()

    synthetic_dir = root_dir / "data" / "synthetic"
    results_csv_dir = root_dir / "results" / "csv"
    figures_dir = root_dir / "figures"
    heatmaps_dir = figures_dir / "heatmaps"
    interventions_dir = figures_dir / "interventions"

    for d in [results_csv_dir, heatmaps_dir, interventions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_file = synthetic_dir / "manifest.json"
    if not manifest_file.exists():
        logger.error("Synthetic manifest not found at %s. Run scripts/run_pipeline.py first.", manifest_file)
        return

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    logger.info("Loaded %d synthetic configurations from manifest.", len(manifest))

    spectral_records = []
    sweep_records = []
    removal_records = []

    tau_multipliers = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    reps = 3
    sim_t_max = 5.0
    fractions = [0.0, 0.05, 0.1, 0.15, 0.2]
    eval_removal_graphs = {"ba_n1000_m6", "er_n1000_p01", "ws_n1000_k8"}

    for item in manifest:
        graph_id = item["graph_id"]
        params = item["parameters"]
        g_type = params["type"]
        edge_file = Path(item["file_path"])

        if not edge_file.exists():
            continue

        G = nx.read_edgelist(edge_file, nodetype=int)
        lam1, tau_c = compute_spectral_threshold_reference(G)

        logger.info("[%s] %s Graph: N=%d, M=%d, lambda_1=%.4f, tau_c=%.4f",
                    graph_id, g_type, G.number_of_nodes(), G.number_of_edges(), lam1, tau_c)

        spectral_records.append({
            "graph_id": graph_id,
            "type": g_type,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "param_value": params.get("p") or params.get("m") or params.get("k"),
            "lambda_1": lam1,
            "tau_c_nimfa": tau_c
        })

        # 1. Parameter sweeps around the spectral threshold: tau = mult * tau_c
        for mult in tau_multipliers:
            tau = mult * tau_c
            beta = tau * 1.0  # gamma = 1.0

            sis_finals = []
            sir_attacks = []
            for r in range(reps):
                seed = 42 + r * 100
                res_sis = simulate_static_gillespie(G, beta=beta, gamma=1.0, model="SIS", t_max=sim_t_max, seed=seed)
                res_sir = simulate_static_gillespie(G, beta=beta, gamma=1.0, model="SIR", t_max=sim_t_max, seed=seed)
                sis_finals.append(res_sis.final_prevalence)
                sir_attacks.append(res_sir.attack_rate)

            sweep_records.append({
                "graph_id": graph_id,
                "type": g_type,
                "param_value": params.get("p") or params.get("m") or params.get("k"),
                "multiplier": mult,
                "tau": tau,
                "tau_c_nimfa": tau_c,
                "mean_sis_prevalence": float(np.mean(sis_finals)),
                "std_sis_prevalence": float(np.std(sis_finals, ddof=1)) if reps > 1 else 0.0,
                "mean_sir_attack_rate": float(np.mean(sir_attacks)),
                "std_sir_attack_rate": float(np.std(sir_attacks, ddof=1)) if reps > 1 else 0.0
            })

        # 2. Node Removal Interventions at supercritical point (tau = 1.5 * tau_c)
        if eval_removal_graphs is None or graph_id in eval_removal_graphs:
            super_tau = 1.5 * tau_c
            rem_df = run_removal_comparison(
                dataset_name=graph_id,
                G_static=G,
                events_df=pd.DataFrame(),  # Synthetic graphs are static
                strategies=["degree", "k_core", "random"],
                fractions=fractions,
                tau=super_tau,
                gamma=1.0,
                model="SIS",
                repetitions=reps,
                seed_base=42,
                t_max=sim_t_max
            )
            rem_df["graph_id"] = graph_id
            rem_df["type"] = g_type
            rem_df["param_value"] = params.get("p") or params.get("m") or params.get("k")
            removal_records.append(rem_df)

    # Save summary DataFrames
    spectral_df = pd.DataFrame(spectral_records)
    spectral_df.to_csv(results_csv_dir / "synthetic_spectral_summary.csv", index=False)

    sweeps_df = pd.DataFrame(sweep_records)
    sweeps_df.to_csv(results_csv_dir / "synthetic_epidemic_sweeps.csv", index=False)

    all_removal_df = pd.concat(removal_records, ignore_index=True) if removal_records else pd.DataFrame()
    all_removal_df.to_csv(results_csv_dir / "synthetic_node_removal.csv", index=False)

    # Generate Heatmaps
    for model_type, col_name, title_suffix in [
        ("ER", "p", "Erdos-Renyi (p)"),
        ("BA", "m", "Barabasi-Albert (m)"),
        ("WS", "k", "Watts-Strogatz (k)")
    ]:
        sub = sweeps_df[sweeps_df["type"] == model_type].copy()
        if not sub.empty:
            plot_sweep_heatmap(
                matrix_df=sub,
                x_col="multiplier",
                y_col="param_value",
                val_col="mean_sis_prevalence",
                title=f"{title_suffix} — Endemic Prevalence (SIS)",
                x_label=r"Normalized Transmission $\tau / \tau_c$",
                y_label="Model Parameter",
                cbar_label="Prevalence $I/N$",
                output_file=heatmaps_dir / f"synthetic_{model_type.lower()}_sis_heatmap.png"
            )

    # Generate Node Removal comparison plot for representative BA and ER graphs
    if not all_removal_df.empty and "dataset" in all_removal_df.columns:
        rep_rem = all_removal_df[all_removal_df["dataset"] == "ba_n1000_m6"]
        if not rep_rem.empty:
            plot_node_removal_comparison(
                removal_df=rep_rem,
                dataset_name="Synthetic Barabasi-Albert (m=6)",
                output_file=interventions_dir / "synthetic_ba_m6_removal.png"
            )

    logger.info("Synthetic network experiments completed successfully.")


if __name__ == "__main__":
    main()
