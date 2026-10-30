"""
Compute Temporal Reachability Gap (TRG) and Aggregation Inflation Ratio (AIR)
across SocioPatterns InVS15, SNAP email-Eu-core-temporal, and SNAP CollegeMsg.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from spm.graphs.reachability import (
    compute_aggregation_inflation_ratio,
    compute_temporal_reachability_gap,
)
from spm.utils.logging_config import setup_logger

logger = setup_logger("scripts.reachability_air")


def main():
    root_dir = Path(__file__).resolve().parent.parent
    processed_dir = root_dir / "data" / "processed"
    results_dir = root_dir / "results" / "csv"
    fig_dir = root_dir / "figures"
    paper_fig_dir = root_dir / "paper" / "figures"

    datasets = [
        {"key": "sociopatterns_invs15", "name": "SocioPatterns InVS15", "n": 217, "sample": None},
        {"key": "email_eu_core_temporal", "name": "email-Eu-core", "n": 986, "sample": 200},
        {"key": "college_msg", "name": "SNAP CollegeMsg", "n": 1899, "sample": 200},
    ]

    records = []

    for d in datasets:
        key = d["key"]
        name = d["name"]
        n = d["n"]
        sample = d["sample"]

        logger.info("Computing reachability for %s...", name)
        edge_file = processed_dir / key / "static_binary_edgelist.txt"
        events_file = processed_dir / key / "clean_events.csv.gz"

        G_static = nx.read_edgelist(edge_file, nodetype=int)
        events_df = pd.read_csv(events_file, compression="gzip")

        # 1. Undirected Reachability
        trg_undir = compute_temporal_reachability_gap(
            G_static=G_static,
            events_df=events_df,
            n_nodes=n,
            sample_sources=sample,
            directed=False,
            seed=42,
            n_jobs=-1
        )

        # 2. Directed Reachability (for communication networks)
        trg_dir = compute_temporal_reachability_gap(
            G_static=G_static,
            events_df=events_df,
            n_nodes=n,
            sample_sources=sample,
            directed=True,
            seed=42,
            n_jobs=-1
        )

        # 3. Read SIR attack rates at tau=0.10 from CSV
        sir_file = results_dir / f"{key}_sir_summary.csv"
        static_ar = np.nan
        temp_ar = np.nan
        air = np.nan

        if sir_file.exists():
            sir_df = pd.read_csv(sir_file)
            row_static = sir_df[(sir_df["sim_type"] == "static") & (np.isclose(sir_df["tau"], 0.1, atol=0.01))]
            row_temp = sir_df[(sir_df["sim_type"] == "temporal") & (np.isclose(sir_df["tau"], 0.1, atol=0.01))]

            if not row_static.empty and not row_temp.empty:
                static_ar = float(row_static["mean_attack_rate"].values[0])
                temp_ar = float(row_temp["mean_attack_rate"].values[0])
                air = compute_aggregation_inflation_ratio(static_ar, temp_ar)

        records.append({
            "dataset_key": key,
            "dataset_name": name,
            "nodes": n,
            "events": len(events_df),
            "static_reachable_pairs": trg_undir["static_reachable_pairs"],
            "undir_temp_reachable_pairs": trg_undir["temporal_reachable_pairs"],
            "undir_reachability_ratio": trg_undir["temporal_reachability_ratio"],
            "undir_reachability_gap": trg_undir["temporal_reachability_gap"],
            "dir_temp_reachable_pairs": trg_dir["temporal_reachable_pairs"],
            "dir_reachability_ratio": trg_dir["temporal_reachability_ratio"],
            "dir_reachability_gap": trg_dir["temporal_reachability_gap"],
            "sir_static_attack_rate_tau01": static_ar,
            "sir_temporal_attack_rate_tau01": temp_ar,
            "aggregation_inflation_ratio_tau01": air
        })

    summary_df = pd.DataFrame(records)
    out_csv = results_dir / "temporal_reachability_and_air.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info("Saved reachability and AIR summary to %s", out_csv)
    print(summary_df[[
        "dataset_name", "undir_reachability_gap", "dir_reachability_gap",
        "sir_static_attack_rate_tau01", "sir_temporal_attack_rate_tau01",
        "aggregation_inflation_ratio_tau01"
    ]])

    # Generate Publication Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    x = np.arange(len(datasets))
    names = [d["name"] for d in datasets]

    # Subplot 1: Temporal Reachability Gap
    width = 0.35
    gap_undir = [r["undir_reachability_gap"] * 100 for r in records]
    gap_dir = [r["dir_reachability_gap"] * 100 for r in records]

    b1 = ax1.bar(x - width/2, gap_undir, width, label="Undirected Contacts", color="#1f77b4", alpha=0.85)
    b2 = ax1.bar(x + width/2, gap_dir, width, label="Directed Ties", color="#ff7f0e", alpha=0.85)

    ax1.set_ylabel("Temporal Reachability Gap (%)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Reachability Gap by Temporal Constraints", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, fontsize=9)
    ax1.set_ylim(0, max(gap_dir) * 1.25)

    for bar in b1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

    # Subplot 2: Aggregation Inflation Ratio (AIR)
    airs = [r["aggregation_inflation_ratio_tau01"] for r in records]
    b3 = ax2.bar(x, airs, width=0.5, color="#2ca02c", alpha=0.85)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="No Inflation (AIR = 1.0)")

    ax2.set_ylabel("Aggregation Inflation Ratio (AIR)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Attack Rate Inflation (SIR at $\\tau = 0.10$)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=9)
    ax2.set_ylim(0, max(airs) * 1.25)

    for bar in b3:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig_file = fig_dir / "temporal_reachability_and_air.png"
    fig.savefig(fig_file)
    logger.info("Saved reachability & AIR figure to %s", fig_file)

    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_fig_dir / "temporal_reachability_and_air.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
