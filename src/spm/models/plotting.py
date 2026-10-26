"""
Plotting and visualization routines for epidemic curves, phase transitions, and threshold comparison.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_plot_style():
    """Apply consistent publication-grade matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.dpi"] = 150


def plot_phase_diagram(
    summary_df: pd.DataFrame,
    tau_c_nimfa: float,
    dataset_name: str,
    model: str,
    output_file: Path
) -> None:
    """
    Plot Order Parameter (prevalence for SIS, attack rate for SIR) vs. tau = beta / gamma,
    comparing Static Aggregation vs. Temporal Sequence alongside NIMFA spectral reference.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))

    static_data = summary_df[summary_df["sim_type"] == "static"].sort_values("tau")
    temporal_data = summary_df[summary_df["sim_type"] == "temporal"].sort_values("tau")

    # Order parameter label
    metric_label = "Endemic Prevalence $\\langle I_\\infty \\rangle / N$" if model == "SIS" else "Final Attack Rate $R_\\infty / N$"

    if not static_data.empty:
        ax.errorbar(
            static_data["tau"],
            static_data["mean_order_param"],
            yerr=static_data["std_order_param"],
            fmt="o-",
            color="#1f77b4",
            label="Static Network (Stochastic SSA)",
            capsize=3,
            linewidth=2,
            markersize=6
        )

    if not temporal_data.empty:
        ax.errorbar(
            temporal_data["tau"],
            temporal_data["mean_order_param"],
            yerr=temporal_data["std_order_param"],
            fmt="s--",
            color="#d62728",
            label="Temporal Sequence (Event-driven)",
            capsize=3,
            linewidth=2,
            markersize=6
        )

    # Static NIMFA spectral threshold reference line
    if 0.0 < tau_c_nimfa < float("inf"):
        ax.axvline(
            x=tau_c_nimfa,
            color="#2ca02c",
            linestyle=":",
            linewidth=2,
            label=f"NIMFA Spectral Ref ($\\tau_c = 1/\\lambda_1 = {tau_c_nimfa:.4f}$)"
        )

    ax.set_xlabel(r"Effective Infection Parameter $\tau = \beta / \gamma$")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{dataset_name} — {model} Epidemic Threshold Comparison\n(Static Aggregation vs. Temporal)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def plot_epidemic_curves(
    static_traj: Dict[str, Any],
    temporal_traj: Dict[str, Any],
    tau: float,
    dataset_name: str,
    model: str,
    output_file: Path
) -> None:
    """
    Plot single-realization epidemic curves I(t)/N over normalized time for Static vs. Temporal.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    if static_traj and "times" in static_traj:
        ax.plot(
            static_traj["times"],
            np.array(static_traj["infected_counts"]) / static_traj["n_nodes"],
            color="#1f77b4",
            linewidth=2,
            label="Static Network"
        )

    if temporal_traj and "times" in temporal_traj:
        ax.plot(
            temporal_traj["times"],
            np.array(temporal_traj["infected_counts"]) / temporal_traj["n_nodes"],
            color="#d62728",
            linewidth=2,
            linestyle="--",
            label="Temporal Sequence"
        )

    ax.set_xlabel("Normalized Simulation Time $t$")
    ax.set_ylabel("Infected Fraction $I(t) / N$")
    ax.set_title(f"{dataset_name} — {model} Epidemic Trajectory ($\\tau = {tau:.3f}$)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def plot_outbreak_distribution(
    runs_df: pd.DataFrame,
    dataset_name: str,
    output_file: Path
) -> None:
    """
    Plot outbreak size / attack rate distribution for SIR across static vs temporal runs.
    """
    set_plot_style()
    sir_runs = runs_df[runs_df["model"] == "SIR"]
    if sir_runs.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.boxplot(
        data=sir_runs,
        x="tau",
        y="attack_rate",
        hue="sim_type",
        palette={"static": "#1f77b4", "temporal": "#d62728"},
        ax=ax
    )
    ax.set_xlabel(r"Effective Transmission Rate $\tau = \beta / \gamma$")
    ax.set_ylabel(r"Outbreak Attack Rate $R_\infty / N$")
    ax.set_title(f"{dataset_name} — SIR Outbreak Size Distributions")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="Network Representation", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def plot_node_removal_comparison(
    removal_df: pd.DataFrame,
    dataset_name: str,
    output_file: Path
) -> None:
    """
    Plot epidemiological and structural impact of node removal strategies:
    Degree vs. K-core vs. Time-aware vs. Random for Static vs. Temporal networks.
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {
        "degree": "#d62728",      # Red
        "k_core": "#ff7f0e",      # Orange
        "time_aware": "#2ca02c",  # Green
        "random": "#7f7f7f"       # Grey
    }
    labels = {
        "degree": "Degree Targeted",
        "k_core": "K-Core Targeted",
        "time_aware": "Time-Aware Activity",
        "random": "Random Uniform"
    }

    # Left Plot: Static network epidemic impact
    for strat, grp in removal_df.groupby("strategy"):
        grp = grp.sort_values("fraction_removed")
        c = colors.get(strat, "#1f77b4")
        lbl = labels.get(strat, strat.title())
        ax1.errorbar(
            grp["fraction_removed"],
            grp["static_mean_impact"],
            yerr=grp["static_std_impact"],
            fmt="o-",
            color=c,
            label=lbl,
            capsize=3,
            linewidth=2,
            markersize=5
        )

    ax1.set_xlabel("Fraction of Nodes Removed ($f$)")
    ax1.set_ylabel("Remaining Epidemic Prevalence / Attack Rate")
    ax1.set_title(f"Static Network — Intervention Efficacy\n({dataset_name})")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Right Plot: Temporal sequence epidemic impact
    for strat, grp in removal_df.groupby("strategy"):
        grp = grp.sort_values("fraction_removed")
        c = colors.get(strat, "#1f77b4")
        lbl = labels.get(strat, strat.title())
        ax2.errorbar(
            grp["fraction_removed"],
            grp["temporal_mean_impact"],
            yerr=grp["temporal_std_impact"],
            fmt="s--",
            color=c,
            label=lbl,
            capsize=3,
            linewidth=2,
            markersize=5
        )

    ax2.set_xlabel("Fraction of Nodes Removed ($f$)")
    ax2.set_ylabel("Remaining Epidemic Prevalence / Attack Rate")
    ax2.set_title(f"Temporal Contact Network — Intervention Efficacy\n({dataset_name})")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def plot_sweep_heatmap(
    matrix_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    val_col: str,
    title: str,
    x_label: str,
    y_label: str,
    cbar_label: str,
    output_file: Path
) -> None:
    """
    Generate 2D parameter sweep heatmap (e.g., parameter vs removal fraction -> epidemic prevalence).
    """
    set_plot_style()
    pivot = matrix_df.pivot(index=y_col, columns=x_col, values=val_col)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f" if pivot.values.max() <= 1.0 else ".2f",
        cmap="YlOrRd",
        cbar_kws={"label": cbar_label},
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)

