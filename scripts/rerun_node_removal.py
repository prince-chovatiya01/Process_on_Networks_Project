"""
Rerun node-removal intervention comparisons with:
1. Contiguous residual-node relabeling (preventing sampling of removed nodes in temporal sim)
2. Independent random removal sets across repetitions (separating policy variance from epidemic variance)
3. Regenerating verified CSVs and publication intervention plots.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

import networkx as nx
import pandas as pd
import shutil

from spm.interventions.node_removal import run_removal_comparison
from spm.models.plotting import plot_node_removal_comparison
from spm.utils.logging_config import setup_logger

logger = setup_logger("scripts.rerun_node_removal")


def main():
    processed_dir = root_dir / "data" / "processed"
    synthetic_dir = root_dir / "data" / "synthetic"
    results_dir = root_dir / "results" / "csv"
    fig_dir = root_dir / "figures" / "interventions"
    paper_fig_dir = root_dir / "paper" / "figures" / "interventions"

    fig_dir.mkdir(parents=True, exist_ok=True)
    paper_fig_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "key": "sociopatterns_invs15",
            "name": "SocioPatterns InVS15",
            "edge_path": processed_dir / "sociopatterns_invs15" / "static_binary_edgelist.txt",
            "events_path": processed_dir / "sociopatterns_invs15" / "clean_events.csv.gz",
            "tau": 0.8,
            "fractions": [0.0, 0.05, 0.1, 0.15, 0.2],
            "reps": 5,
            "t_max": 20.0
        },
        {
            "key": "email_eu_core_temporal",
            "name": "SNAP email-Eu-core-temporal",
            "edge_path": processed_dir / "email_eu_core_temporal" / "static_binary_edgelist.txt",
            "events_path": processed_dir / "email_eu_core_temporal" / "clean_events.csv.gz",
            "tau": 0.2,
            "fractions": [0.0, 0.1, 0.2],
            "reps": 5,
            "t_max": 20.0
        },
        {
            "key": "college_msg",
            "name": "SNAP CollegeMsg",
            "edge_path": processed_dir / "college_msg" / "static_binary_edgelist.txt",
            "events_path": processed_dir / "college_msg" / "clean_events.csv.gz",
            "tau": 0.2,
            "fractions": [0.0, 0.1, 0.2],
            "reps": 5,
            "t_max": 20.0
        },
        {
            "key": "synthetic_ba_m6",
            "name": "Synthetic BA (m=6)",
            "edge_path": synthetic_dir / "ba_n1000_m6" / "edgelist.txt",
            "events_path": None,
            "tau": 0.0652,
            "fractions": [0.0, 0.05, 0.1, 0.15, 0.2],
            "reps": 5,
            "t_max": 20.0
        }
    ]

    for exp in experiments:
        key = exp["key"]
        name = exp["name"]
        logger.info("Rerunning verified interventions for %s...", name)

        G_static = nx.read_edgelist(exp["edge_path"], nodetype=int)
        events_df = pd.read_csv(exp["events_path"], compression="gzip") if exp["events_path"] else pd.DataFrame()

        df_removal = run_removal_comparison(
            dataset_name=name,
            G_static=G_static,
            events_df=events_df,
            strategies=["degree", "k_core", "time_aware", "random"] if not events_df.empty else ["degree", "k_core", "random"],
            fractions=exp["fractions"],
            tau=exp["tau"],
            gamma=1.0,
            model="SIS",
            repetitions=exp["reps"],
            seed_base=42,
            t_max=exp["t_max"]
        )

        csv_file = results_dir / f"{key}_node_removal.csv"
        df_removal.to_csv(csv_file, index=False)
        logger.info("Saved verified removal CSV: %s", csv_file)

        fig_file = fig_dir / f"{key}_removal.png"
        plot_node_removal_comparison(df_removal, dataset_name=name, output_file=fig_file)

        paper_dest = paper_fig_dir / f"{key}_removal.png"
        shutil.copy2(fig_file, paper_dest)
        logger.info("Copied plot to paper figures: %s", paper_dest)

        # Print summary for 20% removal
        df_20 = df_removal[df_removal["fraction_removed"] == 0.2]
        print(f"\n=== {name} at f = 0.20 ===")
        print(df_20[["strategy", "lambda_1_remaining", "static_mean_impact", "static_std_impact", "temporal_mean_impact", "temporal_std_impact"]])


if __name__ == "__main__":
    main()
