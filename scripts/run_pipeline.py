"""
Main execution script for Sprint 1 pipeline:
- Generates all nine synthetic graph configurations
- Attempts dataset acquisition (email-Eu, CollegeMsg, SocioPatterns InVS15)
- Preprocesses and validates any downloaded datasets
- Produces structural/spectral summary reports
"""

import argparse
import sys
from pathlib import Path

# Add src to python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from spm.data.download import acquire_all_datasets
from spm.data.preprocess import process_dataset
from spm.graphs.stats import compute_graph_statistics
from spm.graphs.synthetic import generate_all_synthetic_graphs
from spm.utils.config_loader import get_config
from spm.utils.logging_config import setup_logger

logger = setup_logger("spm.pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="SC452 Sprint 1 Pipeline Runner")
    parser.add_argument("--skip-downloads", action="store_true", help="Skip dataset network downloads")
    parser.add_argument("--synthetic-only", action="store_true", help="Run only synthetic graph generation")
    args = parser.parse_args()

    project_root = root_dir
    data_raw = project_root / "data" / "raw"
    data_processed = project_root / "data" / "processed"
    data_synthetic = project_root / "data" / "synthetic"
    results_dir = project_root / "results"

    for d in [data_raw, data_processed, data_synthetic, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic networks
    logger.info("=== STEP 1: Generating all nine synthetic network ensembles ===")
    synth_cfg = get_config("synthetic.yaml")
    manifest = generate_all_synthetic_graphs(synth_cfg, output_dir=data_synthetic)
    logger.info("Generated %d synthetic networks. Computing spectral thresholds...", len(manifest))
    
    synth_stats = []
    import networkx as nx
    for item in manifest:
        edge_file = Path(item["file_path"])
        G = nx.read_edgelist(edge_file, nodetype=int)
        stats = compute_graph_statistics(G, include_centrality=False)
        stats["graph_id"] = item["graph_id"]
        stats["type"] = item["parameters"]["type"]
        synth_stats.append(stats)
        logger.info("[%s] N=%d, M=%d, lambda_1=%.4f, tau_c=%.4f", 
                    item["graph_id"], stats["nodes"], stats["edges"], stats["lambda_1"], stats["tau_c"] or 0.0)

    if args.synthetic_only:
        logger.info("Synthetic-only flag set. Completed.")
        return

    # 2. Acquire real datasets
    if not args.skip_downloads:
        logger.info("=== STEP 2: Real dataset acquisition ===")
        datasets_cfg = get_config("real_datasets.yaml")
        download_results = acquire_all_datasets(datasets_cfg, raw_root=data_raw)
        
        # 3. Preprocess downloaded datasets
        logger.info("=== STEP 3: Preprocessing and validation ===")
        for key, prov in download_results.items():
            cfg = datasets_cfg["datasets"][key]
            if prov["status"] in ["DOWNLOADED", "VERIFIED_EXISTING"]:
                raw_file = Path(prov["file_path"])
                out_dir = data_processed / key
                logger.info("Processing %s...", key)
                process_dataset(key, raw_file, out_dir, cfg)
            else:
                logger.warning("Skipping processing for %s (Status: %s)", key, prov["status"])


if __name__ == "__main__":
    main()
