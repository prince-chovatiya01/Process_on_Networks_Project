"""
Interventions module: node removal strategies (degree, k-core, time-aware, random).
"""

from spm.interventions.node_removal import (
    apply_node_removal_static,
    apply_node_removal_temporal,
    get_removal_nodes,
    relabel_residual_temporal_system,
    run_removal_comparison,
)

__all__ = [
    "get_removal_nodes",
    "apply_node_removal_static",
    "apply_node_removal_temporal",
    "relabel_residual_temporal_system",
    "run_removal_comparison",
]
