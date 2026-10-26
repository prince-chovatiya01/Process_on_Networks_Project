"""
Epidemic spreading simulation and threshold analysis module (SIS, SIR, NIMFA).
"""

from spm.models.base import SimulationRunResult, State
from spm.models.nimfa import compute_nimfa_trajectory, compute_spectral_threshold_reference
from spm.models.static_sim import simulate_static_gillespie
from spm.models.temporal_sim import simulate_temporal_events

__all__ = [
    "State",
    "SimulationRunResult",
    "simulate_static_gillespie",
    "simulate_temporal_events",
    "compute_nimfa_trajectory",
    "compute_spectral_threshold_reference",
]
