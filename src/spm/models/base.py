"""
Common data structures, states, and result containers for epidemic simulations.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional
import numpy as np


class State(IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2


@dataclass
class SimulationRunResult:
    model: str                  # "SIS" or "SIR"
    network_type: str           # "static" or "temporal"
    beta: float
    gamma: float
    tau: float                  # beta / gamma
    seed: int
    n_nodes: int
    initial_infected: int
    times: List[float] = field(default_factory=list)
    infected_counts: List[int] = field(default_factory=list)
    recovered_counts: List[int] = field(default_factory=list)
    final_infected: int = 0
    final_recovered: int = 0
    attack_rate: float = 0.0    # (final_recovered + final_infected) / N for SIR
    peak_prevalence: float = 0.0 # max(I) / N
    peak_time: float = 0.0
    extinction: bool = False
    duration: float = 0.0

    @property
    def final_prevalence(self) -> float:
        return self.final_infected / self.n_nodes if self.n_nodes > 0 else 0.0
