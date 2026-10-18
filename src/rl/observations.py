"""Observation vector builder for continuous RL observations."""

from typing import Any, Dict, Sequence
import numpy as np
from src.controllers.base import IntersectionObservation


class ObservationBuilder:
    """Builds normalized continuous observation vectors for RL."""

    def __init__(
        self,
        incoming_lanes: Sequence[str],
        num_green_phases: int,
        min_green: float = 5.0,
        max_queue_scale: float = 20.0,
        max_time_scale: float = 60.0,
    ):
        self.incoming_lanes = sorted(list(incoming_lanes))
        self.num_green_phases = int(num_green_phases)
        self.min_green = float(min_green)
        self.max_queue_scale = float(max_queue_scale)
        self.max_time_scale = float(max_time_scale)
        self.observation_dim = len(self.incoming_lanes) + self.num_green_phases + 3

    def build_vector(self, obs: IntersectionObservation) -> np.ndarray:
        queues = [min(max(float(obs.in_lane_queues.get(l, 0.0)) / self.max_queue_scale, 0.0), 1.0) for l in self.incoming_lanes]
        one_hot = [1.0 if i == obs.current_phase else 0.0 for i in range(self.num_green_phases)]
        norm_time = min(max(float(obs.time_in_phase) / self.max_time_scale, 0.0), 1.0)
        in_trans = 1.0 if obs.is_in_transition else 0.0
        min_green_ok = 1.0 if obs.time_in_phase >= self.min_green and not obs.is_in_transition else 0.0

        return np.array(queues + one_hot + [norm_time, in_trans, min_green_ok], dtype=np.float32)

    def get_config(self) -> Dict[str, Any]:
        return {
            "incoming_lanes": list(self.incoming_lanes),
            "num_green_phases": self.num_green_phases,
            "min_green": self.min_green,
            "max_queue_scale": self.max_queue_scale,
            "max_time_scale": self.max_time_scale,
            "observation_dim": self.observation_dim,
        }
