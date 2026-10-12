"""Observation vector building and feature extraction for Deep Q-Network (DQN)."""

from typing import Any, Dict, List, Sequence
import numpy as np
from src.controllers.base import IntersectionObservation


class ObservationBuilder:
    """Builds normalized, fixed-length continuous observation vectors for DQN.

    Vector Structure:
    1. Normalized incoming lane queues: [q_1 / max_queue_scale, ..., q_N / max_queue_scale]
       Length: N_in (number of incoming lanes, sorted deterministically)
       Range: [0.0, 1.0]
    2. One-hot encoding of active green phase: [0, ..., 1, ..., 0]
       Length: K (number of selectable green phases)
    3. Normalized elapsed time in phase: min(time_in_phase / max_time_scale, 1.0)
       Length: 1
       Range: [0.0, 1.0]
    4. Transition indicator flag: 1.0 if in yellow/all-red clearance, else 0.0
       Length: 1
    5. Min-green satisfaction flag: 1.0 if time_in_phase >= min_green, else 0.0
       Length: 1

    Total Dimension: N_in + K + 3
    """

    def __init__(
        self,
        incoming_lanes: Sequence[str],
        num_green_phases: int,
        min_green: float = 5.0,
        max_queue_scale: float = 20.0,
        max_time_scale: float = 60.0,
    ):
        if not incoming_lanes:
            raise ValueError("incoming_lanes must not be empty")
        if num_green_phases <= 0:
            raise ValueError(f"num_green_phases must be positive, got {num_green_phases}")
        if max_queue_scale <= 0:
            raise ValueError(f"max_queue_scale must be positive, got {max_queue_scale}")
        if max_time_scale <= 0:
            raise ValueError(f"max_time_scale must be positive, got {max_time_scale}")

        self.incoming_lanes: List[str] = sorted(list(incoming_lanes))
        self.num_green_phases: int = int(num_green_phases)
        self.min_green: float = float(min_green)
        self.max_queue_scale: float = float(max_queue_scale)
        self.max_time_scale: float = float(max_time_scale)

        self.observation_dim: int = len(self.incoming_lanes) + self.num_green_phases + 3

    def build_vector(self, obs: IntersectionObservation) -> np.ndarray:
        """Converts an IntersectionObservation into a normalized numpy float32 vector.

        Raises:
            ValueError: If the vector contains NaN, Inf, or has unexpected dimensions.
        """
        features: List[float] = []

        # 1. Normalized incoming queues
        for lane_id in self.incoming_lanes:
            q = obs.in_lane_queues.get(lane_id, 0.0)
            norm_q = min(max(float(q) / self.max_queue_scale, 0.0), 1.0)
            features.append(norm_q)

        # 2. One-hot green phase
        one_hot_phase = [0.0] * self.num_green_phases
        curr_p = obs.current_phase
        if 0 <= curr_p < self.num_green_phases:
            one_hot_phase[curr_p] = 1.0
        features.extend(one_hot_phase)

        # 3. Normalized time in phase
        norm_time = min(max(float(obs.time_in_phase) / self.max_time_scale, 0.0), 1.0)
        features.append(norm_time)

        # 4. Transition indicator flag
        features.append(1.0 if obs.is_in_transition else 0.0)

        # 5. Min green satisfaction flag
        features.append(1.0 if obs.time_in_phase >= self.min_green and not obs.is_in_transition else 0.0)

        vec = np.array(features, dtype=np.float32)

        # Rigorous sanity checks
        if vec.shape != (self.observation_dim,):
            raise ValueError(
                f"Generated vector shape {vec.shape} does not match expected ({self.observation_dim},)"
            )
        if np.isnan(vec).any():
            raise ValueError(f"Observation vector contains NaN values: {vec}")
        if np.isinf(vec).any():
            raise ValueError(f"Observation vector contains Infinity values: {vec}")

        return vec

    def get_config(self) -> Dict[str, Any]:
        """Returns observation builder configuration dictionary."""
        return {
            "incoming_lanes": list(self.incoming_lanes),
            "num_green_phases": self.num_green_phases,
            "min_green": self.min_green,
            "max_queue_scale": self.max_queue_scale,
            "max_time_scale": self.max_time_scale,
            "observation_dim": self.observation_dim,
        }
