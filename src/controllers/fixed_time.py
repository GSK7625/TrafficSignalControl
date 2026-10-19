"""Fixed-time traffic signal controller."""

import math
from typing import Any, Dict, List, Optional
from src.controllers.base import BaseTrafficController, IntersectionObservation


class FixedTimeController(BaseTrafficController):
    """Fixed-time signal controller operating on predetermined cyclic schedules."""

    def __init__(
        self,
        default_green_time: float = 30.0,
        phase_durations: Optional[Dict[str, Dict[int, float]]] = None,
        phase_sequences: Optional[Dict[str, List[int]]] = None,
        min_green: Optional[float] = None,
        max_green: Optional[float] = None,
    ):
        super().__init__(name="FixedTime")
        if default_green_time <= 0:
            raise ValueError(f"default_green_time must be positive, got {default_green_time}")

        self.default_green_time = float(default_green_time)
        self.phase_durations = phase_durations or {}
        self.phase_sequences = phase_sequences or {}
        self.min_green = min_green
        self.max_green = max_green
        self.reset()

    def reset(self) -> None:
        """Resets all internal scheduler states."""
        self._target_phases: Dict[str, int] = {}

    def select_action(self, observations: Dict[str, IntersectionObservation]) -> Dict[str, int]:
        """Evaluates elapsed green time against schedule and returns next phase action."""
        actions: Dict[str, int] = {}
        for tls_id, obs in observations.items():
            seq = self.phase_sequences.get(tls_id) or list(range(len(obs.green_phases)))
            if not seq:
                actions[tls_id] = 0
                continue

            if obs.is_in_transition:
                actions[tls_id] = self._target_phases.get(tls_id, obs.current_phase)
                continue

            curr_phase = obs.current_phase if obs.current_phase in seq else seq[0]
            curr_idx = seq.index(curr_phase)
            dur = float(self.phase_durations.get(tls_id, {}).get(curr_phase, self.default_green_time))

            if obs.time_in_phase >= dur and len(seq) > 1:
                next_phase = seq[(curr_idx + 1) % len(seq)]
            else:
                next_phase = curr_phase

            self._target_phases[tls_id] = next_phase
            actions[tls_id] = next_phase
        return actions

    def validate_compatibility(
        self,
        min_green: Optional[float] = None,
        max_green: Optional[float] = None,
        num_phases_per_tls: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        """Validates configuration against constraints."""
        errors: List[str] = []
        if min_green is not None and self.default_green_time < min_green:
            errors.append(f"default_green_time ({self.default_green_time}s) is less than min_green ({min_green}s)")
        if max_green is not None and self.default_green_time > max_green:
            errors.append(f"default_green_time ({self.default_green_time}s) exceeds max_green ({max_green}s)")
        for tls_id, p_map in self.phase_durations.items():
            if isinstance(p_map, dict):
                for p_idx, dur in p_map.items():
                    if dur is None or not (isinstance(dur, (int, float)) and math.isfinite(dur) and dur > 0):
                        errors.append(f"TLS '{tls_id}' phase {p_idx} duration must be positive and finite, got {dur}")
                    elif min_green is not None and dur < min_green:
                        errors.append(f"TLS '{tls_id}' phase {p_idx} duration ({dur}s) is less than min_green ({min_green}s)")
                    elif max_green is not None and dur > max_green:
                        errors.append(f"TLS '{tls_id}' phase {p_idx} duration ({dur}s) exceeds max_green ({max_green}s)")
        if num_phases_per_tls:
            for tls_id, seq in self.phase_sequences.items():
                if any(not 0 <= p < num_phases_per_tls.get(tls_id, 0) for p in seq):
                    errors.append(f"TLS '{tls_id}' phase sequence contains invalid phase index")
        return errors

    def get_config(self) -> Dict[str, Any]:
        """Returns controller configuration dictionary."""
        return {
            "name": self.name,
            "default_green_time": self.default_green_time,
            "phase_durations": self.phase_durations,
            "phase_sequences": self.phase_sequences,
            "min_green": self.min_green,
            "max_green": self.max_green,
        }
