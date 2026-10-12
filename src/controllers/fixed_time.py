"""Fixed-time traffic signal controller."""

from typing import Any, Dict, List, Optional

from src.controllers.base import BaseTrafficController, IntersectionObservation


class FixedTimeController(BaseTrafficController):
    """Fixed-time signal controller operating on predetermined cyclic schedules.

    Features:
    - Independent of instantaneous vehicle queue or density.
    - Configurable green durations per phase and per intersection.
    - Configurable phase sequences and intersection offsets.
    - Respects safe clearance transitions via SafePhaseManager.
    - Fully reproducible given fixed configuration and seed.
    """

    def __init__(
        self,
        default_green_time: float = 30.0,
        phase_durations: Optional[Dict[str, Dict[int, float]]] = None,
        phase_sequences: Optional[Dict[str, List[int]]] = None,
        offsets: Optional[Dict[str, float]] = None,
    ):
        super().__init__(name="FixedTime")
        if default_green_time <= 0:
            raise ValueError(f"default_green_time must be positive, got {default_green_time}")

        self.default_green_time = float(default_green_time)
        self.phase_durations: Dict[str, Dict[int, float]] = phase_durations or {}
        self.phase_sequences: Dict[str, List[int]] = phase_sequences or {}
        self.offsets: Dict[str, float] = offsets or {}

        self._current_seq_index: Dict[str, int] = {}
        self._target_phases: Dict[str, int] = {}
        self._offset_remaining: Dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        """Resets all internal scheduler states."""
        self.history.clear()
        self._current_seq_index.clear()
        self._target_phases.clear()
        self._offset_remaining = {k: float(v) for k, v in self.offsets.items()}

    def _get_sequence(self, tls_id: str, obs: IntersectionObservation) -> List[int]:
        """Resolves the cyclic phase sequence for a given intersection."""
        if tls_id in self.phase_sequences and self.phase_sequences[tls_id]:
            return self.phase_sequences[tls_id]
        num_phases = len(obs.green_phases)
        return list(range(num_phases)) if num_phases > 0 else [0]

    def _get_duration(self, tls_id: str, phase_idx: int) -> float:
        """Resolves the target green duration for an individual phase."""
        tls_durs = self.phase_durations.get(tls_id, {})
        return float(tls_durs.get(phase_idx, self.default_green_time))

    def select_action(
        self, observations: Dict[str, IntersectionObservation]
    ) -> Dict[str, int]:
        """Evaluates elapsed green time against schedule and returns phase action per TLS."""
        actions: Dict[str, int] = {}

        for tls_id, obs in observations.items():
            seq = self._get_sequence(tls_id, obs)
            if not seq:
                actions[tls_id] = 0
                continue

            if tls_id not in self._current_seq_index:
                self._current_seq_index[tls_id] = 0
                self._target_phases[tls_id] = seq[0]

            curr_idx = self._current_seq_index[tls_id]
            curr_phase = seq[curr_idx]
            target_dur = self._get_duration(tls_id, curr_phase)

            # If the intersection is currently undergoing yellow or all-red clearance,
            # maintain the target phase request until transition completes
            if obs.is_in_transition:
                actions[tls_id] = self._target_phases[tls_id]
                continue

            # Check if active green phase matches schedule and duration is reached
            if obs.current_phase == curr_phase:
                if obs.time_in_phase >= target_dur and len(seq) > 1:
                    # Time to cycle to next phase
                    next_idx = (curr_idx + 1) % len(seq)
                    next_phase = seq[next_idx]
                    self._current_seq_index[tls_id] = next_idx
                    self._target_phases[tls_id] = next_phase
                    actions[tls_id] = next_phase

                    self.log_decision(
                        tls_id=tls_id,
                        sim_time=obs.sim_time,
                        step=obs.step,
                        action=next_phase,
                        details={
                            "event": "phase_switch",
                            "prev_phase": curr_phase,
                            "next_phase": next_phase,
                            "elapsed_green": obs.time_in_phase,
                            "target_duration": target_dur,
                        },
                    )
                else:
                    # Continue holding current green phase
                    actions[tls_id] = curr_phase
            else:
                # Synchronize to expected target phase
                actions[tls_id] = self._target_phases[tls_id]

        return actions

    def get_config(self) -> Dict[str, Any]:
        """Returns controller configuration dictionary."""
        return {
            "name": self.name,
            "default_green_time": self.default_green_time,
            "phase_durations": self.phase_durations,
            "phase_sequences": self.phase_sequences,
            "offsets": self.offsets,
        }
