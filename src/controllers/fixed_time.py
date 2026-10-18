"""Fixed-time traffic signal controller."""

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

        self._seq_idx: Dict[str, int] = {}
        self._target_phases: Dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        """Resets all internal scheduler states."""
        self.history.clear()
        self._seq_idx.clear()
        self._target_phases.clear()

    def _get_sequence(self, tls_id: str, obs: IntersectionObservation) -> List[int]:
        if tls_id in self.phase_sequences and self.phase_sequences[tls_id]:
            return self.phase_sequences[tls_id]
        return list(range(len(obs.green_phases))) if obs.green_phases else [0]

    def _get_duration(self, tls_id: str, phase_idx: int) -> float:
        return float(self.phase_durations.get(tls_id, {}).get(phase_idx, self.default_green_time))

    def select_action(self, observations: Dict[str, IntersectionObservation]) -> Dict[str, int]:
        """Evaluates elapsed green time against schedule and returns phase action per TLS."""
        actions: Dict[str, int] = {}

        for tls_id, obs in observations.items():
            seq = self._get_sequence(tls_id, obs)
            if not seq:
                actions[tls_id] = 0
                continue

            if tls_id not in self._seq_idx:
                self._seq_idx[tls_id] = 0
                self._target_phases[tls_id] = seq[0]

            curr_idx = self._seq_idx[tls_id]
            curr_phase = seq[curr_idx]
            target_dur = self._get_duration(tls_id, curr_phase)

            # 1. Đang trong giai đoạn đèn vàng / đỏ dọn ngã tư -> giữ nguyên pha đích
            if obs.is_in_transition:
                actions[tls_id] = self._target_phases[tls_id]
                continue

            # 2. Tự đồng bộ nếu pha thực tế bị lệch khỏi lịch trình (do runner/an toàn can thiệp)
            if obs.current_phase != curr_phase and obs.current_phase in seq:
                curr_idx = seq.index(obs.current_phase)
                self._seq_idx[tls_id] = curr_idx
                self._target_phases[tls_id] = obs.current_phase
                curr_phase = obs.current_phase
                target_dur = self._get_duration(tls_id, curr_phase)

            # 3. Hết thời gian xanh -> chuyển pha kế tiếp trong chu kỳ
            if obs.time_in_phase >= target_dur and len(seq) > 1:
                next_idx = (curr_idx + 1) % len(seq)
                next_phase = seq[next_idx]
                self._seq_idx[tls_id] = next_idx
                self._target_phases[tls_id] = next_phase
                actions[tls_id] = next_phase
                self.log_decision(tls_id, obs.sim_time, obs.step, next_phase)
            else:
                actions[tls_id] = curr_phase

        return actions

    def validate_compatibility(
        self,
        min_green: Optional[float] = None,
        max_green: Optional[float] = None,
        num_phases_per_tls: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        """Validates controller configuration against safety timing constraints."""
        errors: List[str] = []
        if max_green is not None and self.default_green_time > max_green:
            errors.append(f"default_green_time ({self.default_green_time}s) exceeds max_green ({max_green}s)")
        if min_green is not None and self.default_green_time < min_green:
            errors.append(f"default_green_time ({self.default_green_time}s) is less than min_green ({min_green}s)")

        for tls_id, durs in self.phase_durations.items():
            for p_idx, dur in durs.items():
                if max_green is not None and dur > max_green:
                    errors.append(f"TLS '{tls_id}' phase {p_idx} duration ({dur}s) exceeds max_green ({max_green}s)")
                if min_green is not None and dur < min_green:
                    errors.append(f"TLS '{tls_id}' phase {p_idx} duration ({dur}s) is less than min_green ({min_green}s)")

        if num_phases_per_tls:
            for tls_id, seq in self.phase_sequences.items():
                n_phases = num_phases_per_tls.get(tls_id, 0)
                for p_idx in seq:
                    if p_idx >= n_phases:
                        errors.append(f"TLS '{tls_id}' phase sequence contains invalid phase index {p_idx}")

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
