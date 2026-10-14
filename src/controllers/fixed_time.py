"""Fixed-time traffic signal controller."""

import math
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
    - Automatically resynchronizes schedule if actual phase is forced by runner or safety constraints.
    """

    def __init__(
        self,
        default_green_time: float = 30.0,
        phase_durations: Optional[Dict[str, Dict[int, float]]] = None,
        phase_sequences: Optional[Dict[str, List[int]]] = None,
        offsets: Optional[Dict[str, float]] = None,
        min_green: Optional[float] = None,
        max_green: Optional[float] = None,
    ):
        super().__init__(name="FixedTime")

        # 1. Validate default_green_time
        if (
            isinstance(default_green_time, bool)
            or not isinstance(default_green_time, (int, float))
            or not math.isfinite(default_green_time)
            or default_green_time <= 0
        ):
            raise ValueError(
                f"default_green_time must be positive and finite, got {default_green_time}"
            )

        # 2. Validate phase_durations
        if phase_durations is not None:
            if not isinstance(phase_durations, dict):
                raise ValueError(f"phase_durations must be a dict, got {type(phase_durations)}")
            for tls_id, durs in phase_durations.items():
                if not isinstance(tls_id, str):
                    raise ValueError(f"TLS ID in phase_durations must be a string, got {tls_id}")
                if not isinstance(durs, dict):
                    raise ValueError(f"Durations for TLS '{tls_id}' must be a dict, got {type(durs)}")
                for p_idx, dur in durs.items():
                    if isinstance(p_idx, bool) or not isinstance(p_idx, int) or p_idx < 0:
                        raise ValueError(
                            f"Phase index in phase_durations for TLS '{tls_id}' must be a non-negative integer, got {p_idx}"
                        )
                    if (
                        isinstance(dur, bool)
                        or not isinstance(dur, (int, float))
                        or not math.isfinite(dur)
                        or dur <= 0
                    ):
                        raise ValueError(
                            f"Phase duration for TLS '{tls_id}' phase {p_idx} must be positive and finite, got {dur}"
                        )

        # 3. Validate phase_sequences
        if phase_sequences is not None:
            if not isinstance(phase_sequences, dict):
                raise ValueError(f"phase_sequences must be a dict, got {type(phase_sequences)}")
            for tls_id, seq in phase_sequences.items():
                if not isinstance(tls_id, str):
                    raise ValueError(f"TLS ID in phase_sequences must be a string, got {tls_id}")
                if not isinstance(seq, (list, tuple)) or len(seq) == 0:
                    raise ValueError(
                        f"phase_sequences for TLS '{tls_id}' must be a non-empty sequence of phase indices, got {seq}"
                    )
                for p_idx in seq:
                    if isinstance(p_idx, bool) or not isinstance(p_idx, int) or p_idx < 0:
                        raise ValueError(
                            f"Phase index in phase_sequences for TLS '{tls_id}' must be a non-negative integer, got {p_idx}"
                        )

        # 4. Validate offsets
        if offsets is not None:
            if not isinstance(offsets, dict):
                raise ValueError(f"offsets must be a dict, got {type(offsets)}")
            for tls_id, offset in offsets.items():
                if not isinstance(tls_id, str):
                    raise ValueError(f"TLS ID in offsets must be a string, got {tls_id}")
                if (
                    isinstance(offset, bool)
                    or not isinstance(offset, (int, float))
                    or not math.isfinite(offset)
                    or offset < 0
                ):
                    raise ValueError(
                        f"Offset for TLS '{tls_id}' must be a non-negative finite number, got {offset}"
                    )

        # 5. Validate min_green / max_green constraints if provided
        if min_green is not None:
            if (
                isinstance(min_green, bool)
                or not isinstance(min_green, (int, float))
                or not math.isfinite(min_green)
                or min_green < 0
            ):
                raise ValueError(f"min_green must be non-negative and finite, got {min_green}")
        if max_green is not None:
            if (
                isinstance(max_green, bool)
                or not isinstance(max_green, (int, float))
                or not math.isfinite(max_green)
                or max_green <= 0
            ):
                raise ValueError(f"max_green must be positive and finite, got {max_green}")
            if min_green is not None and max_green < min_green:
                raise ValueError(f"max_green ({max_green}) cannot be less than min_green ({min_green})")

        self.default_green_time = float(default_green_time)
        self.phase_durations: Dict[str, Dict[int, float]] = phase_durations or {}
        self.phase_sequences: Dict[str, List[int]] = phase_sequences or {}
        self.offsets: Dict[str, float] = offsets or {}
        self.min_green: Optional[float] = float(min_green) if min_green is not None else None
        self.max_green: Optional[float] = float(max_green) if max_green is not None else None

        self._current_seq_index: Dict[str, int] = {}
        self._target_phases: Dict[str, int] = {}
        self._offset_remaining: Dict[str, float] = {}
        self._has_started: Dict[str, bool] = {}

        if self.min_green is not None or self.max_green is not None:
            errs = self.validate_compatibility(min_green=self.min_green, max_green=self.max_green)
            if errs:
                raise ValueError(f"Incompatible FixedTimeController configuration: {'; '.join(errs)}")

        self.reset()

    def reset(self) -> None:
        """Resets all internal scheduler states."""
        self.history.clear()
        self._current_seq_index.clear()
        self._target_phases.clear()
        self._offset_remaining = {k: float(v) for k, v in self.offsets.items()}
        self._has_started.clear()

    def validate_compatibility(
        self,
        min_green: Optional[float] = None,
        max_green: Optional[float] = None,
        num_phases_per_tls: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        """Validates controller configuration against safety timing constraints and phase counts.

        Args:
            min_green: Optional minimum green constraint.
            max_green: Optional maximum green constraint.
            num_phases_per_tls: Optional mapping from tls_id to count of available green phases.

        Returns:
            List of compatibility error messages (empty if fully compatible).
        """
        errors: List[str] = []

        # Validate default_green_time
        if max_green is not None and self.default_green_time > max_green:
            errors.append(
                f"default_green_time ({self.default_green_time}s) exceeds max_green ({max_green}s)"
            )
        if min_green is not None and self.default_green_time < min_green:
            errors.append(
                f"default_green_time ({self.default_green_time}s) is less than min_green ({min_green}s)"
            )

        # Validate custom phase_durations
        for tls_id, durs in self.phase_durations.items():
            for p_idx, dur in durs.items():
                if max_green is not None and dur > max_green:
                    errors.append(
                        f"TLS '{tls_id}' phase {p_idx} duration ({dur}s) exceeds max_green ({max_green}s)"
                    )
                if min_green is not None and dur < min_green:
                    errors.append(
                        f"TLS '{tls_id}' phase {p_idx} duration ({dur}s) is less than min_green ({min_green}s)"
                    )

        # Validate initial phase duration + offset
        all_tls_ids = set(self.offsets.keys()) | set(self.phase_sequences.keys()) | set(self.phase_durations.keys())
        for tls_id in all_tls_ids:
            offset = self.offsets.get(tls_id, 0.0)
            seq = self.phase_sequences.get(tls_id, [0])
            first_phase = seq[0] if seq else 0
            base_dur = self._get_duration(tls_id, first_phase)
            effective_initial_dur = base_dur + offset
            if max_green is not None and effective_initial_dur > max_green:
                errors.append(
                    f"TLS '{tls_id}' initial green time with offset ({base_dur}s + {offset}s = {effective_initial_dur}s) "
                    f"exceeds max_green ({max_green}s)"
                )

        # Validate phase_sequences against num_phases_per_tls
        if num_phases_per_tls:
            for tls_id, seq in self.phase_sequences.items():
                if tls_id in num_phases_per_tls:
                    n_phases = num_phases_per_tls[tls_id]
                    for p_idx in seq:
                        if p_idx >= n_phases:
                            errors.append(
                                f"TLS '{tls_id}' phase sequence contains phase index {p_idx}, "
                                f"which is out of bounds for {n_phases} green phases [0..{n_phases - 1}]"
                            )

        return errors

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
            base_dur = self._get_duration(tls_id, curr_phase)
            offset_dur = max(0.0, self._offset_remaining.get(tls_id, 0.0))
            effective_dur = base_dur + offset_dur

            # If the intersection is currently undergoing yellow or all-red clearance,
            # maintain the target phase request until transition completes
            if obs.is_in_transition:
                actions[tls_id] = self._target_phases[tls_id]
                continue

            # Check if active green phase matches schedule and duration is reached
            if obs.current_phase == curr_phase:
                self._has_started[tls_id] = True
                if obs.time_in_phase >= effective_dur and len(seq) > 1:
                    # Time to cycle to next phase
                    self._offset_remaining[tls_id] = 0.0
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
                            "target_duration": base_dur,
                            "offset_applied": offset_dur,
                        },
                    )
                else:
                    # Continue holding current green phase
                    actions[tls_id] = curr_phase
            else:
                # Active phase does not match expected curr_phase
                if self._has_started.get(tls_id, False) and obs.current_phase in seq:
                    # An external or runner-forced transition occurred (e.g. max_green cutoff).
                    # Resynchronize schedule to actual active phase.
                    found_idx = None
                    for offset in range(1, len(seq) + 1):
                        cand_idx = (curr_idx + offset) % len(seq)
                        if seq[cand_idx] == obs.current_phase:
                            found_idx = cand_idx
                            break
                    if found_idx is None:
                        found_idx = seq.index(obs.current_phase)

                    prev_target = self._target_phases[tls_id]
                    self._current_seq_index[tls_id] = found_idx
                    self._target_phases[tls_id] = obs.current_phase
                    self._offset_remaining[tls_id] = 0.0

                    curr_idx = found_idx
                    curr_phase = obs.current_phase
                    base_dur = self._get_duration(tls_id, curr_phase)

                    self.log_decision(
                        tls_id=tls_id,
                        sim_time=obs.sim_time,
                        step=obs.step,
                        action=curr_phase,
                        details={
                            "event": "resynchronize_schedule",
                            "reason": "external_or_forced_phase_switch",
                            "actual_phase": obs.current_phase,
                            "previous_target": prev_target,
                            "new_seq_index": found_idx,
                        },
                    )

                    # Evaluate elapsed time on the resynchronized phase
                    if obs.time_in_phase >= base_dur and len(seq) > 1:
                        next_idx = (curr_idx + 1) % len(seq)
                        next_phase = seq[next_idx]
                        self._current_seq_index[tls_id] = next_idx
                        self._target_phases[tls_id] = next_phase
                        actions[tls_id] = next_phase
                    else:
                        actions[tls_id] = curr_phase
                else:
                    # Initial synchronization: request target phase to bring signal into sequence
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
            "min_green": self.min_green,
            "max_green": self.max_green,
        }

