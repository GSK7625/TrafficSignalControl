"""Traffic Light Phase Manager (SafePhaseManager).

Quản lý chuyển pha an toàn: đảm bảo min_green, thời gian đèn vàng dọn nút (yellow clearance),
và toàn đỏ (all-red) nếu có.
"""

from enum import Enum
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import DEFAULT_SAFETY_CONFIG


class PhaseState(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ALL_RED = "ALL_RED"


def _yellow_char(c: str, t: str) -> str:
    if c in "Ggs" and (t in "ruOo" or (c in "Gg" and t == "s")):
        return "y"
    if c in "Gg" and t in "Gg":
        return "g" if "g" in (c, t) else "G"
    return c if c in "sy" else "r"


def build_yellow_state(current_state: str, target_state: str) -> str:
    """Tạo trạng thái đèn vàng chuyển tiếp an toàn giữa 2 pha."""
    if len(current_state) != len(target_state):
        raise ValueError(
            f"State lengths do not match: current({len(current_state)}) vs target({len(target_state)})"
        )
    return "".join(_yellow_char(c, t) for c, t in zip(current_state, target_state))


class StarvationWatchdog:
    """Monitors incoming branches for unserved queues and prevents starvation."""

    def __init__(
        self,
        green_phases: List[str],
        controlled_links: Optional[List[Any]] = None,
        max_red: float = 90.0,
    ):
        if not math.isfinite(max_red) or max_red <= 0:
            raise ValueError(f"max_red must be positive and finite, got {max_red}")
        self.green_phases = list(green_phases)
        self.max_red = float(max_red)
        self.controlled_links: List[Any] = []
        self.branches: List[str] = []
        self.branch_green_phases: Dict[str, Set[int]] = {}
        self.branch_red_timers: Dict[str, float] = {}
        self.in_lane_queues: Dict[str, float] = {}
        self._queues_observed = False
        self._link_items: List[Tuple[str, int]] = []

        if controlled_links:
            self.configure_links(controlled_links)
        else:
            self._setup_fallback_branches()

    def _setup_fallback_branches(self) -> None:
        self.branches = [f"phase_{i}" for i in range(len(self.green_phases))]
        self.branch_green_phases = {b: {i} for i, b in enumerate(self.branches)}
        self.branch_red_timers = {b: 0.0 for b in self.branches}
        self._link_items = []

    def configure_links(self, controlled_links: List[Any]) -> None:
        self.controlled_links = list(controlled_links)
        self._link_items = []
        branches_set = set()
        for link in controlled_links:
            lane = getattr(link, "from_lane", None)
            s_idx = getattr(link, "signal_index", -1)
            if lane:
                b = lane.rsplit("_", 1)[0]
                branches_set.add(b)
                if s_idx >= 0:
                    self._link_items.append((b, s_idx))

        self.branches = sorted(list(branches_set))
        if not self.branches:
            self._setup_fallback_branches()
            return

        self.branch_green_phases = {b: set() for b in self.branches}
        for p_idx, p_str in enumerate(self.green_phases):
            for b, s_idx in self._link_items:
                if s_idx < len(p_str) and p_str[s_idx] in ("G", "g", "s"):
                    self.branch_green_phases[b].add(p_idx)

        self.branch_red_timers = {b: self.branch_red_timers.get(b, 0.0) for b in self.branches}

    def update_queues(self, in_lane_queues: Dict[str, float]) -> None:
        self.in_lane_queues = dict(in_lane_queues)
        self._queues_observed = True

    def update(self, step_duration: float, current_state: str, is_in_transition: bool) -> None:
        active_green_branches = set()
        if not is_in_transition:
            if self._link_items:
                for b, s_idx in self._link_items:
                    if s_idx < len(current_state) and current_state[s_idx] in ("G", "g", "s"):
                        active_green_branches.add(b)
            elif current_state in self.green_phases:
                curr_p = self.green_phases.index(current_state)
                active_green_branches = {b for b, p_set in self.branch_green_phases.items() if curr_p in p_set}

        for b in self.branches:
            self.branch_red_timers[b] = 0.0 if b in active_green_branches else self.branch_red_timers[b] + step_duration

    def get_starving_branches(self) -> List[Tuple[float, str]]:
        is_fallback = not self.controlled_links
        branch_queues: Dict[str, float] = {}
        for lane, q in self.in_lane_queues.items():
            b = lane.rsplit("_", 1)[0]
            branch_queues[b] = branch_queues.get(b, 0.0) + q

        starving = []
        for b in self.branches:
            b_q = sum(self.in_lane_queues.values()) if (is_fallback and b.startswith("phase_")) else branch_queues.get(b, 0.0)
            has_queue = b_q > 0 if self._queues_observed else True
            red_time = self.branch_red_timers.get(b, 0.0)
            if red_time >= self.max_red and has_queue:
                starving.append((red_time, b))
        starving.sort(reverse=True)
        return starving

    def has_starvation(self) -> bool:
        return bool(self.get_starving_branches())

    def get_target_phase(self, valid_actions: List[int]) -> Optional[int]:
        filtered = self.filter_valid_actions(valid_actions)
        return filtered[0] if filtered else None

    def filter_valid_actions(self, valid_actions: List[int]) -> List[int]:
        if not valid_actions:
            return valid_actions
        for _, b in self.get_starving_branches():
            serving = self.branch_green_phases.get(b, set())
            matching = [p for p in valid_actions if p in serving]
            if matching:
                return matching
        return valid_actions


class SafePhaseManager:
    """Quản lý chu kỳ chuyển pha cho từng nút giao thông."""

    def __init__(
        self,
        tls_id: str,
        green_phases: List[str],
        yellow_duration: float = DEFAULT_SAFETY_CONFIG["yellow_duration"],
        all_red_duration: float = 0.0,
        min_green: float = DEFAULT_SAFETY_CONFIG["min_green"],
        max_green: Optional[float] = DEFAULT_SAFETY_CONFIG["max_green"],
        max_red: Optional[float] = None,
        initial_phase: int = 0,
    ):
        if not green_phases:
            raise ValueError("green_phases cannot be empty")
        if any(not phase or len(phase) != len(green_phases[0]) for phase in green_phases):
            raise ValueError("green_phases must have matching non-empty signal lengths")
        if not math.isfinite(yellow_duration) or yellow_duration <= 0:
            raise ValueError("yellow_duration must be positive and finite")
        if not math.isfinite(all_red_duration) or all_red_duration < 0:
            raise ValueError("all_red_duration must be non-negative and finite")
        if not math.isfinite(min_green) or min_green < 0:
            raise ValueError(f"min_green must be non-negative and finite, got {min_green}")
        if max_green is not None:
            if not math.isfinite(max_green):
                raise ValueError(f"max_green must be finite, got {max_green}")
            if max_green <= 0:
                raise ValueError(f"max_green must be positive, got {max_green}")
            if max_green < min_green:
                raise ValueError(f"max_green must be >= min_green ({max_green} < {min_green})")
        if max_red is not None:
            if not math.isfinite(max_red):
                raise ValueError(f"max_red must be finite, got {max_red}")
            if max_red <= 0:
                raise ValueError(f"max_red must be positive, got {max_red}")
        if not (0 <= initial_phase < len(green_phases)):
            raise IndexError("initial_phase out of bounds")

        self.tls_id = tls_id
        self.green_phases = list(green_phases)
        self.yellow_duration = float(yellow_duration)
        self.all_red_duration = float(all_red_duration)
        self.min_green = float(min_green)
        self.max_green = float(max_green) if max_green is not None else None
        self.max_red = float(max_red) if max_red is not None else None
        self.watchdog: Optional[StarvationWatchdog] = (
            StarvationWatchdog(green_phases=self.green_phases, max_red=self.max_red)
            if self.max_red is not None
            else None
        )

        self._current_phase: int = initial_phase
        self._target_phase: int = initial_phase
        self._state_type: PhaseState = PhaseState.GREEN
        self._current_signal_string: str = self.green_phases[initial_phase]
        self._time_in_phase: float = 0.0
        self._yellow_time_remaining: float = 0.0
        self._all_red_time_remaining: float = 0.0

        # Thống kê hoạt động
        self.phase_switch_count: int = 0
        self.phase_durations: dict[int, float] = {i: 0.0 for i in range(len(green_phases))}
        self.total_transition_time: float = 0.0
        self.actual_yellow_time: float = 0.0

    def get_current_phase(self) -> int:
        return self._current_phase

    def get_target_phase(self) -> int:
        return self._target_phase

    def get_current_state(self) -> str:
        return self._current_signal_string

    def is_in_transition(self) -> bool:
        return self._state_type != PhaseState.GREEN

    def time_in_current_phase(self) -> float:
        return self._time_in_phase

    def is_max_green_exceeded(self) -> bool:
        return (
            self.max_green is not None
            and self._state_type == PhaseState.GREEN
            and self._time_in_phase >= self.max_green
        )

    def configure_links(self, controlled_links: List[Any]) -> None:
        if self.watchdog is not None:
            self.watchdog.configure_links(controlled_links)

    def update_queues(self, in_lane_queues: Dict[str, float]) -> None:
        if self.watchdog is not None:
            self.watchdog.update_queues(in_lane_queues)

    def has_starvation(self) -> bool:
        return self.watchdog is not None and self.watchdog.has_starvation()

    def valid_actions(self) -> List[int]:
        if self.is_in_transition():
            return [self._target_phase]
        if self._time_in_phase < self.min_green:
            return [self._current_phase]
        actions = list(range(len(self.green_phases)))
        if self.is_max_green_exceeded() and len(actions) > 1:
            actions.remove(self._current_phase)
        return self.watchdog.filter_valid_actions(actions) if self.watchdog else actions

    def request_phase(self, target_phase_idx: int) -> bool:
        if not (0 <= target_phase_idx < len(self.green_phases)):
            raise IndexError(f"Target phase index {target_phase_idx} out of range")
        if self.is_in_transition() or target_phase_idx not in self.valid_actions():
            return False
        if target_phase_idx == self._current_phase:
            return True

        self._target_phase = target_phase_idx
        curr_state = self.green_phases[self._current_phase]
        targ_state = (
            "r" * len(curr_state)
            if self.all_red_duration > 1e-6
            else self.green_phases[target_phase_idx]
        )
        self._current_signal_string = build_yellow_state(curr_state, targ_state)
        self._state_type = PhaseState.YELLOW
        self._yellow_time_remaining = self.yellow_duration
        self._all_red_time_remaining = 0.0
        self._time_in_phase = 0.0
        self.phase_switch_count += 1
        return True

    def enforce_constraints(self) -> None:
        if not self.is_in_transition():
            actions = self.valid_actions()
            if self._current_phase not in actions:
                self.request_phase(actions[0])

    def update(self, step_duration: float = 1.0, traci_module=None) -> None:
        if not math.isfinite(step_duration) or step_duration <= 0:
            raise ValueError("step_duration must be positive and finite")
        if self.watchdog is not None:
            self.watchdog.update(step_duration, self._current_signal_string, self.is_in_transition())
        self._time_in_phase += step_duration

        if self._state_type == PhaseState.YELLOW:
            self.total_transition_time += step_duration
            self.actual_yellow_time += step_duration
            self._yellow_time_remaining -= step_duration
            if self._yellow_time_remaining <= 1e-6:
                if self.all_red_duration > 1e-6:
                    self._state_type = PhaseState.ALL_RED
                    self._current_signal_string = "r" * len(self.green_phases[self._target_phase])
                    self._all_red_time_remaining = self.all_red_duration
                    self._time_in_phase = 0.0
                else:
                    self._activate_green(self._target_phase)
        elif self._state_type == PhaseState.ALL_RED:
            self.total_transition_time += step_duration
            self._all_red_time_remaining -= step_duration
            if self._all_red_time_remaining <= 1e-6:
                self._activate_green(self._target_phase)
        else:
            self.phase_durations[self._current_phase] = (
                self.phase_durations.get(self._current_phase, 0.0) + step_duration
            )

        if traci_module is not None:
            self.apply_to_traci(traci_module)

    def _activate_green(self, phase_idx: int) -> None:
        self._current_phase = phase_idx
        self._current_signal_string = self.green_phases[phase_idx]
        self._state_type = PhaseState.GREEN
        self._time_in_phase = 0.0

    def sync_external_phase(
        self,
        phase_index: Optional[int] = None,
        signal_state: Optional[str] = None,
        step_duration: float = 1.0,
    ) -> None:
        if signal_state is not None:
            self._current_signal_string = signal_state
            if signal_state in self.green_phases:
                idx = self.green_phases.index(signal_state)
            else:
                self.total_transition_time += step_duration
                self.actual_yellow_time += step_duration if any(c in signal_state for c in ("y", "Y")) else 0.0
                self._state_type = PhaseState.YELLOW if any(c in signal_state for c in ("y", "Y")) else PhaseState.ALL_RED
                self._time_in_phase += step_duration
                return
        elif phase_index is not None and 0 <= phase_index < len(self.green_phases):
            idx = phase_index
            self._current_signal_string = self.green_phases[idx]
        else:
            self._state_type = PhaseState.YELLOW
            self.total_transition_time += step_duration
            self.actual_yellow_time += step_duration
            self._time_in_phase += step_duration
            return

        if idx != self._current_phase:
            self._current_phase = self._target_phase = idx
            self._time_in_phase = 0.0
            self.phase_switch_count += 1
        elif self._state_type != PhaseState.GREEN:
            self._time_in_phase = 0.0

        self._state_type = PhaseState.GREEN
        self._time_in_phase += step_duration
        self.phase_durations[idx] = self.phase_durations.get(idx, 0.0) + step_duration

    def apply_to_traci(self, traci_module) -> None:
        try:
            traci_module.trafficlight.setRedYellowGreenState(
                self.tls_id, self._current_signal_string
            )
        except Exception as e:
            raise RuntimeError(f"Failed to set state for TLS '{self.tls_id}': {e}") from e
