"""Abstract Base Controller and Registry for Traffic Signal Control Algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

T = TypeVar("T", bound="BaseController")
CONTROLLER_REGISTRY: Dict[str, Type["BaseController"]] = {}


def build_yellow_state(
    curr_state: str,
    next_state: Optional[str] = None,
    only_changing: bool = False,
) -> str:
    """Builds a safe yellow transition state for clearance before the next green phase."""
    def _to_yellow(i: int, char: str) -> str:
        if char in ("G", "g"):
            if only_changing and next_state and i < len(next_state) and next_state[i] in ("G", "g"):
                return char
            return "y"
        return char

    return "".join(_to_yellow(i, c) for i, c in enumerate(curr_state))


def build_all_red_state(curr_state: str) -> str:
    """Builds an all-red clearance state where all signal heads display red."""
    return "r" * len(curr_state)


def register_controller(name: str) -> Callable[[Type[T]], Type[T]]:
    """Decorator to register a controller class in the global controller registry."""
    def decorator(cls: Type[T]) -> Type[T]:
        CONTROLLER_REGISTRY[name] = cls
        return cls
    return decorator


def get_controller_class(name: str) -> Type["BaseController"]:
    """Retrieves a registered controller class by its string name."""
    if name not in CONTROLLER_REGISTRY:
        raise KeyError(
            f"Unknown controller type '{name}'. Registered: {list(CONTROLLER_REGISTRY.keys())}"
        )
    return CONTROLLER_REGISTRY[name]


def create_controller(name: str, **kwargs: Any) -> "BaseController":
    """Factory helper to instantiate a controller by its registered name."""
    cls = get_controller_class(name)
    return cls(**kwargs)


class BaseController(ABC):
    """Abstract Base Class for all traffic signal controllers.

    Encapsulates core cycle lifecycle, timing counters, yellow state transitions,
    and vehicle counting utilities.
    """

    build_yellow_state = staticmethod(build_yellow_state)
    build_all_red_state = staticmethod(build_all_red_state)

    def __init__(
        self,
        green_phases: Sequence[Any],
        yellow_time: float = 3.0,
        all_red_time: float = 1.0,
        max_red: float = 90.0,
        tls_id: Optional[str] = None,
        lane_vehicle_fn: Optional[Callable[[str], int]] = None,
    ) -> None:
        if not green_phases:
            raise ValueError("green_phases must contain at least one phase.")
        if yellow_time < 0:
            raise ValueError(f"yellow_time must be non-negative, got {yellow_time}.")
        if all_red_time < 0:
            raise ValueError(f"all_red_time must be non-negative, got {all_red_time}.")
        if max_red <= 0:
            raise ValueError(f"max_red must be positive, got {max_red}.")

        self.green_phases: List[str] = [
            p.state if hasattr(p, "state") else str(p) for p in green_phases
        ]
        self.yellow_time: float = float(yellow_time)
        self.all_red_time: float = float(all_red_time)
        self.max_red: float = float(max_red)
        self.tls_id: Optional[str] = tls_id
        self._lane_vehicle_fn: Optional[Callable[[str], int]] = lane_vehicle_fn

        self.reset()

    # Read-only properties
    @property
    def num_phases(self) -> int:
        return len(self.green_phases)

    @property
    def current_phase(self) -> int:
        return self._current_phase_idx

    @property
    def is_yellow(self) -> bool:
        return self._is_yellow

    @property
    def is_all_red(self) -> bool:
        return self._is_all_red

    @property
    def phase_red_timers(self) -> List[float]:
        return list(self._phase_red_timers)

    @property
    def time_in_phase(self) -> float:
        return self._time_in_phase

    @property
    def total_time(self) -> float:
        return self._total_time

    @property
    def step_count(self) -> int:
        return self._step_count

    # Helper getters
    def get_current_phase(self) -> int:
        return self._current_phase_idx

    def is_in_yellow(self) -> bool:
        return self._is_yellow

    def is_in_all_red(self) -> bool:
        return self._is_all_red

    def get_phase_red_time(self, phase_idx: int) -> float:
        if 0 <= phase_idx < len(self._phase_red_timers):
            return self._phase_red_timers[phase_idx]
        return 0.0

    def get_current_state(self) -> str:
        """Returns the active signal state string for the current step."""
        curr = self.green_phases[self._current_phase_idx]
        if self._is_yellow:
            next_idx = self._next_phase_idx if hasattr(self, "_next_phase_idx") else (self._current_phase_idx + 1) % len(self.green_phases)
            return build_yellow_state(curr, self.green_phases[next_idx])
        if self._is_all_red:
            return build_all_red_state(curr)
        return curr

    def _update_red_timers(self, step_duration: float) -> None:
        """Updates continuous red wait timers for inactive phases."""
        for p in range(len(self._phase_red_timers)):
            if p == self._current_phase_idx:
                self._phase_red_timers[p] = 0.0
            else:
                self._phase_red_timers[p] += step_duration

    def _get_vehicle_count(self, lane_id: str) -> int:
        """Retrieves vehicle count on a given lane with libsumo/TraCI fallback."""
        if not lane_id:
            return 0
        if self._lane_vehicle_fn is not None:
            try:
                return int(self._lane_vehicle_fn(lane_id))
            except Exception:
                return 0
        try:
            try:
                import libsumo
                if libsumo.isLoaded():
                    return int(libsumo.lane.getLastStepVehicleNumber(lane_id))
            except (ImportError, AttributeError, Exception):
                pass

            import traci
            if traci.isLoaded():
                return int(traci.lane.getLastStepVehicleNumber(lane_id))
        except (ImportError, AttributeError, Exception):
            pass
        return 0

    def reset(self) -> None:
        """Resets controller to initial state (phase 0, green, 0 elapsed time)."""
        self._current_phase_idx: int = 0
        self._next_phase_idx: int = 0
        self._is_yellow: bool = False
        self._is_all_red: bool = False
        self._time_in_phase: float = 0.0
        self._total_time: float = 0.0
        self._step_count: int = 0
        self._last_state: Optional[str] = None
        self._phase_red_timers: List[float] = [0.0] * len(self.green_phases)

    @abstractmethod
    def step(self, step_duration: float = 1.0) -> Tuple[str, bool]:
        """Advances controller by step_duration seconds and returns (state, changed)."""
        raise NotImplementedError
