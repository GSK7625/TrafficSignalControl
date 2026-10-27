"""Lớp cơ sở trừu tượng (Base Controller) và Registry quản lý các thuật toán điều khiển tín hiệu giao thông."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

T = TypeVar("T", bound="BaseController")
CONTROLLER_REGISTRY: Dict[str, Type["BaseController"]] = {}


def get_movement_attr(movement: Any, attr: str, default: Any = None) -> Any:
    """Hàm tiện ích lấy an toàn thuộc tính từ đối tượng luồng chuyển động (movement) hoặc dictionary."""
    if isinstance(movement, dict):
        return movement.get(attr, default)
    return getattr(movement, attr, default)


def build_yellow_state(
    curr_state: str,
    next_state: Optional[str] = None,
    only_changing: bool = False,
) -> str:
    """Tạo trạng thái chuyển tiếp đèn vàng an toàn để giải phóng nút giao trước pha xanh tiếp theo."""
    def _to_yellow(i: int, char: str) -> str:
        if char in ("G", "g"):
            if only_changing and next_state and i < len(next_state) and next_state[i] in ("G", "g"):
                return char
            return "y"
        return char

    return "".join(_to_yellow(i, c) for i, c in enumerate(curr_state))


def build_all_red_state(curr_state: str) -> str:
    """Tạo trạng thái đỏ toàn phần (all-red) để giải phóng nút giao, mọi hướng đều hiển thị đèn đỏ."""
    return "r" * len(curr_state)


def register_controller(name: str) -> Callable[[Type[T]], Type[T]]:
    """Decorator đăng ký một lớp controller vào registry chung."""
    def decorator(cls: Type[T]) -> Type[T]:
        CONTROLLER_REGISTRY[name] = cls
        return cls
    return decorator


def get_controller_class(name: str) -> Type["BaseController"]:
    """Lấy lớp controller đã đăng ký theo tên chuỗi định danh."""
    if name not in CONTROLLER_REGISTRY:
        raise KeyError(
            f"Unknown controller type '{name}'. Registered: {list(CONTROLLER_REGISTRY.keys())}"
        )
    return CONTROLLER_REGISTRY[name]


def create_controller(name: str, **kwargs: Any) -> "BaseController":
    """Hàm factory khởi tạo đối tượng controller theo tên định danh đã đăng ký."""
    cls = get_controller_class(name)
    return cls(**kwargs)


def build_default_controller(
    controller_type: str,
    green_phases: Sequence[Any],
    movements: Optional[Sequence[Any]] = None,
    tls_id: Optional[str] = None,
    min_green: float = 20.0,
    max_green: float = 45.0,
    green_time: float = 30.0,
    yellow_time: float = 3.0,
    all_red_time: float = 1.0,
    max_red: float = 90.0,
    decision_interval: float = 5.0,
    **kwargs: Any,
) -> "BaseController":
    """Factory khởi tạo các controller tiêu chuẩn với các tham số mặc định nhất quán."""
    if controller_type not in CONTROLLER_REGISTRY:
        raise ValueError(
            f"Unsupported controller_type: {controller_type}. Must be one of {list(CONTROLLER_REGISTRY.keys())}."
        )
    cls = CONTROLLER_REGISTRY[controller_type]
    common = {
        "green_phases": green_phases,
        "yellow_time": yellow_time,
        "all_red_time": all_red_time,
        "max_red": max_red,
        "tls_id": tls_id,
        **kwargs,
    }
    if controller_type == "fixed_time":
        return cls(green_time=green_time, **common)
    if controller_type == "max_pressure":
        return cls(movements=movements or [], min_green=min_green, max_green=max_green, **common)
    if controller_type == "q_learning":
        return cls(
            movements=movements or [],
            min_green=min_green,
            max_green=max_green,
            decision_interval=decision_interval,
            **common,
        )
    return cls(**common)


class BaseController(ABC):

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

    # Các thuộc tính chỉ đọc (Read-only properties)
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

    def get_phase_red_time(self, phase_idx: int) -> float:
        if 0 <= phase_idx < len(self._phase_red_timers):
            return self._phase_red_timers[phase_idx]
        return 0.0

    def get_current_state(self) -> str:
        """Trả về chuỗi trạng thái tín hiệu đèn hoạt động tại bước hiện tại."""
        curr = self.green_phases[self._current_phase_idx]
        if self._is_yellow:
            next_idx = self._next_phase_idx if hasattr(self, "_next_phase_idx") else (self._current_phase_idx + 1) % len(self.green_phases)
            return build_yellow_state(curr, self.green_phases[next_idx])
        if self._is_all_red:
            return build_all_red_state(curr)
        return curr

    def _update_red_timers(self, step_duration: float) -> None:
        """Cập nhật bộ đếm thời gian chờ đèn đỏ liên tục cho các pha không hoạt động."""
        for p in range(len(self._phase_red_timers)):
            if p == self._current_phase_idx:
                self._phase_red_timers[p] = 0.0
            else:
                self._phase_red_timers[p] += step_duration

    def _get_vehicle_count(self, lane_id: str) -> int:
        """Lấy số lượng xe trên một làn đường qua hàm tùy biến hoặc libsumo/TraCI fallback."""
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

    def _handle_clearance(self) -> bool:
        """Xử lý các khoảng thời gian dọn nút giao khi đang ở pha vàng hoặc đỏ toàn phần.

        Returns:
            True nếu đang trong pha dọn nút giao (vàng/đỏ toàn phần), False nếu đang trong pha xanh bình thường.
        """
        tolerance = 1e-6
        if self._is_yellow:
            if self._time_in_phase >= self.yellow_time - tolerance:
                self._is_yellow = False
                self._time_in_phase = 0.0
                if self.all_red_time > 0:
                    self._is_all_red = True
                else:
                    self._current_phase_idx = self._next_phase_idx
            return True
        if self._is_all_red:
            if self._time_in_phase >= self.all_red_time - tolerance:
                self._is_all_red = False
                self._time_in_phase = 0.0
                self._current_phase_idx = self._next_phase_idx
            return True
        return False

    def _trigger_switch(self, target_phase: int) -> None:
        """Chuyển pha an toàn sang target_phase có kèm giải phóng pha vàng và đỏ toàn phần."""
        if target_phase == self._current_phase_idx:
            return
        self._next_phase_idx = target_phase
        self._time_in_phase = 0.0
        if self.yellow_time > 0:
            self._is_yellow = True
        elif self.all_red_time > 0:
            self._is_all_red = True
        else:
            self._current_phase_idx = target_phase

    def _advance_timers(self, step_duration: float) -> Tuple[str, bool]:
        """Cập nhật bộ đếm thời gian đỏ, kiểm tra thay đổi trạng thái và tăng thời gian tích lũy."""
        self._update_red_timers(step_duration)
        state = self.get_current_state()
        changed = (self._last_state != state)
        self._last_state = state
        self._time_in_phase += step_duration
        self._total_time += step_duration
        self._step_count += 1
        return state, changed

    def reset(self) -> None:
        """Đặt lại bộ điều khiển về trạng thái ban đầu (pha 0, đèn xanh, thời gian trôi qua = 0)."""
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
        """Tiến bộ điều khiển thêm step_duration giây và trả về (trạng_thái_đèn, cờ_đổi_pha)."""
        raise NotImplementedError
