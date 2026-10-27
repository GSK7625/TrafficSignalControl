"""Kiểm thử đơn vị cho BaseController ABC và Controller Registry."""

import pytest
from typing import Tuple
from src.controllers.base import (
    BaseController,
    build_all_red_state,
    register_controller,
    create_controller,
    get_controller_class,
    CONTROLLER_REGISTRY,
)


def test_base_controller_cannot_be_instantiated():
    """BaseController là lớp trừu tượng và không thể khởi tạo trực tiếp."""
    with pytest.raises(TypeError):
        BaseController(green_phases=["Gr", "rG"])  # type: ignore[abstract]


class DummyController(BaseController):
    """Bộ điều khiển cụ thể tối thiểu để kiểm thử BaseController."""

    def step(self, step_duration: float = 1.0) -> Tuple[str, bool]:
        self._time_in_phase += step_duration
        self._total_time += step_duration
        self._step_count += 1
        return self.get_current_state(), False


def test_concrete_controller_properties_and_lifecycle():
    """Kiểm thử vòng đời chuẩn, bộ đếm thời gian, các thuộc tính và hàm reset."""
    ctrl = DummyController(
        green_phases=["GGGrrr", "rrrGGG"],
        yellow_time=3.0,
        tls_id="tls_test",
    )

    assert ctrl.num_phases == 2
    assert ctrl.current_phase == 0
    assert ctrl.is_yellow is False
    assert ctrl.time_in_phase == 0.0
    assert ctrl.total_time == 0.0
    assert ctrl.step_count == 0
    assert ctrl.get_current_state() == "GGGrrr"

    state, changed = ctrl.step(1.5)
    assert state == "GGGrrr"
    assert changed is False
    assert ctrl.time_in_phase == 1.5
    assert ctrl.total_time == 1.5
    assert ctrl.step_count == 1

    ctrl.reset()
    assert ctrl.time_in_phase == 0.0
    assert ctrl.total_time == 0.0
    assert ctrl.step_count == 0


def test_base_controller_validation():
    """Kiểm thử kiểm tra tính hợp lệ của tham số đầu vào trong BaseController."""
    with pytest.raises(ValueError, match="at least one phase"):
        DummyController(green_phases=[])

    with pytest.raises(ValueError, match="non-negative"):
        DummyController(green_phases=["GG", "rr"], yellow_time=-1.0)


def test_get_vehicle_count():
    """Kiểm thử đếm số lượng xe qua lane_vehicle_fn tùy chỉnh và cơ chế dự phòng."""
    # Bộ đếm xe tùy chỉnh
    ctrl = DummyController(
        green_phases=["GG"],
        lane_vehicle_fn=lambda lane_id: 5 if lane_id == "lane_1" else 0,
    )
    assert ctrl._get_vehicle_count("lane_1") == 5
    assert ctrl._get_vehicle_count("lane_2") == 0
    assert ctrl._get_vehicle_count("") == 0

    # Hàm đếm xe gặp lỗi sẽ mặc định trả về 0
    def failing_fn(lane_id: str) -> int:
        raise RuntimeError("Sensor offline")

    ctrl_fail = DummyController(green_phases=["GG"], lane_vehicle_fn=failing_fn)
    assert ctrl_fail._get_vehicle_count("lane_1") == 0


def test_controller_registry():
    """Kiểm thử đăng ký và tạo động các bộ điều khiển."""

    @register_controller("dummy_registered")
    class RegisteredDummy(BaseController):
        def __init__(self, green_phases, custom_arg: str = "default", **kwargs):
            super().__init__(green_phases=green_phases, **kwargs)
            self.custom_arg = custom_arg

        def step(self, step_duration: float = 1.0) -> Tuple[str, bool]:
            return self.get_current_state(), False

    assert "dummy_registered" in CONTROLLER_REGISTRY
    assert get_controller_class("dummy_registered") is RegisteredDummy

    instance = create_controller(
        "dummy_registered",
        green_phases=["GG", "rr"],
        custom_arg="hello",
        tls_id="t1",
    )
    assert isinstance(instance, RegisteredDummy)
    assert instance.custom_arg == "hello"
    assert instance.tls_id == "t1"

    with pytest.raises(KeyError, match="Unknown controller type"):
        get_controller_class("non_existent_controller")


def test_build_all_red_state():
    """Xác nhận build_all_red_state trả về chuỗi giải phóng toàn đỏ."""
    assert build_all_red_state("GGGrrr") == "rrrrrr"
    assert build_all_red_state("yyyrrr") == "rrrrrr"
    assert build_all_red_state("") == ""


def test_base_controller_all_red_and_max_red_defaults():
    """Xác nhận giá trị mặc định và xác thực cho all_red_time và max_red."""
    ctrl = DummyController(green_phases=["GG", "rr"])
    assert ctrl.all_red_time == 1.0
    assert ctrl.max_red == 90.0
    assert ctrl.is_all_red is False
    assert len(ctrl.phase_red_timers) == 2
    assert ctrl.phase_red_timers == [0.0, 0.0]

    with pytest.raises(ValueError, match="all_red_time must be non-negative"):
        DummyController(green_phases=["GG", "rr"], all_red_time=-0.5)

    with pytest.raises(ValueError, match="max_red must be positive"):
        DummyController(green_phases=["GG", "rr"], max_red=-10.0)

