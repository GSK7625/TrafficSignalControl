"""Gói điều khiển tín hiệu đèn giao thông (Traffic Signal Control)."""

from typing import Any

from src.controllers import (
    BaseController,
    FixedTimeController,
    MaxPressureController,
    QLearningController,
    build_default_controller,
    create_controller,
    register_controller,
)
from src.data_inspector import inspect_scenario, ScenarioInfo
from src.metrics import MetricsCollector, StepRecord, TripRecord
from src.simulation_runner import SimulationRunner, SimulationResult

__all__ = [
    "BaseController",
    "FixedTimeController",
    "MaxPressureController",
    "QLearningController",
    "build_default_controller",
    "create_controller",
    "register_controller",
    "inspect_scenario",
    "ScenarioInfo",
    "MetricsCollector",
    "StepRecord",
    "TripRecord",
    "SimulationRunner",
    "SimulationResult",
    "run_benchmark_matrix",
    "run_single_simulation",
]


def __getattr__(name: str) -> Any:
    """Lazy-load các hàm benchmark để tránh cảnh báo import sớm khi chạy CLI."""
    if name in ("run_benchmark_matrix", "run_single_simulation"):
        import src.benchmark as bm
        return getattr(bm, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

