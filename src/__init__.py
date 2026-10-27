"""Traffic Signal Control Package."""

from typing import Any

from src.controllers import (
    BaseController,
    FixedTimeController,
    MaxPressureController,
    QLearningController,
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
    """Lazy-load benchmark functions to avoid premature import warnings when running CLI."""
    if name in ("run_benchmark_matrix", "run_single_simulation"):
        from src.benchmark import run_benchmark_matrix, run_single_simulation

        mapping = {
            "run_benchmark_matrix": run_benchmark_matrix,
            "run_single_simulation": run_single_simulation,
        }
        return mapping[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

