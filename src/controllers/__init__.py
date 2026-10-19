import json
import os
from typing import Any, Optional, Tuple

from src.config import DEFAULT_SAFETY_CONFIG
from src.controllers.base import (
    BaseTrafficController,
    ControlledLink,
    IntersectionObservation,
)
from src.controllers.fixed_time import FixedTimeController
from src.controllers.max_pressure import MaxPressureController, compute_phase_pressure


def algorithm_key(name: str) -> str:
    """Normalizes display names and CLI aliases for all controller entry points."""
    cleaned = name.lower().strip().replace("_", "-").replace(" ", "-")
    aliases = {
        "ft": "fixed-time", "fixedtime": "fixed-time",
        "mp": "max-pressure", "maxpressure": "max-pressure",
        "ql": "q-learning", "qlearning": "q-learning", "original": "none",
    }
    return aliases.get(cleaned, cleaned)


def resolve_q_learning_checkpoint(
    dataset_name: str,
    model_path: Optional[str] = None,
    log_dir: str = "results/training",
    model_dir: str = "models/q_learning",
) -> str:
    """Prefers an explicit checkpoint, then best, then final, returning the expected best path if absent.

    Validates candidate checkpoints against latest training log to avoid using stale historical best checkpoints.
    """
    if model_path:
        return model_path

    best_candidate = f"{model_dir}/q_learning_{dataset_name}_best.json"
    final_candidate = f"{model_dir}/q_learning_{dataset_name}_final.json"
    candidates = [best_candidate, final_candidate]

    log_file = f"{log_dir}/q_learning_{dataset_name}_log.json"
    if os.path.isfile(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            if log_data.get("best_is_viable") is False:
                if os.path.isfile(final_candidate):
                    return final_candidate
                elif os.path.isfile(best_candidate):
                    raise ValueError(
                        f"Latest training log for '{dataset_name}' indicates no viable best checkpoint was found. "
                        f"Historical checkpoint '{best_candidate}' exists from a previous run; please specify model_path explicitly."
                    )
            elif log_data.get("best_is_viable") is True:
                best_path = log_data.get("best_checkpoint_path") or best_candidate
                if os.path.isfile(best_path):
                    return best_path
        except (json.JSONDecodeError, OSError):
            pass

    return next((path for path in candidates if os.path.isfile(path)), candidates[0])


def make_controller(
    algorithm: str,
    dataset_name: str = "cologne1",
    scenario_info: Optional[Any] = None,
    model_path: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Optional[BaseTrafficController], str]:
    """Factory creating traffic signal controllers by algorithm name.

    Returns:
        (controller_instance, control_mode)
    """
    algo_key = algorithm_key(algorithm)

    if algo_key == "fixed-time":
        default_green = float(kwargs.get("default_green_time", kwargs.get("fixed_time_green", 30.0)))
        return FixedTimeController(
            default_green_time=default_green,
            phase_durations=kwargs.get("phase_durations"),
            phase_sequences=kwargs.get("phase_sequences"),
            min_green=kwargs.get("min_green"),
            max_green=kwargs.get("max_green"),
        ), "agent"

    if algo_key == "max-pressure":
        min_green = float(kwargs.get("min_green", DEFAULT_SAFETY_CONFIG["min_green"]))
        max_green = kwargs.get("max_green", DEFAULT_SAFETY_CONFIG["max_green"])
        metric = kwargs.get("pressure_metric", "queue")
        return MaxPressureController(
            min_green=min_green,
            max_green=max_green,
            pressure_metric=metric,
        ), "agent"

    if algo_key == "q-learning":
        from src.controllers.q_learning import QLearningController
        target_path = resolve_q_learning_checkpoint(dataset_name, model_path)
        allowed = (*DEFAULT_SAFETY_CONFIG, "decision_interval", "target_tls_id")
        ctrl_kwargs = {k: kwargs[k] for k in allowed if k in kwargs}

        ctrl = QLearningController.from_checkpoint(
            target_path, eval_mode=kwargs.get("eval_mode", True), seed=kwargs.get("seed"), **ctrl_kwargs,
        )
        return ctrl, "agent"

    if algo_key == "none":
        return None, "original"

    raise ValueError(f"Unsupported algorithm: '{algorithm}'. Supported: 'fixed-time', 'max-pressure', 'q-learning', 'none'.")


def __getattr__(name: str):
    if name == "QLearningController":
        from src.controllers.q_learning import QLearningController
        return QLearningController
    raise AttributeError(name)


__all__ = [
    "BaseTrafficController",
    "ControlledLink",
    "IntersectionObservation",
    "FixedTimeController",
    "MaxPressureController",
    "QLearningController",
    "compute_phase_pressure",
    "make_controller",
    "algorithm_key",
    "resolve_q_learning_checkpoint",
]
