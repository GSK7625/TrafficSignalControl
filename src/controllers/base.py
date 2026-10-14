"""Base classes and data models for traffic signal controllers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ControlledLink:
    """Represents a controlled movement link at an intersection."""
    signal_index: int
    from_lane: str
    to_lane: str
    via_lane: Optional[str] = None


@dataclass
class IntersectionObservation:
    """State observation for a single traffic light intersection."""
    tls_id: str
    current_phase: int
    time_in_phase: float
    is_in_transition: bool
    green_phases: List[str]
    controlled_links: List[ControlledLink]
    in_lane_queues: Dict[str, float]
    out_lane_queues: Dict[str, float]
    in_lane_vehicles: Dict[str, float]
    out_lane_vehicles: Dict[str, float]
    sim_time: float
    step: int
    pending_vehicles: int = 0


class BaseTrafficController(ABC):
    """Abstract base class for all traffic signal control algorithms."""

    def __init__(self, name: str):
        self.name = name
        self.history: List[Dict[str, Any]] = []

    @abstractmethod
    def reset(self) -> None:
        """Resets controller internal states (e.g. at the start of a simulation run)."""
        pass

    @abstractmethod
    def select_action(
        self, observations: Dict[str, IntersectionObservation]
    ) -> Dict[str, int]:
        """Selects target green phase index for each controlled intersection.

        Args:
            observations: Dictionary mapping tls_id to IntersectionObservation.

        Returns:
            Dictionary mapping tls_id to requested target green phase index.
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Returns algorithm configuration parameters and hyper-parameters."""
        pass

    def log_decision(
        self,
        tls_id: str,
        sim_time: float,
        step: int,
        action: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records a control decision for post-run analysis and audit trail."""
        record = {
            "tls_id": tls_id,
            "sim_time": sim_time,
            "step": step,
            "action": action,
            "details": details or {},
        }
        self.history.append(record)
