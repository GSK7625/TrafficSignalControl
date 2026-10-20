"""Base classes and data models for traffic signal controllers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


@dataclass
class ControlledLink:
    """Represents a controlled movement link at an intersection."""
    signal_index: int
    from_lane: str
    to_lane: str
    via_lane: Optional[str] = None


@dataclass(frozen=True)
class IntersectionTopology:
    """Static network topology for an intersection."""
    tls_id: str
    controlled_links: Tuple[ControlledLink, ...]
    in_lanes: Tuple[str, ...]
    out_lanes: Tuple[str, ...]
    branches: Tuple[str, ...]
    branch_to_lanes: Dict[str, Tuple[str, ...]]
    outs_per_in: Dict[str, Set[str]]


def build_intersection_topology(
    tls_id: str,
    controlled_links: Sequence[ControlledLink],
) -> IntersectionTopology:
    """Builds static topology mapping from controlled links."""
    in_lanes_set: Set[str] = set()
    out_lanes_set: Set[str] = set()
    branch_map: Dict[str, Set[str]] = {}
    outs_map: Dict[str, Set[str]] = {}

    for link in controlled_links:
        from_l = link.from_lane
        to_l = link.to_lane
        if from_l:
            in_lanes_set.add(from_l)
            b = from_l.rsplit("_", 1)[0]
            branch_map.setdefault(b, set()).add(from_l)
            if to_l:
                outs_map.setdefault(from_l, set()).add(to_l)
        if to_l:
            out_lanes_set.add(to_l)

    branches = tuple(sorted(branch_map.keys()))
    branch_to_lanes = {b: tuple(sorted(branch_map[b])) for b in branches}

    return IntersectionTopology(
        tls_id=tls_id,
        controlled_links=tuple(controlled_links),
        in_lanes=tuple(sorted(in_lanes_set)),
        out_lanes=tuple(sorted(out_lanes_set)),
        branches=branches,
        branch_to_lanes=branch_to_lanes,
        outs_per_in=outs_map,
    )


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
    valid_actions: Optional[List[int]] = None
    topology: Optional[IntersectionTopology] = None


class BaseTrafficController(ABC):
    """Abstract base class for all traffic signal control algorithms."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def reset(self) -> None:
        """Resets controller internal states."""
        pass

    @abstractmethod
    def select_action(
        self, observations: Dict[str, IntersectionObservation]
    ) -> Dict[str, int]:
        """Selects target green phase index for each controlled intersection."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Returns controller configuration dictionary."""
        pass
