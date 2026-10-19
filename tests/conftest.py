"""Shared pytest fixtures and test data builders for TrafficSignalControl tests."""

from unittest.mock import MagicMock
import pytest

from src.controllers.base import ControlledLink, IntersectionObservation


@pytest.fixture
def mock_traci():
    """Returns a pre-configured MagicMock TraCI module with safe defaults."""
    backend = MagicMock()
    backend.simulation.getDepartedIDList.return_value = []
    backend.simulation.getArrivedIDList.return_value = []
    backend.simulation.getPendingVehicles.return_value = []
    backend.simulation.getTime.return_value = 0.0
    backend.simulation.getStartingTeleportNumber.return_value = 0
    backend.simulation.getEndingTeleportNumber.return_value = 0
    backend.simulation.getCollisions.return_value = []
    backend.vehicle.getIDList.return_value = []
    backend.lane.getLastStepHaltingNumber.return_value = 0
    return backend


@pytest.fixture
def sample_observation():
    """Standard 2-phase IntersectionObservation fixture."""
    links = [
        ControlledLink(signal_index=0, from_lane="lane_N", to_lane="lane_S"),
        ControlledLink(signal_index=1, from_lane="lane_E", to_lane="lane_W"),
    ]
    return IntersectionObservation(
        tls_id="tls_test",
        current_phase=0,
        time_in_phase=10.0,
        is_in_transition=False,
        green_phases=["Gr", "rG"],
        controlled_links=links,
        in_lane_queues={"lane_N": 3.0, "lane_E": 8.0},
        out_lane_queues={"lane_S": 0.0, "lane_W": 0.0},
        in_lane_vehicles={"lane_N": 4.0, "lane_E": 10.0},
        out_lane_vehicles={"lane_S": 1.0, "lane_W": 2.0},
        sim_time=100.0,
        step=100,
    )


@pytest.fixture
def make_observation():
    """Factory fixture returning a callable to build IntersectionObservation objects."""

    def _make(
        tls_id: str = "tls_1",
        current_phase: int = 0,
        time_in_phase: float = 10.0,
        is_in_transition: bool = False,
        green_phases: list = None,
        controlled_links: list = None,
        in_lane_queues: dict = None,
        out_lane_queues: dict = None,
        sim_time: float = 100.0,
        step: int = 100,
        valid_actions: list = None,
    ) -> IntersectionObservation:
        if green_phases is None:
            green_phases = ["GGGrrr", "rrrGGG"]
        if controlled_links is None:
            controlled_links = [
                ControlledLink(signal_index=0, from_lane="in_0", to_lane="out_0"),
                ControlledLink(signal_index=1, from_lane="in_0", to_lane="out_1"),
                ControlledLink(signal_index=2, from_lane="in_1", to_lane="out_2"),
                ControlledLink(signal_index=3, from_lane="in_2", to_lane="out_3"),
                ControlledLink(signal_index=4, from_lane="in_2", to_lane="out_4"),
                ControlledLink(signal_index=5, from_lane="in_3", to_lane="out_5"),
            ]
        if in_lane_queues is None:
            in_lane_queues = {"in_0": 0.0, "in_1": 0.0, "in_2": 0.0, "in_3": 0.0}
        if out_lane_queues is None:
            out_lane_queues = {
                "out_0": 0.0,
                "out_1": 0.0,
                "out_2": 0.0,
                "out_3": 0.0,
                "out_4": 0.0,
                "out_5": 0.0,
            }

        return IntersectionObservation(
            tls_id=tls_id,
            current_phase=current_phase,
            time_in_phase=time_in_phase,
            is_in_transition=is_in_transition,
            green_phases=green_phases,
            controlled_links=controlled_links,
            in_lane_queues=in_lane_queues,
            out_lane_queues=out_lane_queues,
            in_lane_vehicles={k: v for k, v in in_lane_queues.items()},
            out_lane_vehicles={k: v for k, v in out_lane_queues.items()},
            sim_time=sim_time,
            step=step,
            valid_actions=valid_actions,
        )

    return _make
