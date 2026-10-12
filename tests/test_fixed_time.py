"""Unit tests for FixedTimeController."""

import unittest
from src.controllers.base import ControlledLink, IntersectionObservation
from src.controllers.fixed_time import FixedTimeController


def make_mock_obs(
    tls_id: str = "tls_0",
    current_phase: int = 0,
    time_in_phase: float = 0.0,
    is_in_transition: bool = False,
    green_phases: list = None,
    queue: int = 0,
    sim_time: float = 0.0,
    step: int = 0,
) -> IntersectionObservation:
    if green_phases is None:
        green_phases = ["GGGG", "rrrr"]
    return IntersectionObservation(
        tls_id=tls_id,
        current_phase=current_phase,
        time_in_phase=time_in_phase,
        is_in_transition=is_in_transition,
        green_phases=green_phases,
        controlled_links=[],
        in_lane_queues={"lane_0": queue},
        out_lane_queues={"lane_out": 0},
        in_lane_vehicles={"lane_0": queue},
        out_lane_vehicles={"lane_out": 0},
        sim_time=sim_time,
        step=step,
    )


class TestFixedTimeController(unittest.TestCase):
    def test_default_config(self):
        """Verify default configuration has expected parameters."""
        ctrl = FixedTimeController(default_green_time=25.0)
        cfg = ctrl.get_config()
        self.assertEqual(cfg["name"], "FixedTime")
        self.assertEqual(cfg["default_green_time"], 25.0)

    def test_two_phase_cycle_progression(self):
        """Verify 2-phase signal stays green for configured duration then switches."""
        ctrl = FixedTimeController(default_green_time=30.0)
        ctrl.reset()

        # Step at t=0, time_in_phase=0 -> should request phase 0
        obs = make_mock_obs(current_phase=0, time_in_phase=0.0, sim_time=0.0, step=0)
        action = ctrl.select_action({"tls_0": obs})
        self.assertEqual(action["tls_0"], 0)

        # Step at t=20, time_in_phase=20.0 (< 30) -> should keep phase 0
        obs = make_mock_obs(current_phase=0, time_in_phase=20.0, sim_time=20.0, step=20)
        action = ctrl.select_action({"tls_0": obs})
        self.assertEqual(action["tls_0"], 0)

        # Step at t=30, time_in_phase=30.0 (>= 30) -> should request phase 1
        obs = make_mock_obs(current_phase=0, time_in_phase=30.0, sim_time=30.0, step=30)
        action = ctrl.select_action({"tls_0": obs})
        self.assertEqual(action["tls_0"], 1)

        # During yellow transition (time_in_phase resets for yellow, is_in_transition=True)
        obs = make_mock_obs(current_phase=0, time_in_phase=1.0, is_in_transition=True, sim_time=31.0, step=31)
        action = ctrl.select_action({"tls_0": obs})
        self.assertEqual(action["tls_0"], 1)

        # Once target green phase 1 is active
        obs = make_mock_obs(current_phase=1, time_in_phase=0.0, is_in_transition=False, sim_time=33.0, step=33)
        action = ctrl.select_action({"tls_0": obs})
        self.assertEqual(action["tls_0"], 1)

        # Phase 1 for 30s -> should switch back to phase 0
        obs = make_mock_obs(current_phase=1, time_in_phase=30.0, is_in_transition=False, sim_time=63.0, step=63)
        action = ctrl.select_action({"tls_0": obs})
        self.assertEqual(action["tls_0"], 0)

    def test_custom_durations_and_three_phases(self):
        """Verify 3-phase signal (like Ingolstadt1) with custom per-phase durations."""
        # Phase 0: 40s, Phase 1: 15s, Phase 2: 25s
        durations = {"tls_ingolstadt": {0: 40.0, 1: 15.0, 2: 25.0}}
        ctrl = FixedTimeController(phase_durations=durations)
        ctrl.reset()

        green_phases = ["GGgGrGGG", "GGGrrrrr", "rrrGGGrr"]

        # Phase 0: at 39s -> still 0
        obs = make_mock_obs(
            tls_id="tls_ingolstadt",
            current_phase=0,
            time_in_phase=39.0,
            green_phases=green_phases,
        )
        self.assertEqual(ctrl.select_action({"tls_ingolstadt": obs})["tls_ingolstadt"], 0)

        # Phase 0: at 40s -> switches to 1
        obs = make_mock_obs(
            tls_id="tls_ingolstadt",
            current_phase=0,
            time_in_phase=40.0,
            green_phases=green_phases,
        )
        self.assertEqual(ctrl.select_action({"tls_ingolstadt": obs})["tls_ingolstadt"], 1)

        # Phase 1: at 15s -> switches to 2
        obs = make_mock_obs(
            tls_id="tls_ingolstadt",
            current_phase=1,
            time_in_phase=15.0,
            green_phases=green_phases,
        )
        self.assertEqual(ctrl.select_action({"tls_ingolstadt": obs})["tls_ingolstadt"], 2)

        # Phase 2: at 25s -> cycles back to 0
        obs = make_mock_obs(
            tls_id="tls_ingolstadt",
            current_phase=2,
            time_in_phase=25.0,
            green_phases=green_phases,
        )
        self.assertEqual(ctrl.select_action({"tls_ingolstadt": obs})["tls_ingolstadt"], 0)

    def test_independence_from_traffic_volume(self):
        """Verify FixedTime decisions are completely identical regardless of queue size."""
        ctrl1 = FixedTimeController(default_green_time=20.0)
        ctrl2 = FixedTimeController(default_green_time=20.0)

        # Zero vehicles vs 100 queued vehicles
        obs_empty = make_mock_obs(current_phase=0, time_in_phase=10.0, queue=0)
        obs_crowded = make_mock_obs(current_phase=0, time_in_phase=10.0, queue=100)

        a1 = ctrl1.select_action({"tls_0": obs_empty})
        a2 = ctrl2.select_action({"tls_0": obs_crowded})
        self.assertEqual(a1, a2)

    def test_decision_history_logged(self):
        """Verify phase switch decisions are logged to controller history."""
        ctrl = FixedTimeController(default_green_time=10.0)
        ctrl.reset()

        obs1 = make_mock_obs(current_phase=0, time_in_phase=10.0, sim_time=10.0, step=10)
        ctrl.select_action({"tls_0": obs1})

        self.assertGreater(len(ctrl.history), 0)
        last_log = ctrl.history[-1]
        self.assertEqual(last_log["tls_id"], "tls_0")
        self.assertEqual(last_log["action"], 1)
        self.assertEqual(last_log["sim_time"], 10.0)


if __name__ == "__main__":
    unittest.main()
