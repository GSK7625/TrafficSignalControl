"""Unit tests for BaseTrafficController and observation abstractions."""

import unittest
from typing import Dict, Any

from src.controllers.base import (
    BaseTrafficController,
    ControlledLink,
    IntersectionObservation,
)


class DummyController(BaseTrafficController):
    """Simple test controller implementing required methods."""

    def __init__(self, target_phase: int = 1):
        super().__init__(name="DummyController")
        self.target_phase = target_phase
        self.reset_called = False

    def reset(self) -> None:
        self.reset_called = True
        self.history.clear()

    def select_action(self, observations: Dict[str, IntersectionObservation]) -> Dict[str, int]:
        actions = {}
        for tls_id, obs in observations.items():
            actions[tls_id] = self.target_phase
            self.log_decision(
                tls_id=tls_id,
                sim_time=obs.sim_time,
                step=obs.step,
                action=self.target_phase,
                details={"prev_phase": obs.current_phase},
            )
        return actions

    def get_config(self) -> Dict[str, Any]:
        return {"name": self.name, "target_phase": self.target_phase}


class TestControllerBase(unittest.TestCase):
    def test_abstract_base_class_instantiation(self):
        """Verify BaseTrafficController cannot be instantiated directly without implementations."""
        with self.assertRaises(TypeError):
            BaseTrafficController(name="Abstract")  # type: ignore

    def test_dummy_controller_lifecycle(self):
        """Verify concrete controller responds to reset, select_action, and logs decisions."""
        controller = DummyController(target_phase=2)
        self.assertEqual(controller.name, "DummyController")
        self.assertEqual(controller.get_config()["target_phase"], 2)

        controller.reset()
        self.assertTrue(controller.reset_called)
        self.assertEqual(len(controller.history), 0)

        # Create mock observation
        link = ControlledLink(signal_index=0, from_lane="in_0", to_lane="out_0")
        obs = IntersectionObservation(
            tls_id="tls_1",
            current_phase=0,
            time_in_phase=10.0,
            is_in_transition=False,
            green_phases=["GGG", "rrr"],
            controlled_links=[link],
            in_lane_queues={"in_0": 5},
            out_lane_queues={"out_0": 0},
            in_lane_vehicles={"in_0": 5},
            out_lane_vehicles={"out_0": 1},
            sim_time=100.0,
            step=10,
        )

        actions = controller.select_action({"tls_1": obs})
        self.assertEqual(actions, {"tls_1": 2})
        self.assertEqual(len(controller.history), 1)
        self.assertEqual(controller.history[0]["tls_id"], "tls_1")
        self.assertEqual(controller.history[0]["action"], 2)
        self.assertEqual(controller.history[0]["sim_time"], 100.0)


if __name__ == "__main__":
    unittest.main()
