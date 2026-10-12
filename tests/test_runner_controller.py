"""Tests verifying SimulationRunner integration with BaseTrafficController."""

import unittest
from unittest.mock import MagicMock, patch

from src.config import get_scenario_config
from src.controllers.base import BaseTrafficController, ControlledLink, IntersectionObservation
from src.runner import SimulationRunner


class MockController(BaseTrafficController):
    def __init__(self, target_phase: int = 1):
        super().__init__(name="MockController")
        self.target_phase = target_phase
        self.actions_selected = 0

    def reset(self) -> None:
        self.actions_selected = 0
        self.history.clear()

    def select_action(self, observations):
        self.actions_selected += 1
        actions = {}
        for tls_id, obs in observations.items():
            actions[tls_id] = self.target_phase
        return actions

    def get_config(self):
        return {"name": self.name}


class TestRunnerController(unittest.TestCase):
    def test_runner_executes_controller_actions(self):
        """Verify runner queries controller.select_action and passes actions to SafePhaseManager."""
        cfg = get_scenario_config("cologne1")
        controller = MockController(target_phase=1)
        runner = SimulationRunner(cfg, controller=controller)

        tls_id = "cluster_357187_359543"
        mgr = runner.tls_managers[tls_id]
        mgr.update(10.0)  # satisfy min_green

        with patch("src.runner.traci") as mock_traci:
            mock_traci.simulation.getTime.return_value = 25201.0
            mock_traci.simulation.getDepartedIDList.return_value = []
            mock_traci.simulation.getArrivedIDList.return_value = []
            mock_traci.vehicle.getIDList.return_value = []
            mock_traci.lane.getLastStepHaltingNumber.return_value = 0
            mock_traci.lane.getLastStepVehicleNumber.return_value = 0
            mock_traci.trafficlight.getControlledLinks.return_value = [
                ((f"in_{i}", f"out_{i}", f"via_{i}"),) for i in range(20)
            ]

            runner.start()
            runner.step()

            # Controller should have been queried
            self.assertEqual(controller.actions_selected, 1)
            # SafePhaseManager should have initiated transition to phase 1
            self.assertTrue(mgr.is_in_transition())
            self.assertEqual(mgr.get_target_phase(), 1)


if __name__ == "__main__":
    unittest.main()
