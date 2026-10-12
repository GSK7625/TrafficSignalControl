"""Live TraCI integration test for FixedTimeController on Cologne1 and Ingolstadt1."""

import unittest
from src.config import get_scenario_config
from src.controllers.fixed_time import FixedTimeController
from src.runner import SimulationRunner


class TestFixedTimeLive(unittest.TestCase):
    def test_live_fixed_time_cologne1(self):
        """Verify FixedTimeController runs live TraCI on Cologne1 without errors."""
        cfg = get_scenario_config("cologne1", seed=42)
        controller = FixedTimeController(default_green_time=20.0)
        runner = SimulationRunner(cfg, controller=controller)

        summary = runner.run(num_steps=60)
        self.assertTrue(summary["is_valid"])
        self.assertEqual(summary["status"], "OK")
        self.assertEqual(summary["total_steps"], 60)
        self.assertEqual(summary["controller_name"], "FixedTime")
        self.assertGreater(len(controller.history), 0)

    def test_live_fixed_time_ingolstadt1_multi_phase(self):
        """Verify FixedTimeController cycles across 3 phases on Ingolstadt1."""
        cfg = get_scenario_config("ingolstadt1", seed=42)
        controller = FixedTimeController(default_green_time=20.0)
        runner = SimulationRunner(cfg, controller=controller)

        summary = runner.run(num_steps=60)
        self.assertTrue(summary["is_valid"])
        self.assertEqual(summary["status"], "OK")
        self.assertEqual(summary["total_steps"], 60)
        self.assertEqual(summary["controller_name"], "FixedTime")

    def test_reproducibility(self):
        """Verify two FixedTime runs with same seed produce identical summary metrics."""
        cfg1 = get_scenario_config("cologne1", seed=123)
        ctrl1 = FixedTimeController(default_green_time=15.0)
        r1 = SimulationRunner(cfg1, controller=ctrl1)
        s1 = r1.run(num_steps=40)

        cfg2 = get_scenario_config("cologne1", seed=123)
        ctrl2 = FixedTimeController(default_green_time=15.0)
        r2 = SimulationRunner(cfg2, controller=ctrl2)
        s2 = r2.run(num_steps=40)

        self.assertEqual(s1["total_completed_vehicles"], s2["total_completed_vehicles"])
        self.assertEqual(s1["mean_waiting_time"], s2["mean_waiting_time"])
        self.assertEqual(s1["total_time_loss"], s2["total_time_loss"])


if __name__ == "__main__":
    unittest.main()
