"""Live TraCI integration test for MaxPressureController on Cologne1 and Ingolstadt1."""

import unittest
from src.config import get_scenario_config
from src.controllers.max_pressure import MaxPressureController
from src.runner import SimulationRunner


class TestMaxPressureLive(unittest.TestCase):
    def test_live_max_pressure_cologne1(self):
        """Verify MaxPressureController runs live TraCI on Cologne1 cleanly."""
        cfg = get_scenario_config("cologne1", seed=42)
        controller = MaxPressureController(min_green=5.0)
        runner = SimulationRunner(cfg, controller=controller)

        summary = runner.run(num_steps=60)
        self.assertTrue(summary["is_valid"])
        self.assertEqual(summary["status"], "OK")
        self.assertEqual(summary["total_steps"], 60)
        self.assertEqual(summary["controller_name"], "MaxPressure")
        self.assertGreater(len(controller.history), 0)

        # Check that pressures are logged in decision details
        first_log = controller.history[0]
        self.assertIn("pressures", first_log["details"])
        self.assertIn("max_pressure", first_log["details"])

    def test_live_max_pressure_ingolstadt1_multi_phase(self):
        """Verify MaxPressureController runs live TraCI on Ingolstadt1 across 3 phases."""
        cfg = get_scenario_config("ingolstadt1", seed=42)
        controller = MaxPressureController(min_green=5.0)
        runner = SimulationRunner(cfg, controller=controller)

        summary = runner.run(num_steps=60)
        self.assertTrue(summary["is_valid"])
        self.assertEqual(summary["status"], "OK")
        self.assertEqual(summary["total_steps"], 60)
        self.assertEqual(summary["controller_name"], "MaxPressure")

    def test_reproducibility(self):
        """Verify two MaxPressure runs with same seed produce identical summary metrics."""
        cfg1 = get_scenario_config("cologne1", seed=999)
        ctrl1 = MaxPressureController(min_green=5.0)
        r1 = SimulationRunner(cfg1, controller=ctrl1)
        s1 = r1.run(num_steps=50)

        cfg2 = get_scenario_config("cologne1", seed=999)
        ctrl2 = MaxPressureController(min_green=5.0)
        r2 = SimulationRunner(cfg2, controller=ctrl2)
        s2 = r2.run(num_steps=50)

        self.assertEqual(s1["total_completed_vehicles"], s2["total_completed_vehicles"])
        self.assertEqual(s1["mean_waiting_time"], s2["mean_waiting_time"])
        self.assertEqual(s1["total_time_loss"], s2["total_time_loss"])


if __name__ == "__main__":
    unittest.main()
