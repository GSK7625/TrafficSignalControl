import os
import unittest
from src.config import SimulationConfig, get_scenario_config


class TestConfig(unittest.TestCase):
    def test_default_config_cologne1(self):
        cfg = get_scenario_config("cologne1")
        self.assertEqual(cfg.dataset_name, "cologne1")
        self.assertFalse(cfg.gui)
        self.assertEqual(cfg.step_length, 1.0)
        self.assertEqual(cfg.yellow_duration, 3.0)
        self.assertEqual(cfg.min_green, 5.0)
        self.assertTrue(os.path.isfile(cfg.sumocfg_path))
        self.assertEqual(cfg.begin_time, 25200.0)
        self.assertEqual(cfg.end_time, 28800.0)

    def test_gui_enabled(self):
        cfg = get_scenario_config("ingolstadt1", gui=True)
        self.assertTrue(cfg.gui)
        self.assertIn("sumo-gui", cfg.sumo_binary)

    def test_override_parameters(self):
        cfg = get_scenario_config(
            "cologne1",
            yellow_duration=4.0,
            min_green=10.0,
            step_length=0.5,
            max_steps=500,
            output_dir="custom_results",
        )
        self.assertEqual(cfg.yellow_duration, 4.0)
        self.assertEqual(cfg.min_green, 10.0)
        self.assertEqual(cfg.step_length, 0.5)
        self.assertEqual(cfg.max_steps, 500)
        self.assertEqual(cfg.output_dir, "custom_results")

    def test_invalid_dataset(self):
        with self.assertRaises(ValueError):
            get_scenario_config("unknown_dataset_xyz")

    def test_invalid_durations(self):
        with self.assertRaises(ValueError):
            get_scenario_config("cologne1", yellow_duration=-1.0)
        with self.assertRaises(ValueError):
            get_scenario_config("cologne1", min_green=-5.0)
        with self.assertRaises(ValueError):
            get_scenario_config("cologne1", step_length=0)


if __name__ == "__main__":
    unittest.main()
