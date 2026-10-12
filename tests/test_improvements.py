import os
import unittest
from unittest.mock import MagicMock
from src.config import SimulationConfig
from src.metrics import MetricsCollector
from src.phase_manager import SafePhaseManager, build_yellow_state


class TestImprovements(unittest.TestCase):
    def test_direct_simulation_config_loads_begin_time(self):
        # Direct instantiation of SimulationConfig without get_scenario_config
        cfg = SimulationConfig(dataset_name="cologne1")
        self.assertEqual(cfg.begin_time, 25200.0)
        self.assertEqual(cfg.end_time, 28800.0)

    def test_direct_simulation_config_ingolstadt1(self):
        cfg = SimulationConfig(dataset_name="ingolstadt1")
        self.assertEqual(cfg.begin_time, 57600.0)
        self.assertEqual(cfg.end_time, 61200.0)

    def test_exact_yellow_steps(self):
        # With yellow_duration=3.0 and step_duration=1.0, yellow should be displayed
        # for exactly 3 steps.
        mgr = SafePhaseManager(
            tls_id="tls_test",
            green_phases=["GGGG", "rrrr"],
            yellow_duration=3.0,
            min_green=5.0,
        )
        mgr.update(5.0)  # satisfy min_green
        mgr.request_phase(1)
        self.assertTrue(mgr.is_in_transition())

        # Step 1
        mgr.update(1.0)
        self.assertTrue(mgr.is_in_transition())

        # Step 2
        mgr.update(1.0)
        self.assertTrue(mgr.is_in_transition())

        # Step 3
        mgr.update(1.0)
        # After 3 full steps of 1.0s, transition completes to green
        self.assertFalse(mgr.is_in_transition())
        self.assertEqual(mgr.get_current_phase(), 1)

    def test_metrics_delay_not_quadratic(self):
        collector = MetricsCollector("test_delay")
        mock_traci = MagicMock()
        mock_traci.simulation.getDepartedIDList.return_value = ["v1"]
        mock_traci.simulation.getArrivedIDList.return_value = []
        mock_traci.vehicle.getIDList.return_value = ["v1"]
        mock_traci.vehicle.getWaitingTime.return_value = 10.0
        mock_traci.vehicle.getTimeLoss.return_value = 5.0
        # 1 halting vehicle
        mock_traci.lane.getLastStepHaltingNumber.return_value = 1

        for step in range(1, 11):  # 10 steps of 1 vehicle halting
            collector.record_step(
                step=step,
                sim_time=step * 1.0,
                controlled_lanes=["lane1"],
                traci_module=mock_traci,
                step_length=1.0,
            )

        summary = collector.get_summary()
        # 1 halting vehicle for 10 seconds of 1s steps = 10.0 vehicle-seconds total delay
        self.assertEqual(summary["total_waiting_time"], 10.0)

    def test_build_yellow_state_handles_off_signal(self):
        s1 = "GGGG"
        s2 = "oooo"
        yellow = build_yellow_state(s1, s2)
        self.assertEqual(yellow, "yyyy")


if __name__ == "__main__":
    unittest.main()
