"""Tests verifying the fixes for the issues identified in the project audit."""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.config import get_scenario_config
from src.data_inspector import inspect_scenario
from src.metrics import MetricsCollector, parse_tripinfo_output
from src.phase_manager import PhaseState, SafePhaseManager
from src.runner import SimulationRunner


class TestAuditFixes(unittest.TestCase):
    def test_controlled_links_matches_signal_state_length(self):
        """P1 Fix: Verify controlled_links counts signal links (state length), not raw connections."""
        info = inspect_scenario("saltlake2_400sX200w", data_dir="data")
        # In saltlake2_400sX200w, A3 has 21 connections but only 12 signal links (state length 12)
        self.assertEqual(info.controlled_links_per_tls["A3"], 12)
        self.assertEqual(info.controlled_links_per_tls["B3"], 16)

    def test_empty_actions_does_not_trigger_default_progression(self):
        """P0 Fix: Passing actions={} should NOT trigger default 30s progression."""
        cfg = get_scenario_config("cologne1")
        runner = SimulationRunner(cfg, control_mode="agent")

        tls_id = "cluster_357187_359543"
        mgr = runner.tls_managers[tls_id]
        initial_phase = mgr.get_current_phase()

        # Simulate 35 updates with empty actions={}
        with patch("src.runner.traci") as mock_traci:
            mock_traci.simulation.getTime.return_value = 25200.0
            mock_traci.simulation.getDepartedIDList.return_value = []
            mock_traci.simulation.getArrivedIDList.return_value = []
            mock_traci.vehicle.getIDList.return_value = []
            mock_traci.lane.getLastStepHaltingNumber.return_value = 0

            runner.start()
            for _ in range(35):
                # Pass explicit empty dict (meaning: agent does not want to change phase)
                runner.step(actions={})

            # Manager should still be in initial phase and not in transition
            self.assertEqual(mgr.get_current_phase(), initial_phase)
            self.assertFalse(mgr.is_in_transition())

    def test_control_mode_original_does_not_override_tls(self):
        """P0 Fix: control_mode='original' should let SUMO run its native program without setRedYellowGreenState."""
        cfg = get_scenario_config("ingolstadt1")
        runner = SimulationRunner(cfg, control_mode="original")

        with patch("src.runner.traci") as mock_traci:
            mock_traci.simulation.getTime.return_value = 57600.0
            mock_traci.simulation.getDepartedIDList.return_value = []
            mock_traci.simulation.getArrivedIDList.return_value = []
            mock_traci.vehicle.getIDList.return_value = []
            mock_traci.lane.getLastStepHaltingNumber.return_value = 0

            runner.start()
            runner.step()

            # setRedYellowGreenState should NOT be called in 'original' baseline mode
            mock_traci.trafficlight.setRedYellowGreenState.assert_not_called()

    def test_all_red_clearance_support(self):
        """P0/P1 Improvement: Support optional all-red clearance interval in SafePhaseManager."""
        mgr = SafePhaseManager(
            tls_id="tls_test",
            green_phases=["GGGG", "rrrr"],
            yellow_duration=3.0,
            all_red_duration=2.0,
            min_green=5.0,
        )
        mgr.update(5.0)  # satisfy min_green
        accepted = mgr.request_phase(1)
        self.assertTrue(accepted)
        self.assertTrue(mgr.is_in_transition())
        self.assertEqual(mgr.get_current_state(), "yyyy")

        # Step 3s yellow
        mgr.update(1.0)
        mgr.update(1.0)
        mgr.update(1.0)

        # Now should be in ALL-RED state
        self.assertTrue(mgr.is_in_transition())
        self.assertEqual(mgr.get_current_state(), "rrrr")

        # Step 2s all-red
        mgr.update(1.0)
        mgr.update(1.0)

        # Transition finished to target green
        self.assertFalse(mgr.is_in_transition())
        self.assertEqual(mgr.get_current_phase(), 1)
        self.assertEqual(mgr.get_current_state(), "rrrr")

    def test_parse_tripinfo_output(self):
        """P0 Fix: Tripinfo XML parser extracts accurate ground truth metrics."""
        import tempfile
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
    <tripinfo id="v1" depart="0.00" arrival="100.00" duration="100.00" routeLength="1000.00" waitingTime="15.00" waitingCount="2" timeLoss="25.00"/>
    <tripinfo id="v2" depart="10.00" arrival="60.00" duration="50.00" routeLength="500.00" waitingTime="5.00" waitingCount="1" timeLoss="10.00"/>
</tripinfos>
"""
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            tmp_path = f.name

        try:
            stats = parse_tripinfo_output(tmp_path)
            self.assertEqual(stats["tripinfo_completed_vehicles"], 2)
            self.assertAlmostEqual(stats["tripinfo_mean_travel_time"], 75.0)
            self.assertAlmostEqual(stats["tripinfo_mean_waiting_time"], 10.0)
            self.assertAlmostEqual(stats["tripinfo_total_waiting_time"], 20.0)
            self.assertAlmostEqual(stats["tripinfo_mean_time_loss"], 17.5)
            self.assertAlmostEqual(stats["tripinfo_total_time_loss"], 35.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
