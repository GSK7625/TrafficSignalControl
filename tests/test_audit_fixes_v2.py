"""Unit tests for secondary audit bug fixes (TDD Red phase)."""

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock

from src.config import get_scenario_config
from src.data_inspector import inspect_scenario
from src.metrics import MetricsCollector
from src.phase_manager import PhaseState, SafePhaseManager
from src.runner import SimulationRunner


class TestAuditFixesV2(unittest.TestCase):
    def test_all_red_yellow_state_clears_all_green_signals(self):
        """Fix 1: When all_red_duration > 0, ALL currently green links must display yellow.

        No link may remain Green during the yellow phase if it will be forced to Red during all-red.
        """
        # Ingolstadt1 phases 0 and 2 share green on links 0 and 1
        p0 = "GGgGrGGG"
        p2 = "GGGrrrrr"
        mgr = SafePhaseManager(
            tls_id="tls_test",
            green_phases=[p0, p2],
            yellow_duration=3.0,
            all_red_duration=2.0,
            min_green=5.0,
        )
        mgr.update(5.0)  # satisfy min_green
        mgr.request_phase(1)

        # During yellow transition towards all-red, NO link should be 'G' or 'g'
        yellow_state = mgr.get_current_state()
        self.assertTrue(mgr.is_in_transition())
        self.assertNotIn("G", yellow_state, f"Found 'G' in yellow state before all-red: {yellow_state}")
        self.assertNotIn("g", yellow_state, f"Found 'g' in yellow state before all-red: {yellow_state}")

        # Advance through 3s yellow
        mgr.update(3.0)
        # Now in all-red state
        self.assertEqual(mgr.get_current_state(), "rrrrrrrr")

        # Advance through 2s all-red
        mgr.update(2.0)
        # Now in target green phase 2
        self.assertEqual(mgr.get_current_state(), p2)
        self.assertFalse(mgr.is_in_transition())

    def test_original_mode_rejects_external_actions(self):
        """Fix 2: In 'original' control mode, passing external actions must raise ValueError."""
        cfg = get_scenario_config("cologne1", control_mode="original")
        runner = SimulationRunner(cfg)
        runner.is_running = True

        with self.assertRaises(ValueError):
            runner.step(actions={})

        with self.assertRaises(ValueError):
            runner.step(actions={"cluster_357187_359543": 1})

    def test_unique_tripinfo_files_per_runner(self):
        """Fix 3: Multiple runners in the same process must get unique tripinfo paths."""
        cfg = get_scenario_config("cologne1", use_tripinfo=True)
        runner1 = SimulationRunner(cfg)
        runner2 = SimulationRunner(cfg)

        self.assertIsNotNone(runner1.tripinfo_path)
        self.assertIsNotNone(runner2.tripinfo_path)
        self.assertNotEqual(
            runner1.tripinfo_path,
            runner2.tripinfo_path,
            "Tripinfo paths collided across runners in the same process",
        )

    def test_missing_route_file_raises_filenotfounderror(self):
        """Fix 4: inspect_scenario must raise FileNotFoundError if referenced route-file is missing."""
        # Create a mock scenario directory with a .sumocfg referencing a missing route file
        with tempfile.TemporaryDirectory() as tmp_dir:
            scen_name = "test_broken"
            scen_dir = os.path.join(tmp_dir, scen_name)
            os.makedirs(scen_dir)

            # Create dummy net file
            net_path = os.path.join(scen_dir, "test.net.xml")
            with open(net_path, "w", encoding="utf-8") as f:
                f.write("<net><tlLogic id='t1'><phase duration='30' state='GG'/></tlLogic></net>")

            # Create sumocfg referencing non-existent route file
            cfg_path = os.path.join(scen_dir, f"{scen_name}.sumocfg")
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(
                    f"<configuration><input><net-file value='test.net.xml'/>"
                    f"<route-files value='missing_routes.rou.xml'/></input></configuration>"
                )

            with self.assertRaises(FileNotFoundError):
                inspect_scenario(scen_name, data_dir=tmp_dir)

    def test_metrics_uses_accumulated_waiting_time(self):
        """Fix 5: Metrics collector uses accumulated waiting time, not snapshot that resets to 0."""
        collector = MetricsCollector("test_acc_wait")
        mock_traci = MagicMock()
        mock_traci.simulation.getDepartedIDList.return_value = ["v1"]
        mock_traci.simulation.getArrivedIDList.return_value = []
        mock_traci.vehicle.getIDList.return_value = ["v1"]
        mock_traci.lane.getLastStepHaltingNumber.return_value = 0

        # Simulate vehicle that waited 45s total, but is currently moving (waitingTime=0, accumulatedWaitingTime=45)
        mock_traci.vehicle.getWaitingTime.return_value = 0.0
        mock_traci.vehicle.getAccumulatedWaitingTime.return_value = 45.0
        mock_traci.vehicle.getTimeLoss.return_value = 50.0

        collector.record_step(
            step=1,
            sim_time=100.0,
            controlled_lanes=[],
            traci_module=mock_traci,
            step_length=1.0,
        )

        # Vehicle arrives at next step
        mock_traci.simulation.getDepartedIDList.return_value = []
        mock_traci.simulation.getArrivedIDList.return_value = ["v1"]
        mock_traci.vehicle.getIDList.return_value = []

        collector.record_step(
            step=2,
            sim_time=101.0,
            controlled_lanes=[],
            traci_module=mock_traci,
            step_length=1.0,
        )

        summary = collector.get_summary()
        self.assertEqual(summary["total_completed_vehicles"], 1)
        self.assertEqual(summary["total_waiting_time"], 45.0)
        self.assertEqual(summary["mean_waiting_time"], 45.0)


if __name__ == "__main__":
    unittest.main()
