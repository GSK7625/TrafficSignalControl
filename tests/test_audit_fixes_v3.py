"""Unit tests for third audit bug fixes (TDD Red phase)."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from src.config import get_scenario_config
from src.metrics import MetricsCollector
from src.phase_manager import SafePhaseManager
from src.runner import SimulationRunner


class TestAuditFixesV3(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_audit_v3_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_unique_run_folder_names_prevent_collision_in_same_second(self):
        """Fix 1: Two runners with same dataset/output_dir in same second must not share run_folder."""
        cfg = get_scenario_config("cologne1", output_dir=self.test_dir)
        r1 = SimulationRunner(cfg)
        r2 = SimulationRunner(cfg)

        self.assertNotEqual(
            r1.run_folder_name,
            r2.run_folder_name,
            f"Run folder collision: both runners got '{r1.run_folder_name}'",
        )

    def test_summary_includes_metadata_and_network_state(self):
        """Fix 2: Summary must include control_mode, seed, running/pending/teleport/collision stats."""
        collector = MetricsCollector("cologne1")
        mock_traci = MagicMock()
        mock_traci.simulation.getDepartedIDList.return_value = ["v1"]
        mock_traci.simulation.getArrivedIDList.return_value = []
        mock_traci.vehicle.getIDList.return_value = ["v1"]
        mock_traci.vehicle.getAccumulatedWaitingTime.return_value = 10.0
        mock_traci.vehicle.getTimeLoss.return_value = 12.0
        mock_traci.lane.getLastStepHaltingNumber.return_value = 0
        mock_traci.simulation.getPendingVehicles.return_value = ["p1", "p2"]
        mock_traci.simulation.getStartingTeleportNumber.return_value = 0
        mock_traci.simulation.getCollidingVehiclesNumber.return_value = 0

        collector.record_step(
            step=1,
            sim_time=25201.0,
            controlled_lanes=[],
            traci_module=mock_traci,
            step_length=1.0,
        )

        metadata = {
            "control_mode": "original",
            "seed": 42,
            "step_length": 1.0,
            "running_vehicles_at_end": 1,
            "pending_insertion_vehicles": 2,
            "teleport_count": 0,
            "collision_count": 0,
        }
        collector.set_run_metadata(metadata)
        summary = collector.get_summary()

        self.assertEqual(summary["control_mode"], "original")
        self.assertEqual(summary["seed"], 42)
        self.assertEqual(summary["running_vehicles_at_end"], 1)
        self.assertEqual(summary["pending_insertion_vehicles"], 2)
        self.assertEqual(summary["teleport_count"], 0)
        self.assertEqual(summary["collision_count"], 0)

    def test_separation_of_completed_vs_network_waiting_time(self):
        """Fix 3: Distinguish between completed vehicles waiting time and total network waiting time."""
        collector = MetricsCollector("test_separation")
        mock_traci = MagicMock()
        mock_traci.lane.getLastStepHaltingNumber.return_value = 2  # 2 halting vehicles on controlled lane

        # Step 1: v1 and v2 running
        mock_traci.simulation.getDepartedIDList.return_value = ["v1", "v2"]
        mock_traci.simulation.getArrivedIDList.return_value = []
        mock_traci.vehicle.getIDList.return_value = ["v1", "v2"]
        mock_traci.vehicle.getAccumulatedWaitingTime.side_effect = lambda vid: 20.0 if vid == "v1" else 30.0
        mock_traci.vehicle.getTimeLoss.return_value = 25.0

        collector.record_step(1, 100.0, ["lane1"], mock_traci, 1.0)

        # Step 2: v1 arrives, v2 still running
        mock_traci.simulation.getDepartedIDList.return_value = []
        mock_traci.simulation.getArrivedIDList.return_value = ["v1"]
        mock_traci.vehicle.getIDList.return_value = ["v2"]
        mock_traci.vehicle.getAccumulatedWaitingTime.side_effect = lambda vid: 35.0  # v2 waited 35s
        mock_traci.vehicle.getTimeLoss.return_value = 30.0

        collector.record_step(2, 101.0, ["lane1"], mock_traci, 1.0)

        summary = collector.get_summary()
        self.assertEqual(summary["total_completed_vehicles"], 1)
        # v1 completed waiting time was 20.0s
        self.assertEqual(summary["completed_vehicles_waiting_time"], 20.0)
        self.assertEqual(summary["mean_waiting_time"], 20.0)
        # Total network waiting time = completed (20.0) + running v2 (35.0) = 55.0s
        self.assertEqual(summary["total_network_waiting_time"], 55.0)
        # Total queue delay on lane1 = 2 vehicles * 1s * 2 steps = 4.0 vehicle-seconds
        self.assertEqual(summary["total_queue_delay"], 4.0)

    def test_fail_fast_on_traci_state_set_failure(self):
        """Fix 4: SafePhaseManager raises RuntimeError when traci fails to set traffic light state."""
        mgr = SafePhaseManager("tls1", ["GGGG", "rrrr"])
        mock_traci = MagicMock()
        mock_traci.trafficlight.setRedYellowGreenState.side_effect = RuntimeError("Fatal connection loss")

        with self.assertRaises(RuntimeError):
            mgr.apply_to_traci(mock_traci)


if __name__ == "__main__":
    unittest.main()
