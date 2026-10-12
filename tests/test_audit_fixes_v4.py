"""Unit tests for fourth audit bug fixes (TDD Red phase)."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from src.config import get_scenario_config
from src.metrics import MetricsCollector
from src.runner import SimulationRunner


class TestAuditFixesV4(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_audit_v4_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_teleport_and_collision_accumulated_across_all_steps(self):
        """Fix 1: Teleports and collisions occurring at earlier steps must be accumulated

        and not lost even if the final step reports 0.
        """
        cfg = get_scenario_config("cologne1", output_dir=self.test_dir)
        cfg.use_tripinfo = False
        runner = SimulationRunner(cfg)

        mock_traci = MagicMock()
        # Step 1: 1 teleport started, 2 colliding vehicles
        # Step 2: 0 teleports, 0 collisions
        mock_traci.simulation.getStartingTeleportNumber.side_effect = [1, 0]
        mock_traci.simulation.getCollidingVehiclesIDList.side_effect = [["v1", "v2"], []]
        mock_traci.simulation.getDepartedIDList.return_value = []
        mock_traci.simulation.getArrivedIDList.return_value = []
        mock_traci.vehicle.getIDList.return_value = []
        mock_traci.lane.getLastStepHaltingNumber.return_value = 0
        mock_traci.simulation.getPendingVehicles.return_value = []

        import src.runner as runner_module
        orig_traci = runner_module.traci
        runner_module.traci = mock_traci
        try:
            runner.is_running = True
            # Execute step 1 (events occur)
            runner.step()
            # Execute step 2 (no events)
            runner.step()

            self.assertEqual(runner.total_teleports, 1)
            self.assertEqual(runner.total_collisions, 2)
        finally:
            runner_module.traci = orig_traci

    def test_calling_run_twice_on_same_runner_resets_cleanly(self):
        """Fix 2: Calling run() a second time on the same runner instance must reset

        state, generate a new run_folder, and not mix metrics between runs.
        """
        cfg = get_scenario_config("cologne1", output_dir=self.test_dir)
        cfg.use_tripinfo = False
        runner = SimulationRunner(cfg)

        mock_traci = MagicMock()
        mock_traci.simulation.getDepartedIDList.return_value = ["v1"]
        mock_traci.simulation.getArrivedIDList.side_effect = [["v1"], []]
        mock_traci.vehicle.getIDList.return_value = []
        mock_traci.vehicle.getDeparture.return_value = 25200.0
        mock_traci.vehicle.getTimeLoss.return_value = 5.0
        mock_traci.vehicle.getAccumulatedWaitingTime.return_value = 10.0
        mock_traci.lane.getLastStepHaltingNumber.return_value = 0
        mock_traci.simulation.getPendingVehicles.return_value = []
        mock_traci.simulation.getStartingTeleportNumber.return_value = 0
        mock_traci.simulation.getCollidingVehiclesIDList.return_value = []

        import src.runner as runner_module
        orig_traci = runner_module.traci
        runner_module.traci = mock_traci
        try:
            # First run: 1 step, 1 completed vehicle
            res1 = runner.run(num_steps=1)
            self.assertEqual(res1["total_steps"], 1)
            self.assertEqual(res1["total_completed_vehicles"], 1)
            folder1 = res1["output_json"]

            # Second run on SAME runner instance: 1 step, 0 completed vehicles
            mock_traci.simulation.getDepartedIDList.return_value = []
            mock_traci.simulation.getArrivedIDList.side_effect = None
            mock_traci.simulation.getArrivedIDList.return_value = []
            res2 = runner.run(num_steps=1)

            self.assertEqual(res2["total_steps"], 1)
            self.assertEqual(res2["total_completed_vehicles"], 0)
            folder2 = res2["output_json"]

            self.assertNotEqual(folder1, folder2, "Both runs wrote to the same output_json!")
            self.assertTrue(os.path.isfile(folder1), "First run JSON was overwritten or missing!")
            self.assertTrue(os.path.isfile(folder2), "Second run JSON was missing!")
        finally:
            runner_module.traci = orig_traci

    def test_metric_read_failure_marks_invalid_and_tracks_errors(self):
        """Fix 3: When reading metrics from TraCI fails, collector must track measurement errors

        and mark the run as invalid instead of silently returning 0.
        """
        collector = MetricsCollector("test_err_collector")
        mock_traci = MagicMock()
        mock_traci.simulation.getDepartedIDList.return_value = []
        mock_traci.simulation.getArrivedIDList.return_value = []
        # Simulate fatal/unexpected TraCI read error on vehicle query
        mock_traci.vehicle.getIDList.side_effect = RuntimeError("TraCI vehicle query crashed")

        collector.record_step(
            step=1,
            sim_time=100.0,
            controlled_lanes=[],
            traci_module=mock_traci,
            step_length=1.0,
        )

        summary = collector.get_summary()
        self.assertFalse(summary["is_valid"])
        self.assertGreater(summary["measurement_errors"], 0)

    def test_travel_time_accounts_for_step_length(self):
        """Fix 4: Arrival time in TraCI is (sim_time - step_length), so travel time

        matches continuous simulation duration exactly.
        """
        collector = MetricsCollector("test_travel_time")
        mock_traci = MagicMock()
        mock_traci.simulation.getDepartedIDList.return_value = []
        mock_traci.simulation.getArrivedIDList.return_value = ["v1"]
        mock_traci.vehicle.getIDList.return_value = []
        mock_traci.lane.getLastStepHaltingNumber.return_value = 0

        # Departure at 10.0, arrived at step time 35.0 with step_length 1.0 -> duration is (35.0 - 1.0) - 10.0 = 24.0
        collector.register_departed_vehicles({"v1": 10.0})

        collector.record_step(
            step=25,
            sim_time=35.0,
            controlled_lanes=[],
            traci_module=mock_traci,
            step_length=1.0,
        )

        summary = collector.get_summary()
        self.assertEqual(summary["total_completed_vehicles"], 1)
        self.assertEqual(summary["mean_travel_time"], 24.0)


if __name__ == "__main__":
    unittest.main()
