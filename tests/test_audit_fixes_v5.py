"""Unit tests for fifth audit bug fixes (TDD Red phase)."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from src.config import get_scenario_config
from src.metrics import MetricsCollector
from src.runner import SimulationRunner


class TestAuditFixesV5(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_audit_v5_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_manual_close_then_run_resets_state_cleanly(self):
        """Fix 1: Calling start() -> step() -> close() manually, then calling run(1)

        must reset runner state and record total_steps=1, NOT 11.
        """
        cfg = get_scenario_config("cologne1", output_dir=self.test_dir)
        cfg.use_tripinfo = False
        runner = SimulationRunner(cfg)

        mock_traci = MagicMock()
        mock_traci.simulation.getDepartedIDList.return_value = []
        mock_traci.simulation.getArrivedIDList.return_value = []
        mock_traci.vehicle.getIDList.return_value = []
        mock_traci.lane.getLastStepHaltingNumber.return_value = 0
        mock_traci.simulation.getPendingVehicles.return_value = []
        mock_traci.simulation.getStartingTeleportNumber.return_value = 0
        mock_traci.simulation.getCollidingVehiclesIDList.return_value = []
        mock_traci.simulation.getCollisions.return_value = []

        import src.runner as runner_module
        orig_traci = runner_module.traci
        runner_module.traci = mock_traci
        try:
            # Session 1: Manual start -> 10 steps -> close
            runner.start()
            for _ in range(10):
                runner.step()
            runner.close()

            # Session 2: run(1) on same runner instance
            res = runner.run(num_steps=1)
            self.assertEqual(
                res["total_steps"],
                1,
                f"Expected total_steps=1 after re-run, but got {res['total_steps']} (state leaked from manual session!)",
            )
        finally:
            runner_module.traci = orig_traci

    def test_fallback_waiting_time_marks_error_and_invalid(self):
        """Fix 2: When getAccumulatedWaitingTime raises Exception, fallback to getWaitingTime

        must increment measurement_errors and mark is_valid=False / status!=OK.
        """
        collector = MetricsCollector("test_fallback_wait")
        mock_traci = MagicMock()
        mock_traci.simulation.getDepartedIDList.return_value = []
        mock_traci.simulation.getArrivedIDList.return_value = []
        mock_traci.vehicle.getIDList.return_value = ["v1"]
        mock_traci.lane.getLastStepHaltingNumber.return_value = 0

        # Getter tích lũy ném lỗi, getter hiện tại trả 0.0
        mock_traci.vehicle.getAccumulatedWaitingTime.side_effect = RuntimeError("Accumulated wait unavailable")
        mock_traci.vehicle.getWaitingTime.return_value = 0.0
        mock_traci.vehicle.getTimeLoss.return_value = 0.0
        mock_traci.vehicle.getSpeed.return_value = 10.0

        collector.record_step(
            step=1,
            sim_time=100.0,
            controlled_lanes=["lane1"],
            traci_module=mock_traci,
            step_length=1.0,
        )

        summary = collector.get_summary()
        self.assertFalse(summary["is_valid"], "Summary must not be valid when accumulated waiting time query fails!")
        self.assertGreater(summary["measurement_errors"], 0, "Measurement errors must be > 0!")
        self.assertNotEqual(summary["status"], "OK", "Status must not be OK when falling back from broken accumulated wait!")

    def test_collision_incidents_counts_distinct_collisions_in_same_step(self):
        """Fix 3: When 2 distinct collisions occur in the same step (e.g. pairs a-b and c-d),

        collision_incidents must be 2, collision_steps must be 1, and vehicle involvements must be 4.
        """
        cfg = get_scenario_config("cologne1", output_dir=self.test_dir)
        cfg.use_tripinfo = False
        runner = SimulationRunner(cfg)

        mock_traci = MagicMock()
        mock_traci.simulation.getDepartedIDList.return_value = []
        mock_traci.simulation.getArrivedIDList.return_value = []
        mock_traci.vehicle.getIDList.return_value = []
        mock_traci.lane.getLastStepHaltingNumber.return_value = 0
        mock_traci.simulation.getPendingVehicles.return_value = []
        mock_traci.simulation.getStartingTeleportNumber.return_value = 0

        # In step 1: 2 collision incidents (a hitting b, c hitting d)
        mock_collision_1 = MagicMock(collider="a", victim="b")
        mock_collision_2 = MagicMock(collider="c", victim="d")
        mock_traci.simulation.getCollisions.return_value = [mock_collision_1, mock_collision_2]
        mock_traci.simulation.getCollidingVehiclesIDList.return_value = ["a", "b", "c", "d"]

        import src.runner as runner_module
        orig_traci = runner_module.traci
        runner_module.traci = mock_traci
        try:
            res = runner.run(num_steps=1)
            self.assertEqual(res.get("collision_incidents"), 2, "Must record 2 distinct collision incidents!")
            self.assertEqual(res.get("collision_steps"), 1, "Must record 1 simulation step with collisions!")
            self.assertEqual(res.get("collision_vehicle_involvements"), 4, "Must record 4 vehicle collision involvements!")
        finally:
            runner_module.traci = orig_traci


if __name__ == "__main__":
    unittest.main()
