"""Unit tests for sixth audit bug fixes (TDD Red phase)."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from src.config import get_scenario_config
from src.runner import SimulationRunner


class TestAuditFixesV6(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_audit_v6_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_runner_telemetry_failure_marks_run_invalid_and_tracks_errors(self):
        """Fix 1: When runner telemetry getters (teleports, collisions, pending vehicles)

        fail, summary must NOT report is_valid=True with measurement_errors=0 and status=OK.
        It must report is_valid=False, measurement_errors > 0, and status='INVALID'.
        """
        cfg = get_scenario_config("cologne1", output_dir=self.test_dir)
        cfg.use_tripinfo = False
        runner = SimulationRunner(cfg)

        mock_traci = MagicMock()
        mock_traci.simulation.getTime.return_value = 25201.0
        # Normal traffic metrics succeed
        mock_traci.simulation.getDepartedIDList.return_value = []
        mock_traci.simulation.getArrivedIDList.return_value = []
        mock_traci.vehicle.getIDList.return_value = []
        mock_traci.lane.getLastStepHaltingNumber.return_value = 0

        # Telemetry getters raise RuntimeError
        mock_traci.simulation.getStartingTeleportNumber.side_effect = RuntimeError("Teleport query failed")
        mock_traci.simulation.getCollisions.side_effect = RuntimeError("Collisions query failed")
        mock_traci.simulation.getCollidingVehiclesIDList.side_effect = RuntimeError("Colliding vehs query failed")
        mock_traci.simulation.getCollidingVehiclesNumber.side_effect = RuntimeError("Colliding num query failed")
        mock_traci.simulation.getPendingVehicles.side_effect = RuntimeError("Pending vehicles query failed")

        import src.runner as runner_module
        orig_traci = runner_module.traci
        runner_module.traci = mock_traci
        try:
            # Run 2 steps
            summary = runner.run(num_steps=2)

            self.assertFalse(
                summary["is_valid"],
                "Run must be marked invalid when all runner telemetry getters fail!",
            )
            self.assertGreater(
                summary["measurement_errors"],
                0,
                "measurement_errors must be > 0 when telemetry getters raise exceptions!",
            )
            self.assertEqual(
                summary["status"],
                "INVALID",
                "status must be 'INVALID' instead of 'OK' when telemetry fails!",
            )
        finally:
            runner_module.traci = orig_traci

    def test_valid_collision_fallback_does_not_mark_error(self):
        """Fix 2: When getCollisions raises AttributeError/unsupported, but

        getCollidingVehiclesIDList succeeds, fallback is valid and no error is recorded.
        """
        cfg = get_scenario_config("cologne1", output_dir=self.test_dir)
        cfg.use_tripinfo = False
        runner = SimulationRunner(cfg)

        mock_traci = MagicMock()
        mock_traci.simulation.getDepartedIDList.return_value = []
        mock_traci.simulation.getArrivedIDList.return_value = []
        mock_traci.vehicle.getIDList.return_value = []
        mock_traci.lane.getLastStepHaltingNumber.return_value = 0
        mock_traci.simulation.getStartingTeleportNumber.return_value = 0
        mock_traci.simulation.getPendingVehicles.return_value = []

        # getCollisions is unsupported on older API, but getCollidingVehiclesIDList works
        mock_traci.simulation.getCollisions.side_effect = AttributeError("Old TraCI without getCollisions")
        mock_traci.simulation.getCollidingVehiclesIDList.return_value = ["v1", "v2"]

        import src.runner as runner_module
        orig_traci = runner_module.traci
        runner_module.traci = mock_traci
        try:
            summary = runner.run(num_steps=1)

            self.assertTrue(summary["is_valid"])
            self.assertEqual(summary["measurement_errors"], 0)
            self.assertEqual(summary["status"], "OK")
            self.assertEqual(summary["collision_vehicle_involvements"], 2)
            self.assertEqual(summary["collision_incidents"], 1)
        finally:
            runner_module.traci = orig_traci


if __name__ == "__main__":
    unittest.main()
