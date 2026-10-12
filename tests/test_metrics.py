import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock
from src.metrics import MetricsCollector, StepMetrics


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.collector = MetricsCollector(scenario_name="test_scenario")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_step_with_mock_traci(self):
        # Setup mock traci
        mock_traci = MagicMock()
        mock_traci.simulation.getTime.return_value = 100.0
        mock_traci.vehicle.getIDList.return_value = ["v1", "v2"]
        mock_traci.vehicle.getWaitingTime.side_effect = lambda vid: 10.0 if vid == "v1" else 5.0
        mock_traci.vehicle.getTimeLoss.side_effect = lambda vid: 15.0 if vid == "v1" else 8.0
        mock_traci.vehicle.getDeparture.side_effect = lambda vid: 80.0 if vid == "v1" else 85.0

        mock_traci.lane.getLastStepHaltingNumber.side_effect = lambda lid: 2 if lid == "lane1" else 1
        mock_traci.lane.getWaitingTime.side_effect = lambda lid: 12.0 if lid == "lane1" else 3.0

        mock_traci.simulation.getArrivedIDList.return_value = ["v0"]

        controlled_lanes = ["lane1", "lane2"]

        # Track departed vehicles for travel time calculation
        self.collector.register_departed_vehicles({"v0": 50.0})

        metrics = self.collector.record_step(
            step=1,
            sim_time=100.0,
            controlled_lanes=controlled_lanes,
            traci_module=mock_traci,
        )

        self.assertIsInstance(metrics, StepMetrics)
        self.assertEqual(metrics.step, 1)
        self.assertEqual(metrics.sim_time, 100.0)
        self.assertEqual(metrics.running_vehicles, 2)
        self.assertEqual(metrics.queue_length, 3)  # 2 + 1
        self.assertEqual(metrics.waiting_time, 15.0)  # 10 + 5
        self.assertEqual(metrics.time_loss, 23.0)  # 15 + 8
        self.assertEqual(metrics.step_completed_vehicles, 1)
        self.assertEqual(metrics.cumulative_completed_vehicles, 1)

    def test_summary_calculation(self):
        # Manually add some step metrics
        m1 = StepMetrics(
            step=1, sim_time=1.0, running_vehicles=2, queue_length=4,
            waiting_time=10.0, time_loss=5.0, step_completed_vehicles=1,
            cumulative_completed_vehicles=1,
        )
        m2 = StepMetrics(
            step=2, sim_time=2.0, running_vehicles=3, queue_length=6,
            waiting_time=20.0, time_loss=10.0, step_completed_vehicles=1,
            cumulative_completed_vehicles=2,
        )
        self.collector.step_records.extend([m1, m2])
        self.collector.completed_travel_times.extend([30.0, 40.0])

        summary = self.collector.get_summary()
        self.assertEqual(summary["total_steps"], 2)
        self.assertEqual(summary["total_completed_vehicles"], 2)
        self.assertEqual(summary["mean_queue_length"], 5.0)
        self.assertEqual(summary["max_queue_length"], 6)
        self.assertEqual(summary["mean_waiting_time"], 15.0)
        self.assertEqual(summary["mean_travel_time"], 35.0)

    def test_export_results(self):
        m1 = StepMetrics(
            step=1, sim_time=1.0, running_vehicles=1, queue_length=2,
            waiting_time=5.0, time_loss=2.0, step_completed_vehicles=1,
            cumulative_completed_vehicles=1,
        )
        self.collector.step_records.append(m1)
        self.collector.completed_travel_times.append(25.0)

        json_path, csv_path = self.collector.export_results(self.temp_dir, prefix="test")
        self.assertTrue(os.path.isfile(json_path))
        self.assertTrue(os.path.isfile(csv_path))

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["scenario_name"], "test_scenario")
            self.assertEqual(data["total_completed_vehicles"], 1)

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertGreaterEqual(len(lines), 2)  # Header + 1 data line
            self.assertIn("step,sim_time,running_vehicles", lines[0])


if __name__ == "__main__":
    unittest.main()
