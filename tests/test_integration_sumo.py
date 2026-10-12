import json
import os
import shutil
import tempfile
import unittest
from src.config import get_scenario_config
from src.runner import SimulationRunner


class TestIntegrationSumo(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = tempfile.mkdtemp(prefix="sumo_test_results_")

    def tearDown(self):
        shutil.rmtree(self.test_output_dir, ignore_errors=True)

    def test_run_cologne1_live_traci(self):
        cfg = get_scenario_config(
            dataset_name="cologne1",
            gui=False,
            output_dir=self.test_output_dir,
            yellow_duration=3.0,
            min_green=5.0,
            control_mode="agent",
        )

        runner = SimulationRunner(cfg)
        self.assertEqual(len(runner.tls_managers), 1)
        tls_id = "cluster_357187_359543"
        self.assertIn(tls_id, runner.tls_managers)

        try:
            runner.start()
            self.assertTrue(runner.is_running)

            # Step 10 steps in initial phase
            for _ in range(10):
                metrics = runner.step()
                self.assertIsNotNone(metrics)

            # Request switch to phase 1
            mgr = runner.tls_managers[tls_id]
            accepted = runner.step(actions={tls_id: 1})
            # Verify transition is initiated or active
            self.assertTrue(mgr.is_in_transition() or mgr.get_current_phase() == 1)

            # Step another 20 steps
            for _ in range(20):
                runner.step()

            # Verify transition completed
            self.assertFalse(mgr.is_in_transition())
            self.assertEqual(mgr.get_current_phase(), 1)

        finally:
            runner.close()
            self.assertFalse(runner.is_running)

        # Export and verify results
        json_path, csv_path = runner.metrics_collector.export_results(self.test_output_dir, prefix="col1")
        self.assertTrue(os.path.isfile(json_path))
        self.assertTrue(os.path.isfile(csv_path))

        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
            self.assertEqual(summary["scenario_name"], "cologne1")
            self.assertEqual(summary["total_steps"], 31)

    def test_run_ingolstadt1_live_traci(self):
        cfg = get_scenario_config(
            dataset_name="ingolstadt1",
            gui=False,
            output_dir=self.test_output_dir,
            yellow_duration=3.0,
            min_green=5.0,
        )

        runner = SimulationRunner(cfg)
        self.assertEqual(len(runner.tls_managers), 1)
        tls_id = "cluster_274083968_cluster_1200364014_1200364088"
        self.assertIn(tls_id, runner.tls_managers)

        summary = runner.run(num_steps=30)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["scenario_name"], "ingolstadt1")
        self.assertEqual(summary["total_steps"], 30)

        # Verify output files
        self.assertTrue(os.path.isfile(summary["output_json"]))
        self.assertTrue(os.path.isfile(summary["output_csv"]))

    def test_run_cologne3_live_traci(self):
        cfg = get_scenario_config(
            dataset_name="cologne3",
            gui=False,
            output_dir=self.test_output_dir,
            yellow_duration=3.0,
            min_green=5.0,
        )

        runner = SimulationRunner(cfg)
        self.assertEqual(len(runner.tls_managers), 3)

        summary = runner.run(num_steps=20)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["scenario_name"], "cologne3")
        self.assertEqual(summary["total_steps"], 20)


if __name__ == "__main__":
    unittest.main()
