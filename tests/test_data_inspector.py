import os
import unittest
from src.data_inspector import inspect_scenario, inspect_all_scenarios, ScenarioInfo


class TestDataInspector(unittest.TestCase):
    def setUp(self):
        self.data_dir = "data"

    def test_inspect_cologne1(self):
        info = inspect_scenario("cologne1", self.data_dir)
        self.assertIsInstance(info, ScenarioInfo)
        self.assertEqual(info.name, "cologne1")
        self.assertEqual(info.begin_time, 25200.0)
        self.assertEqual(info.end_time, 28800.0)
        self.assertEqual(info.duration, 3600.0)
        self.assertEqual(info.tls_count, 1)
        self.assertIn("cluster_357187_359543", info.tls_ids)
        self.assertEqual(len(info.phases_per_tls["cluster_357187_359543"]), 8)
        self.assertEqual(info.trips_or_vehicles_count, 2015)
        self.assertGreater(len(info.controlled_lanes_per_tls["cluster_357187_359543"]), 0)

    def test_inspect_ingolstadt1(self):
        info = inspect_scenario("ingolstadt1", self.data_dir)
        self.assertEqual(info.name, "ingolstadt1")
        self.assertEqual(info.begin_time, 57600.0)
        self.assertEqual(info.end_time, 61200.0)
        self.assertEqual(info.duration, 3600.0)
        self.assertEqual(info.tls_count, 1)
        tls_id = "cluster_274083968_cluster_1200364014_1200364088"
        self.assertIn(tls_id, info.tls_ids)
        self.assertEqual(len(info.phases_per_tls[tls_id]), 6)
        self.assertEqual(info.trips_or_vehicles_count, 1716)

    def test_inspect_cologne3(self):
        info = inspect_scenario("cologne3", self.data_dir)
        self.assertEqual(info.name, "cologne3")
        self.assertEqual(info.begin_time, 25200.0)
        self.assertEqual(info.end_time, 28800.0)
        self.assertEqual(info.tls_count, 3)
        self.assertEqual(info.trips_or_vehicles_count, 4494)

    def test_inspect_all_scenarios(self):
        all_info = inspect_all_scenarios(self.data_dir)
        self.assertGreaterEqual(len(all_info), 4)
        self.assertIn("cologne1", all_info)
        self.assertIn("ingolstadt1", all_info)
        self.assertIn("cologne3", all_info)
        self.assertIn("grid4x4", all_info)

    def test_inspect_nonexistent_scenario(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            inspect_scenario("non_existent_dataset", self.data_dir)


if __name__ == "__main__":
    unittest.main()
