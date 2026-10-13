import unittest
from unittest.mock import MagicMock, patch
from src.config import get_scenario_config
from src.runner import SimulationRunner, extract_green_phases


class TestRunner(unittest.TestCase):
    def test_extract_green_phases_cologne1(self):
        cfg = get_scenario_config("cologne1")
        runner = SimulationRunner(cfg)
        tls_id = "cluster_357187_359543"
        self.assertIn(tls_id, runner.tls_managers)
        # cologne1 has 8 phases, all are distinct green patterns
        phases = runner.tls_managers[tls_id].green_phases
        self.assertEqual(len(phases), 8)

    def test_extract_green_phases_ingolstadt1(self):
        cfg = get_scenario_config("ingolstadt1")
        runner = SimulationRunner(cfg)
        tls_id = "cluster_274083968_cluster_1200364014_1200364088"
        self.assertIn(tls_id, runner.tls_managers)
        # ingolstadt1 has 6 phases in net.xml: 0, 2, 4 are Green; 1, 3, 5 are Yellow
        phases = runner.tls_managers[tls_id].green_phases
        self.assertEqual(len(phases), 3)

    def test_build_sumo_cmd_headless(self):
        cfg = get_scenario_config("cologne1", gui=False, seed=123)
        runner = SimulationRunner(cfg)
        cmd = runner.build_sumo_cmd()
        self.assertIn("-c", cmd)
        self.assertIn(cfg.sumocfg_path, cmd)
        self.assertIn("--seed", cmd)
        self.assertIn("123", cmd)
        self.assertNotIn("--start", cmd)

    def test_build_sumo_cmd_gui(self):
        cfg = get_scenario_config("cologne1", gui=True)
        runner = SimulationRunner(cfg)
        cmd = runner.build_sumo_cmd()
        self.assertIn("--start", cmd)
        self.assertIn("--quit-on-end", cmd)

        cfg_with_delay = get_scenario_config("cologne1", gui=True, gui_delay=50.0)
        runner_with_delay = SimulationRunner(cfg_with_delay)
        cmd_delay = runner_with_delay.build_sumo_cmd()
        self.assertIn("--delay", cmd_delay)
        self.assertIn("50.0", cmd_delay)

    @patch("src.runner.traci")
    def test_runner_start_step_close(self, mock_traci):
        cfg = get_scenario_config("cologne1", max_steps=5)
        runner = SimulationRunner(cfg)

        mock_traci.simulation.getTime.return_value = 25201.0
        mock_traci.simulation.getDepartedIDList.return_value = []
        mock_traci.simulation.getArrivedIDList.return_value = []
        mock_traci.vehicle.getIDList.return_value = []
        mock_traci.lane.getLastStepHaltingNumber.return_value = 0

        runner.start()
        self.assertTrue(runner.is_running)
        mock_traci.start.assert_called_once()

        step_res = runner.step()
        self.assertEqual(step_res.step, 1)
        mock_traci.simulationStep.assert_called_once()

        runner.close()
        self.assertFalse(runner.is_running)
        mock_traci.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
