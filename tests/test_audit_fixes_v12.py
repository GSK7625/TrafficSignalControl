"""Unit tests for twelfth audit fixes (Readiness and Robustness verification)."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET

from src.controllers.q_learning import QLearningController
from src.data_inspector import inspect_scenario
from src.evaluation.experiment_runner import (
    ExperimentConfig,
    compute_run_fingerprint,
    validate_run_metrics,
    run_single_experiment,
)
from src.rl.environment import TrafficSignalRLEnv


class TestAuditFixesV12(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_audit_v12_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_zero_demand_scenario_validation_fails(self):
        """Zero traffic demand must produce is_valid=False and service_status=ZERO_DEMAND."""
        dummy_metrics = {
            "total_steps": 60,
            "total_completed_vehicles": 0,
            "mean_waiting_time": 0.0,
            "mean_queue_length": 0.0,
        }
        is_valid, anomalies, flags = validate_run_metrics(dummy_metrics, total_demand=0, return_flags=True)
        self.assertFalse(is_valid)
        self.assertEqual(flags["service_status"], "ZERO_DEMAND")
        self.assertEqual(flags["data_validity"], "INVALID")
        self.assertTrue(any("Zero traffic demand" in a for a in anomalies))

    def test_fingerprint_includes_scenario_data_and_version(self):
        """Fingerprint must change when scenario data or configuration changes."""
        cfg1 = ExperimentConfig("ingolstadt1", "Fixed-Time", seed=42)
        cfg2 = ExperimentConfig("cologne1", "Fixed-Time", seed=42)
        fp1 = compute_run_fingerprint(cfg1)
        fp2 = compute_run_fingerprint(cfg2)
        self.assertNotEqual(fp1, fp2)

    def test_multi_intersection_rl_env_guardrail(self):
        """TrafficSignalRLEnv must reject multi-intersection scenarios when target_tls_id is omitted."""
        with self.assertRaises(ValueError) as ctx:
            TrafficSignalRLEnv(scenario_name="cologne3", use_tripinfo=False)
        self.assertIn("contains 3 traffic lights", str(ctx.exception))
        self.assertIn("requires single-intersection networks", str(ctx.exception))

    def test_flow_demand_counting(self):
        """Data inspector must parse flow attributes: number, period, vehsPerHour, probability."""
        data_dir = os.path.join(self.test_dir, "data")
        scen_dir = os.path.join(data_dir, "test_flow")
        os.makedirs(scen_dir, exist_ok=True)

        net_xml = """<net version="1.16" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            <edge id="e1" from="j1" to="j2"><lane id="e1_0" index="0" speed="13.9" length="100.0"/></edge>
            <junction id="j1" type="dead_end" x="0.0" y="0.0" incLanes=""/>
            <junction id="j2" type="traffic_light" x="100.0" y="0.0" incLanes="e1_0"/>
            <tlLogic id="j2" type="static" programID="0" offset="0">
                <phase duration="30" state="G"/>
                <phase duration="3" state="y"/>
            </tlLogic>
        </net>"""
        rou_xml = """<routes>
            <vType id="car" vClass="passenger"/>
            <flow id="f1" begin="0" end="100" number="25" from="e1" to="e1"/>
            <flow id="f2" begin="0" end="100" period="10" from="e1" to="e1"/>
            <trip id="t1" depart="1" from="e1" to="e1"/>
            <vehicle id="v1" depart="2"><route edges="e1"/></vehicle>
        </routes>"""
        cfg_xml = """<configuration>
            <input>
                <net-file value="test_flow.net.xml"/>
                <route-files value="test_flow.rou.xml"/>
            </input>
        </configuration>"""

        with open(os.path.join(scen_dir, "test_flow.net.xml"), "w") as f:
            f.write(net_xml)
        with open(os.path.join(scen_dir, "test_flow.rou.xml"), "w") as f:
            f.write(rou_xml)
        with open(os.path.join(scen_dir, "test_flow.sumocfg"), "w") as f:
            f.write(cfg_xml)

        info = inspect_scenario("test_flow", data_dir=data_dir)
        # Expected: 25 (f1: number=25) + 10 (f2: period=10 over 100s) + 1 (t1) + 1 (v1) = 37 vehicles
        self.assertEqual(info.trips_or_vehicles_count, 37)

    def test_q_learning_tie_break_strategy_configuration(self):
        """QLearningController must support configurable tie_break_strategy."""
        agent_first = QLearningController(num_actions=3, seed=42, eval_mode=True, tie_break_strategy="first")
        self.assertEqual(agent_first.tie_break_strategy, "first")
        self.assertEqual(agent_first.get_config()["tie_break_strategy"], "first")

        agent_random = QLearningController(num_actions=3, seed=42, eval_mode=True, tie_break_strategy="random")
        self.assertEqual(agent_random.tie_break_strategy, "random")
        self.assertEqual(agent_random.get_config()["tie_break_strategy"], "random")
