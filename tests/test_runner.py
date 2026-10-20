"""Unit tests for SimulationRunner and extract_green_phases function."""

import importlib.util
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from src.config import SimulationConfig, get_scenario_config
from src.controllers.base import ControlledLink
from src.controllers.fixed_time import FixedTimeController
from src.controllers.max_pressure import MaxPressureController
from src.controllers.q_learning import QLearningController
from src.data_inspector import PhaseInfo, get_window_demand
from src.phase_manager import SafePhaseManager
from src.runner import SimulationRunner, extract_green_phases


class TestExtractGreenPhases:
    """Test suite for extract_green_phases utility."""

    def test_extract_pure_green_phases(self):
        phases = [
            PhaseInfo(index=0, duration=30.0, state="GGGrrr"),
            PhaseInfo(index=1, duration=3.0, state="yyyrrr"),
            PhaseInfo(index=2, duration=30.0, state="rrrGGG"),
            PhaseInfo(index=3, duration=3.0, state="rrryyy"),
        ]
        greens = extract_green_phases("cologne1", "tl_1", phases)
        assert greens == ["GGGrrr", "rrrGGG"]

    def test_deduplicates_green_phases(self):
        phases = [
            PhaseInfo(index=0, duration=30.0, state="GGGrrr"),
            PhaseInfo(index=1, duration=3.0, state="yyyrrr"),
            PhaseInfo(index=2, duration=15.0, state="GGGrrr"),
        ]
        greens = extract_green_phases("cologne1", "tl_1", phases)
        assert greens == ["GGGrrr"]

    def test_fallback_when_all_phases_have_yellow(self):
        phases = [
            PhaseInfo(index=0, duration=30.0, state="GGGy"),
            PhaseInfo(index=1, duration=3.0, state="yyyy"),
        ]
        greens = extract_green_phases("cologne1", "tl_1", phases)
        assert greens == ["GGGy"]


class TestSimulationRunnerSetup:
    """Test suite for SimulationRunner command building, initialization, and lifecycle."""

    def test_build_sumo_cmd_headless(self):
        cfg = SimulationConfig(
            dataset_name="cologne1",
            begin_time=100.0,
            end_time=200.0,
            step_length=1.0,
            seed=123,
            gui=False,
        )
        runner = SimulationRunner(cfg)
        cmd = runner.build_sumo_cmd()

        assert "-c" in cmd
        assert str(cfg.sumocfg_path) in cmd
        assert "-b" in cmd and "100.0" in cmd
        assert "-e" in cmd and "200.0" in cmd
        assert "--step-length" in cmd and "1.0" in cmd
        assert "--seed" in cmd and "123" in cmd
        assert "--start" not in cmd

    def test_build_sumo_cmd_gui(self):
        cfg = SimulationConfig(
            dataset_name="cologne1",
            gui=True,
            gui_delay=50.0,
        )
        runner = SimulationRunner(cfg)
        cmd = runner.build_sumo_cmd()

        assert "--start" in cmd
        assert "--quit-on-end" in cmd
        assert "--delay" in cmd and "50.0" in cmd

    def test_step_without_start_raises_error(self):
        cfg = SimulationConfig(dataset_name="cologne1")
        runner = SimulationRunner(cfg)
        with pytest.raises(RuntimeError, match="SimulationRunner is not running"):
            runner.step()

    def test_runner_validates_fixed_time_controller_compatibility(self):
        cfg = SimulationConfig(dataset_name="cologne1", max_green=60.0)
        incompatible_ctrl = FixedTimeController(default_green_time=90.0)
        with pytest.raises(ValueError, match="exceeds max_green"):
            SimulationRunner(cfg, controller=incompatible_ctrl)

    def test_runner_selects_libsumo_when_headless(self):
        if importlib.util.find_spec("libsumo") is None:
            pytest.skip("libsumo is not installed in this environment")

        cfg = SimulationConfig(dataset_name="cologne1", gui=False, use_libsumo=True)
        runner = SimulationRunner(cfg)
        assert runner.is_libsumo is True
        assert runner.traci.__name__ == "libsumo"

    def test_runner_forces_traci_when_gui(self):
        cfg = SimulationConfig(dataset_name="cologne1", gui=True)
        runner = SimulationRunner(cfg)
        assert runner.is_libsumo is False
        assert runner.traci.__name__ == "traci"

    def test_runner_forces_traci_when_no_libsumo_requested(self):
        cfg = SimulationConfig(dataset_name="cologne1", gui=False, use_libsumo=False)
        runner = SimulationRunner(cfg)
        assert runner.is_libsumo is False
        assert runner.traci.__name__ == "traci"

    def test_save_run_details_default_is_false(self):
        cfg = SimulationConfig(dataset_name="cologne1")
        assert cfg.save_run_details is False
        runner = SimulationRunner(cfg)
        assert runner.config.save_run_details is False

    def test_runner_finalize_method(self):
        cfg = SimulationConfig(dataset_name="cologne1")
        runner = SimulationRunner(cfg)
        summary = runner.finalize()
        assert isinstance(summary, dict)
        assert "total_steps" in summary
        assert "scenario_name" in summary

    def test_observation_failure_is_not_a_zero_queue(self, mock_traci):
        runner = SimulationRunner(get_scenario_config("cologne1"))
        tls = runner.scenario_info.tls_ids[0]
        runner.tls_controlled_links[tls] = [ControlledLink(0, "in", "out")]
        runner.traci = mock_traci
        runner.traci.lane.getLastStepHaltingNumber.side_effect = RuntimeError("lane missing")
        with pytest.raises(RuntimeError, match="lane missing"):
            runner.get_observation()

    def test_original_mode_does_not_claim_control_of_tls(self):
        runner = SimulationRunner(get_scenario_config("cologne1", control_mode="original"))
        assert runner.get_controlled_tls_ids() == set()

    def test_primary_collision_api_counts_vehicle_involvements(self, mock_traci):
        runner = SimulationRunner(get_scenario_config("cologne1", control_mode="agent"))
        runner.traci = mock_traci
        runner.traci.simulation.getTime.return_value = 1.0
        runner.traci.simulation.getCollisions.return_value = [SimpleNamespace(collider="a", victim="b")]
        runner.metrics_collector.record_step = MagicMock()
        runner.is_running = True
        runner.step({})
        assert runner.total_collision_vehicle_involvements == 2
        assert runner.unique_colliding_vehicles == {"a", "b"}
        events = runner.safety_metadata()["safety_events"]
        assert len(events) == 1
        assert events[0]["type"] == "collision"
        assert events[0]["sim_time"] == 1.0
        assert events[0]["collider"] == "a"

    def test_finalize_is_idempotent_and_closes_on_accounting_error(self, monkeypatch, mock_traci):
        runner = SimulationRunner(get_scenario_config("cologne1"))
        mock_traci.vehicle.getIDList.return_value = ["a"]
        runner.traci = mock_traci
        runner.is_running = True
        assert runner.finalize()["running_vehicles_at_end"] == 1
        assert runner.finalize()["running_vehicles_at_end"] == 1
        runner.reset()
        runner.traci = mock_traci
        runner.is_running = True
        monkeypatch.setattr(
            "src.runner.build_vehicle_accounting",
            lambda **_: (_ for _ in ()).throw(RuntimeError("accounting failed")),
        )
        with pytest.raises(RuntimeError, match="accounting failed"):
            runner.finalize()
        assert runner.is_running is False

    def test_close_error_does_not_hide_simulation_error(self, mock_traci):
        runner = SimulationRunner(get_scenario_config("cologne1"))
        runner.traci = mock_traci
        runner.traci.start.side_effect = RuntimeError("simulation failed")
        with pytest.raises(RuntimeError, match="simulation failed"):
            runner.run()

    def test_runner_checks_q_action_count_before_simulation(self):
        with pytest.raises(ValueError, match="actions"):
            SimulationRunner(get_scenario_config("cologne1"), controller=QLearningController(99))

    def test_controller_cannot_expand_configured_tls_scope(self):
        config = get_scenario_config("cologne3")
        target = config.scenario_info.tls_ids[0]
        config.target_tls_id = target
        runner = SimulationRunner(config, controller=MaxPressureController())
        try:
            runner.start()
            runner.step()
            assert runner.get_controlled_tls_ids() == {target}
            other = next(tls for tls in runner.tls_managers if tls != target)
            with pytest.raises(ValueError, match="controlled"):
                runner.step({other: 0})
        finally:
            runner.close()

    @pytest.mark.parametrize("max_green", [None, 60])
    def test_runner_cannot_bypass_starvation_by_requesting_another_phase(self, mock_traci, max_green):
        runner = SimulationRunner(SimulationConfig("cologne1", control_mode="agent"))
        tls_id = next(iter(runner.tls_managers))
        mgr = SafePhaseManager(tls_id, ["Grr", "rGr", "rrG"], max_red=30, max_green=max_green)
        mgr.configure_links([ControlledLink(i, f"edge{i}_0", f"out{i}_0") for i in range(3)])
        mgr.update_queues({"edge2_0": 3})
        mgr.update(60)
        runner.tls_managers = {tls_id: mgr}
        runner._controlled_tls_ids = {tls_id}
        runner.traci, runner.is_running = mock_traci, True
        try:
            runner.step({tls_id: 1})
            assert mgr.get_target_phase() == 2
        finally:
            runner.close()

    def test_runner_accounts_only_demand_in_short_window(self):
        runner = SimulationRunner(get_scenario_config("cologne1", max_steps=10, use_tripinfo=True))
        summary = runner.run()
        expected = get_window_demand(runner.scenario_info, runner.config.begin_time, runner.sim_time)["scheduled_demand_in_window"]
        assert summary["scheduled_demand_in_window"] == expected
        assert summary["demand_denominator"] == expected
        assert summary["tripinfo_reconciliation_status"] == "MATCH"

    def test_retired_cyclic_mode_is_rejected_in_runner_override(self):
        with pytest.raises(ValueError, match="control_mode"):
            SimulationRunner(get_scenario_config("cologne1"), control_mode="cyclic")

