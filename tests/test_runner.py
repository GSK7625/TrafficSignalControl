"""Unit tests for SimulationRunner and extract_green_phases function."""

import pytest

from src.config import SimulationConfig
from src.data_inspector import PhaseInfo
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
        # Should filter out 'yyyrrr' and 'rrryyy'
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
        # Fallback keeps phase with fewer yellow than green: "GGGy" (1 y, 3 G)
        assert greens == ["GGGy"]


class TestSimulationRunnerSetup:
    """Test suite for SimulationRunner command building and initialization."""

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
        from src.controllers.fixed_time import FixedTimeController
        cfg = SimulationConfig(dataset_name="cologne1", max_green=60.0)
        incompatible_ctrl = FixedTimeController(default_green_time=90.0)
        with pytest.raises(ValueError, match="exceeds max_green"):
            SimulationRunner(cfg, controller=incompatible_ctrl)

    def test_runner_selects_libsumo_when_headless(self):
        import importlib.util
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


