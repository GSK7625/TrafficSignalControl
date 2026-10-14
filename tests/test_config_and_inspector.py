"""Unit tests for SimulationConfig, YAML loader, and dataset inspection."""

import os
import tempfile
import pytest

from src.config import SimulationConfig, load_yaml_config
from src.data_inspector import inspect_all_scenarios, inspect_scenario


class TestSimulationConfig:
    """Test suite for SimulationConfig."""

    def test_valid_config_cologne1(self):
        cfg = SimulationConfig(dataset_name="cologne1")
        assert cfg.dataset_name == "cologne1"
        assert cfg.step_length == 1.0
        assert cfg.min_green == 5.0
        assert cfg.yellow_duration == 3.0
        assert os.path.isfile(cfg.sumocfg_path)

    def test_invalid_step_length(self):
        with pytest.raises(ValueError, match="step_length must be positive"):
            SimulationConfig(dataset_name="cologne1", step_length=0.0)

    def test_invalid_decision_interval(self):
        with pytest.raises(ValueError, match="decision_interval must be positive"):
            SimulationConfig(dataset_name="cologne1", decision_interval=-1.0)

    def test_invalid_yellow_duration(self):
        with pytest.raises(ValueError, match="yellow_duration must be positive"):
            SimulationConfig(dataset_name="cologne1", yellow_duration=-1.0)

    def test_missing_dataset_raises_error(self):
        with pytest.raises(ValueError, match="Dataset 'non_existent_dataset' not found"):
            SimulationConfig(dataset_name="non_existent_dataset")


class TestYamlConfigLoader:
    """Test suite for load_yaml_config."""

    def test_missing_file_raises_error(self):
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_yaml_config("non_existent.yaml")

    def test_invalid_yaml_format(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("- item1\n- item2\n")
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="expected dictionary"):
                load_yaml_config(tmp_path)
        finally:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

    def test_load_existing_project_configs(self):
        exp_cfg = load_yaml_config("configs/experiments.yaml")
        assert isinstance(exp_cfg, dict)
        assert "scenarios" in exp_cfg
        assert "algorithms" in exp_cfg

        ql_cfg = load_yaml_config("configs/q_learning.yaml")
        assert isinstance(ql_cfg, dict)
        assert ql_cfg.get("algorithm") == "QLearning"


class TestDataInspector:
    """Test suite for dataset inspection functions."""

    def test_inspect_cologne1(self):
        info = inspect_scenario("cologne1")
        assert info.name == "cologne1"
        assert info.tls_count == 1
        assert len(info.tls_ids) == 1
        assert info.trips_or_vehicles_count > 0
        assert info.duration == 3600.0

    def test_inspect_ingolstadt1(self):
        info = inspect_scenario("ingolstadt1")
        assert info.name == "ingolstadt1"
        assert info.tls_count == 1
        assert len(info.tls_ids) == 1
        assert info.trips_or_vehicles_count > 0

    def test_inspect_missing_scenario(self):
        with pytest.raises(FileNotFoundError, match="Scenario directory not found"):
            inspect_scenario("non_existent_scen")

    def test_inspect_all_scenarios(self):
        all_info = inspect_all_scenarios()
        assert "cologne1" in all_info
        assert "ingolstadt1" in all_info
