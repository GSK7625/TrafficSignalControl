"""Unit tests for SimulationConfig, YAML loader, and dataset inspection."""

import os
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
        assert cfg.all_red_duration == 1.0
        assert os.path.isfile(cfg.sumocfg_path)

    @pytest.mark.parametrize("invalid_kwargs,error_match", [
        ({"step_length": 0.0}, "step_length must be positive"),
        ({"step_length": -1.0}, "step_length must be positive"),
        ({"decision_interval": -1.0}, "decision_interval must be positive"),
        ({"yellow_duration": -1.0}, "yellow_duration must be positive"),
        ({"yellow_duration": 0.0}, "yellow_duration must be positive"),
    ])
    def test_invalid_numerical_parameters(self, invalid_kwargs, error_match):
        with pytest.raises(ValueError, match=error_match):
            SimulationConfig(dataset_name="cologne1", **invalid_kwargs)

    @pytest.mark.parametrize("invalid_kwargs,error_match", [
        ({"min_green": 10.0, "max_green": 5.0}, "max_green must be >= min_green"),
        ({"max_green": -1.0}, "max_green must be positive"),
        ({"min_green": float("nan")}, "finite"),
        ({"max_green": float("nan")}, "finite"),
        ({"min_green": float("inf")}, "finite"),
    ])
    def test_invalid_green_time_constraints(self, invalid_kwargs, error_match):
        with pytest.raises(ValueError, match=error_match):
            SimulationConfig(dataset_name="cologne1", **invalid_kwargs)

    def test_missing_dataset_raises_error(self):
        with pytest.raises(ValueError, match="Dataset 'non_existent_dataset' not found"):
            SimulationConfig(dataset_name="non_existent_dataset")

    @pytest.mark.parametrize("options", [
        {"step_length": 0}, {"step_length": -1}, {"decision_interval": -1}, {"yellow_duration": 0},
    ])
    def test_config_rejects_invalid_inputs_once(self, options):
        with pytest.raises(ValueError):
            SimulationConfig("cologne1", **options)

    @pytest.mark.parametrize("dec_int,step_len,should_pass", [
        (5.0, 1.0, True),
        (2.0, 0.5, True),
        (1.0, 1.0, True),
        (5.0, 2.0, False),  # 2.5x -> must be rejected
        (0.5, 1.0, False),  # 0.5x < 1 -> must be rejected
        (3.0, 2.0, False),  # 1.5x -> must be rejected
    ])
    def test_decision_interval_step_length_multiple(self, dec_int, step_len, should_pass):
        if should_pass:
            cfg = SimulationConfig("cologne1", decision_interval=dec_int, step_length=step_len)
            assert cfg.decision_interval == dec_int
            assert cfg.step_length == step_len
        else:
            with pytest.raises(ValueError, match="must be a positive integer multiple"):
                SimulationConfig("cologne1", decision_interval=dec_int, step_length=step_len)

    def test_config_rejects_negative_max_steps(self):
        with pytest.raises(ValueError, match="max_steps must be non-negative"):
            SimulationConfig("cologne1", max_steps=-5)



class TestYamlConfigLoader:
    """Test suite for load_yaml_config."""

    def test_missing_file_raises_error(self):
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_yaml_config("non_existent.yaml")

    def test_invalid_yaml_format(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected dictionary"):
            load_yaml_config(str(bad_yaml))

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

    def test_simulation_config_custom_sumocfg_inspects_custom_config(self, tmp_path):
        from src.config import SimulationConfig
        custom_rou = tmp_path / "custom.rou.xml"
        custom_rou.write_text("<routes></routes>", encoding="utf-8")

        real_net = os.path.abspath("data/cologne1/cologne1.net.xml").replace("\\", "/")
        custom_cfg = tmp_path / "custom.sumocfg"
        custom_cfg.write_text(f"""<configuration>
    <input>
        <net-file value="{real_net}"/>
        <route-files value="{custom_rou.name}"/>
    </input>
    <time>
        <begin value="100"/>
        <end value="200"/>
    </time>
</configuration>""", encoding="utf-8")

        cfg = SimulationConfig(dataset_name="cologne1", sumocfg_path=str(custom_cfg))
        assert cfg.scenario_info.trips_or_vehicles_count == 0
        assert cfg.scenario_info.begin_time == 100.0
        assert cfg.scenario_info.end_time == 200.0
        assert cfg.begin_time == 100.0
        assert cfg.end_time == 200.0

    def test_inspect_scenario_missing_net_file_raises_value_error(self, tmp_path):
        cfg = tmp_path / "test.sumocfg"
        cfg.write_text("<configuration><input></input></configuration>", encoding="utf-8")
        with pytest.raises(ValueError, match="net-file"):
            inspect_scenario("test", cfg_path=str(cfg))
