"""Unit tests for FixedTime, MaxPressure, and QLearning controllers."""

import json
import os
from types import SimpleNamespace
import numpy as np
import pytest

from src.controllers import make_controller, FixedTimeController, MaxPressureController, QLearningController
from src.controllers.base import ControlledLink, IntersectionObservation
from src.controllers.max_pressure import compute_phase_pressure


def create_dummy_observation(
    tls_id: str = "tls_1",
    current_phase: int = 0,
    time_in_phase: float = 10.0,
    is_in_transition: bool = False,
    green_phases: list = None,
    controlled_links: list = None,
    in_lane_queues: dict = None,
    out_lane_queues: dict = None,
    sim_time: float = 100.0,
    step: int = 100,
    valid_actions: list = None,
) -> IntersectionObservation:
    """Helper to create a standard IntersectionObservation for testing."""
    if green_phases is None:
        green_phases = ["GGGrrr", "rrrGGG"]
    if controlled_links is None:
        controlled_links = [
            ControlledLink(signal_index=0, from_lane="in_0", to_lane="out_0"),
            ControlledLink(signal_index=1, from_lane="in_0", to_lane="out_1"),
            ControlledLink(signal_index=2, from_lane="in_1", to_lane="out_2"),
            ControlledLink(signal_index=3, from_lane="in_2", to_lane="out_3"),
            ControlledLink(signal_index=4, from_lane="in_2", to_lane="out_4"),
            ControlledLink(signal_index=5, from_lane="in_3", to_lane="out_5"),
        ]
    if in_lane_queues is None:
        in_lane_queues = {"in_0": 0.0, "in_1": 0.0, "in_2": 0.0, "in_3": 0.0}
    if out_lane_queues is None:
        out_lane_queues = {"out_0": 0.0, "out_1": 0.0, "out_2": 0.0, "out_3": 0.0, "out_4": 0.0, "out_5": 0.0}

    return IntersectionObservation(
        tls_id=tls_id,
        current_phase=current_phase,
        time_in_phase=time_in_phase,
        is_in_transition=is_in_transition,
        green_phases=green_phases,
        controlled_links=controlled_links,
        in_lane_queues=in_lane_queues,
        out_lane_queues=out_lane_queues,
        in_lane_vehicles={k: v for k, v in in_lane_queues.items()},
        out_lane_vehicles={k: v for k, v in out_lane_queues.items()},
        sim_time=sim_time,
        step=step,
        valid_actions=valid_actions,
    )


class TestFixedTimeController:
    """Test suite for FixedTimeController."""

    def test_invalid_default_green_time(self):
        with pytest.raises(ValueError, match="default_green_time must be positive"):
            FixedTimeController(default_green_time=0.0)

    def test_normal_cycle_progression(self):
        controller = FixedTimeController(default_green_time=30.0)
        obs = create_dummy_observation(current_phase=0, time_in_phase=10.0)

        # Before 30s elapsed: holds current phase 0
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 0

        # At 30s: cycles to next phase (phase 1)
        obs.time_in_phase = 30.0
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 1

        # While transitioning to phase 1: keeps requesting target phase 1
        obs.is_in_transition = True
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 1

    def test_custom_sequence(self):
        controller = FixedTimeController(
            default_green_time=20.0,
            phase_sequences={"tls_1": [1, 0]},
        )
        obs = create_dummy_observation(current_phase=1, time_in_phase=0.0)
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 1

        obs.time_in_phase = 15.0
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 1

        obs.time_in_phase = 20.0
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 0

    def test_schedule_resynchronization_on_forced_phase_switch(self):
        controller = FixedTimeController(default_green_time=90.0)
        obs0 = create_dummy_observation(current_phase=0, time_in_phase=59.0)
        actions0 = controller.select_action({"tls_1": obs0})
        assert actions0["tls_1"] == 0

        obs1 = create_dummy_observation(current_phase=1, time_in_phase=0.0, is_in_transition=False)
        actions1 = controller.select_action({"tls_1": obs1})
        assert actions1["tls_1"] == 1

        obs1_5 = create_dummy_observation(current_phase=1, time_in_phase=5.0, is_in_transition=False)
        actions1_5 = controller.select_action({"tls_1": obs1_5})
        assert actions1_5["tls_1"] == 1

    def test_compatibility_validation_with_max_and_min_green(self):
        controller = FixedTimeController(default_green_time=90.0)
        errors = controller.validate_compatibility(min_green=5.0, max_green=60.0)
        assert len(errors) > 0
        assert any("exceeds max_green" in e for e in errors)

        controller_min = FixedTimeController(default_green_time=2.0)
        errors_min = controller_min.validate_compatibility(min_green=5.0, max_green=60.0)
        assert len(errors_min) > 0
        assert any("less than min_green" in e for e in errors_min)

        controller_seq = FixedTimeController(phase_sequences={"tls_1": [0, 3]})
        errors_seq = controller_seq.validate_compatibility(num_phases_per_tls={"tls_1": 2})
        assert any("out of bounds" in e or "invalid" in e for e in errors_seq)


class TestMaxPressureController:
    """Test suite for MaxPressureController and compute_phase_pressure."""

    def test_compute_phase_pressure_empty_inputs(self):
        assert compute_phase_pressure("", [], {}, {}) == 0.0

    def test_compute_phase_pressure_math(self):
        links = [
            ControlledLink(signal_index=0, from_lane="in_A", to_lane="out_A"),
            ControlledLink(signal_index=1, from_lane="in_A", to_lane="out_B"),
            ControlledLink(signal_index=2, from_lane="in_C", to_lane="out_C"),
        ]
        in_q = {"in_A": 10.0, "in_C": 4.0}
        out_q = {"out_A": 2.0, "out_B": 0.0, "out_C": 1.0}

        p0 = compute_phase_pressure("GGr", links, in_q, out_q)
        assert pytest.approx(p0, 1e-6) == 8.0

        p1 = compute_phase_pressure("rrG", links, in_q, out_q)
        assert pytest.approx(p1, 1e-6) == 3.0

    def test_min_green_constraint_held(self):
        controller = MaxPressureController(min_green=5.0)
        in_q = {"in_0": 0.0, "in_1": 0.0, "in_2": 20.0, "in_3": 20.0}
        obs = create_dummy_observation(
            current_phase=0, time_in_phase=3.0, in_lane_queues=in_q
        )
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 0

    def test_switches_to_max_pressure_phase(self):
        controller = MaxPressureController(min_green=5.0)
        in_q = {"in_0": 1.0, "in_1": 1.0, "in_2": 8.0, "in_3": 7.0}
        obs = create_dummy_observation(
            current_phase=0, time_in_phase=6.0, in_lane_queues=in_q
        )
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 1

    def test_tie_breaking_keeps_current_phase(self):
        controller = MaxPressureController(min_green=5.0)
        obs = create_dummy_observation(current_phase=0, time_in_phase=6.0)
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 0

    def test_compute_phase_pressure_with_s_signal(self):
        links = [
            ControlledLink(signal_index=0, from_lane="in_A", to_lane="out_A"),
            ControlledLink(signal_index=1, from_lane="in_B", to_lane="out_B"),
        ]
        in_q = {"in_A": 6.0, "in_B": 8.0}
        out_q = {"out_A": 1.0, "out_B": 2.0}

        p_s = compute_phase_pressure("sr", links, in_q, out_q)
        assert pytest.approx(p_s, 1e-6) == 5.0

        p_G = compute_phase_pressure("Gr", links, in_q, out_q)
        assert pytest.approx(p_s, 1e-6) == p_G


class TestQLearningController:
    """Test suite for QLearningController."""

    def test_initialization_validation(self):
        with pytest.raises(ValueError, match="num_actions must be positive"):
            QLearningController(num_actions=0)

        with pytest.raises(ValueError, match="learning_rate must be in"):
            QLearningController(num_actions=2, learning_rate=0.0)

        with pytest.raises(ValueError, match="gamma must be in"):
            QLearningController(num_actions=2, gamma=-0.1)

    def test_action_selection_epsilon_greedy(self):
        ctrl = QLearningController(num_actions=3, epsilon=0.0, eval_mode=True, tie_break_strategy="first")
        state = (0, 1, 2, 1)
        ctrl.q_table[state] = np.array([0.5, 2.5, 1.0], dtype=np.float64)

        action = ctrl.select_action_for_state(state)
        assert action == 1

        action_masked = ctrl.select_action_for_state(state, valid_actions=[0, 2])
        assert action_masked == 2

    def test_tie_breaking_prefers_current_phase(self):
        ctrl = QLearningController(num_actions=3, epsilon=0.0, eval_mode=True)
        state = (1, 0, 0, 1)
        ctrl.q_table[state] = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        assert ctrl.select_action_for_state(state, current_phase=1) == 1
        assert ctrl.select_action_for_state(state, current_phase=2) == 2
        assert ctrl.select_action_for_state(state, current_phase=None) == 0

    def test_bellman_update(self):
        ctrl = QLearningController(num_actions=2, learning_rate=0.5, gamma=0.9, epsilon=0.0)
        s0 = (0, 0, 0, 1)
        s1 = (1, 0, 0, 1)

        ctrl.q_table[s0] = np.array([0.0, 0.0], dtype=np.float64)
        ctrl.q_table[s1] = np.array([10.0, 5.0], dtype=np.float64)

        td_error = ctrl.update(state=s0, action=0, reward=2.0, next_state=s1, done=False)
        assert pytest.approx(ctrl.q_table[s0][0], 1e-6) == 5.5
        assert pytest.approx(td_error, 1e-6) == 11.0

        td_term = ctrl.update(state=s0, action=1, reward=4.0, next_state=s1, done=True)
        assert pytest.approx(ctrl.q_table[s0][1], 1e-6) == 2.0
        assert pytest.approx(td_term, 1e-6) == 4.0

    def test_epsilon_decay(self):
        ctrl = QLearningController(num_actions=2, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
        assert ctrl.decay_epsilon() == 0.5
        assert ctrl.decay_epsilon() == 0.25
        assert ctrl.decay_epsilon() == 0.125
        assert ctrl.decay_epsilon() == 0.1

    def test_save_and_load(self, tmp_path):
        ctrl = QLearningController(num_actions=2, learning_rate=0.2, gamma=0.95)
        s = (0, 1, 1, 1)
        ctrl.q_table[s] = np.array([1.23, 4.56], dtype=np.float64)
        ctrl.total_updates = 42

        model_file = str(tmp_path / "model.json")
        ctrl.save(model_file)
        assert os.path.isfile(model_file)

        new_ctrl = QLearningController(num_actions=2)
        new_ctrl.load(model_file)

        assert new_ctrl.num_actions == 2
        assert new_ctrl.learning_rate == 0.2
        assert new_ctrl.gamma == 0.95
        assert new_ctrl.total_updates == 42
        assert s in new_ctrl.q_table
        assert pytest.approx(new_ctrl.q_table[s][0], 1e-6) == 1.23
        assert pytest.approx(new_ctrl.q_table[s][1], 1e-6) == 4.56

    def test_from_checkpoint(self, tmp_path):
        ctrl = QLearningController(num_actions=3, min_green=7.0, eval_mode=False)
        s = (1, 0, 2, 1)
        ctrl.q_table[s] = np.array([0.5, 1.5, 2.5], dtype=np.float64)

        model_file = str(tmp_path / "model.json")
        ctrl.save(model_file)

        loaded = QLearningController.from_checkpoint(model_file, eval_mode=True)
        assert loaded.num_actions == 3
        assert loaded.min_green == 7.0
        assert loaded.eval_mode is True
        assert loaded.epsilon == 0.0
        assert s in loaded.q_table
        assert pytest.approx(loaded.q_table[s][2], 1e-6) == 2.5

    def test_checkpoint_shape_is_checked_at_load(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"config": {"num_actions": 3}, "q_table": {"(0, 0, 0, 1)": [10]}}))
        with pytest.raises(ValueError, match="shape"):
            QLearningController.from_checkpoint(str(path))

    def test_checkpoint_rejects_unknown_override_and_changed_state_encoding(self, tmp_path):
        path = str(tmp_path / "agent.json")
        QLearningController(2).save(path)
        with pytest.raises(TypeError):
            QLearningController.from_checkpoint(path, min_grean=9)
        with pytest.raises(ValueError, match="discretizer"):
            QLearningController.from_checkpoint(path, min_green=9)

    def test_q_cli_respects_the_environment_action_mask(self):
        agent = QLearningController(3, eval_mode=True)
        agent.q_table[(0, 0, 0, 2)] = np.array([100.0, 1.0, 50.0])
        obs = SimpleNamespace(
            current_phase=0,
            time_in_phase=60.0,
            is_in_transition=False,
            green_phases=["Grr", "rGr", "rrG"],
            controlled_links=[],
            in_lane_queues={},
            valid_actions=[1, 2],
        )
        assert agent.select_action({"tls": obs}) == {"tls": 2}

    @pytest.mark.parametrize("actions", [[], [999]])
    def test_empty_or_invalid_action_mask_is_not_expanded(self, actions):
        with pytest.raises(ValueError):
            QLearningController(2, eval_mode=True).select_action_for_state((0, 0, 0, 0), actions)

    def test_load_checkpoint_restores_optional_target(self, tmp_path):
        path = str(tmp_path / "agent.json")
        original = QLearningController(num_actions=3, target_tls_id="junction-2")
        original.save(path)
        loaded = QLearningController(num_actions=2)
        loaded.load(path)
        assert loaded.target_tls_id == "junction-2"
        assert loaded.num_actions == 3
        assert loaded.action_counts.shape == (3,)

    def test_checkpoint_override_can_disable_max_green(self, tmp_path):
        path = str(tmp_path / "agent.json")
        QLearningController(num_actions=2).save(path)
        assert QLearningController.from_checkpoint(path, max_green=None).max_green is None

    def test_legacy_checkpoint_uses_safety_and_discretizer_defaults(self, tmp_path):
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps({"config": {"num_actions": 2, "min_green": 7}, "q_table": {}}))
        agent = QLearningController.from_checkpoint(str(path))
        assert agent.max_green == 60.0
        assert agent.discretizer.min_green == 7.0

    def test_checkpoint_metadata_keeps_explicit_disabled_max_green(self, tmp_path):
        from src.evaluation.evaluate_rl import checkpoint_env_params
        path = str(tmp_path / "agent.json")
        QLearningController(num_actions=2, max_green=None).save(path)
        assert checkpoint_env_params(QLearningController.from_checkpoint(path).get_config())["max_green"] is None

    @pytest.mark.parametrize("bad_cfg,match", [
        ({"learning_rate": -1.0}, "learning_rate"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"gamma": 2.0}, "gamma"),
        ({"gamma": -0.1}, "gamma"),
        ({"learning_rate": float("nan")}, "finite"),
        ({"gamma": float("nan")}, "finite"),
        ({"num_actions": 0}, "num_actions"),
        ({"min_green": -5.0}, "min_green"),
        ({"min_green": 10.0, "max_green": 5.0}, "max_green"),
    ])
    def test_checkpoint_loading_rejects_invalid_hyperparameters(self, tmp_path, bad_cfg, match):
        base_cfg = {
            "num_actions": 2,
            "learning_rate": 0.1,
            "gamma": 0.95,
            "epsilon": 0.1,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "min_green": 5.0,
            "max_green": 60.0,
            "yellow_duration": 3.0,
            "all_red_duration": 1.0,
            "decision_interval": 5.0,
            "tie_break_strategy": "first",
        }
        base_cfg.update(bad_cfg)
        path = tmp_path / "bad_checkpoint.json"
        path.write_text(json.dumps({"config": base_cfg, "q_table": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            QLearningController.from_checkpoint(str(path))



class TestMakeController:
    """Test suite for make_controller factory function."""

    def test_make_controller_fixed_time(self):
        ctrl, mode = make_controller("fixed-time", "cologne1", default_green_time=25.0)
        assert isinstance(ctrl, FixedTimeController)
        assert ctrl.default_green_time == 25.0
        assert mode == "agent"

    def test_make_controller_max_pressure(self):
        ctrl, mode = make_controller("max-pressure", "cologne1", min_green=8.0)
        assert isinstance(ctrl, MaxPressureController)
        assert ctrl.min_green == 8.0
        assert mode == "agent"

    def test_make_controller_none(self):
        ctrl, mode = make_controller("none", "cologne1")
        assert ctrl is None
        assert mode == "original"

    def test_make_controller_invalid_raises_error(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            make_controller("invalid-algo", "cologne1")

    def test_factory_does_not_silently_create_untrained_policy(self):
        with pytest.raises(FileNotFoundError):
            make_controller("q-learning", "missing-dataset", num_actions=6)

    @pytest.mark.parametrize("algorithm", ["max-pressure", "q-learning"])
    def test_factory_distinguishes_default_and_disabled_max_green(self, algorithm, tmp_path):
        path = str(tmp_path / "agent.json")
        QLearningController(2).save(path)
        default, _ = make_controller(algorithm, model_path=path)
        disabled, _ = make_controller(algorithm, model_path=path, max_green=None)
        assert default.max_green == 60.0
        assert disabled.max_green is None

    @pytest.mark.parametrize("name", ["ql", "Q Learning", "q_learning", "QLearning"])
    def test_algorithm_aliases_use_same_factory_and_benchmark_path(self, name, tmp_path):
        from src.evaluation.experiment_runner import ExperimentConfig, resolve_effective_decision_interval, resolve_model_path
        path = str(tmp_path / "agent.json")
        QLearningController(2).save(path)
        agent, _ = make_controller(name, model_path=path)
        assert isinstance(agent, QLearningController)
        cfg = ExperimentConfig(scenario_name="cologne1", algorithm=name)
        assert resolve_effective_decision_interval(cfg) == 5.0
        assert resolve_model_path(cfg) == "models/q_learning/q_learning_cologne1_best.json"

    def test_resolve_q_learning_checkpoint_avoids_stale_best_when_validation_failed(self, tmp_path):
        from src.controllers import resolve_q_learning_checkpoint
        model_dir = tmp_path / "models" / "q_learning"
        log_dir = tmp_path / "results" / "training"
        model_dir.mkdir(parents=True)
        log_dir.mkdir(parents=True)

        old_best = model_dir / "q_learning_cologne1_best.json"
        old_best.write_text('{"min_green": 5}', encoding="utf-8")

        new_final = model_dir / "q_learning_cologne1_final.json"
        new_final.write_text('{"min_green": 7}', encoding="utf-8")

        log_file = log_dir / "q_learning_cologne1_log.json"
        log_file.write_text(json.dumps({
            "scenario_name": "cologne1",
            "best_is_viable": False,
            "best_checkpoint_path": None,
            "final_checkpoint_path": str(new_final),
        }), encoding="utf-8")

        resolved = resolve_q_learning_checkpoint("cologne1", log_dir=str(log_dir), model_dir=str(model_dir))
        from pathlib import Path
        assert Path(resolved) == new_final

        new_final.unlink()
        with pytest.raises(ValueError, match="no viable best checkpoint"):
            resolve_q_learning_checkpoint("cologne1", log_dir=str(log_dir), model_dir=str(model_dir))

    def test_fixed_time_validate_compatibility_checks_phase_durations(self):
        ctrl = FixedTimeController(
            default_green_time=30.0,
            phase_durations={"tls_1": {0: 90.0, 1: 3.0, 2: float("nan")}},
        )
        errors = ctrl.validate_compatibility(min_green=5.0, max_green=60.0)
        assert any("exceeds max_green" in err for err in errors)
        assert any("less than min_green" in err for err in errors)
        assert any("finite" in err for err in errors)
