"""Unit tests for MetricsAlignedReward module."""

import math
import pytest
from src.rl.rewards import MetricsAlignedReward


def test_metrics_aligned_reward_initialization_defaults():
    """Verify default initialization values."""
    reward_fn = MetricsAlignedReward()
    config = reward_fn.get_config()

    assert config["scale"] == 1.0
    assert config["w_completed"] == 2.0
    assert config["w_queue"] == 0.05
    assert config["w_waiting"] == 0.02
    assert config["w_time_loss"] == 0.01
    assert config["w_switch"] == 0.5


def test_metrics_aligned_reward_invalid_weights():
    """Verify ValueError is raised on negative or non-finite weights."""
    with pytest.raises(ValueError):
        MetricsAlignedReward(scale=-1.0)

    with pytest.raises(ValueError):
        MetricsAlignedReward(w_completed=float("nan"))

    with pytest.raises(ValueError):
        MetricsAlignedReward(w_queue=float("inf"))

    with pytest.raises(ValueError):
        MetricsAlignedReward(w_switch=-0.1)


def test_empty_step_calculation():
    """Verify 0.0 reward when no steps were recorded."""
    reward_fn = MetricsAlignedReward()
    assert reward_fn.calculate_reward() == 0.0


def test_quadratic_queue_penalty():
    """Verify quadratic queue penalty penalizes unbalanced long queues heavier than balanced ones."""
    reward_fn1 = MetricsAlignedReward(w_completed=0.0, w_waiting=0.0, w_time_loss=0.0, w_switch=0.0, w_queue=1.0)
    reward_fn2 = MetricsAlignedReward(w_completed=0.0, w_waiting=0.0, w_time_loss=0.0, w_switch=0.0, w_queue=1.0)

    # Balanced: queue 5 on lane A, 5 on lane B (sum = 10, sum_sq = 25 + 25 = 50)
    reward_fn1.record_step(in_lane_queues={"lane_1": 5.0, "lane_2": 5.0})
    rew_balanced = reward_fn1.calculate_reward()

    # Unbalanced: queue 10 on lane A, 0 on lane B (sum = 10, sum_sq = 100 + 0 = 100)
    reward_fn2.record_step(in_lane_queues={"lane_1": 10.0, "lane_2": 0.0})
    rew_unbalanced = reward_fn2.calculate_reward()

    assert rew_balanced == -50.0
    assert rew_unbalanced == -100.0
    assert rew_unbalanced < rew_balanced  # More penalty for severe bottleneck


def test_throughput_reward_positive():
    """Verify completed vehicles give positive reward boost."""
    reward_fn = MetricsAlignedReward(
        w_completed=5.0,
        w_queue=0.0,
        w_waiting=0.0,
        w_time_loss=0.0,
        w_switch=0.0,
    )
    reward_fn.record_step(completed_vehicles=3)
    reward_fn.record_step(completed_vehicles=2)
    # Total completed = 5 -> reward = 5.0 * 5 = 25.0
    assert reward_fn.calculate_reward() == 25.0


def test_waiting_time_and_time_loss_penalties():
    """Verify waiting time and time loss contribute negative penalties."""
    reward_fn = MetricsAlignedReward(
        w_completed=0.0,
        w_queue=0.0,
        w_waiting=1.0,
        w_time_loss=0.5,
        w_switch=0.0,
    )
    # 2 steps: waiting = 10s and 20s (mean = 15s), time_loss = 4s and 6s (mean = 5s)
    reward_fn.record_step(waiting_time=10.0, time_loss=4.0)
    reward_fn.record_step(waiting_time=20.0, time_loss=6.0)

    expected = -(1.0 * 15.0 + 0.5 * 5.0)
    assert reward_fn.calculate_reward() == pytest.approx(expected)


def test_switch_penalty_applied_once():
    """Verify switch penalty is subtracted if a switch happened during interval."""
    reward_fn = MetricsAlignedReward(
        w_completed=0.0,
        w_queue=0.0,
        w_waiting=0.0,
        w_time_loss=0.0,
        w_switch=3.0,
    )
    reward_fn.record_step(is_switch=False)
    reward_fn.record_step(is_switch=True)
    reward_fn.record_step(is_switch=False)

    assert reward_fn.calculate_reward() == -3.0


def test_reset_and_reset_interval():
    """Verify reset clears internal step accumulations."""
    reward_fn = MetricsAlignedReward()
    reward_fn.record_step(
        in_lane_queues={"lane_1": 4.0},
        waiting_time=10.0,
        completed_vehicles=2,
        is_switch=True,
    )
    assert reward_fn.calculate_reward() != 0.0

    reward_fn.reset()
    assert reward_fn.calculate_reward() == 0.0
